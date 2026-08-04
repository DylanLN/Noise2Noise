"""Controller：五线程模型接线（回调→队列→DSP线程→响应）。纯逻辑可用 feed_chunk 测试。"""
from __future__ import annotations

import random
import sys
import threading
import time
from pathlib import Path

from audio import AudioIn, AudioOut, SimAudioIn
from config import Config, load_config
from detectors import DETECTOR_CLASSES, LoudnessDetector
from dsp import AudioFilter, Baseline, FeatureExtractor
from engine import EventManager, MuteGate, ResponseEngine, ScheduleManager
import paths

_REASON_CN = {
    "outside_window": "不在活跃时段",
    "not_fresh": "距上次响应未满冷却",
    "max_responses": "已达本时段响应上限",
    "manual_pending": "等待人工确认",
}


def _pick_sound(sounds_dir: str) -> str | None:
    """从反馈音目录递归随机选一个（wav/mp3/flac，含子目录如 sounds/default/）。"""
    d = Path(sounds_dir)
    if not d.exists():
        return None
    files = [p for p in d.rglob("*") if p.is_file()
             and p.suffix.lower() in (".wav", ".mp3", ".flac")]
    return str(random.choice(files)) if files else None


class Controller:
    """把 滤波→特征→标定→检测→事件→时间窗→响应 串成一条 DSP 流水线。"""

    def __init__(self, cfg: Config, on_feature=None, on_log=None, sim_mode: bool = False):
        self.cfg = cfg
        self.on_feature = on_feature
        self.on_log = on_log
        self.sim_mode = sim_mode                # 无音频硬件时用模拟输入测试检测链路
        self.log_path: str | None = None        # 设置后所有日志同时写入该文件
        self.sample_rate = cfg.sample_rate
        self.filter = AudioFilter(self.sample_rate,
                                  lowpass_hz=5000.0 if self.sample_rate >= 16000 else 3500.0)
        self.extractor = FeatureExtractor(self.sample_rate,
                                          cfg.short_window_ms, cfg.long_window_ms)
        self.baseline = Baseline(self.sample_rate)
        # 优先级 = 类顺序倒排（Impact=6 最高 … Loudness=0 最低），与设计 §九 一致
        n = len(DETECTOR_CLASSES)
        self.detectors = []
        for i, c in enumerate(DETECTOR_CLASSES):
            name = c.__name__.replace("Detector", "")
            if c is LoudnessDetector:
                # 响度触发要抓短促敲击（1-2 帧），用快速滞回；其余检测器保持默认
                self.detectors.append(c(name, n - i, self.sample_rate,
                                        cfg.short_window_ms, n1=1, n2=1, n3=5))
            else:
                self.detectors.append(c(name, n - i, self.sample_rate, cfg.short_window_ms))
        self.loud = next(d for d in self.detectors if isinstance(d, LoudnessDetector))
        self.em = EventManager(
            self.detectors,
            arbitration_window_ms=cfg.arbitration_window_ms,
            default_confirm_count=cfg.confirm_count,
            default_confirm_window_sec=cfg.confirm_window_sec,
            on_episode=lambda name, count, target:
                self._log(f"Episode[{name}] {count}/{target}"))
        self.schedule = ScheduleManager(cfg)
        self.gate = MuteGate(cfg.mute_ignore_ms, measured_latency_ms=None)
        self.sounds = str(paths.sounds_dir(cfg.sounds_dir))
        self.audio_out = AudioOut(device=cfg.output_device or None, volume=cfg.volume)
        self.response = ResponseEngine(
            cfg, self.gate, self._play,
            on_respond=lambda tr, plan: self._log(f"响应 {tr.detector}"),
            on_log=self._log)
        self.audio_in = AudioIn(device=cfg.input_device or None,
                                sample_rate=self.sample_rate)
        self._running = False
        self._thread: threading.Thread | None = None

    # ── 生命周期 ──
    def start(self) -> None:
        self._log("开始监听")
        if self.sim_mode:
            self._log("模拟输入模式：无音频硬件，用生成的脉冲测试检测链路")
            self.audio_in = SimAudioIn(sample_rate=self.sample_rate,
                                       chunk_size=self.extractor.short_n)
        self.audio_in.start()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.audio_in.stop()
        self._log("停止监听")

    def _loop(self) -> None:
        while self._running:
            chunk = self.audio_in.get(timeout=0.1)
            if chunk is not None:
                self.process(chunk)

    # ── 处理 ──
    def process(self, samples) -> Feature | None:
        """单帧处理（DSP 线程调用；测试可直接喂）。"""
        clean = self.filter.process(samples)
        feat = self.extractor.push(clean)
        if not self.baseline.calibrated:
            self.baseline.feed(feat.rms, feat.low_energy_ratio, feat.ts)
            if self.on_feature:
                self.on_feature(feat)              # 标定期也推送，让音量条有反应
            return None
        trig_ratio = self.em.non_idle_ratio()
        self.baseline.feed(feat.rms, feat.low_energy_ratio, feat.ts, trig_ratio)
        self.extractor.set_baseline(self.baseline.baseline_rms,
                                    self.baseline.baseline_low_ratio)
        self.extractor.set_rms_threshold(self.baseline.threshold(self.cfg.sensitivity))
        self.extractor.set_peak_threshold(self.baseline.peak_threshold(
            self.cfg.sensitivity, floor=1.0))
        # 响度触发器阈值随容忍度更新；并刷新归一化 RMS（用最新 baseline）
        self.loud.set_rms_norm_min(self.baseline.threshold_norm(self.cfg.sensitivity))
        feat.rms_norm = feat.rms / max(self.baseline.baseline_rms, 1e-9)
        self.em.update(feat)
        for trig in self.em.take_triggers():
            now = time.time()                        # 时间窗口/冷却用真实时钟
            verdict = self.schedule.decide(now)
            if verdict.allowed:
                self.response.handle(trig, now, set_muted=self.audio_in.set_muted)
                self.schedule.record_response(now)
            else:
                self._log(f"事件 {trig.detector} 未响应：{_REASON_CN.get(verdict.reason, verdict.reason)}")
        if self.on_feature:
            self.on_feature(feat)
        return feat

    def _play(self, trigger) -> float:
        # 指定了回击音文件 → 用它；否则从 sounds/ 随机
        fb = self.cfg.feedback_file.strip()
        path = fb if fb and Path(fb).exists() else _pick_sound(self.sounds)
        if path is None:
            self._log("无反馈音文件，跳过播放")
            return 0.0
        self._log(f"播放 {Path(path).name}{'（模拟，无音频硬件）' if self.sim_mode else ''}")
        if self.sim_mode:
            return 500.0                       # 模拟模式不实际出声
        return self.audio_out.play(path)

    def _log(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        if self.on_log:
            self.on_log(line)
        if self.log_path:
            p = Path(self.log_path)
            p.parent.mkdir(parents=True, exist_ok=True)   # logs/ 目录可能不存在
            with open(p, "a", encoding="utf-8") as f:
                f.write(line + "\n")


def main() -> int:
    cfg = load_config()
    ctrl = Controller(cfg, on_log=print)
    # 日志同时写文件（GUI 模式与打包 --windowed 也生效），并在窗口显示路径
    ctrl.log_path = str(paths.app_dir() / "logs" / "app.log")
    ctrl._log(f"采样率 {cfg.sample_rate}Hz，短窗 {cfg.short_window_ms}ms")
    ctrl._log(f"日志文件：{ctrl.log_path}")
    try:
        from PyQt6.QtWidgets import QApplication
        from gui import MainWindow
        app = QApplication(sys.argv)
        win = MainWindow(ctrl, cfg)
        win.show()
        return app.exec()
    except Exception as e:                           # GUI 不可用（无显示/缺依赖）→ 控制台
        ctrl._log(f"GUI 不可用，退回控制台模式：{e}")
        ctrl.start()
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            ctrl.stop()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
