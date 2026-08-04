"""Controller：五线程模型接线（回调→队列→DSP线程→响应）。纯逻辑可用 feed_chunk 测试。"""
from __future__ import annotations

import random
import sys
import threading
import time
from pathlib import Path

from audio import AudioIn, AudioOut
from config import Config, load_config
from detectors import DETECTOR_CLASSES
from dsp import AudioFilter, Baseline, FeatureExtractor
from engine import EventManager, MuteGate, ResponseEngine, ScheduleManager
import paths


def _pick_sound(sounds_dir: str) -> str | None:
    """从反馈音目录随机选一个（wav/mp3/flac）。无音效时返回 None。"""
    d = Path(sounds_dir)
    if not d.exists():
        return None
    files = [p for p in d.iterdir() if p.suffix.lower() in (".wav", ".mp3", ".flac")]
    return str(random.choice(files)) if files else None


class Controller:
    """把 滤波→特征→标定→检测→事件→时间窗→响应 串成一条 DSP 流水线。"""

    def __init__(self, cfg: Config, on_feature=None, on_log=None):
        self.cfg = cfg
        self.on_feature = on_feature
        self.on_log = on_log
        self.log_path: str | None = None        # 设置后所有日志同时写入该文件
        self.sample_rate = cfg.sample_rate
        self.filter = AudioFilter(self.sample_rate,
                                  lowpass_hz=5000.0 if self.sample_rate >= 16000 else 3500.0)
        self.extractor = FeatureExtractor(self.sample_rate,
                                          cfg.short_window_ms, cfg.long_window_ms)
        self.baseline = Baseline(self.sample_rate)
        # 优先级 = 类顺序倒排（Impact=6 最高 … Chair=1 最低），与设计 §九 一致
        n = len(DETECTOR_CLASSES)
        self.detectors = [c(c.__name__.replace("Detector", ""), n - i,
                            self.sample_rate, cfg.short_window_ms)
                          for i, c in enumerate(DETECTOR_CLASSES)]
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
        self.em.update(feat)
        for trig in self.em.take_triggers():
            verdict = self.schedule.decide(feat.ts)
            if verdict.allowed:
                self.response.handle(trig, feat.ts, set_muted=self.audio_in.set_muted)
                self.schedule.record_response(feat.ts)
            else:
                self._log(f"事件 {trig.detector} 未响应：{verdict.reason}")
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
        self._log(f"播放 {Path(path).name}")
        return self.audio_out.play(path)

    def _log(self, msg: str) -> None:
        if self.on_log:
            self.on_log(msg)
        if self.log_path:
            line = f"{time.strftime('%H:%M:%S')} {msg}"
            p = Path(self.log_path)
            p.parent.mkdir(parents=True, exist_ok=True)   # logs/ 目录可能不存在
            with open(p, "a", encoding="utf-8") as f:
                f.write(line + "\n")


def main() -> int:
    cfg = load_config()
    ctrl = Controller(cfg, on_log=print)
    # 日志同时写文件（GUI 模式与打包 --windowed 也生效）
    ctrl.log_path = str(paths.app_dir() / "logs" / "app.log")
    ctrl._log(f"采样率 {cfg.sample_rate}Hz，短窗 {cfg.short_window_ms}ms")
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
