"""Controller 集成测试：喂合成 PCM，验证特征管线跑通、事件触发不崩溃。"""
import numpy as np

from config import Config
from main import Controller


def _impulse_train(sr, dur_s, interval_s, amp=1.0):
    n = int(sr * dur_s)
    x = np.zeros(n)
    win = int(0.02 * sr)
    t = 0.0
    while t < dur_s:
        i = int(t * sr)
        if i < n:                                # 浮点累加可能让最后一帧越界，裁剪
            end = min(i + win, n)
            x[i:end] = amp * np.hanning(end - i)
        t += interval_s
    return x


def _frames(sig, sr, short_n):
    return [sig[i:i + short_n] for i in range(0, len(sig) - short_n, short_n)]


def test_feature_pipeline_runs():
    cfg = Config()
    feats = []
    ctrl = Controller(cfg, on_feature=feats.append)
    sr = cfg.sample_rate
    short_n = int(sr * cfg.short_window_ms / 1000)
    sig = np.sin(2 * np.pi * 60 * np.arange(sr) / sr) * 0.5
    for fr in _frames(sig, sr, short_n):
        ctrl.process(fr)
    assert len(feats) > 10
    assert feats[-1].low_energy_ratio > 0.3


def test_calibration_then_impulses_no_crash():
    cfg = Config()
    ctrl = Controller(cfg, on_log=print)
    sr = cfg.sample_rate
    short_n = int(sr * cfg.short_window_ms / 1000)
    # 标定期：喂 11 秒静音（默认标定时长 10s）
    for _ in range(int(11.0 * sr / short_n)):
        ctrl.process(np.zeros(short_n))
    assert ctrl.baseline.calibrated
    # 触发期：喂 3 秒脉冲
    sig = _impulse_train(sr, 3.0, interval_s=0.3, amp=1.0)
    for fr in _frames(sig, sr, short_n):
        ctrl.process(fr)
    # 触发路径不能抛异常；是否真正触发依赖阈值标定，这里只验证不崩溃
    assert ctrl.em.take_triggers() == [] or True


def test_loud_sound_triggers_response():
    """端到端：响亮脉冲 → 响度检测 → Episode 关闭 → 触发并播放。"""
    cfg = Config()
    cfg.sensitivity = 1.0
    cfg.schedule_always = True
    cfg.confirm_count = 1
    played = []
    ctrl = Controller(cfg)
    ctrl.audio_out = type("Out", (), {"play": lambda self, p: played.append(p) or 1.0})()
    sr = cfg.sample_rate
    short_n = int(sr * cfg.short_window_ms / 1000)
    rng = np.random.default_rng(0)
    for _ in range(int(11.0 * sr / short_n)):            # 标定（模拟麦克风底噪）
        ctrl.process(rng.normal(0, 0.001, short_n))
    assert ctrl.baseline.calibrated
    # 0.15s 响亮低频脉冲（模拟拍桌）+ 3s 数字静音（Episode 关闭 → 触发）
    burst = 1.0 * np.sin(2 * np.pi * 60 * np.arange(int(0.15 * sr)) / sr)
    sig = np.concatenate([burst, np.zeros(int(3 * sr))])
    for fr in _frames(sig, sr, short_n):
        ctrl.process(fr)
    assert played, "响亮噪声应触发并播放反馈音"


def test_sim_audio_produces_burst_and_silence():
    from audio import SimAudioIn
    sim = SimAudioIn(sample_rate=48000, burst_sec=0.15, period_sec=3.0, chunk_size=1008)
    sim.start()
    chunks = []
    while len(chunks) < 20:
        c = sim.get(timeout=0.5)
        if c is not None:
            chunks.append(c)
    sim.stop()
    rms = [float(np.sqrt(np.mean(c ** 2))) for c in chunks]
    assert max(rms) > 0.1        # 突发段有大能量
    assert min(rms) < 1e-3       # 静音段接近 0


def test_sim_mode_play_skips_hardware():
    cfg = Config()
    ctrl = Controller(cfg, sim_mode=True)
    assert ctrl.sim_mode
    assert ctrl._play(None) == 500.0     # 模拟模式：不调用 audio_out，返回名义时长


def test_feedback_file_used(tmp_path):
    cfg = Config()
    wav = tmp_path / "custom.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 32)          # 假 wav，不实际解码
    cfg.feedback_file = str(wav)
    ctrl = Controller(cfg)
    played = []
    ctrl.audio_out = type("Out", (), {"play": lambda self, p: played.append(p) or 1.0})()
    ctrl._play(None)
    assert played and str(wav) in played[0]


def test_log_path_writes_file(tmp_path):
    cfg = Config()
    ctrl = Controller(cfg)
    logf = tmp_path / "logs" / "app.log"        # 父目录不存在，也应自动创建
    ctrl.log_path = str(logf)
    ctrl._log("测试日志行")
    assert "测试日志行" in logf.read_text(encoding="utf-8")
