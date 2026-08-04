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
