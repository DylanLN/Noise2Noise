"""DSP 核心 sanity 测试。"""
import numpy as np

from dsp import AudioFilter, FeatureExtractor, Baseline, measure_attack_decay, crest_floor


def _tone(freq, sr, dur, amp=1.0):
    t = np.arange(int(sr * dur)) / sr
    return amp * np.sin(2 * np.pi * freq * t)


# ── AudioFilter ──
def test_dc_removed():
    f = AudioFilter(48000)
    out = f.process(_tone(200, 48000, 0.5) + 5.0)      # 5V 直流偏置
    assert abs(np.mean(out)) < 0.05


def test_highpass_attenuates_low():
    f = AudioFilter(48000, highpass_hz=20.0)
    low = f.process(_tone(5, 48000, 1.0))
    high = f.process(_tone(500, 48000, 1.0))
    assert np.std(high) > 5 * np.std(low)


def test_lowpass_attenuates_high():
    f = AudioFilter(16000, lowpass_hz=3500.0)
    hi = f.process(_tone(7000, 16000, 1.0))
    lo = f.process(_tone(800, 16000, 1.0))
    assert np.std(lo) > 10 * np.std(hi)


# ── FeatureExtractor ──
def _impulse_train(sr, dur_s, interval_s, amp=1.0):
    x = np.zeros(int(sr * dur_s))
    t = 0.0
    while t < dur_s:
        i = int(t * sr)
        x[i:i + int(0.02 * sr)] = amp * np.hanning(int(0.02 * sr))
        t += interval_s
    return x


def test_feature_values():
    sr = 48000
    ex = FeatureExtractor(sr)
    sig = np.sin(2 * np.pi * 60 * np.arange(sr) / sr) * 0.5
    feats = [ex.push(sig[i:i + 1024]) for i in range(0, len(sig) - 1024, 1024)]
    last = feats[-1]
    assert last.rms > 0.1
    assert last.crest_factor > 1.0
    assert last.low_energy_ratio > 0.3


def test_peak_count_and_interval():
    sr = 16000
    ex = FeatureExtractor(sr, short_window_ms=21.0)
    ex.set_peak_threshold(0.3)
    sig = _impulse_train(sr, 3.0, interval_s=0.25, amp=1.0)
    feats = [ex.push(sig[i:i + 336]) for i in range(0, len(sig) - 336, 336)]
    last = feats[-1]
    assert last.peak_count >= 3
    if last.peak_interval is not None and len(last.peak_interval):
        assert abs(float(np.mean(last.peak_interval)) - 0.25) < 0.1


def test_normalized_fields():
    ex = FeatureExtractor(16000)
    ex.set_baseline(rms=0.1, low_ratio=0.3)
    feat = ex.push(np.zeros(ex.short_n))
    assert abs(feat.rms_norm) < 1e-6


# ── attack/decay ──
def test_attack_decay():
    sr = 16000
    t = np.arange(0, 1.0, 1 / sr)
    env = np.interp(t, [0, 0.05, 0.25], [0.0, 1.0, 0.0])     # 上升50ms 下降200ms
    at, dt = measure_attack_decay(env, 1 / sr)
    assert at is not None and 30 < at < 70
    assert dt is not None and 150 < dt < 250


# ── Baseline ──
def test_baseline_calibrates_p10():
    b = Baseline(48000, calib_duration_sec=5.0)
    t = 0.0
    for _ in range(5 * 50 + 1):                  # 喂到 ts=5.0 触发标定
        b.feed(0.1, 0.3, t)
        t += 0.02
    assert b.calibrated
    assert abs(b.baseline_rms - 0.1) < 0.02


def test_baseline_freezes_when_triggering():
    b = Baseline(48000, calib_duration_sec=2.0, update_interval_sec=1.0)
    t = 0.0
    for _ in range(2 * 50 + 1):                  # 喂到 ts=2.0 完成标定
        b.feed(0.1, 0.3, t); t += 0.02
    base = b.baseline_rms
    assert base > 0.05                            # 标定应已完成
    for _ in range(60 * 50):
        b.feed(1.0, 0.3, t, trigger_ratio=0.8); t += 0.02
    assert b.baseline_rms / max(base, 1e-9) < 1.5       # 触发中 → 冻结，不会追上 1.0


# ── crest floor ──
def test_crest_floor_math():
    assert abs(crest_floor(16000, 21.0) - (2 * np.log(336)) ** 0.5) < 0.05
