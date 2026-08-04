"""DSP 核心：滤波、特征提取、本底噪声标定。纯 numpy/scipy，无平台依赖。"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

# ── 频带（与设计文档 §六 一致）──
LOW_BAND = (20.0, 150.0)
MID_BAND = (150.0, 1000.0)


def crest_floor(sample_rate: int, short_window_ms: float) -> float:
    """随机噪声在短窗 N 点内的预期峰均比 √(2·ln N)。判定阈值必须避开它。"""
    n = max(2, int(sample_rate * short_window_ms / 1000))
    return float((2.0 * np.log(n)) ** 0.5)


# ── 滤波 ──────────────────────────────────────────────────────────────

class AudioFilter:
    """流式 DC 去除 + 高通(20Hz) + 可选低通。带 SOS 状态，逐帧调用。"""

    def __init__(self, sample_rate: int, highpass_hz: float = 20.0,
                 lowpass_hz: float | None = None):
        self.sr = sample_rate
        self._dc_prev = 0.0
        self._hp = signal.butter(2, highpass_hz, "highpass", fs=sample_rate, output="sos")
        self._hp_zi = np.zeros((self._hp.shape[0], 2))
        self._lp = None
        self._lp_zi = None
        if lowpass_hz is not None:
            self._lp = signal.butter(4, lowpass_hz, "lowpass", fs=sample_rate, output="sos")
            self._lp_zi = np.zeros((self._lp.shape[0], 2))

    def process(self, samples: np.ndarray) -> np.ndarray:
        x = np.asarray(samples, dtype=np.float64)
        # 一阶 DC blocker：y[n]=x[n]-x[n-1]+α·y[n-1]，α=0.999 对 ≥20Hz 无影响
        out = np.empty_like(x)
        yp = 0.0
        prev = self._dc_prev
        for i, v in enumerate(x):
            y = v - prev + 0.999 * yp
            out[i] = y
            prev = v
            yp = y
        self._dc_prev = prev
        y, self._hp_zi = signal.sosfilt(self._hp, out, zi=self._hp_zi)
        if self._lp is not None:
            y, self._lp_zi = signal.sosfilt(self._lp, y, zi=self._lp_zi)
        return y


# ── 特征 ──────────────────────────────────────────────────────────────

@dataclass
class Feature:
    ts: float
    rms: float = 0.0
    peak: float = 0.0
    crest_factor: float = 0.0
    zcr: float = 0.0
    envelope: np.ndarray = field(default_factory=lambda: np.zeros(0))
    attack_time_ms: float | None = None
    decay_time_ms: float | None = None
    spectral_centroid: float = 0.0
    spectral_flux: float = 0.0
    spectral_flatness: float = 0.0
    low_energy_ratio: float = 0.0
    mid_energy_ratio: float = 0.0
    high_energy_ratio: float = 0.0
    duration: float = 0.0
    peak_count: int = 0
    peak_interval: np.ndarray | None = None
    interval_variance: float = 0.0
    rms_norm: float = 1.0
    low_energy_norm: float = 1.0


def _sub_envelope(x: np.ndarray, sr: int, step_sec: float = 0.005) -> np.ndarray:
    """按 step 步长取 |x| 峰值的亚帧包络（原始形状，供 attack/decay 测量）。"""
    step = max(1, int(sr * step_sec))
    n = max(1, len(x) // step)
    out = np.empty(n)
    for i in range(n):
        out[i] = float(np.max(np.abs(x[i * step:(i + 1) * step]))) if n else 0.0
    return out


def measure_attack_decay(env: np.ndarray, dt: float) -> tuple[float | None, float | None]:
    """dt=包络采样间隔(秒)。返回 (attack_ms, decay_ms)；无显著峰值时 (None, None)。"""
    if len(env) < 3:
        return None, None
    pk = int(np.argmax(env))
    peak = float(env[pk])
    if peak < 1e-9:
        return None, None
    lo, hi = 0.10 * peak, 0.90 * peak
    rise_start = rise_end = None
    for i in range(pk + 1):
        if rise_start is None and env[i] >= lo:
            rise_start = i
        if env[i] >= hi:
            rise_end = i
            break
    attack = (rise_end - rise_start) * dt * 1000.0 if (rise_start is not None and rise_end is not None) else None
    fall_start = fall_end = None
    for i in range(pk, len(env)):
        if fall_start is None and env[i] <= hi:
            fall_start = i
        if env[i] <= lo:
            fall_end = i
            break
    decay = (fall_end - fall_start) * dt * 1000.0 if (fall_start is not None and fall_end is not None) else None
    return attack, decay


class FeatureExtractor:
    """双窗口特征：短窗(约21ms)算包络/峰均比/过零率，长窗(约85ms)算频谱。
    维护 5 秒环缓冲统计峰值个数/间隔/持续时长。"""

    def __init__(self, sample_rate: int, short_window_ms: float = 21.0,
                 long_window_ms: float = 85.0, ring_sec: float = 5.0,
                 min_peak_interval_ms: float = 100.0):
        self.sr = sample_rate
        self.short_n = max(2, int(sample_rate * short_window_ms / 1000))
        self.long_n = max(4, int(sample_rate * long_window_ms / 1000))
        self.hop_n = max(1, self.long_n // 2)          # 50% overlap
        self.ring_sec = ring_sec
        self.min_peak_interval = min_peak_interval_ms / 1000.0
        self.peak_threshold: float | None = None
        self.rms_threshold: float | None = None
        self._long_buf = np.zeros(self.long_n)
        self._since_spectral = self.hop_n
        self._prev_mag: np.ndarray | None = None
        self._spec = [0.0] * 6                        # centroid,flux,flatness,low,mid,high
        self._fft: np.ndarray | None = None
        self._peaks: deque[tuple[float, float]] = deque()
        self._last_peak_ts: float | None = None
        self._over_start: float | None = None
        self._ts = 0.0
        self._baseline_rms = 1.0
        self._baseline_low = 1.0

    def set_peak_threshold(self, v: float | None) -> None:
        self.peak_threshold = v

    def set_rms_threshold(self, v: float | None) -> None:
        self.rms_threshold = v

    def set_baseline(self, rms: float, low_ratio: float) -> None:
        self._baseline_rms = rms if rms > 1e-9 else 1.0
        self._baseline_low = low_ratio if low_ratio > 1e-9 else 1.0

    def push(self, samples: np.ndarray) -> Feature:
        x = np.asarray(samples, dtype=np.float64)
        self._ts += len(x) / self.sr
        rms = float(np.sqrt(np.mean(x ** 2)))
        peak = float(np.max(np.abs(x)))
        crest = peak / rms if rms > 1e-9 else 0.0
        zcr = float(np.sum(np.diff(np.signbit(x)) != 0)) / len(x)
        env = _sub_envelope(x, self.sr)
        at, dt = measure_attack_decay(env, 1.0 / self.sr * self.short_n / max(1, len(env)))

        # 频谱：滑窗 + 满 hop 计算一次
        self._long_buf = np.roll(self._long_buf, -len(x))
        self._long_buf[-len(x):] = x
        self._since_spectral += len(x)
        if self._since_spectral >= self.hop_n:
            self._since_spectral = 0
            self._compute_spectral()

        # 峰值检测：超过峰值门槛且与上一个峰值间隔足够
        if self.peak_threshold is not None and peak > self.peak_threshold:
            if self._last_peak_ts is None or (self._ts - self._last_peak_ts) >= self.min_peak_interval:
                self._last_peak_ts = self._ts
                self._peaks.append((self._ts, peak))
                self._trim()

        # 持续时长：连续超过 RMS 门槛
        if self.rms_threshold is not None and rms > self.rms_threshold:
            if self._over_start is None:
                self._over_start = self._ts
            duration = self._ts - self._over_start
        else:
            self._over_start = None
            duration = 0.0

        interval = np.array([self._peaks[i + 1][0] - self._peaks[i][0]
                             for i in range(max(0, len(self._peaks) - 1))])
        ivar = float(np.std(interval) / (np.mean(interval) + 1e-9)) if len(interval) > 1 else 0.0
        centroid, flux, flat, low, mid, high = self._spec
        return Feature(
            ts=self._ts, rms=rms, peak=peak, crest_factor=crest, zcr=zcr,
            envelope=env, attack_time_ms=at, decay_time_ms=dt,
            spectral_centroid=centroid, spectral_flux=flux, spectral_flatness=flat,
            low_energy_ratio=low, mid_energy_ratio=mid, high_energy_ratio=high,
            duration=duration, peak_count=len(self._peaks),
            peak_interval=interval if len(interval) else None, interval_variance=ivar,
            rms_norm=rms / self._baseline_rms, low_energy_norm=low / self._baseline_low,
        )

    def _compute_spectral(self) -> None:
        xw = self._long_buf * signal.windows.hann(self.long_n)
        fft = np.fft.rfft(xw)
        mag = np.abs(fft)
        freqs = np.fft.rfftfreq(self.long_n, d=1.0 / self.sr)
        low = _band_energy(mag, freqs, LOW_BAND)
        mid = _band_energy(mag, freqs, MID_BAND)
        high = _band_energy(mag, freqs, (MID_BAND[1], self.sr / 2.0))
        total = low + mid + high
        eps = 1e-12
        centroid = float(np.sum(freqs * mag) / (np.sum(mag) + eps))
        flux = 0.0
        if self._prev_mag is not None and len(self._prev_mag) == len(mag):
            pn = self._prev_mag / (np.sum(self._prev_mag) + eps)
            cn = mag / (np.sum(mag) + eps)
            flux = float(np.sum(np.abs(cn - pn)))
        self._prev_mag = mag
        flat = float(np.exp(np.mean(np.log(mag + eps))) / (np.mean(mag) + eps))
        self._fft = fft
        self._spec = [centroid, flux, flat,
                      low / (total + eps), mid / (total + eps), high / (total + eps)]

    def _trim(self) -> None:
        while self._peaks and self._ts - self._peaks[0][0] > self.ring_sec:
            self._peaks.popleft()


def _band_energy(mag: np.ndarray, freqs: np.ndarray, band: tuple[float, float]) -> float:
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return float(np.sum(mag[mask] ** 2)) if np.any(mask) else 0.0


# ── 本底噪声标定 ─────────────────────────────────────────────────────

class Baseline:
    """环境本底噪声：启动采集 P10 定标；运行时按窗口取 P10 慢更新，触发占比高时冻结。"""

    def __init__(self, sample_rate: int, calib_duration_sec: float = 10.0,
                 percentile: float = 10.0, update_interval_sec: float = 30.0,
                 stall_trigger_ratio: float = 0.5):
        self.sr = sample_rate
        self.calib_sec = calib_duration_sec
        self.percentile = percentile
        self.update_interval = update_interval_sec
        self.stall_ratio = stall_trigger_ratio
        self.baseline_rms = 0.0
        self.baseline_std = 1.0
        self.baseline_low_ratio = 0.0
        self.calibrated = False
        self._hist: deque[tuple[float, float]] = deque()   # (ts, rms)
        self._last_update = 0.0

    def feed(self, rms: float, low_ratio: float, ts: float, trigger_ratio: float = 0.0) -> None:
        self._hist.append((ts, rms))
        self._trim_hist(ts, window=self.calib_sec if not self.calibrated else self.update_interval)
        if not self.calibrated:
            # 容差 1ms：规避浮点累加导致最后一个 ts 略小于 calib_sec
            if ts >= self.calib_sec - 1e-3:
                self._apply(np.percentile([r for _, r in self._hist], self.percentile))
                self._last_update = ts
                self.calibrated = True
            return
        if ts - self._last_update >= self.update_interval:
            self._last_update = ts
            if trigger_ratio <= self.stall_ratio:
                self._apply(np.percentile([r for _, r in self._hist], self.percentile))

    def _apply(self, target: float) -> None:
        recent = [r for _, r in self._hist]
        self.baseline_rms = float(target)
        self.baseline_std = float(np.std(recent)) if recent else 1.0

    def _trim_hist(self, ts: float, window: float) -> None:
        while self._hist and ts - self._hist[0][0] > window:
            self._hist.popleft()

    def threshold(self, sensitivity: float) -> float:
        return self.baseline_rms + sensitivity * self.baseline_std

    def threshold_norm(self, sensitivity: float) -> float:
        return 1.0 + sensitivity * (self.baseline_std / (self.baseline_rms + 1e-9))

    def peak_threshold(self, sensitivity: float, floor: float) -> float:
        return max(self.threshold(sensitivity) * 1.5, floor)
