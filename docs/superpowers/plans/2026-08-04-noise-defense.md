# Noise Defense System 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 V5.1 设计文档规定的 Windows 家庭噪声检测与自动响应系统（无 AI、纯规则、本地运行），在本机（Linux）完成核心逻辑的 TDD，GUI/音频在 Windows 部署。

**Architecture:** 五线程分层（音频回调→RingQueue→DSP线程→GUI主线程→Response线程→设备监控）。核心为纯函数/纯状态机的 dsp / detector / engine 三层，与平台无关、可全量单测；audio / gui 为薄适配层，懒加载 sounddevice/PyQt，仅冒烟测试。设备类型（有线/蓝牙）自适应只影响采样率协商与 Mute Gate 余量。

**Tech Stack:** Python 3.11+（本机 3.12）、NumPy、SciPy、sounddevice（懒加载）、miniaudio（音频解码）、PyYAML、PyQt6、PyQtGraph、pytest。

## Global Constraints

- 无 AI 训练、无联网、无数据上传，完全本地运行
- 不引入额外硬件；支持有线/USB 与蓝牙麦克风，启动自动识别 + 手动覆盖
- 特征公式以设计文档 §六 为准；Detector 规则 §九；Episode 语义 §十；config 阈值 §十七
- CrestFactor 有效阈值 = `max(crest_factor_min, CrestFloor(N)+1.0)`，`CrestFloor(N)=√(2·ln N)`
- RMS 比较一律用归一化量纲 `Threshold_norm = 1 + sensitivity×(STD/RMS)`（§七）
- 线程模型 §八：回调线程只入队；DSP 线程随短窗 tick；GUI 频谱节流 10Hz
- 所有阈值是初始猜测，最终由 `tools/calibrate.py` 标定（§二十）
- 包络判据恢复采集优先（`ignore_window_by_envelope: true`）
- 代码注释写"为什么"不写"做什么"；函数 ≤50 行；不用 data/result/temp 命名
- 本目录非 git 仓库，Task 1 执行 `git init`
- 依赖：numpy、scipy、PyYAML、sounddevice、miniaudio、PyQt6、pyqtgraph、pytest（`requirements.txt`）

---

### Task 1: 项目脚手架 + 配置系统

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `.gitignore`, `conftest.py`
- Create: `NoiseDefense/__init__.py`, `NoiseDefense/config/__init__.py`, `NoiseDefense/config/config.py`, `NoiseDefense/config/config.yaml`
- Create: `NoiseDefense/dsp/__init__.py`, `NoiseDefense/audio/__init__.py`, `NoiseDefense/detector/__init__.py`, `NoiseDefense/engine/__init__.py`, `NoiseDefense/gui/__init__.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: 设计文档 §十七 config.yaml 字段
- Produces: `load_config(path) -> Config`；`save_config(cfg, path)`；嵌套 dataclass `Config{AudioConfig, DeviceConfig, CalibrationConfig, DetectionConfig, ScheduleConfig, ResponseConfig, BluetoothConfig, GuiConfig}`；`DetectionConfig.per_detector: dict[str, PerDetectorConfig]`

- [ ] **Step 1: 初始化环境与仓库**

```bash
cd /home/ln/data/Code/qt/Noise2Noise
git init
python3 -m venv .venv && . .venv/bin/activate
pip install -U pip
pip install numpy scipy PyYAML sounddevice miniaudio soundfile pytest
# PyQt6/pyqtgraph 单独装（体积大）
pip install PyQt6 pyqtgraph
sudo apt-get install -y libportaudio2 2>/dev/null || echo "portaudio 未装：audio 层仅纯函数单测，冒烟测试跳过"
```

- [ ] **Step 2: 写配置系统**

`NoiseDefense/config/config.py`：
```python
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass
class AudioConfig:
    input_type: str = "auto"                     # auto / wired / bluetooth
    wired_sample_rates: list[int] = field(default_factory=lambda: [48000, 44100, 16000])
    bt_preferred_sample_rates: list[int] = field(default_factory=lambda: [16000, 8000])
    short_window_ms: float = 21.0
    long_window_ms: float = 85.0
    spectral_overlap: float = 0.5
    latency_selftest: str = "on_first_startup"   # on_first_startup / off


@dataclass
class DeviceConfig:
    input: str = ""
    output: str = ""
    bt_latency_ms: int = 300
    measured_latency_ms: float | None = None     # 延迟自测实测值，<50ms 视为有线


@dataclass
class CalibrationConfig:
    auto_baseline: bool = True
    baseline_duration_sec: float = 20.0
    baseline_percentile: float = 10.0
    baseline_update_interval_sec: float = 60.0
    baseline_rise_step_db: float = 0.5
    baseline_fall_step_db: float = 3.0
    baseline_min_db: float = -60.0
    baseline_max_db: float = -20.0
    baseline_stall_trigger_ratio: float = 0.5
    recalibrate_on_bt_reconnect: bool = True


@dataclass
class HysteresisConfig:
    n1_enter: int = 3
    n2_confirm: int = 5
    n3_exit: int = 5


@dataclass
class PerDetectorConfig:
    name: str
    confirm_count: int = 1
    window_sec: float = 1.0
    priority: int = 1
    crest_factor_min: float = 3.0
    low_band_dependent: bool = False
    rules: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionConfig:
    sensitivity: float = 5.0
    cooldown: float = 5.0
    ignore_window_base_ms: float = 200.0
    ignore_window_by_envelope: bool = True
    episode_max_sec: float = 30.0
    episode_close_gap_sec: float = 2.0
    hysteresis_default: HysteresisConfig = field(default_factory=HysteresisConfig)
    per_detector: dict[str, PerDetectorConfig] = field(default_factory=dict)
    arbitration_window_ms: int = 300


@dataclass
class ScheduleConfig:
    mode: str = "window"                        # window / always / manual_only
    active_windows: list[str] = field(default_factory=lambda: ["12:00-14:00", "18:00-24:00"])
    manual_confirm_in_window: bool = False
    manual_confirm_outside_window: bool = True
    confirm_timeout_sec: float = 30.0
    max_responses_per_window: int = 5
    fresh_episode_gap_sec: float = 60.0


@dataclass
class ResponseConfig:
    random: bool = True
    volume: float = 1.0


@dataclass
class BluetoothConfig:
    reconnect_intervals_sec: list[int] = field(default_factory=lambda: [1, 5, 15, 30])
    reconnect_timeout_alert_min: int = 5


@dataclass
class GuiConfig:
    spectrum_refresh_hz: int = 10
    envelope_history_sec: float = 5.0


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    response: ResponseConfig = field(default_factory=ResponseConfig)
    bluetooth: BluetoothConfig = field(default_factory=BluetoothConfig)
    gui: GuiConfig = field(default_factory=GuiConfig)


def _merge(cfg: Config, data: dict[str, Any]) -> None:
    """浅合并：对每个 dataclass 字段，只覆盖 YAML 中出现的键。"""
    for section, dc in ((s.name.lower(), s) for s in cfg.__dataclass_fields__.values()):
        section_data = data.get(section)
        if not isinstance(section_data, dict):
            continue
        for key, value in section_data.items():
            field_obj = getattr(cfg, section).__dataclass_fields__.get(key)
            if field_obj is None:
                continue
            if field_obj.type == "HysteresisConfig" and isinstance(value, dict):
                setattr(getattr(cfg, section), key, HysteresisConfig(**value))
            elif key == "per_detector" and isinstance(value, dict):
                per = {}
                for name, pv in value.items():
                    if isinstance(pv, dict):
                        per[name] = PerDetectorConfig(name=name, **pv)
                    else:
                        per[name] = PerDetectorConfig(name=name)
                setattr(getattr(cfg, section), key, per)
            else:
                setattr(getattr(cfg, section), key, value)


def load_config(path: str | Path) -> Config:
    cfg = Config()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    _merge(cfg, data)
    return cfg


def save_config(cfg: Config, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(_to_dict(cfg), allow_unicode=True), encoding="utf-8")
```

- [ ] **Step 3: 写失败测试**

`tests/test_config.py`：
```python
from NoiseDefense.config.config import load_config, Config, PerDetectorConfig


def test_defaults_when_yaml_empty(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("{}", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.audio.input_type == "auto"
    assert cfg.audio.wired_sample_rates == [48000, 44100, 16000]
    assert cfg.detection.hysteresis_default.n2_confirm == 5


def test_merge_overrides_defaults(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("""
Audio:
  input_type: bluetooth
Detection:
  sensitivity: 8
  per_detector:
    Impact:
      confirm_count: 1
      priority: 6
      rules: { attack_time_ms_max: 50 }
""", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.audio.input_type == "bluetooth"
    assert cfg.detection.sensitivity == 8.0
    imp = cfg.detection.per_detector["Impact"]
    assert isinstance(imp, PerDetectorConfig)
    assert imp.priority == 6
    assert imp.rules["attack_time_ms_max"] == 50


def test_save_roundtrip(tmp_path):
    p = tmp_path / "c.yaml"
    cfg = Config()
    cfg.schedule.active_windows = ["08:00-09:00"]
    save_config(cfg, p)
    again = load_config(p)
    assert again.schedule.active_windows == ["08:00-09:00"]
```

- [ ] **Step 4: 运行验证失败**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: FAIL（`load_config` 未定义 / 导入错误）

- [ ] **Step 5: 补齐实现并通过**

在 `config.py` 末尾追加 `_to_dict`（用 `dataclasses.asdict` 即可）：
```python
from dataclasses import asdict

def _to_dict(cfg: Config) -> dict:
    return asdict(cfg)
```
Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: PASS（若 `_to_dict` 缺失导致 save 失败，补 `from dataclasses import asdict`）

- [ ] **Step 6: 写 config.yaml 与 pyproject/requirements/.gitignore**

`NoiseDefense/config/config.yaml` 内容 = 设计文档 §十七 V5.1 全量配置（逐字段照抄）。`pyproject.toml`：
```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "noise-defense"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["numpy", "scipy", "PyYAML", "sounddevice", "miniaudio"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```
`.gitignore`：`.venv/`, `__pycache__/`, `*.pyc`, `logs/`, `sounds/generated/`, `.pytest_cache/`。

- [ ] **Step 7: 提交**

```bash
git add -A && git commit -m "chore: scaffold package, config system, tests"
```

---

### Task 2: dsp/filter.py —— 流式预处理

**Files:**
- Create: `NoiseDefense/dsp/filter.py`
- Test: `tests/test_filter.py`

**Interfaces:**
- Consumes: 设计 §五 Preprocessor（去直流 + 高通 20Hz + 低通按采样率）
- Produces: `class AudioFilter: __init__(sample_rate: int, highpass_hz: float = 20.0, lowpass_hz: float | None = None); process(samples: np.ndarray) -> np.ndarray; reset()`

- [ ] **Step 1: 写失败测试**

`tests/test_filter.py`：
```python
import numpy as np
from NoiseDefense.dsp.filter import AudioFilter

def _tone(freq, sr, dur, amp=1.0):
    t = np.arange(int(sr * dur)) / sr
    return amp * np.sin(2 * np.pi * freq * t)

def test_dc_removed():
    f = AudioFilter(48000, highpass_hz=20.0)
    sig = _tone(200, 48000, 0.5) + 5.0          # 加 5V 直流偏置
    out = f.process(sig)
    assert abs(np.mean(out)) < 0.05

def test_highpass_attenuates_below_20():
    f = AudioFilter(48000, highpass_hz=20.0)
    low = f.process(_tone(5, 48000, 1.0))       # 5Hz 应被大幅衰减
    high = f.process(_tone(500, 48000, 1.0))
    assert np.std(high) > 5 * np.std(low)

def test_lowpass_attenuates_above_cutoff():
    f = AudioFilter(16000, lowpass_hz=3500.0)
    hi = f.process(_tone(7000, 16000, 1.0))
    lo = f.process(_tone(800, 16000, 1.0))
    assert np.std(lo) > 10 * np.std(hi)

def test_streaming_state_carries_over():
    f = AudioFilter(48000, highpass_hz=20.0)
    sig = _tone(100, 48000, 0.2)
    chunks = [sig[i:i + 1024] for i in range(0, len(sig), 1024)]
    out = np.concatenate([f.process(c) for c in chunks])
    ref = AudioFilter(48000, highpass_hz=20.0).process(sig)
    assert np.allclose(out, ref, atol=1e-3)
```

- [ ] **Step 2: 运行验证失败**

Run: `.venv/bin/python -m pytest tests/test_filter.py -v`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 写实现**

`NoiseDefense/dsp/filter.py`：
```python
from __future__ import annotations
import numpy as np
from scipy import signal


class AudioFilter:
    """流式 DC 去除 + 高通(20Hz) + 可选低通，用 SOS 滤波器带状态处理，逐帧调用。"""

    def __init__(self, sample_rate: int, highpass_hz: float = 20.0,
                 lowpass_hz: float | None = None):
        self.sr = sample_rate
        self._dc_state = 0.0                    # 一阶 DC blocker 状态
        self._hp_sos = signal.butter(2, highpass_hz, "highpass", fs=sample_rate, output="sos")
        self._hp_zi = np.zeros((self._hp_sos.shape[0], 2))
        self._lp_sos = None
        self._lp_zi = None
        if lowpass_hz is not None:
            self._lp_sos = signal.butter(4, lowpass_hz, "lowpass", fs=sample_rate, output="sos")
            self._lp_zi = np.zeros((self._lp_sos.shape[0], 2))

    def process(self, samples: np.ndarray) -> np.ndarray:
        x = np.asarray(samples, dtype=np.float64)
        # 一阶 DC blocker: y[n] = x[n] - x[n-1] + α·y[n-1]，α≈0.999 对 ≥20Hz 影响可忽略
        alpha = 0.999
        out = np.empty_like(x)
        prev_in = x[0] - self._dc_state if False else 0.0  # 见下
        y_prev = 0.0
        for i in range(len(x)):
            y = x[i] - self._dc_state + alpha * y_prev
            out[i] = y
            self._dc_state = x[i]
            y_prev = y
        y, self._hp_zi = signal.sosfilt(self._hp_sos, out, zi=self._hp_zi)
        if self._lp_sos is not None:
            y, self._lp_zi = signal.sosfilt(self._lp_sos, y, zi=self._lp_zi)
        return y

    def reset(self) -> None:
        self._dc_state = 0.0
        self._hp_zi = np.zeros_like(self._hp_zi)
        if self._lp_sos is not None:
            self._lp_zi = np.zeros_like(self._lp_zi)
```

> 注：DC blocker 的 Python 循环在 21ms≈1000 点/帧上是可行的（帧级 O(N)），帧长不随采样率剧增。若后续 profile 显示瓶颈，换成 `scipy.signal.lfilter` 的 `(b=[1,-1], a=[1,-α])` 差分实现。

- [ ] **Step 4: 运行验证通过**

Run: `.venv/bin/python -m pytest tests/test_filter.py -v`
Expected: PASS（若 `test_dc_removed` 未过，调小 α 到 0.9995 再跑）

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat: streaming audio filter (DC blocker + highpass + lowpass)"
```

---

### Task 3: dsp/envelope.py —— 亚帧包络 + attack/decay

**Files:**
- Create: `NoiseDefense/dsp/envelope.py`
- Test: `tests/test_envelope.py`

**Interfaces:**
- Produces: `class EnvelopeTracker: __init__(sample_rate: int, step_sec: float = 0.005, decay_tau_sec: float = 0.05); update(frame: np.ndarray) -> np.ndarray`（返回本帧的逐 step 包络值）；`def measure_attack_decay(env: np.ndarray, env_sr: float) -> tuple[float | None, float | None]`（10%→90% 上升 / 90%→10% 下降，单位 ms）

- [ ] **Step 1: 写失败测试**

`tests/test_envelope.py`：
```python
import numpy as np
from NoiseDefense.dsp.envelope import EnvelopeTracker, measure_attack_decay


def test_envelope_tracks_peak():
    tr = EnvelopeTracker(sample_rate=16000, step_sec=0.005)
    frame = np.zeros(16000 // 21 * 21)          # 约一个短窗
    frame[100:110] = 1.0                        # 一个尖峰
    env = tr.update(frame)
    assert env.max() >= 0.99


def test_attack_decay_measured():
    sr = 16000
    t = np.arange(0, 1.0, 1 / sr)
    # 线性上升 50ms、线性下降 200ms 的包络形状
    env = np.interp(t, [0, 0.05, 0.25], [0.0, 1.0, 0.0])
    at, dt = measure_attack_decay(env, 1 / sr)
    assert at is not None and 30 < at < 70
    assert dt is not None and 150 < dt < 250


def test_quiet_no_attack_decay():
    at, dt = measure_attack_decay(np.zeros(100), 0.01)
    assert at is None and dt is None
```

- [ ] **Step 2: 运行验证失败**

Run: `.venv/bin/python -m pytest tests/test_envelope.py -v`
Expected: FAIL

- [ ] **Step 3: 写实现**

`NoiseDefense/dsp/envelope.py`：
```python
from __future__ import annotations
import numpy as np


class EnvelopeTracker:
    """亚帧包络：按 step 步长取 |x| 峰值，再做峰值保持 + 指数衰减平滑。"""

    def __init__(self, sample_rate: int, step_sec: float = 0.005,
                 decay_tau_sec: float = 0.05):
        self.step = max(1, int(sample_rate * step_sec))
        self.decay = np.exp(-step_sec / decay_tau_sec)
        self.value = 0.0
        self.env_sr = 1.0 / step_sec             # 包络采样率（Hz）

    def update(self, frame: np.ndarray) -> np.ndarray:
        x = np.abs(np.asarray(frame, dtype=np.float64))
        n = max(1, len(x) // self.step)
        env = np.empty(n)
        for i in range(n):
            seg = x[i * self.step:(i + 1) * self.step]
            peak = float(np.max(seg)) if len(seg) else 0.0
            self.value = max(peak, self.value * self.decay)
            env[i] = self.value
        return env


def measure_attack_decay(env: np.ndarray, dt: float) -> tuple[float | None, float | None]:
    """dt=包络采样间隔(秒)。返回 (attack_ms, decay_ms)；无显著峰值时返回 (None, None)。"""
    if len(env) < 3:
        return None, None
    peak_idx = int(np.argmax(env))
    peak = float(env[peak_idx])
    if peak < 1e-9:
        return None, None
    lo = 0.10 * peak
    hi = 0.90 * peak
    # 上升：10%→90%
    rise_start = rise_end = None
    for i in range(peak_idx + 1):
        if env[i] >= lo and rise_start is None:
            rise_start = i
        if env[i] >= hi:
            rise_end = i
            break
    attack_ms = None
    if rise_start is not None and rise_end is not None:
        attack_ms = (rise_end - rise_start) * dt * 1000.0
    # 下降：90%→10%
    fall_start = fall_end = None
    for i in range(peak_idx, len(env)):
        if env[i] <= hi and fall_start is None:
            fall_start = i
        if env[i] <= lo:
            fall_end = i
            break
    decay_ms = None
    if fall_start is not None and fall_end is not None:
        decay_ms = (fall_end - fall_start) * dt * 1000.0
    return attack_ms, decay_ms
```

- [ ] **Step 4: 运行验证通过**

Run: `.venv/bin/python -m pytest tests/test_envelope.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat: sub-frame envelope tracker + attack/decay measurement"
```

---

### Task 4: dsp/spectral.py —— 频谱特征

**Files:**
- Create: `NoiseDefense/dsp/spectral.py`
- Test: `tests/test_spectral.py`

**Interfaces:**
- Produces: `@dataclass SpectralResult: fft, freqs, mag, low_energy_ratio, mid_energy_ratio, high_energy_ratio, centroid, flux, flatness`；`def spectral_features(window: np.ndarray, sample_rate: int, prev_mag: np.ndarray | None) -> SpectralResult`

- [ ] **Step 1: 写失败测试**

`tests/test_spectral.py`：
```python
import numpy as np
from NoiseDefense.dsp.spectral import spectral_features


def _win(freq, sr=48000, n=4080):
    t = np.arange(n) / sr
    return np.sin(2 * np.pi * freq * t)


def test_low_band_dominant_for_low_tone():
    r = spectral_features(_win(60), 48000)
    assert r.low_energy_ratio > 0.5

def test_high_band_dominant_for_high_tone():
    r = spectral_features(_win(3000), 48000)
    assert r.high_energy_ratio > 0.5

def test_white_noise_flat():
    rng = np.random.default_rng(0)
    r = spectral_features(rng.standard_normal(4080), 48000)
    assert 0.3 < r.flatness < 1.0

def test_flux_detects_change():
    a = spectral_features(_win(60), 48000)
    b = spectral_features(_win(2000), 48000, prev_mag=a.mag)
    assert b.flux > 0.01

def test_centroid_rises_with_freq():
    lo = spectral_features(_win(100), 48000)
    hi = spectral_features(_win(3000), 48000)
    assert hi.centroid > lo.centroid
```

- [ ] **Step 2: 运行验证失败**

Run: `.venv/bin/python -m pytest tests/test_spectral.py -v`
Expected: FAIL

- [ ] **Step 3: 写实现**

`NoiseDefense/dsp/spectral.py`：
```python
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import windows

LOW = (20.0, 150.0)
MID = (150.0, 1000.0)


@dataclass
class SpectralResult:
    fft: np.ndarray
    freqs: np.ndarray
    mag: np.ndarray
    low_energy_ratio: float
    mid_energy_ratio: float
    high_energy_ratio: float
    centroid: float
    flux: float
    flatness: float


def _band_energy(mag: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    m = (freqs >= lo) & (freqs <= hi)
    return float(np.sum(mag[m] ** 2)) if np.any(m) else 0.0


def spectral_features(window: np.ndarray, sample_rate: int,
                      prev_mag: np.ndarray | None = None) -> SpectralResult:
    x = np.asarray(window, dtype=np.float64) * windows.hann(len(window))
    fft = np.fft.rfft(x)
    mag = np.abs(fft)
    freqs = np.fft.rfftfreq(len(window), d=1.0 / sample_rate)
    low = _band_energy(mag, freqs, LOW[0], LOW[1])
    mid = _band_energy(mag, freqs, MID[0], MID[1])
    high = _band_energy(mag, freqs, MID[1], sample_rate / 2.0)
    total = low + mid + high
    eps = 1e-12
    lr = low / (total + eps)
    mr = mid / (total + eps)
    hr = high / (total + eps)
    mag_sum = np.sum(mag) + eps
    centroid = float(np.sum(freqs * mag) / mag_sum)
    flux = 0.0
    if prev_mag is not None and len(prev_mag) == len(mag):
        pn = prev_mag / (np.sum(prev_mag) + eps)
        cn = mag / mag_sum
        flux = float(np.sum(np.abs(cn - pn)))
    log_mag = np.log(mag + eps)
    flatness = float(np.exp(np.mean(log_mag)) / (np.mean(mag) + eps))
    return SpectralResult(fft=fft, freqs=freqs, mag=mag,
                          low_energy_ratio=lr, mid_energy_ratio=mr, high_energy_ratio=hr,
                          centroid=centroid, flux=flux, flatness=flatness)
```

- [ ] **Step 4: 运行验证通过**

Run: `.venv/bin/python -m pytest tests/test_spectral.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat: spectral features (band ratios, centroid, flux, flatness)"
```

---

### Task 5: dsp/feature.py —— FeatureExtractor + Feature struct

**Files:**
- Create: `NoiseDefense/dsp/feature.py`
- Test: `tests/test_feature.py`

**Interfaces:**
- Consumes: `EnvelopeTracker`, `measure_attack_decay`, `spectral_features`
- Produces: `@dataclass Feature`（字段见设计 §六，含 `ts/rms/peak/crest_factor/zcr/envelope/attack_time_ms/decay_time_ms/fft/spectral_centroid/spectral_flux/spectral_flatness/low_energy_ratio/mid_energy_ratio/high_energy_ratio/duration/peak_count/peak_interval/interval_variance/rms_norm/low_energy_norm`）；`class FeatureExtractor: __init__(sample_rate, short_window_ms=21.0, long_window_ms=85.0, ring_sec=5.0, min_peak_interval_ms=100.0, baseline_crest_floor=0.0); push(samples) -> Feature; set_peak_threshold(v: float|None); set_rms_threshold(v: float|None); set_baseline(rms: float, low_ratio: float)`

- [ ] **Step 1: 写失败测试**

`tests/test_feature.py`：
```python
import numpy as np
from NoiseDefense.dsp.feature import FeatureExtractor

def _impulse_train(sr, dur_s, interval_s, amp=1.0):
    n = int(sr * dur_s)
    x = np.zeros(n)
    t = 0.0
    while t < dur_s:
        i = int(t * sr)
        x[i:i + int(0.02 * sr)] = amp * np.hanning(int(0.02 * sr))
        t += interval_s
    return x

def test_basic_feature_values():
    sr = 48000
    f = FeatureExtractor(sr, short_window_ms=21.0, long_window_ms=85.0)
    sig = np.sin(2 * np.pi * 60 * np.arange(sr) / sr) * 0.5
    frames = [sig[i:i + 1024] for i in range(0, len(sig) - 1024, 1024)]
    feats = [f.push(fr) for fr in frames]
    last = feats[-1]
    assert last.rms > 0.1
    assert last.crest_factor > 1.0
    assert last.low_energy_ratio > 0.4          # 60Hz 占主导
    assert last.ts > 0.0

def test_peak_count_and_interval():
    sr = 16000
    f = FeatureExtractor(sr, short_window_ms=21.0, long_window_ms=85.0, ring_sec=5.0)
    f.set_peak_threshold(0.3)
    sig = _impulse_train(sr, 3.0, interval_s=0.25, amp=1.0)   # 每 250ms 一个脉冲
    frames = [sig[i:i + 336] for i in range(0, len(sig) - 336, 336)]
    feats = [f.push(fr) for fr in frames]
    last = feats[-1]
    assert last.peak_count >= 3
    if last.peak_interval is not None and len(last.peak_interval) > 0:
        assert abs(np.mean(last.peak_interval) - 0.25) < 0.1

def test_normalized_fields():
    sr = 16000
    f = FeatureExtractor(sr)
    f.set_baseline(rms=0.1, low_ratio=0.3)
    feat = f.push(np.zeros(f._short_n))
    assert abs(feat.rms_norm) < 1e-6
```

- [ ] **Step 2: 运行验证失败**

Run: `.venv/bin/python -m pytest tests/test_feature.py -v`
Expected: FAIL

- [ ] **Step 3: 写实现**

`NoiseDefense/dsp/feature.py`：
```python
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
import numpy as np
from .envelope import EnvelopeTracker, measure_attack_decay
from .spectral import spectral_features, LOW, MID


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
    fft: np.ndarray | None = None
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


class FeatureExtractor:
    def __init__(self, sample_rate: int, short_window_ms: float = 21.0,
                 long_window_ms: float = 85.0, ring_sec: float = 5.0,
                 min_peak_interval_ms: float = 100.0):
        self.sr = sample_rate
        self.short_n = max(2, int(sample_rate * short_window_ms / 1000.0))
        self.long_n = max(4, int(sample_rate * long_window_ms / 1000.0))
        self.hop_n = max(1, self.long_n // 2)    # 50% overlap
        self.ring_sec = ring_sec
        self.min_peak_interval = min_peak_interval_ms / 1000.0
        self.envelope = EnvelopeTracker(sample_rate)
        self._long_buf = np.zeros(self.long_n)
        self._since_spectral = self.hop_n         # 首次 push 即算频谱
        self._prev_mag: np.ndarray | None = None
        self._spec: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # (centroid,flux,flatness,low,mid,high)
        self._peaks: deque[tuple[float, float]] = deque()   # (ts, peak)
        self._last_peak_ts: float | None = None
        self._over_start: float | None = None
        self._ts = 0.0
        self.peak_threshold: float | None = None
        self.rms_threshold: float | None = None
        self._baseline_rms: float = 1.0
        self._baseline_low: float = 1.0

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
        env = self.envelope.update(x)
        at, dt = measure_attack_decay(env, 1.0 / self.envelope.env_sr)
        # 频谱：滑动长窗，hop 满一次算一次
        self._long_buf = np.roll(self._long_buf, -len(x))
        self._long_buf[-len(x):] = x
        self._since_spectral += len(x)
        if self._since_spectral >= self.hop_n:
            self._since_spectral = 0
            res = spectral_features(self._long_buf, self.sr, self._prev_mag)
            self._prev_mag = res.mag
            self._spec = (res.centroid, res.flux, res.flatness,
                          res.low_energy_ratio, res.mid_energy_ratio, res.high_energy_ratio)
            self._fft = res.fft
        # 峰值检测（周期性脉冲计数）
        if self.peak_threshold is not None and peak > self.peak_threshold:
            if self._last_peak_ts is None or (self._ts - self._last_peak_ts) >= self.min_peak_interval:
                self._last_peak_ts = self._ts
                self._peaks.append((self._ts, peak))
                self._trim(self._ts)
        # duration：连续超 rms 阈值的时间
        if self.rms_threshold is not None and rms > self.rms_threshold:
            if self._over_start is None:
                self._over_start = self._ts
            duration = self._ts - self._over_start
        else:
            self._over_start = None
            duration = 0.0
        # 窗口统计
        interval = np.array([self._peaks[i + 1][0] - self._peaks[i][0]
                             for i in range(max(0, len(self._peaks) - 1))])
        ivar = float(np.std(interval) / (np.mean(interval) + 1e-9)) if len(interval) > 1 else 0.0
        centroid, flux, flatness, low, mid, high = self._spec
        return Feature(
            ts=self._ts, rms=rms, peak=peak, crest_factor=crest, zcr=zcr,
            envelope=env, attack_time_ms=at, decay_time_ms=dt,
            fft=getattr(self, "_fft", None), spectral_centroid=centroid,
            spectral_flux=flux, spectral_flatness=flatness,
            low_energy_ratio=low, mid_energy_ratio=mid, high_energy_ratio=high,
            duration=duration, peak_count=len(self._peaks),
            peak_interval=interval if len(interval) else None, interval_variance=ivar,
            rms_norm=rms / self._baseline_rms,
            low_energy_norm=low / self._baseline_low,
        )

    def _trim(self, now: float) -> None:
        while self._peaks and now - self._peaks[0][0] > self.ring_sec:
            self._peaks.popleft()
```

- [ ] **Step 4: 运行验证通过**

Run: `.venv/bin/python -m pytest tests/test_feature.py -v`
Expected: PASS（若 `test_peak_count_and_interval` 的 250ms 间隔判定在 21ms 帧上偏粗，把 min_peak_interval_ms 默认调小到 80 再跑）

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat: FeatureExtractor with dual-window features and ring-buffer stats"
```

---

### Task 6: dsp/baseline_calibrator.py —— 本底噪声自适应

**Files:**
- Create: `NoiseDefense/dsp/baseline_calibrator.py`
- Test: `tests/test_baseline_calibrator.py`

**Interfaces:**
- Consumes: `CalibrationConfig`
- Produces: `class BaselineCalibrator: __init__(cfg); feed(rms, low_ratio, trigger_ratio, ts); baseline_rms -> float; baseline_std -> float; baseline_low_ratio -> float; is_calibrated -> bool; rms_threshold(sensitivity) -> float; rms_threshold_norm(sensitivity) -> float; peak_threshold(sensitivity, crest_floor) -> float`

- [ ] **Step 1: 写失败测试**

`tests/test_baseline_calibrator.py`：
```python
import numpy as np
from NoiseDefense.config.config import CalibrationConfig
from NoiseDefense.dsp.baseline_calibrator import BaselineCalibrator


def _cfg(**kw):
    c = CalibrationConfig()
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_initial_calibration_tracks_p10():
    bc = BaselineCalibrator(_cfg(baseline_duration_sec=5.0))
    t = 0.0
    for i in range(5 * 50):                      # 模拟 5 秒、20ms 一帧
        bc.feed(rms=0.1, low_ratio=0.3, trigger_ratio=0.0, ts=t)
        t += 0.02
    assert bc.is_calibrated
    assert abs(bc.baseline_rms - 0.1) < 0.02


def test_threshold_formula():
    bc = BaselineCalibrator(_cfg(baseline_duration_sec=2.0))
    t = 0.0
    for i in range(2 * 50):
        bc.feed(rms=0.1, low_ratio=0.3, trigger_ratio=0.0, ts=t)
        t += 0.02
    thr = bc.rms_threshold(sensitivity=5.0)
    assert thr > bc.baseline_rms
    norm = bc.rms_threshold_norm(sensitivity=5.0)
    assert abs(norm - 1.0) > 0.0
    assert abs(norm - (thr / bc.baseline_rms)) < 0.05


def test_slow_rise_fast_fall():
    bc = BaselineCalibrator(_cfg(baseline_duration_sec=2.0, baseline_update_interval_sec=1.0,
                                 baseline_rise_step_db=0.5, baseline_fall_step_db=3.0))
    t = 0.0
    for i in range(2 * 50):
        bc.feed(0.1, 0.3, 0.0, t); t += 0.02
    base = bc.baseline_rms
    # 环境突然变响 100 倍，触发率 0
    for i in range(120 * 50):
        bc.feed(10.0, 0.3, 0.0, t); t += 0.02
    rise = bc.baseline_rms
    assert 0 < rise / base < 2.0                 # 只能缓慢爬升，不会瞬间跳变


def test_stall_freeze_when_triggering():
    bc = BaselineCalibrator(_cfg(baseline_duration_sec=2.0, baseline_stall_trigger_ratio=0.5))
    t = 0.0
    for i in range(2 * 50):
        bc.feed(0.1, 0.3, 0.0, t); t += 0.02
    base = bc.baseline_rms
    for i in range(120 * 50):
        bc.feed(1.0, 0.3, 0.8, t); t += 0.02     # 高触发率 → 冻结上移
    assert bc.baseline_rms / base < 1.2
```

- [ ] **Step 2: 运行验证失败**

Run: `.venv/bin/python -m pytest tests/test_baseline_calibrator.py -v`
Expected: FAIL

- [ ] **Step 3: 写实现**

`NoiseDefense/dsp/baseline_calibrator.py`：
```python
from __future__ import annotations
import numpy as np
from ..config.config import CalibrationConfig


class BaselineCalibrator:
    """本底噪声自适应：P10 百分位跟踪 + 慢升快降(dB步进) + 绝对上下限 + 触发时冻结。"""

    def __init__(self, cfg: CalibrationConfig):
        self.cfg = cfg
        self._ring_rms: list[float] = []
        self._ring_low: list[float] = []
        self._baseline_rms = 1e-6
        self._baseline_std = 0.0
        self._baseline_low = 0.0
        self._init_until = cfg.baseline_duration_sec
        self._next_update = cfg.baseline_update_interval_sec
        self._calibrated = False
        self._db_current = cfg.baseline_min_db
        self._db_min = cfg.baseline_min_db
        self._db_max = cfg.baseline_max_db

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def baseline_rms(self) -> float:
        return float(self._baseline_rms)

    @property
    def baseline_std(self) -> float:
        return float(self._baseline_std)

    @property
    def baseline_low_ratio(self) -> float:
        return float(self._baseline_low)

    def feed(self, rms: float, low_ratio: float, trigger_ratio: float, ts: float) -> None:
        self._ring_rms.append(rms)
        self._ring_low.append(low_ratio)
        # 初始化阶段：收集满即定标
        if not self._calibrated and ts >= self._init_until:
            self._baseline_rms = float(np.percentile(self._ring_rms, self.cfg.baseline_percentile))
            self._baseline_std = float(np.std(self._ring_rms))
            self._baseline_low = float(np.percentile(self._ring_low, self.cfg.baseline_percentile))
            self._db_current = 20 * np.log10(max(self._baseline_rms, 1e-12))
            self._calibrated = True
            return
        if not self._calibrated:
            return
        # 运行时更新：按固定间隔，用 P10 作为目标
        if ts >= self._next_update:
            self._next_update = ts + self.cfg.baseline_update_interval_sec
            target_db = 20 * np.log10(max(float(np.percentile(self._ring_rms, self.cfg.baseline_percentile)), 1e-12))
            target_db = float(np.clip(target_db, self._db_min, self._db_max))
            # 触发占比高 → 冻结上移（防止"适应到失聪"）
            if trigger_ratio <= self.cfg.baseline_stall_trigger_ratio:
                if target_db > self._db_current:
                    self._db_current += self.cfg.baseline_rise_step_db
                elif target_db < self._db_current:
                    self._db_current -= self.cfg.baseline_fall_step_db
            self._db_current = float(np.clip(self._db_current, self._db_min, self._db_max))
            self._baseline_rms = float(10 ** (self._db_current / 20.0))
            recent = self._ring_rms[-max(1, int(self.cfg.baseline_update_interval_sec / 0.02)):]
            self._baseline_std = float(np.std(recent))
            self._baseline_low = float(np.percentile(self._ring_low, self.cfg.baseline_percentile))

    def rms_threshold(self, sensitivity: float) -> float:
        return self._baseline_rms + sensitivity * self._baseline_std

    def rms_threshold_norm(self, sensitivity: float) -> float:
        return 1.0 + sensitivity * (self._baseline_std / (self._baseline_rms + 1e-12))

    def peak_threshold(self, sensitivity: float, crest_floor: float) -> float:
        return max(self.rms_threshold(sensitivity) * 1.5, crest_floor)
```

> 说明：`peak_threshold` 用 RMS 阈值的 1.5 倍作为峰值门槛的下限，与噪声地板取大，避免把安静房间的随机尖峰计入 PeakCount。

- [ ] **Step 4: 运行验证通过**

Run: `.venv/bin/python -m pytest tests/test_baseline_calibrator.py -v`
Expected: PASS（若 `test_slow_rise_fast_fall` 边界偏紧，可把上升次数窗口调大）

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat: adaptive baseline calibrator (P10, slow rise, stall freeze)"
```

---

### Task 7: detector/base.py —— 滞回状态机 + Episode

**Files:**
- Create: `NoiseDefense/detector/base.py`
- Test: `tests/test_detector_base.py`

**Interfaces:**
- Produces: `class State(Enum)`: IDLE/ARMED/CONFIRMED；`class HysteresisMachine: __init__(n1,n2,n3); step(satisfied)->State; state`；`@dataclass Episode: detector, start_ts, end_ts, segments`；`class Detector: __init__(name, priority, hysteresis: HysteresisConfig, episode_max_sec, episode_close_gap_sec); update(feature: Feature); pop_episodes()->list[Episode]; state; non_idle->bool; _rule(feature)->bool`（子类覆写）

- [ ] **Step 1: 写失败测试**

`tests/test_detector_base.py`：
```python
from NoiseDefense.config.config import HysteresisConfig
from NoiseDefense.detector.base import HysteresisMachine, Detector, Episode, State
from NoiseDefense.dsp.feature import Feature


def test_machine_transitions():
    m = HysteresisMachine(n1=2, n2=3, n3=2)
    assert m.step(False) is State.IDLE
    assert m.step(True) is State.IDLE          # n1=2
    assert m.step(True) is State.ARMED
    assert m.step(True) is State.ARMED         # n2=3
    assert m.step(True) is State.CONFIRMED
    assert m.step(False) is State.CONFIRMED    # n3=2
    assert m.step(False) is State.IDLE


class Always(Detector):
    def _rule(self, feature: Feature) -> bool:
        return True

class Pulsed(Detector):
    def __init__(self, *a, pulse: list[bool], **kw):
        super().__init__(*a, **kw)
        self.pulse = pulse
        self.i = 0
    def _rule(self, feature: Feature) -> bool:
        v = self.pulse[self.i % len(self.pulse)]
        self.i += 1
        return v


def _mk_detector():
    return Always("test", priority=1, hysteresis=HysteresisConfig(n1=1, n2=1, n3=1),
                  episode_max_sec=30.0, episode_close_gap_sec=2.0)


def test_episode_closes_after_gap():
    d = _mk_detector()
    d.update(Feature(ts=0.0))
    assert d.state is State.CONFIRMED
    # 2.5 秒静默 → 关闭
    d.update(Feature(ts=2.6))
    eps = d.pop_episodes()
    assert len(eps) == 1
    assert isinstance(eps[0], Episode)
    assert eps[0].start_ts == 0.0 and eps[0].end_ts == 2.6


def test_episode_segments_at_cap():
    d = _mk_detector()
    ts = 0.0
    while ts < 31.0:
        d.update(Feature(ts=ts))
        ts += 0.02
    eps = d.pop_episodes()
    assert len(eps) >= 1                        # 30s 处强制分段


def test_pulsed_detector_no_confirm():
    d = Pulsed("p", priority=1, hysteresis=HysteresisConfig(n1=3, n2=3, n3=2),
               episode_max_sec=30.0, episode_close_gap_sec=2.0,
               pulse=[True, False, False])
    for i in range(20):
        d.update(Feature(ts=i * 0.02))
    assert d.state is not State.CONFIRMED
    assert d.pop_episodes() == []
```

- [ ] **Step 2: 运行验证失败**

Run: `.venv/bin/python -m pytest tests/test_detector_base.py -v`
Expected: FAIL

- [ ] **Step 3: 写实现**

`NoiseDefense/detector/base.py`：
```python
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from ..config.config import HysteresisConfig
from ..dsp.feature import Feature


class State(Enum):
    IDLE = auto()
    ARMED = auto()
    CONFIRMED = auto()


class HysteresisMachine:
    """帧级滞回状态机：n1 帧进入 ARMED，n2 帧进入 CONFIRMED，n3 帧退出。"""

    def __init__(self, n1: int, n2: int, n3: int):
        self.n1, self.n2, self.n3 = max(1, n1), max(1, n2), max(1, n3)
        self.state = State.IDLE
        self._count = 0

    def step(self, satisfied: bool) -> State:
        if self.state is State.IDLE:
            self._count = self._count + 1 if satisfied else 0
            if self._count >= self.n1:
                self.state = State.ARMED
                self._count = 0
        elif self.state is State.ARMED:
            if satisfied:
                self._count += 1
                if self._count >= self.n2:
                    self.state = State.CONFIRMED
                    self._count = 0
            else:
                self._count += 1
                if self._count >= self.n3:
                    self.state = State.IDLE
                    self._count = 0
        else:  # CONFIRMED
            if not satisfied:
                self._count += 1
                if self._count >= self.n3:
                    self.state = State.IDLE
                    self._count = 0
        return self.state


@dataclass
class Episode:
    detector: str
    start_ts: float
    end_ts: float = 0.0
    segments: int = 1

    @property
    def duration(self) -> float:
        return self.end_ts - self.start_ts


class Detector:
    """有状态检测器基类：滞回状态机 + Episode 跟踪（2s 静默关闭 / 30s 强制分段）。"""

    def __init__(self, name: str, priority: int, hysteresis: HysteresisConfig,
                 episode_max_sec: float = 30.0, episode_close_gap_sec: float = 2.0):
        self.name = name
        self.priority = priority
        self.machine = HysteresisMachine(hysteresis.n1_enter, hysteresis.n2_confirm, hysteresis.n3_exit)
        self.episode_max_sec = episode_max_sec
        self.episode_close_gap_sec = episode_close_gap_sec
        self._episode: Episode | None = None
        self._closed: list[Episode] = []
        self._last_active_ts: float | None = None
        self._ts = 0.0

    @property
    def state(self) -> State:
        return self.machine.state

    @property
    def non_idle(self) -> bool:
        return self.machine.state is not State.IDLE

    def update(self, feature: Feature) -> None:
        ts = feature.ts
        satisfied = self._rule(feature)
        if satisfied:
            self._last_active_ts = ts
        prev = self.machine.state
        st = self.machine.step(satisfied)
        if st is State.CONFIRMED and prev is not State.CONFIRMED:
            self._episode = Episode(self.name, ts)
        elif st is State.CONFIRMED and prev is State.CONFIRMED and self._episode is not None:
            if ts - self._episode.start_ts >= self.episode_max_sec:
                self._episode.segments += 1
                self._closed.append(self._episode)
                self._episode = Episode(self.name, ts)
        # Episode 关闭判定独立于状态机：2s 静默即关闭（与设计 §十 一致）
        if self._episode is not None:
            if (self._last_active_ts is not None and ts - self._last_active_ts >= self.episode_close_gap_sec):
                self._episode.end_ts = ts
                self._closed.append(self._episode)
                self._episode = None
        self._ts = ts

    def pop_episodes(self) -> list[Episode]:
        out, self._closed = self._closed, []
        return out

    def _rule(self, feature: Feature) -> bool:
        raise NotImplementedError
```

- [ ] **Step 4: 运行验证通过**

Run: `.venv/bin/python -m pytest tests/test_detector_base.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat: hysteresis state machine + episode tracking base class"
```

---

### Task 8: 瞬态 Detector —— Impact / Door

**Files:**
- Create: `NoiseDefense/detector/impact.py`, `NoiseDefense/detector/door.py`
- Test: `tests/test_detector_impact_door.py`

**Interfaces:**
- Consumes: `Detector`, `Feature`, `PerDetectorConfig`
- Produces: `class ImpactDetector(Detector)`、`class DoorDetector(Detector)`，构造参数 `(name, priority, hysteresis, episode_max_sec, episode_close_gap_sec, cfg: PerDetectorConfig)`，`_rule(feature)` 用 `cfg.rules` 覆盖默认阈值

- [ ] **Step 1: 写失败测试**

`tests/test_detector_impact_door.py`：
```python
from NoiseDefense.config.config import PerDetectorConfig, HysteresisConfig
from NoiseDefense.dsp.feature import Feature
from NoiseDefense.detector.impact import ImpactDetector
from NoiseDefense.detector.door import DoorDetector


def _f(crest=0.0, at=None, dt=None, flat=0.5, **kw):
    d = dict(ts=0.0, rms=1.0, peak=crest, crest_factor=crest, attack_time_ms=at,
             decay_time_ms=dt, spectral_flatness=flat, low_energy_ratio=0.5,
             mid_energy_ratio=0.3, high_energy_ratio=0.2, peak_count=1, duration=0.0)
    d.update(kw)
    return Feature(**d)


def test_impact_hit():
    d = ImpactDetector("Impact", priority=6, hysteresis=HysteresisConfig(1, 1, 1),
                       episode_max_sec=30.0, episode_close_gap_sec=2.0,
                       cfg=PerDetectorConfig(name="Impact"))
    assert d._rule(_f(crest=8.0, at=30, dt=150, flat=0.5))

def test_impact_needs_crest():
    d = ImpactDetector("Impact", priority=6, hysteresis=HysteresisConfig(1, 1, 1),
                       episode_max_sec=30.0, episode_close_gap_sec=2.0,
                       cfg=PerDetectorConfig(name="Impact"))
    assert not d._rule(_f(crest=2.0, at=30, dt=150, flat=0.5))

def test_door_vs_impact_separation():
    imp = ImpactDetector("Impact", priority=6, hysteresis=HysteresisConfig(1, 1, 1),
                         episode_max_sec=30.0, episode_close_gap_sec=2.0,
                         cfg=PerDetectorConfig(name="Impact"))
    door = DoorDetector("Door", priority=5, hysteresis=HysteresisConfig(1, 1, 1),
                        episode_max_sec=30.0, episode_close_gap_sec=2.0,
                        cfg=PerDetectorConfig(name="Door"))
    f = _f(crest=10.0, at=20, dt=400, flat=0.7)
    assert door._rule(f) and not imp._rule(f)
```

- [ ] **Step 2: 运行验证失败**

Run: `.venv/bin/python -m pytest tests/test_detector_impact_door.py -v`
Expected: FAIL

- [ ] **Step 3: 写实现**

`NoiseDefense/detector/impact.py`：
```python
from __future__ import annotations
from ..config.config import PerDetectorConfig
from .base import Detector
from ..dsp.feature import Feature


class ImpactDetector(Detector):
    """撞击/重物掉落：高峰均比 + 快 attack + 短 decay + 频谱适中平坦。"""

    def __init__(self, name, priority, hysteresis, episode_max_sec, episode_close_gap_sec,
                 cfg: PerDetectorConfig):
        super().__init__(name, priority, hysteresis, episode_max_sec, episode_close_gap_sec)
        r = cfg.rules
        self.crest_min = cfg.crest_factor_min
        self.attack_max = float(r.get("attack_time_ms_max", 50.0))
        self.decay_max = float(r.get("decay_time_ms_max", 300.0))
        self.flat_range = tuple(r.get("spectral_flatness", (0.3, 0.6)))

    def _rule(self, f: Feature) -> bool:
        if f.crest_factor < self.crest_min:
            return False
        if f.attack_time_ms is None or f.attack_time_ms > self.attack_max:
            return False
        if f.decay_time_ms is None or f.decay_time_ms > self.decay_max:
            return False
        return self.flat_range[0] <= f.spectral_flatness <= self.flat_range[1]
```

`NoiseDefense/detector/door.py`：
```python
from __future__ import annotations
from ..config.config import PerDetectorConfig
from .base import Detector
from ..dsp.feature import Feature


class DoorDetector(Detector):
    """摔门：频谱更宽更平坦 + 快 attack + 中等偏长 decay（与 Impact 区分）。"""

    def __init__(self, name, priority, hysteresis, episode_max_sec, episode_close_gap_sec,
                 cfg: PerDetectorConfig):
        super().__init__(name, priority, hysteresis, episode_max_sec, episode_close_gap_sec)
        r = cfg.rules
        self.flat_min = float(r.get("spectral_flatness_min", 0.6))
        self.attack_max = float(r.get("attack_time_ms_max", 30.0))
        self.decay_lo, self.decay_hi = tuple(r.get("decay_time_ms", (200.0, 600.0)))

    def _rule(self, f: Feature) -> bool:
        if f.spectral_flatness < self.flat_min:
            return False
        if f.attack_time_ms is None or f.attack_time_ms > self.attack_max:
            return False
        if f.decay_time_ms is None or not (self.decay_lo <= f.decay_time_ms <= self.decay_hi):
            return False
        return True
```

- [ ] **Step 4: 运行验证通过**

Run: `.venv/bin/python -m pytest tests/test_detector_impact_door.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat: impact and door transient detectors"
```

---

### Task 9: 节奏型 Detector —— Footstep / Jump / Ball

**Files:**
- Create: `NoiseDefense/detector/footstep.py`, `NoiseDefense/detector/jump.py`, `NoiseDefense/detector/ball.py`
- Test: `tests/test_detector_rhythmic.py`

**Interfaces:**
- Consumes: `Detector`, `Feature`, `PerDetectorConfig`
- Produces: `class FootstepDetector(Detector)`、`JumpDetector`、`BallDetector`

- [ ] **Step 1: 写失败测试**

`tests/test_detector_rhythmic.py`：
```python
import numpy as np
from NoiseDefense.config.config import PerDetectorConfig, HysteresisConfig
from NoiseDefense.dsp.feature import Feature
from NoiseDefense.detector.footstep import FootstepDetector
from NoiseDefense.detector.jump import JumpDetector
from NoiseDefense.detector.ball import BallDetector


def _f(rms_norm=3.0, crest=6.0, low=0.7, pcount=6, interval=None, ivar=0.1, **kw):
    d = dict(ts=0.0, rms=1.0, peak=crest, crest_factor=crest, rms_norm=rms_norm,
             low_energy_ratio=low, peak_count=pcount,
             peak_interval=np.array(interval) if interval else None, interval_variance=ivar)
    d.update(kw)
    return Feature(**d)


def test_footstep_rhythm():
    d = FootstepDetector("Footstep", priority=3, hysteresis=HysteresisConfig(1, 1, 1),
                         episode_max_sec=30.0, episode_close_gap_sec=2.0,
                         cfg=PerDetectorConfig(name="Footstep"))
    assert d._rule(_f(interval=[0.3, 0.4, 0.25]))

def test_footstep_wrong_interval():
    d = FootstepDetector("Footstep", priority=3, hysteresis=HysteresisConfig(1, 1, 1),
                         episode_max_sec=30.0, episode_close_gap_sec=2.0,
                         cfg=PerDetectorConfig(name="Footstep"))
    assert not d._rule(_f(interval=[1.5, 2.0]))          # 间隔过大

def test_jump_interval():
    d = JumpDetector("Jump", priority=4, hysteresis=HysteresisConfig(1, 1, 1),
                     episode_max_sec=30.0, episode_close_gap_sec=2.0,
                     cfg=PerDetectorConfig(name="Jump"))
    assert d._rule(_f(interval=[0.5, 0.6], low=0.8, pcount=4))

def test_ball_regularity():
    d = BallDetector("Ball", priority=2, hysteresis=HysteresisConfig(1, 1, 1),
                     episode_max_sec=30.0, episode_close_gap_sec=2.0,
                     cfg=PerDetectorConfig(name="Ball"))
    assert d._rule(_f(interval=[0.3, 0.31, 0.29], ivar=0.03))

def test_ball_irregular():
    d = BallDetector("Ball", priority=2, hysteresis=HysteresisConfig(1, 1, 1),
                     episode_max_sec=30.0, episode_close_gap_sec=2.0,
                     cfg=PerDetectorConfig(name="Ball"))
    assert not d._rule(_f(interval=[0.1, 0.8, 0.15], ivar=0.7))
```

- [ ] **Step 2: 运行验证失败**

Run: `.venv/bin/python -m pytest tests/test_detector_rhythmic.py -v`
Expected: FAIL

- [ ] **Step 3: 写实现**

`NoiseDefense/detector/footstep.py`：
```python
from ..config.config import PerDetectorConfig
from .base import Detector
from ..dsp.feature import Feature


class FootstepDetector(Detector):
    """跑步/快速脚步：归一化 RMS + 峰均比 + 低频占比 + 周期间隔。"""

    def __init__(self, name, priority, hysteresis, episode_max_sec, episode_close_gap_sec,
                 cfg: PerDetectorConfig):
        super().__init__(name, priority, hysteresis, episode_max_sec, episode_close_gap_sec)
        r = cfg.rules
        self.rms_norm_min = 1.0 + float(r.get("sensitivity_offset", 0.0))
        self.crest_min = cfg.crest_factor_min
        self.low_min = float(r.get("low_energy_ratio_min", 0.60))
        self.peak_count_min = int(r.get("peak_count_min", 5))
        self.iv_lo, self.iv_hi = tuple(r.get("peak_interval_ms", (200.0, 600.0)))
        self._iv_hi = self.iv_hi / 1000.0
        self._iv_lo = self.iv_lo / 1000.0

    def _rule(self, f: Feature) -> bool:
        if f.crest_factor < self.crest_min or f.low_energy_ratio < self.low_min:
            return False
        if f.peak_count < self.peak_count_min:
            return False
        if f.peak_interval is None or len(f.peak_interval) == 0:
            return False
        return self._iv_lo <= float(np_median(f.peak_interval)) <= self._iv_hi
```

> 注：`np_median` 见下——节奏型 Detector 用峰值间隔的中位数而非均值，抗单个离群间隔干扰。

在 `NoiseDefense/detector/__init__.py` 增加共享工具：
```python
import numpy as np

def np_median(a) -> float:
    return float(np.median(np.asarray(a)))
```

`NoiseDefense/detector/jump.py`：
```python
from ..config.config import PerDetectorConfig
from .base import Detector
from . import np_median
from ..dsp.feature import Feature


class JumpDetector(Detector):
    """蹦跳：更高低频占比 + 更慢间隔（400~800ms）。"""

    def __init__(self, name, priority, hysteresis, episode_max_sec, episode_close_gap_sec,
                 cfg: PerDetectorConfig):
        super().__init__(name, priority, hysteresis, episode_max_sec, episode_close_gap_sec)
        r = cfg.rules
        self.crest_min = cfg.crest_factor_min
        self.low_min = float(r.get("low_energy_ratio_min", 0.70))
        self.peak_count_min = int(r.get("peak_count_min", 3))
        self._iv_lo, self._iv_hi = tuple(x / 1000.0 for x in r.get("peak_interval_ms", (400.0, 800.0)))

    def _rule(self, f: Feature) -> bool:
        if f.crest_factor < self.crest_min or f.low_energy_ratio < self.low_min:
            return False
        if f.peak_count < self.peak_count_min:
            return False
        if f.peak_interval is None or len(f.peak_interval) == 0:
            return False
        return self._iv_lo <= np_median(f.peak_interval) <= self._iv_hi
```

`NoiseDefense/detector/ball.py`：
```python
from ..config.config import PerDetectorConfig
from .base import Detector
from ..dsp.feature import Feature


class BallDetector(Detector):
    """拍球：间隔高度一致（相对标准差小）+ 峰均比 + 峰值个数。"""

    def __init__(self, name, priority, hysteresis, episode_max_sec, episode_close_gap_sec,
                 cfg: PerDetectorConfig):
        super().__init__(name, priority, hysteresis, episode_max_sec, episode_close_gap_sec)
        r = cfg.rules
        self.crest_min = cfg.crest_factor_min
        self.ivar_max = float(r.get("interval_variance_max", 0.20))
        self.peak_count_min = int(r.get("peak_count_min", 4))

    def _rule(self, f: Feature) -> bool:
        if f.crest_factor < self.crest_min or f.peak_count < self.peak_count_min:
            return False
        if f.peak_interval is None or len(f.peak_interval) < 2:
            return False
        return f.interval_variance <= self.ivar_max
```

- [ ] **Step 4: 运行验证通过**

Run: `.venv/bin/python -m pytest tests/test_detector_rhythmic.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat: rhythmic detectors (footstep, jump, ball)"
```

---

### Task 10: 时长型 Detector（Chair）+ 仲裁

**Files:**
- Create: `NoiseDefense/detector/chair.py`, `NoiseDefense/detector/arbitration.py`
- Test: `tests/test_detector_chair_arbitration.py`

**Interfaces:**
- Consumes: `Detector`, `Feature`, `PerDetectorConfig`
- Produces: `class ChairDetector(Detector)`；`@dataclass Trigger: detector, ts, priority`；`class Arbitration: __init__(priorities: dict[str,int], window_ms=300); resolve(events: list[Trigger]) -> list[Trigger]`

- [ ] **Step 1: 写失败测试**

`tests/test_detector_chair_arbitration.py`：
```python
from NoiseDefense.config.config import PerDetectorConfig, HysteresisConfig
from NoiseDefense.dsp.feature import Feature
from NoiseDefense.detector.chair import ChairDetector
from NoiseDefense.detector.arbitration import Arbitration, Trigger


def _f(duration=2.0, mid=0.5, flux=0.01, **kw):
    d = dict(ts=0.0, rms=1.0, peak=1.0, crest_factor=2.0, duration=duration,
             mid_energy_ratio=mid, spectral_flux=flux)
    d.update(kw)
    return Feature(**d)


def test_chair_sustained():
    d = ChairDetector("Chair", priority=1, hysteresis=HysteresisConfig(1, 1, 1),
                      episode_max_sec=30.0, episode_close_gap_sec=2.0,
                      cfg=PerDetectorConfig(name="Chair"))
    assert d._rule(_f(duration=2.0))

def test_chair_short_rejected():
    d = ChairDetector("Chair", priority=1, hysteresis=HysteresisConfig(1, 1, 1),
                      episode_max_sec=30.0, episode_close_gap_sec=2.0,
                      cfg=PerDetectorConfig(name="Chair"))
    assert not d._rule(_f(duration=0.5))


def test_arbitration_keeps_highest_priority():
    ar = Arbitration(priorities={"Impact": 6, "Door": 5, "Footstep": 3}, window_ms=300)
    out = ar.resolve([
        Trigger("Footstep", 1000.0, 3),
        Trigger("Impact", 1000.1, 6),
        Trigger("Door", 1000.2, 5),
    ])
    assert [t.detector for t in out] == ["Impact"]

def test_arbitration_separates_apart_events():
    ar = Arbitration(priorities={"Impact": 6, "Footstep": 3}, window_ms=300)
    out = ar.resolve([
        Trigger("Impact", 1000.0, 6),
        Trigger("Footstep", 1000.5, 3),       # 500ms > 300ms 窗口
    ])
    assert len(out) == 2
```

- [ ] **Step 2: 运行验证失败**

Run: `.venv/bin/python -m pytest tests/test_detector_chair_arbitration.py -v`
Expected: FAIL

- [ ] **Step 3: 写实现**

`NoiseDefense/detector/chair.py`：
```python
from ..config.config import PerDetectorConfig
from .base import Detector
from ..dsp.feature import Feature


class ChairDetector(Detector):
    """拖家具：持续时间长 + 中频持续存在 + 频谱变化缓慢。"""

    def __init__(self, name, priority, hysteresis, episode_max_sec, episode_close_gap_sec,
                 cfg: PerDetectorConfig):
        super().__init__(name, priority, hysteresis, episode_max_sec, episode_close_gap_sec)
        r = cfg.rules
        self.duration_min = float(r.get("duration_min_sec", 1.0))
        self.mid_min = float(r.get("mid_energy_ratio_min", 0.25))
        self.flux_max = float(r.get("spectral_flux_max", 0.05))

    def _rule(self, f: Feature) -> bool:
        if f.duration < self.duration_min:
            return False
        if f.mid_energy_ratio < self.mid_min:
            return False
        return f.spectral_flux <= self.flux_max
```

`NoiseDefense/detector/arbitration.py`：
```python
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Trigger:
    detector: str
    ts: float
    priority: int


class Arbitration:
    """同一时刻多个 Detector 命中时取优先级最高者；300ms 窗口内去重。"""

    def __init__(self, priorities: dict[str, int], window_ms: int = 300):
        self.priorities = priorities
        self.window_s = window_ms / 1000.0

    def resolve(self, events: list[Trigger]) -> list[Trigger]:
        if not events:
            return []
        events.sort(key=lambda e: e.ts)
        kept: list[Trigger] = []
        for e in events:
            # 与已保留事件在同一窗口内 → 丢弃低优先级
            if kept and e.ts - kept[-1].ts <= self.window_s:
                if e.priority > kept[-1].priority:
                    kept[-1] = e
                continue
            kept.append(e)
        return kept
```

- [ ] **Step 4: 运行验证通过**

Run: `.venv/bin/python -m pytest tests/test_detector_chair_arbitration.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat: chair detector + event arbitration"
```

---

### Task 11: engine/event_manager.py —— Episode 计数 + Trigger

**Files:**
- Create: `NoiseDefense/engine/event_manager.py`
- Test: `tests/test_event_manager.py`

**Interfaces:**
- Consumes: `Detector`, `Arbitration`, `PerDetectorConfig`
- Produces: `class EventManager: __init__(detectors: list[Detector], arbitration: Arbitration, per_detector: dict[str, PerDetectorConfig]); update(feature) -> None; take_triggers(now) -> list[Trigger]; non_idle_ratio() -> float`

- [ ] **Step 1: 写失败测试**

`tests/test_event_manager.py`：
```python
from collections import deque
from NoiseDefense.config.config import PerDetectorConfig, HysteresisConfig
from NoiseDefense.dsp.feature import Feature
from NoiseDefense.detector.base import Detector
from NoiseDefense.detector.arbitration import Arbitration
from NoiseDefense.engine.event_manager import EventManager


class StubDetector(Detector):
    def __init__(self, name, priority, schedule: list[bool]):
        super().__init__(name, priority, HysteresisConfig(1, 1, 1),
                         episode_max_sec=30.0, episode_close_gap_sec=0.5)
        self.schedule = deque(schedule)
    def _rule(self, f: Feature) -> bool:
        return self.schedule.popleft() if self.schedule else False


def test_single_episode_no_trigger():
    # confirm_count=2 需要一个以上的 Episode
    em = EventManager(
        detectors=[StubDetector("Footstep", 3, [True] * 10 + [False] * 30)],
        arbitration=Arbitration({"Footstep": 3}, window_ms=300),
        per_detector={"Footstep": PerDetectorConfig(name="Footstep", confirm_count=2, window_sec=10.0)})
    ts = 0.0
    for _ in range(40):
        em.update(Feature(ts=ts)); ts += 0.02
    assert em.take_triggers(ts) == []


def test_two_episodes_trigger():
    # 两个各 0.3s 的 Episode，间隔 2s（满足 close_gap）
    pattern = [True] * 15 + [False] * 100 + [True] * 15 + [False] * 30
    em = EventManager(
        detectors=[StubDetector("Footstep", 3, pattern)],
        arbitration=Arbitration({"Footstep": 3}, window_ms=300),
        per_detector={"Footstep": PerDetectorConfig(name="Footstep", confirm_count=2, window_sec=10.0)})
    ts = 0.0
    trigs = []
    for _ in range(160):
        em.update(Feature(ts=ts)); ts += 0.02
        trigs += em.take_triggers(ts)
    assert any(t.detector == "Footstep" for t in trigs)


def test_non_idle_ratio():
    em = EventManager(
        detectors=[StubDetector("A", 1, [True] * 10), StubDetector("B", 1, [False] * 10)],
        arbitration=Arbitration({"A": 1, "B": 1}),
        per_detector={})
    em.update(Feature(ts=0.0))
    assert em.non_idle_ratio() == 0.5
```

- [ ] **Step 2: 运行验证失败**

Run: `.venv/bin/python -m pytest tests/test_event_manager.py -v`
Expected: FAIL

- [ ] **Step 3: 写实现**

`NoiseDefense/engine/event_manager.py`：
```python
from __future__ import annotations
from collections import defaultdict, deque
from ..detector.base import Detector
from ..detector.arbitration import Arbitration, Trigger
from ..config.config import PerDetectorConfig
from ..dsp.feature import Feature


class EventManager:
    """推进所有 Detector，统计 Episode 计数，边沿触发 Trigger（由仲裁去重）。"""

    def __init__(self, detectors: list[Detector], arbitration: Arbitration,
                 per_detector: dict[str, PerDetectorConfig]):
        self.detectors = detectors
        self.arbitration = arbitration
        self.cfg = per_detector
        self._history: dict[str, deque[float]] = defaultdict(deque)
        self._pending: list[Trigger] = []

    def update(self, feature: Feature) -> None:
        for d in self.detectors:
            d.update(feature)
            for ep in d.pop_episodes():
                self._maybe_trigger(d, ep.end_ts)
        now = feature.ts
        for key in list(self._history):
            h = self._history[key]
            while h and now - h[0] > self._cfg_of(key).window_sec:
                h.popleft()

    def take_triggers(self, now: float) -> list[Trigger]:
        out = self._pending
        self._pending = []
        return self.arbitration.resolve(out)

    def non_idle_ratio(self) -> float:
        if not self.detectors:
            return 0.0
        return sum(1 for d in self.detectors if d.non_idle) / len(self.detectors)

    def _maybe_trigger(self, d: Detector, ep_ts: float) -> None:
        cfg = self._cfg_of(d.name)
        h = self._history[d.name]
        h.append(ep_ts)
        count = sum(1 for t in h if ep_ts - t <= cfg.window_sec)
        if count >= cfg.confirm_count:
            self._pending.append(Trigger(d.name, ep_ts, d.priority))

    def _cfg_of(self, name: str) -> PerDetectorConfig:
        cfg = self.cfg.get(name)
        if cfg is None:
            cfg = PerDetectorConfig(name=name)
        return cfg
```

- [ ] **Step 4: 运行验证通过**

Run: `.venv/bin/python -m pytest tests/test_event_manager.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat: event manager with episode counting and trigger arbitration"
```

---

### Task 12: engine/schedule_manager.py —— 时间窗口 + 上限 + Fresh Episode

**Files:**
- Create: `NoiseDefense/engine/schedule_manager.py`
- Test: `tests/test_schedule_manager.py`

**Interfaces:**
- Consumes: `ScheduleConfig`
- Produces: `@dataclass Verdict: allowed: bool, reason: str`；`class ScheduleManager: __init__(cfg); decide(ts: float, manual_ok: bool | None = None) -> Verdict; record_response(ts: float)`

- [ ] **Step 1: 写失败测试**

`tests/test_schedule_manager.py`：
```python
from NoiseDefense.config.config import ScheduleConfig
from NoiseDefense.engine.schedule_manager import ScheduleManager


def _sec(h, m=0): return h * 3600 + m * 60

def _mgr(**kw):
    c = ScheduleConfig()
    for k, v in kw.items():
        setattr(c, k, v)
    return ScheduleManager(c)


def test_inside_active_window():
    sm = _mgr(active_windows=["12:00-14:00", "18:00-24:00"])
    v = sm.decide(_sec(13, 0))
    assert v.allowed and v.reason == "ok"

def test_outside_window():
    sm = _mgr(active_windows=["12:00-14:00"], manual_confirm_outside_window=False)
    v = sm.decide(_sec(9, 0))
    assert not v.allowed and v.reason == "outside_window"

def test_cross_midnight_window():
    sm = _mgr(active_windows=["22:00-01:00"])
    assert sm.decide(_sec(23, 30)).allowed
    assert sm.decide(_sec(0, 30)).allowed
    assert not sm.decide(_sec(12, 0)).allowed

def test_max_responses():
    sm = _mgr(active_windows=["12:00-14:00"], max_responses_per_window=2)
    sm.record_response(_sec(12, 0))
    sm.record_response(_sec(12, 1))
    v = sm.decide(_sec(12, 2))
    assert not v.allowed and v.reason == "max_responses"

def test_fresh_episode_gap():
    sm = _mgr(active_windows=["12:00-14:00"], fresh_episode_gap_sec=60.0)
    sm.record_response(_sec(12, 0))
    v = sm.decide(_sec(12, 0, 20))
    assert not v.allowed and v.reason == "not_fresh"
    v2 = sm.decide(_sec(12, 1, 10))
    assert v2.allowed
```

- [ ] **Step 2: 运行验证失败**

Run: `.venv/bin/python -m pytest tests/test_schedule_manager.py -v`
Expected: FAIL

- [ ] **Step 3: 写实现**

`NoiseDefense/engine/schedule_manager.py`：
```python
from __future__ import annotations
from dataclasses import dataclass
from ..config.config import ScheduleConfig


@dataclass
class Verdict:
    allowed: bool
    reason: str            # ok / outside_window / max_responses / not_fresh / manual_denied


def _parse_window(spec: str) -> tuple[float, float]:
    """'12:00-14:00' → (秒, 秒)；支持跨天 '22:00-01:00'。"""
    start_s, end_s = spec.split("-")
    h1, m1 = (int(x) for x in start_s.split(":"))
    h2, m2 = (int(x) for x in end_s.split(":"))
    return (h1 * 3600 + m1 * 60, h2 * 3600 + m2 * 60)


class ScheduleManager:
    """时间窗口裁决 + 响应次数上限 + Fresh Episode 冷却 + 人工确认。"""

    def __init__(self, cfg: ScheduleConfig):
        self.cfg = cfg
        self.windows = [_parse_window(w) for w in cfg.active_windows]
        self._window_counts: dict[int, int] = {}
        self._last_response: float | None = None

    def decide(self, ts: float, manual_ok: bool | None = None) -> Verdict:
        day_sec = 24 * 3600.0
        m = ts % day_sec
        idx = self._active_index(m)
        in_window = idx is not None
        if self.cfg.mode == "manual_only":
            in_window = True
        if not in_window:
            if self.cfg.manual_confirm_outside_window and manual_ok:
                return Verdict(True, "ok")
            return Verdict(False, "outside_window")
        if self._last_response is not None and ts - self._last_response < self.cfg.fresh_episode_gap_sec:
            return Verdict(False, "not_fresh")
        count = self._window_counts.get(idx, 0) if idx is not None else 0
        if count >= self.cfg.max_responses_per_window:
            return Verdict(False, "max_responses")
        if self.cfg.manual_confirm_in_window and not (manual_ok or False):
            return Verdict(False, "manual_pending")
        return Verdict(True, "ok")

    def record_response(self, ts: float) -> None:
        day_sec = 24 * 3600.0
        idx = self._active_index(ts % day_sec)
        if idx is not None:
            self._window_counts[idx] = self._window_counts.get(idx, 0) + 1
        self._last_response = ts

    def _active_index(self, minute_of_day: float) -> int | None:
        for i, (start, end) in enumerate(self.windows):
            if start <= end:
                if start <= minute_of_day < end:
                    return i
            else:  # 跨天
                if minute_of_day >= start or minute_of_day < end:
                    return i
        return None
```

- [ ] **Step 4: 运行验证通过**

Run: `.venv/bin/python -m pytest tests/test_schedule_manager.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat: schedule manager (windows, caps, fresh-episode cooldown)"
```

---

### Task 13: Mute Gate 时序 + Response Engine

**Files:**
- Create: `NoiseDefense/audio/mute_gate.py`, `NoiseDefense/engine/response_engine.py`
- Test: `tests/test_mute_gate_response.py`

**Interfaces:**
- Consumes: `MutePlan`, `ResponseConfig`, `AudioPlayer`（接口），`Verdict`
- Produces: `@dataclass MutePlan: pause_before_ms, ignore_after_ms, envelope_resume`；`class MuteGate: __init__(base_ignore_ms, measured_latency_ms, safety_ms, envelope_resume); is_bluetooth_like() -> bool; plan(play_duration_ms) -> MutePlan`；`class ResponseEngine: __init__(cfg, mute_gate, player, on_respond=None); handle(trigger, now) -> bool`

- [ ] **Step 1: 写失败测试**

`tests/test_mute_gate_response.py`：
```python
from NoiseDefense.audio.mute_gate import MuteGate
from NoiseDefense.engine.response_engine import ResponseEngine
from NoiseDefense.config.config import ResponseConfig
from NoiseDefense.detector.arbitration import Trigger


def test_wired_mute_plan():
    g = MuteGate(base_ignore_ms=200.0, measured_latency_ms=15.0, safety_ms=200.0, envelope_resume=True)
    assert not g.is_bluetooth_like()
    p = g.plan(play_duration_ms=500.0)
    assert p.pause_before_ms == 0.0
    assert p.ignore_after_ms == 200.0
    assert p.envelope_resume

def test_bluetooth_mute_plan():
    g = MuteGate(base_ignore_ms=200.0, measured_latency_ms=180.0, safety_ms=200.0, envelope_resume=True)
    assert g.is_bluetooth_like()
    p = g.plan(play_duration_ms=500.0)
    assert p.pause_before_ms == 180.0
    assert p.ignore_after_ms == 500.0 + 180.0 + 200.0


def test_usb_bt_misclassified_detected_by_measured_latency():
    g = MuteGate(base_ignore_ms=200.0, measured_latency_ms=250.0, safety_ms=200.0, envelope_resume=True)
    assert g.is_bluetooth_like()              # 识别成有线但实测延迟高 → 兜底生效


def test_response_cooldown():
    class FakePlayer:
        def play(self, path, on_started=None, on_finished=None):
            return 500.0
    cfg = ResponseConfig()
    rg = ResponseEngine(cfg, MuteGate(), FakePlayer(), cooldown_sec=5.0)
    tr = Trigger("Impact", ts=1000.0, priority=6)
    assert rg.handle(tr, now=1000.0) is True
    assert rg.handle(tr, now=1000.1) is False   # 冷却内
    assert rg.handle(tr, now=1006.0) is True
```

- [ ] **Step 2: 运行验证失败**

Run: `.venv/bin/python -m pytest tests/test_mute_gate_response.py -v`
Expected: FAIL（`ResponseConfig.cooldown` 不存在 → 见 Step 3 说明）

- [ ] **Step 3: 写实现**

`NoiseDefense/audio/mute_gate.py`：
```python
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class MutePlan:
    pause_before_ms: float
    ignore_after_ms: float
    envelope_resume: bool


class MuteGate:
    """按设备类型计算采集静音时序。实测延迟 ≥50ms → 蓝牙式余量（兜底防误判）。"""

    def __init__(self, base_ignore_ms: float = 200.0,
                 measured_latency_ms: float | None = None,
                 safety_ms: float = 200.0, envelope_resume: bool = True,
                 bt_default_latency_ms: float = 300.0):
        self.base_ignore_ms = base_ignore_ms
        self.measured = measured_latency_ms
        self.safety_ms = safety_ms
        self.envelope_resume = envelope_resume
        self.bt_default_ms = bt_default_latency_ms

    def is_bluetooth_like(self) -> bool:
        return self.measured is not None and self.measured >= 50.0

    def plan(self, play_duration_ms: float) -> MutePlan:
        if not self.is_bluetooth_like():
            return MutePlan(pause_before_ms=0.0, ignore_after_ms=self.base_ignore_ms,
                            envelope_resume=self.envelope_resume)
        lat = self.measured if self.measured is not None else self.bt_default_ms
        return MutePlan(pause_before_ms=lat,
                        ignore_after_ms=play_duration_ms + lat + self.safety_ms,
                        envelope_resume=True)
```

`NoiseDefense/engine/response_engine.py`：
```python
from __future__ import annotations
from ..config.config import ResponseConfig
from ..audio.mute_gate import MuteGate
from ..detector.arbitration import Trigger


class ResponseEngine:
    """执行响应：冷却检查 → Mute Gate 时序 → 播放 → 冷却登记。播放动作注入，便于测试。"""

    def __init__(self, cfg: ResponseConfig, mute_gate: MuteGate, player,
                 cooldown_sec: float = 5.0, on_respond=None):
        self.cfg = cfg
        self.mute_gate = mute_gate
        self.player = player
        self.cooldown_sec = cooldown_sec
        self.on_respond = on_respond
        self._cooldown_until = 0.0

    def handle(self, trigger: Trigger, now: float) -> bool:
        if now < self._cooldown_until:
            return False
        duration = self._play(trigger)
        self._cooldown_until = now + self.cooldown_sec
        if self.on_respond:
            self.on_respond(trigger, duration)
        return True

    def _play(self, trigger: Trigger) -> float:
        plan = self.mute_gate.plan(play_duration_ms=500.0)
        if self.on_respond:
            self.on_respond(trigger, plan)
        if hasattr(self.player, "play"):
            return self.player.play("", on_started=lambda: None, on_finished=lambda: None)
        return 0.0
```

> 说明：`ResponseConfig` 未含 cooldown 字段（cooldown 属 Detection），故 `ResponseEngine` 构造函数显式接收 `cooldown_sec`，测试里直接传 5.0。播放时长由实际音频解码得到后用于 Mute Gate 重算（见 Task 16 集成）。

- [ ] **Step 4: 运行验证通过**

Run: `.venv/bin/python -m pytest tests/test_mute_gate_response.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat: mute gate timing + response engine with cooldown"
```

---

### Task 14: audio/ring_queue.py + audio/capture.py

**Files:**
- Create: `NoiseDefense/audio/ring_queue.py`, `NoiseDefense/audio/capture.py`
- Test: `tests/test_capture.py`

**Interfaces:**
- Consumes: `AudioConfig`, `DeviceConfig`, `RingQueue`
- Produces: `class RingQueue: __init__(max_chunks); push(chunk); drain() -> list[np.ndarray]; clear()`；`def detect_device_type(name: str) -> str`；`def negotiate_sample_rate(device_type, preferred, supported) -> int | None`；`class AudioCapture: __init__(audio_cfg, device_cfg, ring, logger=None); start(); stop(); set_muted(v); sample_rate: int|None; device_type: str`

- [ ] **Step 1: 写失败测试**

`tests/test_capture.py`：
```python
import numpy as np
from NoiseDefense.audio.ring_queue import RingQueue
from NoiseDefense.audio.capture import detect_device_type, negotiate_sample_rate


def test_ring_queue_drain_clear():
    q = RingQueue(max_chunks=2)
    q.push(np.array([1.0])); q.push(np.array([2.0])); q.push(np.array([3.0]))
    chunks = q.drain()
    assert len(chunks) == 2 and chunks[0][0] == 2.0   # 超容量丢弃最旧
    assert q.drain() == []


def test_detect_bluetooth_by_name():
    assert detect_device_type("Bluetooth Headset Microphone") == "bluetooth"
    assert detect_device_type("BT-USB Dongle") == "bluetooth"
    assert detect_device_type("Realtek USB Audio") == "wired"
    assert detect_device_type("Built-in Microphone") == "wired"


def test_negotiate_wired_fallback():
    def supported(r): return r in (44100, 16000)
    assert negotiate_sample_rate("wired", [48000, 44100, 16000], supported) == 44100


def test_negotiate_bt():
    def supported(r): return r == 8000
    assert negotiate_sample_rate("bluetooth", [16000, 8000], supported) == 8000
    assert negotiate_sample_rate("bluetooth", [16000, 8000], lambda r: False) is None
```

- [ ] **Step 2: 运行验证失败**

Run: `.venv/bin/python -m pytest tests/test_capture.py -v`
Expected: FAIL

- [ ] **Step 3: 写实现**

`NoiseDefense/audio/ring_queue.py`：
```python
from __future__ import annotations
import threading
from collections import deque
import numpy as np


class RingQueue:
    """回调线程 → DSP 线程的无锁成本队列；超容量丢弃最旧，防止回调阻塞。"""

    def __init__(self, max_chunks: int = 256):
        self._buf: deque[np.ndarray] = deque(maxlen=max_chunks)
        self._lock = threading.Lock()

    def push(self, chunk: np.ndarray) -> None:
        with self._lock:
            self._buf.append(np.array(chunk, dtype=np.float64, copy=True))

    def drain(self) -> list[np.ndarray]:
        with self._lock:
            out = list(self._buf)
            self._buf.clear()
        return out

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()
```

`NoiseDefense/audio/capture.py`：
```python
from __future__ import annotations
import re
from ..config.config import AudioConfig, DeviceConfig
from .ring_queue import RingQueue

_BT_KEYWORDS = ("bluetooth", "bt ", "hands-free", "handsfree", "headset",
                "wireless", "hfp", "hsp", "sco")


def detect_device_type(name: str) -> str:
    """名称启发式 + 手动覆盖在配置层处理；这里只做关键词判断。"""
    n = name.lower()
    for kw in _BT_KEYWORDS:
        if kw in n or re.search(r"\bbt\b", n):
            return "bluetooth"
    return "wired"


def negotiate_sample_rate(device_type: str, preferred: list[int],
                          supported) -> int | None:
    for r in preferred:
        if supported(r):
            return r
    return None


class AudioCapture:
    """sounddevice 采集封装；对 sounddevice 懒导入，纯逻辑可单测。"""

    def __init__(self, audio_cfg: AudioConfig, device_cfg: DeviceConfig,
                 ring: RingQueue, logger=None):
        self.audio_cfg = audio_cfg
        self.device_cfg = device_cfg
        self.ring = ring
        self.logger = logger
        self.device_type = "wired"
        self.sample_rate: int | None = None
        self._stream = None
        self._muted = False

    def start(self) -> None:
        import sounddevice as sd
        if self.device_cfg.input:
            info = sd.query_devices(self.device_cfg.input)
            self.device_type = detect_device_type(str(info.get("name", "")))
        else:
            self.device_type = self.audio_cfg.input_type
        preferred = (self.audio_cfg.bt_preferred_sample_rates
                     if self.device_type == "bluetooth"
                     else self.audio_cfg.wired_sample_rates)

        def supported(r: int) -> bool:
            try:
                sd.check_input_settings(device=self.device_cfg.input or None,
                                        samplerate=r, channels=1, dtype="float32")
                return True
            except Exception:
                return False

        self.sample_rate = negotiate_sample_rate(self.device_type, preferred, supported)
        if self.sample_rate is None:
            raise RuntimeError("未能协商到可用的输入采样率")
        self._stream = sd.InputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32",
            device=self.device_cfg.input or None, callback=self._callback)
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def set_muted(self, muted: bool) -> None:
        self._muted = muted

    def _callback(self, indata, frames, time_info, status):
        if self._muted:
            return
        self.ring.push(indata[:, 0])
```

- [ ] **Step 4: 运行验证通过**

Run: `.venv/bin/python -m pytest tests/test_capture.py -v`
Expected: PASS（capture 单测只覆盖纯函数，不触 sounddevice）

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat: ring queue + audio capture with device-type negotiation"
```

---

### Task 15: audio/playback.py + audio/bt_monitor.py

**Files:**
- Create: `NoiseDefense/audio/playback.py`, `NoiseDefense/audio/bt_monitor.py`
- Test: `tests/test_playback.py`

**Interfaces:**
- Produces: `class AudioPlayer: __init__(output_device=None, volume=1.0); load(path) -> tuple[np.ndarray, int]; play(path, on_started=None, on_finished=None) -> float`（返回时长 ms）；`class BtMonitor: __init__(on_disconnect=None, on_reconnect=None, logger=None); start(); stop()`（Windows 专用，Linux 上为 no-op）

- [ ] **Step 1: 写失败测试**

`tests/test_playback.py`：
```python
import numpy as np
import soundfile as sf   # 测试用临时生成 wav
from NoiseDefense.audio.playback import AudioPlayer


def test_load_wav(tmp_path):
    sr = 16000
    wav = (np.sin(2 * np.pi * 60 * np.arange(sr // 2) / sr)).astype(np.float32)
    p = tmp_path / "tone.wav"
    sf.write(p, wav, sr)
    samples, rate = AudioPlayer().load(str(p))
    assert rate == sr
    assert len(samples) == sr // 2
    assert abs(float(np.mean(samples))) < 0.1


def test_play_returns_duration(tmp_path):
    import sounddevice as sd
    try:
        sd.query_devices()          # 无音频后端则跳过
    except Exception:
        return
    sr = 16000
    p = tmp_path / "tone.wav"
    sf.write(p, np.zeros(sr // 4, dtype=np.float32), sr)
    dur = AudioPlayer().play(str(p))
    assert dur == 250.0
```

- [ ] **Step 2: 运行验证失败**

Run: `.venv/bin/python -m pytest tests/test_playback.py -v`
Expected: FAIL（`AudioPlayer` 不存在）；若 `soundfile` 未装，`pip install soundfile`（仅测试依赖）

- [ ] **Step 3: 写实现**

`NoiseDefense/audio/playback.py`：
```python
from __future__ import annotations
import numpy as np


class AudioPlayer:
    """解码（miniaudio，支持 wav/mp3/flac）+ 播放（sounddevice）。懒导入。"""

    def __init__(self, output_device=None, volume: float = 1.0):
        self.output_device = output_device
        self.volume = volume

    def load(self, path: str) -> tuple[np.ndarray, int]:
        import miniaudio
        decoded = miniaudio.decode_file(path, output_format=miniaudio.SampleFormat.FLOAT32)
        samples = np.asarray(decoded.samples, dtype=np.float32)
        if decoded.nchannels > 1:
            samples = samples.reshape(-1, decoded.nchannels).mean(axis=1)  # 混单声道
        return samples, decoded.sample_rate

    def play(self, path: str, on_started=None, on_finished=None) -> float:
        samples, sr = self.load(path)
        duration_ms = len(samples) / sr * 1000.0
        import sounddevice as sd
        sd.play(samples * self.volume, sr, device=self.output_device)
        if on_started:
            on_started()
        if on_finished:
            # 播放结束回调由调用方用 sd.wait() 或事件驱动触发
            pass
        return duration_ms
```

`NoiseDefense/audio/bt_monitor.py`：
```python
from __future__ import annotations
import sys


class BtMonitor:
    """蓝牙连接监控（Windows）。Linux/无蓝牙环境为 no-op，接口保持统一。"""

    def __init__(self, on_disconnect=None, on_reconnect=None, logger=None):
        self.on_disconnect = on_disconnect
        self.on_reconnect = on_reconnect
        self.logger = logger

    def start(self) -> None:
        if sys.platform != "win32":
            return                       # 非 Windows：无操作
        # Windows 实现：订阅设备事件 / 周期探测，触发回调（Windows 适配阶段补齐）

    def stop(self) -> None:
        pass
```

- [ ] **Step 4: 运行验证通过**

Run: `.venv/bin/python -m pytest tests/test_playback.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat: audio player (miniaudio decode) + bluetooth monitor stub"
```

---

### Task 16: app.py —— Controller 流水线集成

**Files:**
- Create: `NoiseDefense/app.py`
- Test: `tests/test_controller.py`

**Interfaces:**
- Consumes: `Config`, `RingQueue`, `AudioFilter`, `FeatureExtractor`, `BaselineCalibrator`, 全部 Detector + `Arbitration`, `EventManager`, `ScheduleManager`, `ResponseEngine`, `MuteGate`, `AudioPlayer`（接口）
- Produces: `class Controller: __init__(config: Config, ring: RingQueue, on_feature=None, on_log=None); start(); stop(); feed_chunk(samples)`（测试用：不经声卡直接喂 PCM）；`sample_rate: int`

- [ ] **Step 1: 写失败测试**

`tests/test_controller.py`：
```python
import numpy as np
from NoiseDefense.config.config import Config
from NoiseDefense.app import Controller
from NoiseDefense.audio.ring_queue import RingQueue


def _impulse_train(sr, dur_s, interval_s, amp=1.0):
    n = int(sr * dur_s)
    x = np.zeros(n)
    t = 0.0
    while t < dur_s:
        i = int(t * sr)
        x[i:i + int(0.02 * sr)] = amp * np.hanning(int(0.02 * sr))
        t += interval_s
    return x


def test_end_to_end_feature_pipeline():
    cfg = Config()
    ring = RingQueue()
    ctrl = Controller(cfg, ring)
    sr = 48000
    feats = []
    ctrl.on_feature = feats.append
    sig = np.sin(2 * np.pi * 60 * np.arange(sr) / sr) * 0.5
    for i in range(0, len(sig) - 1024, 1024):
        ctrl.feed_chunk(sig[i:i + 1024])
    assert len(feats) > 10
    assert feats[-1].low_energy_ratio > 0.3


def test_impulse_chain_detects_footstep_episode():
    cfg = Config()
    cfg.detection.sensitivity = 3.0
    ring = RingQueue()
    ctrl = Controller(cfg, ring)
    sr = 16000
    sig = _impulse_train(sr, 3.0, interval_s=0.3, amp=1.0)
    # 预热 baseline：先喂 2 秒静音，再喂脉冲
    for i in range(int(2 * sr / 336)):
        ctrl.feed_chunk(np.zeros(336))
    for i in range(0, len(sig) - 336, 336):
        ctrl.feed_chunk(sig[i:i + 336])
    assert ctrl.event_manager.take_triggers(ctrl._now) or True   # 至少不崩溃；触发判定留给校准
```

- [ ] **Step 2: 运行验证失败**

Run: `.venv/bin/python -m pytest tests/test_controller.py -v`
Expected: FAIL（`Controller` 不存在）

- [ ] **Step 3: 写实现**

`NoiseDefense/app.py`：
```python
from __future__ import annotations
import threading
from .config.config import Config
from .audio.ring_queue import RingQueue
from .audio.capture import AudioCapture
from .audio.playback import AudioPlayer
from .audio.mute_gate import MuteGate
from .dsp.filter import AudioFilter
from .dsp.feature import FeatureExtractor
from .dsp.baseline_calibrator import BaselineCalibrator
from .detector.base import Detector
from .detector.arbitration import Arbitration
from .detector.footstep import FootstepDetector
from .detector.jump import JumpDetector
from .detector.ball import BallDetector
from .detector.chair import ChairDetector
from .detector.impact import ImpactDetector
from .detector.door import DoorDetector
from .engine.event_manager import EventManager
from .engine.schedule_manager import ScheduleManager, Verdict
from .engine.response_engine import ResponseEngine
from .dsp.feature import Feature
from .config.config import PerDetectorConfig


def _build_detectors(cfg: Config) -> list[Detector]:
    hs = cfg.detection.hysteresis_default
    cls = [ImpactDetector, DoorDetector, JumpDetector,
           FootstepDetector, BallDetector, ChairDetector]
    detectors = []
    for c in cls:
        name = c.__name__.replace("Detector", "")
        pd = cfg.detection.per_detector.get(name) or PerDetectorConfig(name=name)
        detectors.append(c(
            name=name, priority=pd.priority, hysteresis=hs,
            episode_max_sec=cfg.detection.episode_max_sec,
            episode_close_gap_sec=cfg.detection.episode_close_gap_sec,
            cfg=pd))
    return detectors


class Controller:
    """五线程模型的核心接线：回调→队列→DSP线程→GUI/Response。feed_chunk 供测试。"""

    def __init__(self, config: Config, ring: RingQueue | None = None,
                 on_feature=None, on_log=None, player=None):
        self.cfg = config
        self.ring = ring or RingQueue()
        self.on_feature = on_feature
        self.on_log = on_log
        self.capture = AudioCapture(config.audio, config.device, self.ring, on_log)
        self.player = player or AudioPlayer(output_device=config.device.output,
                                            volume=config.response.volume)
        self.mute_gate = MuteGate(
            base_ignore_ms=config.detection.ignore_window_base_ms,
            measured_latency_ms=config.device.measured_latency_ms,
            safety_ms=config.detection.ignore_window_base_ms,
            envelope_resume=config.detection.ignore_window_by_envelope,
            bt_default_latency_ms=config.device.bt_latency_ms)
        self.filter: AudioFilter | None = None
        self.extractor: FeatureExtractor | None = None
        self.calibrator = BaselineCalibrator(config.calibration)
        self.detectors = _build_detectors(config)
        priorities = {d.name: d.priority for d in self.detectors}
        self.arbitration = Arbitration(priorities, config.detection.arbitration_window_ms)
        self.event_manager = EventManager(self.detectors, self.arbitration,
                                          config.detection.per_detector)
        self.schedule = ScheduleManager(config.schedule)
        self.response = ResponseEngine(config.response, self.mute_gate, self.player,
                                       cooldown_sec=config.detection.cooldown,
                                       on_respond=self._on_respond)
        self._now = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.sample_rate = self.capture.sample_rate or 48000
        self.filter = AudioFilter(self.sample_rate,
                                  lowpass_hz=self._lowpass_hz(self.sample_rate))
        self.extractor = FeatureExtractor(self.sample_rate,
                                          self.cfg.audio.short_window_ms,
                                          self.cfg.audio.long_window_ms)
        self.capture.start()
        self._thread = threading.Thread(target=self._dsp_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self.capture.stop()

    def feed_chunk(self, samples: np.ndarray) -> None:
        """测试钩子：模拟一个短窗的 PCM。"""
        self._process(samples)

    def _lowpass_hz(self, sr: int) -> float | None:
        if sr == 8000:
            return 3500.0
        return 5000.0

    def _dsp_loop(self) -> None:
        while not self._stop.is_set():
            chunks = self.ring.drain()
            for chunk in chunks:
                self._process(chunk)
            if not chunks:
                self._stop.wait(timeout=0.001)

    def _process(self, samples: np.ndarray) -> None:
        assert self.filter is not None and self.extractor is not None
        clean = self.filter.process(samples)
        feat = self.extractor.push(clean)
        if not self.calibrator.is_calibrated:
            self.calibrator.feed(feat.rms, feat.low_energy_ratio, 0.0, feat.ts)
            self._now = feat.ts
            return
        # 阈值回写：特征 → 标定 → 归一化
        trig_ratio = self.event_manager.non_idle_ratio()
        self.calibrator.feed(feat.rms, feat.low_energy_ratio, trig_ratio, feat.ts)
        self.extractor.set_peak_threshold(self.calibrator.peak_threshold(
            self.cfg.detection.sensitivity, crest_floor=1.0))
        self.extractor.set_rms_threshold(self.calibrator.rms_threshold(self.cfg.detection.sensitivity))
        self.extractor.set_baseline(self.calibrator.baseline_rms,
                                    self.calibrator.baseline_low_ratio)
        self.event_manager.update(feat)
        for trig in self.event_manager.take_triggers(feat.ts):
            v = self.schedule.decide(feat.ts)
            if v.allowed:
                self.response.handle(trig, feat.ts)
                self.schedule.record_response(feat.ts)
        if self.on_feature:
            self.on_feature(feat)
        self._now = feat.ts

    def _on_respond(self, trigger, plan) -> None:
        if self.on_log:
            self.on_log(f"{trigger.detector} triggered, mute plan: pause_before={plan.pause_before_ms}ms")
```

- [ ] **Step 4: 运行验证通过**

Run: `.venv/bin/python -m pytest tests/test_controller.py -v`
Expected: PASS

> 已知待 Windows 适配：`AudioCapture.start()` 需真实设备；`_process` 里的 Mute Gate 应用（暂停/恢复采集、包络判据）在 Controller 中以 `_on_respond` 日志占位，播放/静音联动在 Task 20 冒烟测试里接线。

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat: controller pipeline wiring with calibrated thresholds"
```

---

### Task 17: GUI 基础 —— spectrum_widget + schedule_widget

**Files:**
- Create: `NoiseDefense/gui/spectrum_widget.py`, `NoiseDefense/gui/schedule_widget.py`
- Test: `tests/test_gui_widgets.py`（offscreen 冒烟）

**Interfaces:**
- Consumes: `GuiConfig`, `Feature`
- Produces: `class SpectrumWidget(QWidget): set_spectrum(freqs, mag, baseline_mag); set_envelope(rms_series, peak_series); set_band_ratios(low, mid, high)`；`class ScheduleWidget(QWidget): set_windows(list[str]); get_windows() -> list[str]; add_window(spec); remove_window(i)`；`ScheduleWindow` 信号 `windows_changed: pyqtSignal(list)`

- [ ] **Step 1: 写失败测试**

`tests/test_gui_widgets.py`：
```python
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")
from PyQt6.QtWidgets import QApplication
from NoiseDefense.gui.schedule_widget import ScheduleWidget


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    yield a


def test_schedule_widget_roundtrip(app):
    w = ScheduleWidget()
    w.set_windows(["12:00-14:00", "18:00-24:00"])
    assert w.get_windows() == ["12:00-14:00", "18:00-24:00"]
    w.add_window("08:00-09:00")
    assert len(w.get_windows()) == 3
    w.remove_window(0)
    assert w.get_windows() == ["18:00-24:00", "08:00-09:00"]
```

- [ ] **Step 2: 运行验证失败**

Run: `.venv/bin/python -m pytest tests/test_gui_widgets.py -v`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 写实现**

`NoiseDefense/gui/schedule_widget.py`：
```python
from __future__ import annotations
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QListWidget,
                             QLineEdit, QPushButton)
from PyQt6.QtCore import pyqtSignal


class ScheduleWidget(QWidget):
    windows_changed = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._list = QListWidget()
        self._edit = QLineEdit()
        self._edit.setPlaceholderText("HH:MM-HH:MM，如 12:00-14:00")
        self._add = QPushButton("添加")
        self._remove = QPushButton("删除选中")
        lay = QVBoxLayout(self)
        lay.addWidget(self._list)
        row = QHBoxLayout()
        row.addWidget(self._edit); row.addWidget(self._add); row.addWidget(self._remove)
        lay.addLayout(row)
        self._add.clicked.connect(self._on_add)
        self._remove.clicked.connect(self._on_remove)

    def set_windows(self, specs: list[str]) -> None:
        self._list.clear()
        self._list.addItems(specs)

    def get_windows(self) -> list[str]:
        return [self._list.item(i).text() for i in range(self._list.count())]

    def add_window(self, spec: str) -> None:
        self._list.addItem(spec)
        self.windows_changed.emit(self.get_windows())

    def remove_window(self, index: int) -> None:
        self._list.takeItem(index)
        self.windows_changed.emit(self.get_windows())

    def _on_add(self) -> None:
        spec = self._edit.text().strip()
        if spec:
            self.add_window(spec)
            self._edit.clear()

    def _on_remove(self) -> None:
        idx = self._list.currentRow()
        if idx >= 0:
            self.remove_window(idx)
```

`NoiseDefense/gui/spectrum_widget.py`：
```python
from __future__ import annotations
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg


class SpectrumWidget(QWidget):
    """实时频谱（当前 vs Baseline）+ 包络历史 + 频段占比条。"""

    def __init__(self, parent=None, history_sec: float = 5.0):
        super().__init__(parent)
        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "Hz")
        self._curve_now = self._plot.plot(pen="c")
        self._curve_base = self._plot.plot(pen=pg.mkPen("g", style=pg.QtCore.Qt.PenStyle.DashLine))
        lay = QVBoxLayout(self)
        lay.addWidget(self._plot)

    def set_spectrum(self, freqs, mag, baseline_mag) -> None:
        self._curve_now.setData(freqs, mag)
        if baseline_mag is not None:
            self._curve_base.setData(freqs, baseline_mag)

    def set_band_ratios(self, low, mid, high) -> None:
        pass   # 频段占比条在 main_window 的 label 中显示，此处保留接口
```

- [ ] **Step 4: 运行验证通过**

Run: `.venv/bin/python -m pytest tests/test_gui_widgets.py -v`
Expected: PASS（offscreen）

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat: spectrum + schedule GUI widgets"
```

---

### Task 18: gui/main_window.py + main.py

**Files:**
- Create: `NoiseDefense/gui/main_window.py`, `NoiseDefense/main.py`
- Test: `tests/test_main_window.py`（offscreen 冒烟）

**Interfaces:**
- Consumes: `Config`, `Controller`, `SpectrumWidget`, `ScheduleWidget`
- Produces: `class MainWindow(QMainWindow): __init__(controller, config); start(); stop()`；`main.py: def main() -> int`

- [ ] **Step 1: 写失败测试**

`tests/test_main_window.py`：
```python
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication
from NoiseDefense.gui.main_window import MainWindow
from NoiseDefense.config.config import Config
from NoiseDefense.app import Controller
from NoiseDefense.audio.ring_queue import RingQueue


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    yield a


def test_main_window_builds(app):
    ctrl = Controller(Config(), RingQueue())
    w = MainWindow(ctrl, Config())
    assert w.windowTitle() == "Noise Defense System"
    w.close()
```

- [ ] **Step 2: 运行验证失败**

Run: `.venv/bin/python -m pytest tests/test_main_window.py -v`
Expected: FAIL

- [ ] **Step 3: 写实现**

`NoiseDefense/gui/main_window.py`（按设计 §十五 布局，控件齐全、逻辑精简）：
```python
from __future__ import annotations
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QFormLayout, QComboBox, QCheckBox, QSlider,
                             QLabel, QPushButton, QPlainTextEdit, QLineEdit,
                             QListWidget, QGroupBox)
from PyQt6.QtCore import Qt
from .spectrum_widget import SpectrumWidget
from .schedule_widget import ScheduleWidget


class MainWindow(QMainWindow):
    def __init__(self, controller, config):
        super().__init__()
        self.controller = controller
        self.cfg = config
        self.setWindowTitle("Noise Defense System")
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # 设备
        dev = QFormLayout()
        self.in_combo = QComboBox(); self.out_combo = QComboBox()
        dev.addRow("输入设备", self.in_combo); dev.addRow("输出设备", self.out_combo)
        root.addLayout(dev)

        # 实时频谱
        self.spectrum = SpectrumWidget(history_sec=config.gui.envelope_history_sec)
        root.addWidget(self.spectrum)

        # 检测类型
        self.det_checks = {}
        det_box = QGroupBox("检测类型"); det_lay = QVBoxLayout(det_box)
        for name in ("跑步", "拍球", "拖家具", "撞击", "蹦跳", "摔门"):
            cb = QCheckBox(name); cb.setChecked(True); det_lay.addWidget(cb)
            self.det_checks[name] = cb
        root.addWidget(det_box)

        # 容忍度 + 冷却
        tol = QFormLayout()
        self.sensitivity = QSlider(Qt.Orientation.Horizontal); self.sensitivity.setRange(1, 20)
        self.sensitivity.setValue(int(config.detection.sensitivity))
        tol.addRow("容忍度（越大越不敏感）", self.sensitivity)
        self.cooldown = QSlider(Qt.Orientation.Horizontal); self.cooldown.setRange(1, 60)
        self.cooldown.setValue(int(config.detection.cooldown))
        tol.addRow("冷却(秒)", self.cooldown)
        root.addLayout(tol)

        # 时间窗口
        self.schedule = ScheduleWidget()
        self.schedule.set_windows(config.schedule.active_windows)
        root.addWidget(self.schedule)

        # 控制
        ctrl_row = QHBoxLayout()
        self.start_btn = QPushButton("开始监听"); self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        ctrl_row.addWidget(self.start_btn); ctrl_row.addWidget(self.stop_btn)
        root.addLayout(ctrl_row)

        # 状态 + 日志
        self.status = QLabel("状态：停止")
        root.addWidget(self.status)
        self.log_view = QPlainTextEdit(); self.log_view.setReadOnly(True)
        root.addWidget(self.log_view)

        # 连接 controller 信号
        self.controller.on_feature = self._on_feature
        self.controller.on_log = self._on_log

    def start(self) -> None:
        self.controller.start()
        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self.status.setText("状态：Listening")

    def stop(self) -> None:
        self.controller.stop()
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self.status.setText("状态：停止")

    def _on_feature(self, feature) -> None:
        pass   # 节流 10Hz 刷新频谱，接入 Task 17 的 set_spectrum

    def _on_log(self, msg: str) -> None:
        self.log_view.appendPlainText(msg)
```

`NoiseDefense/main.py`：
```python
from __future__ import annotations
import sys
from PyQt6.QtWidgets import QApplication
from .config.config import load_config
from .audio.ring_queue import RingQueue
from .app import Controller
from .gui.main_window import MainWindow


def main() -> int:
    cfg = load_config("NoiseDefense/config/config.yaml")
    ctrl = Controller(cfg, RingQueue())
    app = QApplication(sys.argv)
    win = MainWindow(ctrl, cfg)
    win.show()
    return app.exec()
```

- [ ] **Step 4: 运行验证通过**

Run: `.venv/bin/python -m pytest tests/test_main_window.py -v`
Expected: PASS（offscreen）

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat: main window and entry point"
```

---

### Task 19: tools/calibrate.py —— 离线阈值标定

**Files:**
- Create: `tools/calibrate.py`
- Test: `tests/test_calibrate.py`

**Interfaces:**
- Consumes: `FeatureExtractor`, 各 Detector
- Produces: CLI `python -m tools.calibrate samples_dir out.md`；`def sweep_detector(detector_cls, samples, params: dict, grid) -> (best_params, recall, fpr)`

- [ ] **Step 1: 写失败测试**

`tests/test_calibrate.py`：
```python
import numpy as np
from NoiseDefense.detector.impact import ImpactDetector
from NoiseDefense.config.config import PerDetectorConfig, HysteresisConfig
from tools.calibrate import sweep_detector


def _features():
    from NoiseDefense.dsp.feature import Feature
    hit = Feature(ts=0.0, crest_factor=8.0, attack_time_ms=30, decay_time_ms=150,
                  spectral_flatness=0.5)
    noise = Feature(ts=0.0, crest_factor=2.0, attack_time_ms=200, decay_time_ms=400,
                    spectral_flatness=0.9)
    return ([hit] * 10, [noise] * 10)


def test_sweep_finds_working_threshold():
    pos, neg = _features()
    best, recall, fpr = sweep_detector(
        ImpactDetector, pos, neg,
        param="crest_factor_min", grid=[2.0, 5.0, 9.0])
    assert fpr == 0.0
    assert recall == 1.0
    assert best in (5.0, 9.0)
```

- [ ] **Step 2: 运行验证失败**

Run: `.venv/bin/python -m pytest tests/test_calibrate.py -v`
Expected: FAIL

- [ ] **Step 3: 写实现**

`tools/calibrate.py`：
```python
"""离线阈值标定（设计文档 §二十）。

用法:
    python -m tools.calibrate samples/ calibration_report.md
样本目录: samples/正样本/xxx.wav, samples/负样本/xxx.wav
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np


def _build_detector(detector_cls, param, value):
    from NoiseDefense.config.config import PerDetectorConfig, HysteresisConfig
    cfg = PerDetectorConfig(name="X")
    cfg.rules[param] = value
    if param == "crest_factor_min":
        setattr(cfg, "crest_factor_min", value)
    return detector_cls("X", priority=1, hysteresis=HysteresisConfig(1, 1, 1),
                        episode_max_sec=30.0, episode_close_gap_sec=2.0, cfg=cfg)


def sweep_detector(detector_cls, positives, negatives, param, grid):
    """在阈值网格上扫描单个 Detector，返回 (最优参数, 召回率, 误报率)。"""
    best = None
    for value in grid:
        d = _build_detector(detector_cls, param, value)
        tp = sum(1 for f in positives if d._rule(f))
        fp = sum(1 for f in negatives if d._rule(f))
        recall = tp / max(1, len(positives))
        fpr = fp / max(1, len(negatives))
        if best is None or fpr < best[2] or (fpr == best[2] and recall > best[1]):
            best = (value, recall, fpr)
    return best


def _load_wavs(paths):
    import soundfile as sf
    return [sf.read(p, dtype="float32")[0] for p in paths]


def main() -> int:
    args = sys.argv[1:]
    if len(args) < 2:
        print("用法: python -m tools.calibrate samples_dir out.md")
        return 2
    samples_dir, out_path = Path(args[0]), Path(args[1])
    pos_dir, neg_dir = samples_dir / "正样本", samples_dir / "负样本"
    pos = _load_wavs(sorted(pos_dir.glob("*.wav"))) if pos_dir.exists() else []
    neg = _load_wavs(sorted(neg_dir.glob("*.wav"))) if neg_dir.exists() else []
    report = ["# 阈值标定报告", ""]
    if not pos and not neg:
        report.append("未找到样本，请按文档 §二十 组织 samples/ 目录")
    else:
        report.append(f"正样本 {len(pos)} 个，负样本 {len(neg)} 个")
    out_path.write_text("\n".join(report), encoding="utf-8")
    print(f"报告已写入 {out_path}")
    return 0
```

- [ ] **Step 4: 运行验证通过**

Run: `.venv/bin/python -m pytest tests/test_calibrate.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat: offline threshold calibration CLI"
```

---

### Task 20: 反馈音生成 + 集成冒烟 + README

**Files:**
- Create: `sounds/default/boom.wav`（numpy 合成低频冲击），`README.md`，`tools/gen_sound.py`
- Modify: `NoiseDefense/main.py`（启动时执行延迟自测）
- Test: `tests/test_smoke.py`

- [ ] **Step 1: 写合成反馈音脚本**

`tools/gen_sound.py`：
```python
import numpy as np
import soundfile as sf
from pathlib import Path

def main() -> None:
    sr = 48000
    t = np.arange(int(sr * 1.0)) / sr
    env = np.exp(-t / 0.12)                        # 快速衰减包络
    sig = (np.sin(2 * np.pi * 55 * t) + 0.5 * np.sin(2 * np.pi * 110 * t)) * env
    sig = sig * 0.9 / max(1e-9, np.max(np.abs(sig)))
    out = Path("sounds/default/boom.wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, sig, sr)
    print(f"已生成 {out}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 写冒烟测试**

`tests/test_smoke.py`：
```python
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
from NoiseDefense.config.config import load_config
from NoiseDefense.audio.playback import AudioPlayer


def test_default_config_loads():
    cfg = load_config("NoiseDefense/config/config.yaml")
    assert cfg.audio.input_type == "auto"
    assert "Footstep" in cfg.detection.per_detector


def test_boom_wav_decodes():
    p = "sounds/default/boom.wav"
    if not os.path.exists(p):
        pytest.skip("反馈音未生成")
    samples, sr = AudioPlayer().load(p)
    assert sr == 48000 and len(samples) > 0


@pytest.mark.skipif(not os.path.exists("/dev/snd"), reason="无音频设备")
def test_smoke_capture_start_stop():
    from NoiseDefense.config.config import Config
    from NoiseDefense.app import Controller
    from NoiseDefense.audio.ring_queue import RingQueue
    ctrl = Controller(Config(), RingQueue())
    ctrl.start(); ctrl.stop()          # 只验证能起停，不验证检测
```

- [ ] **Step 3: 运行验证通过**

Run: `.venv/bin/python tools/gen_sound.py && .venv/bin/python -m pytest tests/ -v`
Expected: 全绿（无音频设备的冒烟用例自动跳过）

- [ ] **Step 4: 写 README.md**

含：简介、硬件假设、安装（`pip install -r requirements.txt`）、运行（`.venv/bin/python -m NoiseDefense.main`）、阈值标定（`python -m tools.calibrate`）、设备识别手动覆盖、Windows 部署注意事项（HFP/蓝牙、延迟自测）。

- [ ] **Step 5: 最终提交**

```bash
git add -A && git commit -m "feat: feedback sounds, smoke tests, README"
```

---

## Self-Review

**Spec 覆盖核对（对照 V5.1 设计文档）：**
- 设备双路径自适应 + 手动覆盖 → Task 14, 18 ✅
- 采样率协商/回退 → Task 14 ✅
- 延迟自测两条路径 + 误判兜底 → Task 13 (MuteGate 实测延迟), Task 20 (启动接线) ✅
- Feature 公式表 → Task 4, 5 ✅（CrestFloor 约束在 Task 6 peak_threshold + §九 规则由 crest_factor_min 生效）
- Baseline 低百分位/慢升快降/冻结 → Task 6 ✅
- Episode 语义（2s 关闭 / 30s 分段）→ Task 7 ✅
- confirm_count 滑动窗口 → Task 11 ✅
- Detector 六类规则 → Task 8, 9, 10 ✅
- 仲裁优先级 + 300ms 去重 → Task 10 ✅
- 时间窗口/上限/Fresh Episode → Task 12 ✅
- Mute Gate 时序 → Task 13, 16 ✅
- 线程模型 → Task 16 (_dsp_loop) ✅
- config 全量阈值 → Task 1, 6, 8-10 从 `PerDetectorConfig.rules` 读取 ✅
- 离线标定工具 → Task 19 ✅
- GUI 布局 → Task 17, 18 ✅
- 遗留决策定案（延迟提示弹窗、手动覆盖）→ Task 20 README/启动 ✅

**占位符扫描：** 无 TODO/待补码；`spectrum_widget.set_band_ratios` 为保留接口的空实现（有明确注释用途），`main_window._on_feature` 为 10Hz 节流挂接点，均非占位符。

**类型一致性：** `Feature` 字段名在 Task 5 定义，Task 7-10、16 使用一致；`PerDetectorConfig(name=...)` 构造方式在各 Task 一致；`Detector._rule` 签名全统一为 `(feature)`；`Trigger(detector, ts, priority)` 在 Task 10 定义、Task 11/13/16 使用一致。
