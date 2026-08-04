"""检测器：滞回状态机 + Episode 跟踪 + 6 类规则。规则阈值均为初始猜测，见设计 §九。"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from dsp import Feature, crest_floor


class State(Enum):
    IDLE = auto()
    ARMED = auto()
    CONFIRMED = auto()


class HysteresisMachine:
    """帧级滞回：n1 帧进入 ARMED，n2 帧进入 CONFIRMED，n3 帧退出。"""

    def __init__(self, n1: int = 3, n2: int = 5, n3: int = 5):
        self.n1, self.n2, self.n3 = max(1, n1), max(1, n2), max(1, n3)
        self.state = State.IDLE
        self._count = 0

    def step(self, ok: bool) -> State:
        s = self.state
        if s is State.IDLE:
            self._count = self._count + 1 if ok else 0
            if self._count >= self.n1:
                self.state, self._count = State.ARMED, 0
        elif s is State.ARMED:
            if ok:
                self._count += 1
                if self._count >= self.n2:
                    self.state, self._count = State.CONFIRMED, 0
            else:
                self._count += 1
                if self._count >= self.n3:
                    self.state, self._count = State.IDLE, 0
        else:  # CONFIRMED
            if not ok:
                self._count += 1
                if self._count >= self.n3:
                    self.state, self._count = State.IDLE, 0
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

    #: 各检测器默认规则参数，子类覆写；config 覆盖值经 overrides 传入
    defaults: dict = {}

    def __init__(self, name: str, priority: int, sample_rate: int,
                 short_window_ms: float, n1: int = 3, n2: int = 5, n3: int = 5,
                 episode_max_sec: float = 30.0, episode_close_gap_sec: float = 2.0,
                 overrides: dict | None = None):
        self.name = name
        self.priority = priority
        self.enabled = True                        # GUI 检测类型复选框控制
        self.machine = HysteresisMachine(n1, n2, n3)
        self.episode_max_sec = episode_max_sec
        self.episode_close_gap_sec = episode_close_gap_sec
        self.rules = {**self.defaults, **(overrides or {})}
        # 有效 Crest 阈值 = max(配置下限, 噪声地板+1)，避免随机噪声周期性越线
        self.crest_min = max(self.rules.get("crest_factor_min", 3.0),
                             crest_floor(sample_rate, short_window_ms) + 1.0)
        self._episode: Episode | None = None
        self._closed: list[Episode] = []
        self._last_active: float | None = None

    @property
    def state(self) -> State:
        return self.machine.state

    @property
    def non_idle(self) -> bool:
        return self.machine.state is not State.IDLE

    def update(self, f: Feature) -> None:
        ok = self.rule(f)
        if ok:
            self._last_active = f.ts
        prev = self.machine.state
        st = self.machine.step(ok)
        if st is State.CONFIRMED and prev is not State.CONFIRMED:
            self._episode = Episode(self.name, f.ts)
        elif (st is State.CONFIRMED and prev is State.CONFIRMED
              and self._episode is not None
              and f.ts - self._episode.start_ts >= self.episode_max_sec):
            # 30s 强制分段：旧段入账，新段从当前时刻重新开
            self._episode.segments += 1
            self._closed.append(self._episode)
            self._episode = Episode(self.name, f.ts)
        # Episode 关闭独立于状态机：2s 静默即关闭（设计 §十）
        if self._episode is not None and self._last_active is not None:
            if f.ts - self._last_active >= self.episode_close_gap_sec:
                self._episode.end_ts = f.ts
                self._closed.append(self._episode)
                self._episode = None

    def pop_episodes(self) -> list[Episode]:
        out, self._closed = self._closed, []
        return out

    def rule(self, f: Feature) -> bool:
        raise NotImplementedError


def _interval_ok(f: Feature, lo_ms: float, hi_ms: float) -> bool:
    """峰值间隔落在 [lo,hi]（ms）内，用中位数抗单个离群间隔。"""
    if f.peak_interval is None or len(f.peak_interval) == 0:
        return False
    med = float(np.median(f.peak_interval)) * 1000.0
    return lo_ms <= med <= hi_ms


class ImpactDetector(Detector):
    """撞击/重物掉落：高峰均比 + 快 attack + 短 decay + 频谱适中平坦。
    flat_range 下界放宽到 0.1：木质桌面的敲击是窄带共振，严格 0.3 会把它们漏掉。
    单次瞬态即触发（confirm_count=1），与设计 §十 一致。"""
    defaults = {"crest_factor_min": 6.0, "attack_time_ms_max": 50.0,
                "decay_time_ms_max": 300.0, "flat_range": (0.1, 0.7)}

    def rule(self, f: Feature) -> bool:
        if f.crest_factor < self.crest_min:
            return False
        if f.attack_time_ms is None or f.attack_time_ms > self.rules["attack_time_ms_max"]:
            return False
        if f.decay_time_ms is None or f.decay_time_ms > self.rules["decay_time_ms_max"]:
            return False
        lo, hi = self.rules["flat_range"]
        return lo <= f.spectral_flatness <= hi


class DoorDetector(Detector):
    """摔门：频谱更宽更平坦 + 快 attack + 中等偏长 decay（与 Impact 区分）。
    单次瞬态即触发。"""
    defaults = {"crest_factor_min": 4.0, "flat_min": 0.6, "attack_time_ms_max": 30.0,
                "decay_ms": (200.0, 600.0)}

    def rule(self, f: Feature) -> bool:
        if f.spectral_flatness < self.rules["flat_min"]:
            return False
        if f.attack_time_ms is None or f.attack_time_ms > self.rules["attack_time_ms_max"]:
            return False
        if f.decay_time_ms is None:
            return False
        lo, hi = self.rules["decay_ms"]
        return lo <= f.decay_time_ms <= hi


class FootstepDetector(Detector):
    """跑步/快速脚步：归一化 RMS + 峰均比 + 低频占比 + 周期间隔 200~600ms。
    需 2 个 Episode 才触发（避免偶发单段噪声误报）。"""
    defaults = {"crest_factor_min": 4.0, "low_energy_ratio_min": 0.60,
                "peak_count_min": 5, "peak_interval_ms": (200.0, 600.0),
                "rms_norm_min": 2.0}

    def rule(self, f: Feature) -> bool:
        if f.crest_factor < self.crest_min:
            return False
        if f.low_energy_ratio < self.rules["low_energy_ratio_min"]:
            return False
        if f.peak_count < self.rules["peak_count_min"]:
            return False
        if f.rms_norm < self.rules["rms_norm_min"]:
            return False
        return _interval_ok(f, *self.rules["peak_interval_ms"])


class JumpDetector(Detector):
    """蹦跳：更高低频占比 + 更慢间隔 400~800ms。需 2 个 Episode。"""
    defaults = {"crest_factor_min": 5.0, "low_energy_ratio_min": 0.70,
                "peak_count_min": 3, "peak_interval_ms": (400.0, 800.0)}

    def rule(self, f: Feature) -> bool:
        if f.crest_factor < self.crest_min:
            return False
        if f.low_energy_ratio < self.rules["low_energy_ratio_min"]:
            return False
        if f.peak_count < self.rules["peak_count_min"]:
            return False
        return _interval_ok(f, *self.rules["peak_interval_ms"])


class BallDetector(Detector):
    """拍球：间隔高度一致（相对标准差小）+ 峰均比 + 峰值个数。需 2 个 Episode。"""
    defaults = {"crest_factor_min": 4.0, "interval_variance_max": 0.20,
                "peak_count_min": 4}

    def rule(self, f: Feature) -> bool:
        if f.crest_factor < self.crest_min:
            return False
        if f.peak_count < self.rules["peak_count_min"]:
            return False
        if f.peak_interval is None or len(f.peak_interval) < 2:
            return False
        return f.interval_variance <= self.rules["interval_variance_max"]


class ChairDetector(Detector):
    """拖家具：持续时间长 + 中频持续存在 + 频谱变化缓慢。时长型，1 个 Episode 即触发。"""
    defaults = {"crest_factor_min": 2.0, "duration_min_sec": 1.0,
                "mid_energy_ratio_min": 0.25, "spectral_flux_max": 0.05}

    def rule(self, f: Feature) -> bool:
        if f.duration < self.rules["duration_min_sec"]:
            return False
        if f.mid_energy_ratio < self.rules["mid_energy_ratio_min"]:
            return False
        return f.spectral_flux <= self.rules["spectral_flux_max"]


#: 构造顺序即触发优先级（高优先在前），仲裁用 priority 数值
DETECTOR_CLASSES = [ImpactDetector, DoorDetector, JumpDetector,
                    FootstepDetector, BallDetector, ChairDetector]
