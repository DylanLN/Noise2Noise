"""检测器与状态机 sanity 测试。"""
import numpy as np

from dsp import Feature
from detectors import (HysteresisMachine, State, Detector, Episode,
                       ImpactDetector, DoorDetector, FootstepDetector,
                       JumpDetector, BallDetector, ChairDetector)


def _f(**kw):
    d = dict(ts=0.0, rms=1.0, peak=1.0, crest_factor=1.0, low_energy_ratio=0.5,
             mid_energy_ratio=0.3, high_energy_ratio=0.2, spectral_flatness=0.5,
             spectral_flux=0.0, peak_count=1, peak_interval=None, interval_variance=0.0,
             duration=0.0, rms_norm=1.0, attack_time_ms=None, decay_time_ms=None)
    d.update(kw)
    return Feature(**d)


# ── 状态机 ──
def test_machine_transitions():
    m = HysteresisMachine(n1=2, n2=3, n3=2)
    assert m.step(False) is State.IDLE
    assert m.step(True) is State.IDLE          # count=1
    assert m.step(True) is State.ARMED         # count=2 >= n1
    assert m.step(True) is State.ARMED         # ARMED count=1
    assert m.step(True) is State.ARMED         # ARMED count=2
    assert m.step(True) is State.CONFIRMED     # ARMED count=3 >= n2
    assert m.step(False) is State.CONFIRMED    # CONFIRMED count=1
    assert m.step(False) is State.IDLE         # CONFIRMED count=2 >= n3


class Always(Detector):
    def rule(self, f): return True


class Burst(Detector):
    """只在 ts<=until 满足规则，用于制造"静默间隔"。"""
    def __init__(self, *a, until=0.5, **kw):
        super().__init__(*a, **kw)
        self.until = until
    def rule(self, f): return f.ts <= self.until


def _mk(name="t", pri=1, cls=Always, **kw):
    return cls(name, pri, sample_rate=16000, short_window_ms=21.0,
               n1=1, n2=1, n3=1, episode_max_sec=30.0, episode_close_gap_sec=2.0, **kw)


def test_episode_closes_after_gap():
    d = _mk(cls=Burst, until=0.5)
    d.update(_f(ts=0.0))                       # True → ARMED
    d.update(_f(ts=0.02))                      # True → CONFIRMED → Episode 开始
    assert d.state is State.CONFIRMED
    d.update(_f(ts=2.5))                       # False，距上次满足 2.48s ≥ 2s → 关闭
    eps = d.pop_episodes()
    assert len(eps) == 1 and isinstance(eps[0], Episode)
    assert eps[0].start_ts == 0.02 and eps[0].end_ts == 2.5


def test_episode_segments_at_cap():
    d = _mk()
    ts = 0.0
    while ts < 31.0:
        d.update(_f(ts=ts)); ts += 0.02
    assert len(d.pop_episodes()) >= 1


# ── 瞬态区分：Impact vs Door ──
def test_impact_door_separation():
    imp = ImpactDetector("Impact", 6, 16000, 21.0, n1=1, n2=1, n3=1)
    door = DoorDetector("Door", 5, 16000, 21.0, n1=1, n2=1, n3=1)
    impact_like = _f(crest_factor=8.0, attack_time_ms=30, decay_time_ms=150, spectral_flatness=0.5)
    door_like = _f(crest_factor=10.0, attack_time_ms=20, decay_time_ms=400, spectral_flatness=0.7)
    assert imp.rule(impact_like) and not door.rule(impact_like)
    assert door.rule(door_like) and not imp.rule(door_like)


def test_impact_rejects_low_crest():
    imp = ImpactDetector("Impact", 6, 16000, 21.0, n1=1, n2=1, n3=1)
    assert not imp.rule(_f(crest_factor=2.0, attack_time_ms=30, decay_time_ms=150, spectral_flatness=0.5))


# ── 节奏型 ──
def test_footstep_rhythm():
    d = FootstepDetector("Footstep", 3, 16000, 21.0, n1=1, n2=1, n3=1)
    f = _f(crest_factor=6.0, low_energy_ratio=0.7, peak_count=6, rms_norm=3.0,
           peak_interval=np.array([0.3, 0.4, 0.25]))
    assert d.rule(f)


def test_footstep_wrong_interval():
    d = FootstepDetector("Footstep", 3, 16000, 21.0, n1=1, n2=1, n3=1)
    f = _f(crest_factor=6.0, low_energy_ratio=0.7, peak_count=6, rms_norm=3.0,
           peak_interval=np.array([1.5, 2.0]))
    assert not d.rule(f)


def test_jump_interval():
    d = JumpDetector("Jump", 4, 16000, 21.0, n1=1, n2=1, n3=1)
    f = _f(crest_factor=7.0, low_energy_ratio=0.8, peak_count=4,
           peak_interval=np.array([0.5, 0.6]))
    assert d.rule(f)


def test_ball_regularity():
    d = BallDetector("Ball", 2, 16000, 21.0, n1=1, n2=1, n3=1)
    f = _f(crest_factor=6.0, peak_count=5, peak_interval=np.array([0.3, 0.31, 0.29]),
           interval_variance=0.03)
    assert d.rule(f)
    f2 = _f(crest_factor=6.0, peak_count=5, peak_interval=np.array([0.1, 0.8, 0.15]),
            interval_variance=0.7)
    assert not d.rule(f2)


# ── 时长型 ──
def test_chair_sustained():
    d = ChairDetector("Chair", 1, 16000, 21.0, n1=1, n2=1, n3=1)
    assert d.rule(_f(duration=2.0, mid_energy_ratio=0.5, spectral_flux=0.01))
    assert not d.rule(_f(duration=0.5, mid_energy_ratio=0.5, spectral_flux=0.01))
