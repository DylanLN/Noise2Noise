"""引擎层 sanity 测试：EventManager / ScheduleManager / MuteGate / ResponseEngine。"""
from collections import deque

from config import Config
from dsp import Feature
from detectors import Detector
from engine import EventManager, ScheduleManager, MuteGate, ResponseEngine, Trigger


class StubDetector(Detector):
    def __init__(self, name, pri, pattern):
        super().__init__(name, pri, sample_rate=16000, short_window_ms=21.0,
                         n1=1, n2=1, n3=1, episode_max_sec=30.0,
                         episode_close_gap_sec=0.5)
        self.pattern = deque(pattern)
    def rule(self, f):
        return self.pattern.popleft() if self.pattern else False


# ── EventManager ──
def test_single_episode_no_trigger():
    em = EventManager([StubDetector("Footstep", 3, [True] * 10 + [False] * 30)],
                      default_confirm_count=2, default_confirm_window_sec=10.0)
    ts = 0.0
    for _ in range(40):
        em.update(Feature(ts=ts)); ts += 0.02
    assert em.take_triggers() == []


def test_two_episodes_trigger():
    em = EventManager([StubDetector("Footstep", 3, [True] * 15 + [False] * 100
                                    + [True] * 15 + [False] * 30)],
                      default_confirm_count=2, default_confirm_window_sec=10.0)
    ts = 0.0
    trigs = []
    for _ in range(160):
        em.update(Feature(ts=ts)); ts += 0.02
        trigs += em.take_triggers()
    assert any(t.detector == "Footstep" for t in trigs)


def test_arbitration_highest_priority():
    em = EventManager(
        [StubDetector("Footstep", 3, [True] * 10 + [False] * 10),
         StubDetector("Impact", 6, [True] * 10 + [False] * 10)],
        default_confirm_count=1, default_confirm_window_sec=10.0, arbitration_window_ms=300)
    ts = 0.0
    trigs = []
    for _ in range(40):                      # 需跑过 0.5s 静默关闭 Episode
        em.update(Feature(ts=ts)); ts += 0.02
        trigs += em.take_triggers()
    # 同窗口内两个 Trigger 只留优先级高的 Impact
    assert trigs and all(t.detector == "Impact" for t in trigs)


# ── ScheduleManager ──
def _sec(h, m=0): return h * 3600 + m * 60


def test_window_inside_outside():
    c = Config(); c.active_windows = ["12:00-14:00", "18:00-24:00"]
    sm = ScheduleManager(c)
    assert sm.decide(_sec(13, 0)).allowed
    assert not sm.decide(_sec(9, 0)).allowed


def test_cross_midnight_window():
    c = Config(); c.active_windows = ["22:00-01:00"]
    sm = ScheduleManager(c)
    assert sm.decide(_sec(23, 30)).allowed
    assert sm.decide(_sec(0, 30)).allowed
    assert not sm.decide(_sec(12, 0)).allowed


def test_max_responses():
    c = Config(); c.active_windows = ["12:00-14:00"]; c.max_responses_per_window = 2
    sm = ScheduleManager(c)
    sm.record_response(_sec(12, 0)); sm.record_response(_sec(12, 1))
    v = sm.decide(_sec(12, 2))
    assert not v.allowed and v.reason == "max_responses"


def test_fresh_episode_gap():
    c = Config(); c.active_windows = ["12:00-14:00"]; c.fresh_episode_gap_sec = 60.0
    sm = ScheduleManager(c)
    sm.record_response(_sec(12, 0))
    v = sm.decide(_sec(12, 0) + 30)                          # 30s 后 → 未过冷却
    assert not v.allowed and v.reason == "not_fresh"
    assert sm.decide(_sec(12, 2)).allowed                     # 2分钟后 → 允许


# ── MuteGate ──
def test_wired_mute_plan():
    g = MuteGate(base_ignore_ms=200.0, measured_latency_ms=15.0)
    assert not g.is_bluetooth_like()
    p = g.plan(500.0)
    assert p["pause_before_ms"] == 0.0 and p["ignore_after_ms"] == 200.0


def test_bluetooth_mute_plan():
    g = MuteGate(base_ignore_ms=200.0, measured_latency_ms=180.0, safety_ms=200.0)
    assert g.is_bluetooth_like()
    p = g.plan(500.0)
    assert p["pause_before_ms"] == 180.0 and p["ignore_after_ms"] == 500.0 + 180.0 + 200.0


def test_misclassified_device_caught_by_measured_latency():
    g = MuteGate(base_ignore_ms=200.0, measured_latency_ms=250.0)
    assert g.is_bluetooth_like()             # 名称误判有线，实测延迟兜底


# ── ResponseEngine ──
def test_response_cooldown():
    calls = []
    def play(tr): calls.append(tr); return 300.0
    c = Config(); c.cooldown = 5.0; c.mute_ignore_ms = 200.0
    re_ = ResponseEngine(c, MuteGate(base_ignore_ms=200.0), play)
    muted = []
    tr = Trigger("Impact", 1000.0, 6)
    assert re_.handle(tr, 1000.0, set_muted=muted.append) is True
    assert muted == [True, False]
    assert re_.handle(tr, 1000.1, set_muted=muted.append) is False   # 冷却内
    assert re_.handle(tr, 1006.0, set_muted=muted.append) is True    # 冷却结束
