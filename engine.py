"""事件 → 时间窗口 → 响应。EventManager / ScheduleManager / ResponseEngine。"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass

from config import Config
from dsp import Feature
from detectors import Detector


# ── 事件管理 ──────────────────────────────────────────────────────────

@dataclass
class Trigger:
    detector: str
    ts: float
    priority: int


class EventManager:
    """推进所有 Detector；Episode 计数达标才产生 Trigger；多 Detector 同时命中按优先级仲裁。"""

    def __init__(self, detectors: list[Detector], arbitration_window_ms: int = 300,
                 default_confirm_count: int = 1, default_confirm_window_sec: float = 65.0,
                 on_episode=None):
        self.detectors = detectors
        self.default_confirm_count = default_confirm_count
        self.default_confirm_window = default_confirm_window_sec
        self.arb_window = arbitration_window_ms / 1000.0
        self.on_episode = on_episode                  # (name, count, target) 调试回调
        self._history: dict[str, deque[float]] = defaultdict(deque)
        self._pending: list[Trigger] = []

    def update(self, f: Feature) -> None:
        for d in self.detectors:
            if not d.enabled:
                d.pop_episodes()                   # 丢弃禁用期间的遗留 Episode
                continue
            d.update(f)
            for ep in d.pop_episodes():
                self._maybe_trigger(d, ep.end_ts)

    def take_triggers(self) -> list[Trigger]:
        out = self._pending
        self._pending = []
        if not out:
            return []
        # 仲裁：同一时刻（arb_window 内）多个 Trigger 只留优先级最高者
        out.sort(key=lambda t: (t.ts, -t.priority))
        kept: list[Trigger] = []
        for t in out:
            if kept and t.ts - kept[-1].ts <= self.arb_window:
                if t.priority > kept[-1].priority:
                    kept[-1] = t
                continue
            kept.append(t)
        return kept

    def non_idle_ratio(self) -> float:
        enabled = [d for d in self.detectors if d.enabled]
        if not enabled:
            return 0.0
        return sum(1 for d in enabled if d.non_idle) / len(enabled)

    def _maybe_trigger(self, d: Detector, ep_ts: float) -> None:
        # 每个 Detector 自带 confirm_count / window（见 detector.defaults），全局为兜底
        cc = int(d.rules.get("confirm_count", self.default_confirm_count))
        win = float(d.rules.get("confirm_window_sec", self.default_confirm_window))
        h = self._history[d.name]
        h.append(ep_ts)
        count = sum(1 for t in h if ep_ts - t <= win)
        if self.on_episode:
            self.on_episode(d.name, count, cc)
        # 滑动窗口内 Episode 数达标即产生 Trigger（episode 关闭本身是边沿事件，
        # 不会逐帧重复；重复响应由 ResponseEngine 冷却 / Schedule 上限 / Fresh 间隔控制）
        if count >= cc:
            self._pending.append(Trigger(d.name, ep_ts, d.priority))


# ── 时间窗口 ──────────────────────────────────────────────────────────

@dataclass
class Verdict:
    allowed: bool
    reason: str          # ok / outside_window / max_responses / not_fresh


def _parse_window(spec: str) -> tuple[float, float]:
    """'12:00-14:00' → (秒, 秒)；支持跨天 '22:00-01:00'。"""
    s, e = spec.split("-")
    h1, m1 = (int(x) for x in s.split(":"))
    h2, m2 = (int(x) for x in e.split(":"))
    return h1 * 3600 + m1 * 60, h2 * 3600 + m2 * 60


class ScheduleManager:
    """时间窗口 + 响应次数上限 + Fresh Episode 冷却。ts 用时间戳（%24h 取时刻）。"""

    def __init__(self, cfg: Config):
        self.windows = [_parse_window(w) for w in cfg.active_windows]
        self.max_responses = cfg.max_responses_per_window
        self.fresh_gap = cfg.fresh_episode_gap_sec
        self.always = cfg.schedule_always              # 测试用：忽略时间窗口
        self._counts: dict[int, int] = {}
        self._last_response: float | None = None

    def decide(self, ts: float) -> Verdict:
        idx = self._active_index(ts % (24 * 3600.0))
        if idx is None and not self.always:
            return Verdict(False, "outside_window")
        if idx is None:
            idx = -1                                   # always 模式用统一计数桶
        if self._last_response is not None and ts - self._last_response < self.fresh_gap:
            return Verdict(False, "not_fresh")
        if self._counts.get(idx, 0) >= self.max_responses:
            return Verdict(False, "max_responses")
        return Verdict(True, "ok")

    def set_windows(self, specs: list[str]) -> None:
        """GUI 运行时修改时间窗口：重新解析并清空各窗口计数。"""
        self.windows = [_parse_window(w) for w in specs]
        self._counts = {}

    def record_response(self, ts: float) -> None:
        idx = self._active_index(ts % (24 * 3600.0))
        if idx is not None:
            self._counts[idx] = self._counts.get(idx, 0) + 1
        self._last_response = ts

    def _active_index(self, minute: float) -> int | None:
        for i, (start, end) in enumerate(self.windows):
            if start <= end:
                if start <= minute < end:
                    return i
            else:  # 跨天
                if minute >= start or minute < end:
                    return i
        return None


# ── 响应 ──────────────────────────────────────────────────────────────

class MuteGate:
    """按设备类型计算采集静音时序。实测延迟 ≥50ms 视为蓝牙式（防"USB转蓝牙"误判）。"""

    def __init__(self, base_ignore_ms: float = 200.0,
                 measured_latency_ms: float | None = None,
                 safety_ms: float = 200.0):
        self.base_ignore_ms = base_ignore_ms
        self.measured = measured_latency_ms
        self.safety_ms = safety_ms

    def is_bluetooth_like(self) -> bool:
        return self.measured is not None and self.measured >= 50.0

    def plan(self, play_duration_ms: float) -> dict:
        """返回 {pause_before_ms, ignore_after_ms}。"""
        if not self.is_bluetooth_like():
            return {"pause_before_ms": 0.0, "ignore_after_ms": self.base_ignore_ms}
        lat = self.measured if self.measured is not None else 300.0
        return {"pause_before_ms": lat,
                "ignore_after_ms": play_duration_ms + lat + self.safety_ms}


class ResponseEngine:
    """冷却检查 → 采集静音 → 阻塞播放到结束 → 保持静音覆盖 ignore 窗口 → 恢复采集。
    play_fn 必须阻塞到播放结束并返回时长(ms)，避免尾音重新进入检测造成自触发。"""

    def __init__(self, cfg: Config, mute_gate: MuteGate, play_fn,
                 on_respond=None, on_log=None):
        self.cfg = cfg
        self.gate = mute_gate
        self.play_fn = play_fn
        self.on_respond = on_respond
        self.on_log = on_log
        self._cooldown_until = 0.0

    def handle(self, trigger: Trigger, now: float, set_muted=None) -> bool:
        if now < self._cooldown_until:
            return False
        if set_muted:
            set_muted(True)
        try:
            duration_ms = self.play_fn(trigger)
        finally:
            plan = self.gate.plan(duration_ms or 0.0)
            time.sleep(plan["ignore_after_ms"] / 1000.0)   # 保持静音覆盖播放尾音/混响
            if set_muted:
                set_muted(False)
        self._cooldown_until = now + self.cfg.cooldown
        if self.on_respond:
            self.on_respond(trigger, plan)
        if self.on_log:
            self.on_log(f"触发 {trigger.detector}，静音窗口 {plan}")
        return True
