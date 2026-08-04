"""配置加载：Config dataclass + YAML 覆盖。缺省值即 V1 默认配置。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Config:
    # 音频
    input_device: str = ""
    output_device: str = ""
    input_type: str = "auto"          # auto / wired / bluetooth
    sample_rate: int = 48000
    short_window_ms: float = 21.0
    long_window_ms: float = 85.0

    # 检测
    sensitivity: float = 5.0
    cooldown: float = 5.0
    confirm_count: int = 2
    confirm_window_sec: float = 65.0
    episode_max_sec: float = 30.0
    episode_close_gap_sec: float = 2.0
    arbitration_window_ms: int = 300
    mute_ignore_ms: float = 200.0

    # 时间窗口
    active_windows: list[str] = field(default_factory=lambda: ["12:00-14:00", "18:00-24:00"])
    max_responses_per_window: int = 5
    fresh_episode_gap_sec: float = 60.0

    # 响应
    volume: float = 1.0
    sounds_dir: str = "sounds"


def load_config(path: str | Path = "config.yaml") -> Config:
    """加载 YAML；未出现的键保持默认值。文件不存在时返回默认配置。"""
    cfg = Config()
    p = Path(path)
    if not p.exists():
        return cfg
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for key, value in data.items():
        if hasattr(cfg, key) and value is not None:
            setattr(cfg, key, value)
    return cfg
