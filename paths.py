"""打包(frozen)与开发环境的路径解析。
打包后 config.yaml / sounds/ 放在可执行文件旁边，方便用户编辑配置。"""
from __future__ import annotations

import sys
from pathlib import Path


def app_dir() -> Path:
    """frozen(打包)时返回 exe 所在目录；开发时返回项目根目录。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def config_path() -> Path:
    return app_dir() / "config.yaml"


def sounds_dir(sounds: str) -> Path:
    p = Path(sounds)
    return p if p.is_absolute() else app_dir() / p
