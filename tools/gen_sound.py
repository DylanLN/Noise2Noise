"""合成默认反馈音：低频冲击 boom（55Hz + 110Hz，快速衰减包络）。
用法: python tools/gen_sound.py [输出目录]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sounds/default")
    out_dir.mkdir(parents=True, exist_ok=True)
    sr = 48000
    t = np.arange(int(sr * 1.0)) / sr
    env = np.exp(-t / 0.12)                              # ~120ms 衰减
    sig = (np.sin(2 * np.pi * 55 * t) + 0.5 * np.sin(2 * np.pi * 110 * t)) * env
    sig = sig * 0.9 / max(1e-9, np.max(np.abs(sig)))
    pcm = (sig * 32767).astype(np.int16)                 # 16-bit PCM
    path = out_dir / "boom.wav"
    wavfile.write(path, sr, pcm)
    print(f"已生成 {path}（{len(sig) / sr:.1f}s @ {sr}Hz）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
