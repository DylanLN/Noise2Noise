"""音频采集与播放。sounddevice/miniaudio 懒加载：无 PortAudio 环境也能 import 本模块。"""
from __future__ import annotations

import queue
import threading

import numpy as np


class AudioIn:
    """sounddevice 采集 → 队列。回调线程只入队；DSP 线程消费。muted 时丢弃数据。"""

    def __init__(self, device: str | None = None, sample_rate: int = 48000,
                 max_chunks: int = 256):
        self.device = device or None
        self.sample_rate = sample_rate
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_chunks)
        self._stream = None
        self._muted = False
        self._lock = threading.Lock()

    def start(self) -> None:
        import sounddevice as sd
        def callback(indata, frames, time_info, status):
            if self._muted:
                return
            chunk = np.array(indata[:, 0], dtype=np.float64, copy=True)
            try:
                self._q.put_nowait(chunk)
            except queue.Full:
                pass                                   # 丢弃最旧处理窗口，避免阻塞回调
        self._stream = sd.InputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32",
            device=self.device, callback=callback)
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def set_muted(self, muted: bool) -> None:
        with self._lock:
            self._muted = muted

    def get(self, timeout: float = 0.1) -> np.ndarray | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None


class AudioOut:
    """反馈音播放：miniaudio 解码（wav/mp3/flac）→ sounddevice 播放（阻塞到结束）。"""

    def __init__(self, device: str | None = None, volume: float = 1.0):
        self.device = device or None
        self.volume = volume

    def play(self, path: str) -> float:
        import miniaudio
        decoded = miniaudio.decode_file(path, output_format=miniaudio.SampleFormat.FLOAT32)
        samples = np.asarray(decoded.samples, dtype=np.float32)
        if decoded.nchannels > 1:
            samples = samples.reshape(-1, decoded.nchannels).mean(axis=1)
        import sounddevice as sd
        sd.play(samples * self.volume, decoded.sample_rate, device=self.device)
        sd.wait()                                      # 阻塞到播放结束，防止尾音自触发
        return len(samples) / decoded.sample_rate * 1000.0


# ── 设备枚举（GUI 用）──────────────────────────────────────────────────

def list_input_devices() -> list[str]:
    import sounddevice as sd
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            out.append(f"{i}: {d.get('name', '')}")
    return out


def list_output_devices() -> list[str]:
    import sounddevice as sd
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_output_channels", 0) > 0:
            out.append(f"{i}: {d.get('name', '')}")
    return out
