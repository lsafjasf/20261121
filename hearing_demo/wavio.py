"""WAV 读写（标准库 wave 模块），采样在内部统一为 float [-1, 1]。"""

from __future__ import annotations

import array
import math
import sys
import wave


def _clamp(x: float) -> float:
    if x > 1.0:
        return 1.0
    if x < -1.0:
        return -1.0
    return x


def write_wav(path: str, samples, sample_rate: int = 48000) -> None:
    """写 16-bit PCM 单声道 WAV。samples 为 float 序列。"""
    pcm = array.array("h")
    n = 0
    for x in samples:
        pcm.append(int(_clamp(x) * 32767.0))
        n += 1
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def read_wav(path: str):
    """读取单声道 WAV，返回 (sample_rate, list[float])，支持 8/16/24/32-bit。"""
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    if channels != 1:
        raise ValueError(f"仅支持单声道 WAV，收到 {channels} 声道")
    if width == 2:
        raw = array.array("h")
        raw.frombytes(frames)
        if sys_byteorder_big():
            raw.byteswap()
        return sample_rate, [x / 32768.0 for x in raw]
    if width == 1:
        return sample_rate, [(b - 128) / 128.0 for b in frames]
    if width == 4:
        raw = array.array("i")
        raw.frombytes(frames)
        if sys_byteorder_big():
            raw.byteswap()
        return sample_rate, [x / 2147483648.0 for x in raw]
    if width == 3:
        out = []
        for i in range(0, len(frames), 3):
            b0, b1, b2 = frames[i], frames[i + 1], frames[i + 2]
            v = b0 | (b1 << 8) | (b2 << 16)
            if v & 0x800000:
                v -= 1 << 24
            out.append(v / 8388608.0)
        return sample_rate, out
    raise ValueError(f"不支持的采样位宽: {width}")


def sys_byteorder_big() -> bool:
    return sys.byteorder == "big"


def rms_dbfs(samples) -> float:
    if not samples:
        return -math.inf
    acc = 0.0
    for x in samples:
        acc += x * x
    return 10.0 * math.log10(acc / len(samples) + 1e-12)


def peak_dbfs(samples) -> float:
    peak = 0.0
    for x in samples:
        a = abs(x)
        if a > peak:
            peak = a
    return 20.0 * math.log10(peak + 1e-12)

