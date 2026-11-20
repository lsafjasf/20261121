"""Small dependency-free WAV reader/writer for the recognition pipeline."""

from __future__ import annotations

import array
import math
import sys
import wave
from pathlib import Path


def read_wav(path: str | Path) -> tuple[list[float], int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    if width == 1:
        raw = array.array("B", frames)
        values = [(sample - 128) / 128.0 for sample in raw]
    elif width == 2:
        raw = array.array("h")
        raw.frombytes(frames)
        if sys.byteorder != "little":
            raw.byteswap()
        values = [sample / 32768.0 for sample in raw]
    elif width == 3:
        values = []
        for index in range(0, len(frames), 3):
            sample = int.from_bytes(frames[index : index + 3], "little", signed=True)
            values.append(sample / 8388608.0)
    elif width == 4:
        raw = array.array("i")
        raw.frombytes(frames)
        if sys.byteorder != "little":
            raw.byteswap()
        values = [sample / 2147483648.0 for sample in raw]
    else:
        raise ValueError(f"unsupported PCM sample width: {width} bytes")

    if channels > 1:
        mono = []
        for index in range(0, len(values), channels):
            chunk = values[index : index + channels]
            mono.append(sum(chunk) / len(chunk))
        values = mono

    peak = max((abs(sample) for sample in values), default=0.0)
    if peak > 0:
        values = [sample / peak for sample in values]
    return values, rate


def write_wav(path: str | Path, samples: list[float], rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = max((abs(sample) for sample in samples), default=1.0)
    scale = 0.96 / peak if peak > 0 else 1.0
    pcm = array.array(
        "h",
        (max(-32768, min(32767, round(sample * scale * 32767))) for sample in samples),
    )
    if sys.byteorder != "little":
        pcm.byteswap()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


def resample_linear(samples: list[float], source_rate: int, target_rate: int) -> list[float]:
    if source_rate == target_rate:
        return list(samples)
    count = int(len(samples) * target_rate / source_rate)
    scale = source_rate / target_rate
    result = [0.0] * count
    last = len(samples) - 1
    for index in range(count):
        position = index * scale
        left = int(position)
        right = min(last, left + 1)
        fraction = position - left
        result[index] = samples[left] * (1.0 - fraction) + samples[right] * fraction
    return result


def apply_speed(samples: list[float], rate: int, speed: float) -> tuple[list[float], int]:
    """Play the file faster without changing its nominal sample rate."""
    if speed <= 0:
        raise ValueError("speed must be positive")
    count = int(len(samples) / speed)
    result = [0.0] * count
    for index in range(count):
        position = index * speed
        left = int(position)
        right = min(len(samples) - 1, left + 1)
        fraction = position - left
        result[index] = samples[left] * (1.0 - fraction) + samples[right] * fraction
    return result, rate


def lowpass(samples: list[float], rate: int, cutoff_hz: float) -> list[float]:
    alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff_hz / rate)
    result = [0.0] * len(samples)
    state = 0.0
    for index, sample in enumerate(samples):
        state += alpha * (sample - state)
        result[index] = state
    return result


def rms(samples: list[float]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def normalize(samples: list[float], peak: float = 0.92) -> list[float]:
    current = max((abs(sample) for sample in samples), default=0.0)
    if current == 0:
        return samples
    gain = peak / current
    return [sample * gain for sample in samples]


def write_wave_header_preview(samples: list[float], buckets: int = 640) -> list[float]:
    if not samples:
        return []
    size = max(1, len(samples) // buckets)
    peaks = []
    for start in range(0, len(samples), size):
        chunk = samples[start : start + size]
        peaks.append(max((abs(sample) for sample in chunk), default=0.0))
        if len(peaks) >= buckets:
            break
    top = max(peaks, default=1.0) or 1.0
    return [round(value / top, 5) for value in peaks]
