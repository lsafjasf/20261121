"""Small, dependency-free PCM WAV helpers used by the laboratory."""

from __future__ import annotations

import array
import math
import sys
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioClip:
    samples: list[float]
    sample_rate: int

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate


def _decode_pcm(raw: bytes, sample_width: int, channels: int) -> list[float]:
    if sample_width != 2:
        raise ValueError(f"only 16-bit PCM WAV is supported, got {sample_width * 8}-bit")
    pcm = array.array("h")
    pcm.frombytes(raw)
    if sys.byteorder != "little":
        pcm.byteswap()
    if channels == 1:
        return [value / 32768.0 for value in pcm]
    scale = 1.0 / (32768.0 * channels)
    return [
        sum(pcm[index : index + channels]) * scale
        for index in range(0, len(pcm), channels)
    ]


def read_mono_wav(path: str | Path) -> AudioClip:
    with wave.open(str(path), "rb") as handle:
        if handle.getcomptype() != "NONE":
            raise ValueError(f"compressed WAV is not supported: {path}")
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    return AudioClip(_decode_pcm(raw, sample_width, channels), sample_rate)


def _pcm16(samples: list[float]) -> array.array:
    pcm = array.array("h")
    pcm.extend(
        max(-32768, min(32767, int(round(value * 32767.0)))) for value in samples
    )
    if sys.byteorder != "little":
        pcm.byteswap()
    return pcm


def write_wav(path: str | Path, channels: list[list[float]], sample_rate: int) -> None:
    if not channels:
        raise ValueError("at least one channel is required")
    frames = len(channels[0])
    if any(len(channel) != frames for channel in channels):
        raise ValueError("all channels must have the same frame count")
    interleaved: list[float] = []
    for index in range(frames):
        interleaved.extend(channel[index] for channel in channels)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(len(channels))
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(_pcm16(interleaved).tobytes())


def write_mono_wav(path: str | Path, samples: list[float], sample_rate: int) -> None:
    write_wav(path, [samples], sample_rate)


def write_stereo_ab(
    path: str | Path,
    reference: list[float],
    decoded: list[float],
    sample_rate: int,
) -> None:
    frames = min(len(reference), len(decoded))
    write_wav(path, [reference[:frames], decoded[:frames]], sample_rate)


def dbfs(energy: float, floor_db: float = -180.0) -> float:
    if energy <= 0.0 or not math.isfinite(energy):
        return floor_db
    return max(floor_db, 10.0 * math.log10(energy))
