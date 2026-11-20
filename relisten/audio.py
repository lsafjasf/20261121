"""Minimal PCM WAV reader/writer using only the Python standard library.

Sample-accurate slicing is the whole point of this package, so no resampling
is ever performed: clips keep their source track's sample rate and every cut
is expressed in integer source samples.
"""

from __future__ import annotations

import math
import struct
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Track:
    """One decoded speaker track as mono float64 samples in [-1, 1]."""

    speaker_id: str
    path: Path
    sample_rate: int
    samples: list[float]

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate


def _decode(raw: bytes, sample_width: int, channels: int) -> list[float]:
    if sample_width == 1:
        # 8-bit PCM is unsigned, centred at 128.
        decoded = [(byte - 128) / 128.0 for byte in raw]
    elif sample_width == 2:
        count = len(raw) // 2
        decoded = [
            value / 32768.0
            for value in struct.unpack(f"<{count}h", raw)
        ]
    elif sample_width == 3:
        count = len(raw) // 3
        decoded = []
        for offset in range(count):
            base = offset * 3
            value = (
                raw[base]
                | (raw[base + 1] << 8)
                | ((raw[base + 2] ^ 0x80) << 16)
            ) - 0x800000
            decoded.append(value / 8388608.0)
    elif sample_width == 4:
        count = len(raw) // 4
        decoded = [
            max(-1.0, min(1.0, value / 2147483648.0))
            for value in struct.unpack(f"<{count}i", raw)
        ]
    else:
        raise ValueError(f"unsupported sample width: {sample_width} bytes")

    if channels == 1:
        return decoded

    # Downmix multi-channel tracks to mono.
    frame_count = len(decoded) // channels
    return [
        sum(decoded[i * channels : (i + 1) * channels]) / channels
        for i in range(frame_count)
    ]


def read_wav(path: str | Path, speaker_id: str | None = None) -> Track:
    path = Path(path)
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        raw = wav.readframes(frames)

    samples = _decode(raw, sample_width, channels)
    return Track(
        speaker_id=speaker_id if speaker_id is not None else path.stem,
        path=path,
        sample_rate=sample_rate,
        samples=samples,
    )


def write_wav(
    path: str | Path, samples: list[float], sample_rate: int
) -> None:
    """Write 16-bit PCM WAV (clipped, no dithering needed for test material)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    integer = [
        32767
        if sample >= 1.0
        else -32768
        if sample <= -1.0
        else int(math.copysign(math.floor(abs(sample) * 32768.0 + 0.5), sample))
        for sample in samples
    ]
    packed = struct.pack(f"<{len(integer)}h", *integer)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(packed)
