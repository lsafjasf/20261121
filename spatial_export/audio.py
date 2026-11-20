"""Offline signal synthesis, block-accurate gain rendering and WAV I/O.

Only the standard library is used (``wave``, ``struct``). No audio device,
no third-party dependency and no network access are required.
"""

from __future__ import annotations

import math
import struct
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RenderConfig:
    sample_rate: int = 48000
    block_size: int = 512
    amplitude: float = 0.6
    bit_depth: int = 24


def synthesize_mono(
    num_samples: int, sample_rate: int, amplitude: float = 0.6
) -> list[float]:
    """Deterministic band-limited-ish test signal: three sines, no DC."""
    freqs = (220.0, 440.0, 880.0)
    weights = (0.55, 0.3, 0.15)
    out: list[float] = []
    for n in range(num_samples):
        t = n / sample_rate
        value = sum(
            w * math.sin(2.0 * math.pi * f * t) for f, w in zip(freqs, weights)
        )
        # Gentle fade in/out avoids boundary clicks in every rendered file.
        fade = min(1.0, n / 2048.0, (num_samples - 1 - n) / 2048.0)
        out.append(amplitude * fade * value)
    return out


def render_channels(
    signal: list[float],
    gain_table: list[tuple[float, ...]],
    block_size: int,
    num_channels: int,
) -> list[list[float]]:
    """Apply block gains with a short linear crossfade at block boundaries.

    ``gain_table[b]`` holds the target gains for block ``b``. Targets hold
    across each block; a linear crossfade of ``block_size // 2`` samples
    bridges consecutive blocks, and every blended gain vector is renormalized
    to unit power. The result is click-free, continuous across hand-offs and
    energy-conserving (total gain power exactly 1 at every sample).
    """
    num_samples = len(signal)
    channels = [[0.0] * num_samples for _ in range(num_channels)]
    fade_len = max(1, block_size // 2)
    for block_index, target in enumerate(gain_table):
        start = block_index * block_size
        end = min(start + block_size, num_samples)
        if start >= num_samples:
            break
        previous = (
            gain_table[block_index - 1]
            if block_index > 0
            else (0.0,) * num_channels
        )
        for n in range(start, end):
            offset = n - start
            if offset < fade_len:
                frac = (offset + 1) / fade_len
                blended = tuple(
                    previous[ch] * (1.0 - frac) + target[ch] * frac
                    for ch in range(num_channels)
                )
                power = math.sqrt(sum(g * g for g in blended))
                gain = (
                    tuple(g / power for g in blended)
                    if power > 1e-15
                    else target
                )
            else:
                gain = target
            sample = signal[n]
            for ch in range(num_channels):
                channels[ch][n] = sample * gain[ch]
    return channels


def block_rms(buffer: list[float], block_size: int) -> list[float]:
    """Per-block RMS used by the energy-continuity verification."""
    result: list[float] = []
    for start in range(0, len(buffer), block_size):
        block = buffer[start : start + block_size]
        if not block:
            continue
        result.append(math.sqrt(sum(x * x for x in block) / len(block)))
    return result


def _encode_sample(value: float, bit_depth: int) -> bytes:
    value = max(-1.0, min(1.0, value))
    if bit_depth == 16:
        return struct.pack("<h", int(round(value * 32767.0)))
    if bit_depth == 24:
        q = int(round(value * 8388607.0))
        return struct.pack("<i", q)[0:3]
    if bit_depth == 32:
        return struct.pack("<i", int(round(value * 2147483647.0)))
    raise ValueError(f"unsupported bit depth: {bit_depth}")


def _decode_sample(raw: bytes, bit_depth: int) -> float:
    if bit_depth == 16:
        return struct.unpack("<h", raw)[0] / 32767.0
    if bit_depth == 24:
        padded = raw + (b"\xff" if raw[2] & 0x80 else b"\x00")
        return struct.unpack("<i", padded)[0] / 8388607.0
    if bit_depth == 32:
        return struct.unpack("<i", raw)[0] / 2147483647.0
    raise ValueError(f"unsupported bit depth: {bit_depth}")


def write_multichannel_wav(
    path: Path,
    channels: list[list[float]],
    sample_rate: int,
    bit_depth: int = 24,
) -> None:
    """Write one interleaved, multi-channel PCM WAV (offline, local file)."""
    num_channels = len(channels)
    num_frames = min(len(ch) for ch in channels)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(num_channels)
        wav.setsampwidth(bit_depth // 8)
        wav.setframerate(sample_rate)
        frames = bytearray()
        for frame in range(num_frames):
            for ch in range(num_channels):
                frames.extend(_encode_sample(channels[ch][frame], bit_depth))
        wav.writeframes(bytes(frames))


def write_mono_wav(
    path: Path, buffer: list[float], sample_rate: int, bit_depth: int = 24
) -> None:
    write_multichannel_wav(path, [buffer], sample_rate, bit_depth)


def read_wav(path: Path) -> tuple[list[list[float]], int, int]:
    """Read a PCM WAV back as deinterleaved float channels."""
    with wave.open(str(path), "rb") as wav:
        num_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    bit_depth = sample_width * 8
    channels: list[list[float]] = [[] for _ in range(num_channels)]
    step = sample_width
    frame_count = len(raw) // (sample_width * num_channels)
    for frame in range(frame_count):
        base = frame * sample_width * num_channels
        for ch in range(num_channels):
            chunk = raw[base + ch * step : base + (ch + 1) * step]
            channels[ch].append(_decode_sample(chunk, bit_depth))
    return channels, sample_rate, bit_depth
