"""Gain-envelope rendering: full renders and affected-range-only re-renders.

Both paths evaluate the same envelope function per frame and quantize with
the same deterministic rounding, so an incremental render spliced into a
previous master is sample-identical to a full re-render, provided the
envelope is unchanged outside the re-rendered range.
"""
from __future__ import annotations

import math

from .bwf import WAVE_FORMAT_IEEE_FLOAT, BwfFile
from .envelope import Envelope


def to_float(value, audio_format: int, bit_depth: int) -> float:
    if audio_format == WAVE_FORMAT_IEEE_FLOAT:
        return float(value)
    return value / float(1 << (bit_depth - 1))


def from_float(value: float, audio_format: int, bit_depth: int):
    if audio_format == WAVE_FORMAT_IEEE_FLOAT:
        return float(value)
    peak = 1 << (bit_depth - 1)
    scaled = int(round(value * peak))
    return max(-peak, min(peak - 1, scaled))


def render_frames(
    src: BwfFile, env: Envelope, start_frame: int, end_frame: int
) -> list:
    """Render interleaved output samples for frames [start_frame, end_frame)."""
    start_frame = max(0, start_frame)
    end_frame = min(src.frame_count, end_frame)
    ch = src.channels
    sr = src.sample_rate
    fmt, bits = src.audio_format, src.bit_depth
    gains = [
        math.pow(10.0, env.eval_db(f / sr) / 20.0)
        for f in range(start_frame, end_frame)
    ]
    out = []
    samples = src.samples
    for i, gain in enumerate(gains):
        base = (start_frame + i) * ch
        for c in range(ch):
            value = to_float(samples[base + c], fmt, bits) * gain
            out.append(from_float(value, fmt, bits))
    return out


def render_full(src: BwfFile, env: Envelope) -> BwfFile:
    """Render the whole master from source and envelope."""
    return BwfFile(
        sample_rate=src.sample_rate,
        channels=src.channels,
        bit_depth=src.bit_depth,
        audio_format=src.audio_format,
        samples=render_frames(src, env, 0, src.frame_count),
        bext=src.bext,
        extra_chunks=list(src.extra_chunks),
    )


def affected_frames(time_range: tuple, src: BwfFile) -> tuple:
    """Convert an affected time range to a clamped frame range [start, end).

    Both ends are inclusive-safe: any frame whose timestamp falls inside
    [t0, t1] is covered, including a zero-width range that lands exactly
    on a frame boundary (a single-point envelope change).
    """
    t0, t1 = time_range
    start = max(0, int(math.floor(t0 * src.sample_rate)))
    end = min(src.frame_count, int(math.floor(t1 * src.sample_rate)) + 1)
    return (start, end)


def render_incremental(
    src: BwfFile, env: Envelope, master: BwfFile, frame_range: tuple
) -> BwfFile:
    """Re-render only frames [start, end) of the master; copy the rest.

    Frames outside frame_range are taken verbatim from the previous master,
    which equals a full re-render there as long as the envelope change is
    confined to frame_range (see Envelope.affected_range).
    """
    start, end = frame_range
    ch = src.channels
    total = src.frame_count
    start = max(0, min(start, total))
    end = max(start, min(end, total))
    samples = (
        master.samples[: start * ch]
        + render_frames(src, env, start, end)
        + master.samples[end * ch :]
    )
    return BwfFile(
        sample_rate=src.sample_rate,
        channels=src.channels,
        bit_depth=src.bit_depth,
        audio_format=src.audio_format,
        samples=samples,
        bext=src.bext,
        extra_chunks=list(src.extra_chunks),
    )
