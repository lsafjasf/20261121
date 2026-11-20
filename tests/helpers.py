"""Shared fixtures: synthetic BWF files for tests."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bwf_batch.bwf import (  # noqa: E402
    WAVE_FORMAT_IEEE_FLOAT,
    WAVE_FORMAT_PCM,
    BextChunk,
    BwfFile,
    write_bwf,
)

SAMPLE_RATE = 8000


def make_bwf(
    frames=1000,
    channels=2,
    bit_depth=16,
    audio_format=WAVE_FORMAT_PCM,
    sample_rate=SAMPLE_RATE,
    with_bext=True,
):
    """Deterministic two-tone + ramp test signal, interleaved."""
    samples = []
    for f in range(frames):
        t = f / sample_rate
        left = 0.5 * math.sin(2 * math.pi * 440 * t)
        right = 0.25 * math.sin(2 * math.pi * 1000 * t) + 0.1 * (f / frames)
        pair = (left, right) if channels == 2 else (left,) * channels
        for value in pair[:channels]:
            if audio_format == WAVE_FORMAT_IEEE_FLOAT:
                samples.append(float(value))
            else:
                peak = 1 << (bit_depth - 1)
                samples.append(max(-peak, min(peak - 1, int(round(value * peak)))))
    bext = None
    if with_bext:
        bext = BextChunk(
            description="test material",
            originator="bwf_batch tests",
            originator_reference="TESTREF0001",
            origination_date="2026-11-21",
            origination_time="12:00:00",
            time_reference=0,
        )
    return BwfFile(
        sample_rate=sample_rate,
        channels=channels,
        bit_depth=bit_depth,
        audio_format=audio_format,
        samples=samples,
        bext=bext,
    )


def write_source(tmpdir, name="src.wav", **kwargs):
    path = os.path.join(tmpdir, name)
    write_bwf(path, make_bwf(**kwargs))
    return path
