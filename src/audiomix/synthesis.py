"""Procedural source-signal synthesis (pure stdlib).

All signals are mono float32 in the range [-1, 1], sampled at the session
sample rate.  Three distinct timbres are generated so the three mixer
channels can be told apart both by ear and in the recorded waveform:

* ambient bed  - filtered brown noise (slow, soft, continuous)
* narration    - a syllable train of voiced formant pulses with gaps
* announcement - sharp two-tone alert chirps (hard transient onsets)
"""

from __future__ import annotations

import math
import random


def _phase_inc(freq: float, sample_rate: int) -> float:
    return 2.0 * math.pi * freq / sample_rate


def brown_noise(n: int, sample_rate: int, level: float = 0.5,
                cutoff_hz: float = 400.0, seed: int = 7) -> list[float]:
    """Low-pass filtered brown noise: a calm, rumbling ambient bed."""
    rng = random.Random(seed)
    # One-pole low pass coefficient for the requested cutoff.
    alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff_hz / sample_rate)
    out: list[float] = []
    integrated = 0.0
    filtered = 0.0
    for _ in range(n):
        integrated = 0.995 * integrated + 0.05 * (rng.random() * 2.0 - 1.0)
        filtered += alpha * (integrated - filtered)
        out.append(filtered * level)
    peak = max((abs(s) for s in out), default=1.0) or 1.0
    return [s / peak * level for s in out]


def syllable_train(n: int, sample_rate: int, level: float = 0.9,
                   syllable_hz: float = 3.2, seed: int = 11) -> list[float]:
    """Voiced syllables: 120 Hz glottal pulse through a moving formant.

    A syllable is an amplitude-shaped voiced burst; gaps between syllables
    make the "first syllable after a duck" easy to inspect.
    """
    rng = random.Random(seed)
    period = sample_rate / syllable_hz
    voiced_frac = 0.62  # remainder of each period is silence (a word gap)
    glottal = 110.0
    out: list[float] = []
    phase = 0.0
    for i in range(n):
        pos = (i % period) / period
        if pos < voiced_frac:
            local = pos / voiced_frac
            # Fast attack, slower release envelope per syllable.
            env = math.sin(math.pi * min(1.0, local * 1.15)) ** 1.5
            formant = 700.0 + 260.0 * math.sin(_phase_inc(1.7, sample_rate) * i)
            phase += _phase_inc(glottal, sample_rate)
            wobble = 1.0 + 0.02 * math.sin(phase * 0.37 + rng.random() * 0.0)
            sample = env * math.sin(phase) * 0.6
            sample += env * 0.4 * math.sin(_phase_inc(formant, sample_rate) * i * wobble)
            out.append(max(-1.0, min(1.0, sample)) * level)
        else:
            # Keep the oscillator phase advancing through the gap.
            phase += _phase_inc(glottal, sample_rate)
            out.append(0.0)
    return out


def alert_chirp(n: int, sample_rate: int, level: float = 1.0,
                f0: float = 880.0, f1: float = 1320.0) -> list[float]:
    """Two-tone alert with a hard onset transient (stresses the attack)."""
    out: list[float] = []
    split = n // 2
    phase = 0.0
    prev_f = f0
    for i in range(n):
        freq = f0 if i < split else f1
        phase += _phase_inc(freq, sample_rate)
        # 8 ms attack so the very first sample is not a full-scale click,
        # but still fast enough to expose sidechain overshoot.
        attack = min(1.0, i / (0.008 * sample_rate))
        release = min(1.0, (n - i) / (0.02 * sample_rate))
        env = attack * release
        # Add a crisp inharmonic transient at the onset.
        tick = math.exp(-i / (0.004 * sample_rate)) * math.sin(
            _phase_inc(2600.0, sample_rate) * i)
        out.append(max(-1.0, min(1.0,
                     env * math.sin(phase) * 0.85 + tick * 0.3)) * level)
    return out


def pad_signal(signal: list[float], n: int) -> list[float]:
    """Right-pad with silence (or truncate) to exactly ``n`` samples."""
    if len(signal) >= n:
        return signal[:n]
    return signal + [0.0] * (n - len(signal))
