"""Synthetic tape material for validating wow/flutter restoration.

Generates two channels through the *same* physical time-base warp:

* a pilot channel: a single stable reference tone,
* a program channel: a stable note plus a note with intentional performer
  vibrato, so validation can prove that machine wow is removed while the
  musician's own vibrato survives.

The warp combines slow drift, low-frequency wow and faster flutter.  Pilot
dropouts (level fades to silence/noise) can be injected to exercise the
bounded extrapolation path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .resample import _sinc_table, _interpolate


@dataclass
class SyntheticTape:
    sample_rate: float
    pilot: np.ndarray
    program: np.ndarray
    clean_program: np.ndarray
    speed_error: np.ndarray
    input_position: np.ndarray
    dropout_mask: np.ndarray


def _warp_speed(t: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(7)
    slow = 0.008 * np.sin(2 * np.pi * 0.07 * t)          # slow drift
    wow = 0.004 * np.sin(2 * np.pi * 1.7 * t + 0.6)     # wow
    flutter = 0.0015 * np.sin(2 * np.pi * 6.3 * t)      # flutter
    jitter = 0.0008 * rng.standard_normal(len(t))       # tape noise floor
    return slow + wow + flutter + jitter


def generate_tape(
    fs: float = 16000.0,
    duration: float = 8.0,
    *,
    pilot_freq: float = 1000.0,
    stable_note: float = 392.0,
    vibrato_note: float = 523.25,
    vibrato_rate: float = 5.5,
    vibrato_depth_cents: float = 35.0,
    dropout_intervals: tuple[tuple[float, float], ...] = ((2.4, 3.1), (5.2, 5.7)),
    noise_level: float = 0.01,
) -> SyntheticTape:
    """Build warped pilot/program signals and the ground-truth warp."""
    rng = np.random.default_rng(11)
    n = int(round(fs * duration))
    index = np.arange(n)
    t = index / fs

    speed = _warp_speed(t)
    position = index + np.cumsum(speed)
    table = _sinc_table(16, 64)

    # Performer vibrato: a deliberate frequency modulation that must survive.
    depth = vibrato_depth_cents / 1200.0
    vib_phase = 2 * np.pi * vibrato_note * (
        t + depth / (2 * np.pi * vibrato_rate) * np.sin(2 * np.pi * vibrato_rate * t)
    )
    clean_program = (
        0.6 * np.sin(2 * np.pi * stable_note * t)
        + 0.4 * np.sin(vib_phase)
    )
    clean_pilot = np.sin(2 * np.pi * pilot_freq * t)

    # Apply the shared physical warp to both channels.
    warped_program = _interpolate(clean_program, position, table)
    warped_pilot = _interpolate(clean_pilot, position, table)

    dropout = np.zeros(n, dtype=bool)
    envelope = np.ones(n)
    for start, end in dropout_intervals:
        lo, hi = int(start * fs), int(end * fs)
        dropout[lo:hi] = True
        fade = int(0.03 * fs)
        ramp = np.linspace(1.0, 0.0, fade)
        envelope[lo:lo + fade] = ramp
        envelope[hi - fade:hi] = ramp[::-1]
        envelope[lo + fade:hi - fade] = 0.0

    noisy_pilot = warped_pilot * envelope
    noisy_pilot += noise_level * rng.standard_normal(n)
    warped_program += noise_level * rng.standard_normal(n)

    return SyntheticTape(
        sample_rate=fs,
        pilot=noisy_pilot,
        program=warped_program,
        clean_program=clean_program,
        speed_error=speed,
        input_position=position,
        dropout_mask=dropout,
    )
