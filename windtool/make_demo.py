"""Generate a stereo demo recording that overlays all three phenomena.

Layout (8 s, 48 kHz, two capsules with a small level/phase mismatch):

* 0.5-2.5 s  correlated kick-drum hits (shared source, stable phase),
* 3.0-5.5 s  independent wind gusts (decorrelated, sustained, < 150 Hz),
* 6.0-7.6 s  wind again plus one-sided handling bumps on channel 1,
with a low-level ambience bed throughout so silence frames are realistic.
"""

from __future__ import annotations

import numpy as np

from .io import save_stereo

SR = 48000
DURATION = 8.0


def _time_axis(n: int) -> np.ndarray:
    return np.arange(n) / SR


def _ambience(n: int, rng: np.random.Generator) -> np.ndarray:
    bed = rng.standard_normal((2, n)) * 0.004
    # Gentle low-pass so the bed is not pure white hiss.
    from scipy import signal

    sos = signal.butter(2, 1200.0, btype="lowpass", fs=SR, output="sos")
    return signal.sosfiltfilt(sos, bed, axis=-1)


def _kick(t: np.ndarray, hit_time: float, rng: np.random.Generator) -> np.ndarray:
    """One kick hit: pitch-dropping sine burst plus a short click, in stereo."""
    dt = t - hit_time
    env = np.exp(-dt * 22.0) * (dt >= 0)
    pitch = 45.0 + 35.0 * np.exp(-dt * 30.0)
    phase = 2.0 * np.pi * np.cumsum(pitch) / SR
    body = np.sin(phase) * env
    click = rng.standard_normal(len(t)) * np.exp(-dt * 180.0) * (dt >= 0) * 0.25
    mono = body + click
    stereo = np.stack([mono, mono * 0.92], axis=0)
    # Tiny physical offset between capsules: ~0.4 ms.
    shift = int(0.0004 * SR)
    stereo[1] = np.roll(stereo[1], shift)
    return stereo * 0.55


def _wind(n: int, start: float, end: float, rng: np.random.Generator) -> np.ndarray:
    """Sustained, decorrelated low-frequency gusts per capsule."""
    from scipy import signal

    t = _time_axis(n)
    gate = ((t >= start) & (t <= end)).astype(float)
    # Slow amplitude modulation gives the gust a natural envelope.
    gust = 0.6 + 0.4 * np.sin(2.0 * np.pi * 1.7 * t + rng.uniform(0, 6))
    sos = signal.butter(4, [28.0, 140.0], btype="bandpass", fs=SR, output="sos")
    channels = []
    for scale in (1.0, 0.85):
        noise = rng.standard_normal(n)
        colored = signal.sosfiltfilt(sos, noise)
        channels.append(colored * gust * gate * 0.32 * scale)
    return np.stack(channels, axis=0)


def _bump(t: np.ndarray, hit_time: float, rng: np.random.Generator) -> np.ndarray:
    """Low-frequency thump present almost only on channel 1."""
    dt = t - hit_time
    env = np.exp(-dt * 28.0) * (dt >= 0)
    thump = np.sin(2.0 * np.pi * 55.0 * dt) * env
    thump += rng.standard_normal(len(t)) * env * 0.15
    ch1 = thump * 0.8
    ch2 = thump * 0.05  # >20 dB channel separation
    return np.stack([ch1, ch2], axis=0)


def build_demo(path: str, seed: int = 20261121) -> dict:
    rng = np.random.default_rng(seed)
    n = int(DURATION * SR)
    t = _time_axis(n)
    mix = _ambience(n, rng)

    ground_truth = {"drum": [], "wind": [], "bump": []}
    for hit in (0.7, 1.3, 1.9):
        mix += _kick(t, hit, rng)
        ground_truth["drum"].append([round(hit, 3), round(hit + 0.18, 3)])

    mix += _wind(n, 3.0, 5.5, rng)
    ground_truth["wind"].append([3.0, 5.5])
    mix += _wind(n, 6.0, 7.6, rng)
    ground_truth["wind"].append([6.0, 7.6])

    for hit in (6.35, 7.05):
        mix += _bump(t, hit, rng)
        ground_truth["bump"].append([round(hit, 3), round(hit + 0.15, 3)])

    save_stereo(path, mix, SR)
    return {"sample_rate": SR, "duration_sec": DURATION, "ground_truth": ground_truth}
