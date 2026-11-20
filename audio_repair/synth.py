"""Synthetic fixtures: crossed chirps, transients, narrowband hum."""

from __future__ import annotations

import numpy as np
from scipy.signal import chirp


def crossed_chirps(sr: int = 44100, dur: float = 4.0,
                   seed: int = 7) -> dict:
    """Two crossing chirps + clicks + hum over a quiet background.

    Returns the mixed signal, the isolated components and ground-truth
    trajectories so verification can build ideal strokes.
    """
    t = np.arange(int(sr * dur)) / sr
    f_lo, f_hi = 400.0, 3200.0
    up = chirp(t, f0=f_lo, f1=f_hi, t1=dur, method="linear") * 0.5
    down = chirp(t, f0=f_hi, f1=f_lo, t1=dur, method="linear") * 0.5

    # transients: short decaying clicks at fixed times
    rng = np.random.default_rng(seed)
    clicks = np.zeros_like(t)
    click_times = [0.5, 1.3, 2.1, 2.9, 3.6]
    for tc in click_times:
        i0 = int(tc * sr)
        n = int(0.02 * sr)
        env = np.exp(-np.linspace(0, 8, n))
        clicks[i0:i0 + n] += env * rng.standard_normal(n) * 0.8

    hum = 0.3 * np.sin(2 * np.pi * 900.0 * t)  # narrowband noise to remove
    bed = 0.05 * rng.standard_normal(len(t))   # quiet background

    mix = up + down + clicks + hum + bed
    mix *= 0.9 / max(1.0, np.max(np.abs(mix)))

    up_traj = [(float(ti), float(f_lo + (f_hi - f_lo) * ti / dur))
               for ti in np.linspace(0.02, dur - 0.02, 40)]
    down_traj = [(float(ti), float(f_hi - (f_hi - f_lo) * ti / dur))
                 for ti in np.linspace(0.02, dur - 0.02, 40)]
    return {
        "sr": sr, "dur": dur, "mix": mix,
        "components": {"up": up, "down": down, "clicks": clicks,
                       "hum": hum, "bed": bed},
        "trajectories": {"up": up_traj, "down": down_traj},
        "click_times": click_times,
        "hum_freq": 900.0,
    }


def hum_noise(sr: int = 44100, dur: float = 3.0, seed: int = 3) -> dict:
    """Tonal hum + harmonics over a broadband signal, for hum removal."""
    t = np.arange(int(sr * dur)) / sr
    rng = np.random.default_rng(seed)
    music = 0.4 * np.sin(2 * np.pi * 250 * t) \
        + 0.2 * np.sin(2 * np.pi * 350 * t) \
        + 0.1 * rng.standard_normal(len(t))
    hum = sum(0.25 / k * np.sin(2 * np.pi * 100.0 * k * t)
              for k in (1, 2, 3, 4))
    mix = music + np.asarray(hum)
    return {"sr": sr, "dur": dur, "mix": mix,
            "components": {"music": music, "hum": np.asarray(hum)},
            "music_freqs": [250.0, 350.0],
            "hum_freqs": [100.0 * k for k in (1, 2, 3, 4)]}
