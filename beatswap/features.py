"""Onset strength envelope, tempo estimation and beat tracking."""

from __future__ import annotations

import numpy as np
from scipy import signal

HOP = 256
NFFT = 1024


def onset_envelope(x: np.ndarray, sr: int, hop: int = HOP) -> tuple[np.ndarray, float]:
    """Spectral-flux onset strength envelope.

    Returns (envelope, frames_per_second). Loudness-independent because it
    works on normalised log-magnitude differences.
    """
    window = signal.windows.hann(NFFT)
    n_frames = 1 + max(0, (len(x) - NFFT) // hop)
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, NFFT),
        strides=(x.strides[0] * hop, x.strides[0]),
    )
    mag = np.abs(np.fft.rfft(frames * window, axis=1))
    log_mag = np.log1p(1000.0 * mag)
    flux = np.diff(log_mag, axis=0, prepend=log_mag[:1])
    np.maximum(flux, 0.0, out=flux)
    env = flux.sum(axis=1)
    med = signal.medfilt(env, kernel_size=31)
    env = np.maximum(env - med, 0.0)
    fps = sr / hop
    return env, fps


def estimate_tempo(env: np.ndarray, fps: float,
                   bpm_min: float = 60.0, bpm_max: float = 200.0) -> float:
    """Comb-filterbank tempo estimate with a mild 120 BPM prior."""
    e = env - env.mean()
    ac = np.correlate(e, e, mode="full")[len(e) - 1:]
    ac = ac / (ac[0] + 1e-12)
    lag_min = max(2, int(fps * 60.0 / bpm_max))
    lag_max = min(int(fps * 60.0 / bpm_min), (len(ac) - 1) // 4)
    if lag_max <= lag_min:
        return 120.0
    best_lag, best_score = lag_min, -np.inf
    for lag in range(lag_min, lag_max):
        # comb: reward harmonically related lags, normalise by term count
        ks = np.arange(1, 5)
        idx = lag * ks
        idx = idx[idx < len(ac)]
        score = float(ac[idx].sum()) / len(idx)
        bpm = 60.0 * fps / lag
        score += 0.05 * np.exp(-0.5 * ((bpm - 120.0) / 40.0) ** 2)
        if score > best_score:
            best_score, best_lag = score, lag
    lag = best_lag
    if lag_min < lag < lag_max:
        y0, y1, y2 = ac[lag - 1], ac[lag], ac[lag + 1]
        denom = (y0 - 2 * y1 + y2)
        if abs(denom) > 1e-12:
            lag = lag + 0.5 * (y0 - y2) / denom
    return 60.0 * fps / lag


def track_beats(env: np.ndarray, fps: float, bpm: float) -> np.ndarray:
    """Dynamic-programming beat tracker (Ellis 2007 style).

    Returns beat positions in envelope frames (float, ascending).
    """
    period = fps * 60.0 / bpm
    e = env / (env.max() + 1e-12)
    n = len(e)
    cumscore = np.full(n, -np.inf)
    backlink = np.full(n, -1, dtype=int)
    tightness = 100.0
    lo = int(round(period * 0.5))
    hi = int(round(period * 2.0))
    local = np.zeros(n)
    for i in range(n):
        j0 = max(0, i - hi)
        j1 = max(0, i - lo)
        if j1 > j0:
            js = np.arange(j0, j1)
            penalty = -tightness * np.log((i - js) / period) ** 2
            cand = cumscore[js] + penalty
            best = int(np.argmax(cand))
            local[i] = cand[best]
            backlink[i] = js[best]
        else:
            local[i] = 0.0
        cumscore[i] = e[i] + local[i]
    tail_start = max(0, n - int(round(period)))
    last = int(np.argmax(cumscore[tail_start:])) + tail_start
    beats = []
    b = last
    while b >= 0:
        beats.append(b)
        b = backlink[b]
    beats = np.array(beats[::-1], dtype=float)
    return beats
