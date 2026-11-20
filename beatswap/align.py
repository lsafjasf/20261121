"""DTW alignment of two onset envelopes and beat-time mapping."""

from __future__ import annotations

import numpy as np


def _znorm(e):
    e = np.asarray(e, dtype=float)
    s = float(e.std())
    if s < 1e-12:
        s = 1.0
    return (e - e.mean()) / s


def dtw_path(env_a, env_b, max_ratio=1.12):
    """Full DTW with a Sakoe-Chiba band; returns (path_a, path_b, cost)."""
    a = _znorm(env_a)
    b = _znorm(env_b)
    n, m = len(a), len(b)
    band = int(max(n, m) * (max_ratio - 1.0) + 0.02 * max(n, m)) + 4
    k = 3
    a_s = np.convolve(a, np.ones(k) / k, mode="same")
    b_s = np.convolve(b, np.ones(k) / k, mode="same")
    INF = 1e18
    D = np.full((n, m), INF)
    D[0, 0] = abs(a_s[0] - b_s[0])
    for i in range(n):
        j_lo = max(0, i - band)
        j_hi = min(m, i + band + 1)
        for j in range(j_lo, j_hi):
            if i == 0 and j == 0:
                continue
            cost = abs(a_s[i] - b_s[j])
            best = INF
            if i > 0:
                best = min(best, D[i - 1, j])
            if j > 0:
                best = min(best, D[i, j - 1])
            if i > 0 and j > 0:
                best = min(best, D[i - 1, j - 1])
            D[i, j] = cost + best
    if D[n - 1, m - 1] >= INF / 2:
        raise RuntimeError("DTW failed: band too narrow for these files")
    i, j = n - 1, m - 1
    pa, pb = [i], [j]
    total = abs(a_s[i] - b_s[j])
    steps = 1
    while i > 0 or j > 0:
        cands = []
        if i > 0 and j > 0:
            cands.append((D[i - 1, j - 1], i - 1, j - 1))
        if i > 0:
            cands.append((D[i - 1, j], i - 1, j))
        if j > 0:
            cands.append((D[i, j - 1], i, j - 1))
        cands.sort(key=lambda t: t[0])
        _, i, j = cands[0]
        pa.append(i)
        pb.append(j)
        total += abs(a_s[i] - b_s[j])
        steps += 1
    pa = np.array(pa[::-1])
    pb = np.array(pb[::-1])
    return pa, pb, total / steps


def make_frame_mapper(pa, pb):
    """Build a function mapping A-frame -> B-frame (float) from the path."""
    sums = np.zeros(int(pa.max()) + 1)
    counts = np.zeros(int(pa.max()) + 1)
    np.add.at(sums, pa, pb)
    np.add.at(counts, pa, 1)
    valid = counts > 0
    xs = np.nonzero(valid)[0]
    ys = sums[valid] / counts[valid]

    def map_frame(fa):
        return float(np.interp(fa, xs, ys))

    return map_frame


def map_beats(beats_a, mapper):
    """Map beat frame positions of A onto B's frame clock."""
    return np.array([mapper(f) for f in beats_a])
