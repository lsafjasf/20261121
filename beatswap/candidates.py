"""Splice-candidate search and mis-match warnings."""

from __future__ import annotations

import numpy as np


def _ncc(x, y, max_lag):
    """Max normalised cross-correlation of x against y over +/-max_lag."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = min(len(x), len(y))
    if n < 4:
        return 0.0, 0
    x = x[:n] - x[:n].mean()
    y0 = y[:n] - y[:n].mean()
    best, best_lag = -2.0, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = x[lag:], y0[:n - lag] if lag else y0
        else:
            a, b = x[:n + lag], y0[-lag:]
        m = min(len(a), len(b))
        if m < 4:
            continue
        a, b = a[:m], b[:m]
        d = np.sqrt((a ** 2).sum() * (b ** 2).sum()) + 1e-12
        c = float((a * b).sum() / d)
        if c > best:
            best, best_lag = c, lag
    return best, best_lag


def _window(env, center_frame, length_frames):
    lo = int(max(0, round(center_frame)))
    hi = int(min(len(env), lo + int(length_frames)))
    return env[lo:hi]


def _window_resampled(env, lo_frame, hi_frame, out_len):
    lo = int(max(0, round(lo_frame)))
    hi = int(min(len(env), round(hi_frame)))
    seg = env[lo:hi]
    if len(seg) < 2 or out_len < 2:
        return np.zeros(max(int(out_len), 2))
    return np.interp(np.linspace(0, 1, int(out_len)),
                     np.linspace(0, 1, len(seg)), seg)


def find_candidates(env_a, beats_a, env_b, beats_b, mapper, fps,
                    bars=8, beats_per_bar=4, top_k=5, search_step=None):
    """Score every entry beat of A for a swap of `bars` bars against B."""
    span = bars * beats_per_bar
    if search_step is None:
        search_step = beats_per_bar  # bar-aligned entries by default
    n_beats_a = len(beats_a)
    if n_beats_a < span + 2:
        raise RuntimeError("not enough beats in A for the requested span")
    med_a = float(np.median(np.diff(beats_a)))
    med_b = float(np.median(np.diff(beats_b)))
    tempo_ratio = med_a / med_b  # >1 means B is faster
    win = int(round(med_a * 2))  # 2-beat comparison windows
    lag = max(2, int(round(med_a * 0.15)))
    candidates = []
    for e in range(0, n_beats_a - span, search_step):
        fa0, fa1 = beats_a[e], beats_a[e + span]
        fb0, fb1 = mapper(fa0), mapper(fa1)
        tol = 0.5 * med_b
        inside = int(np.sum((beats_b >= fb0 - tol) & (beats_b < fb1 - tol)))
        # span duration measured in B-beat units catches inserted/dropped
        # material even when the beat grid itself stays phase-locked
        b_span_beats = (fb1 - fb0) / med_b
        warns = []
        if abs(b_span_beats - span) > 0.5 or inside != span:
            diff = b_span_beats - span
            kind = "extra" if diff > 0 else "missing"
            warns.append(
                f"beat_count_mismatch: mapped B span is "
                f"{b_span_beats:.2f} beat-durations (grid count {inside}), "
                f"expected {span} ({abs(diff):.2f} {kind} beat(s) -> likely "
                f"inserted/dropped beat in this region)")
        local_ratio = (fa1 - fa0) / max(fb1 - fb0, 1e-9)
        if abs(local_ratio / tempo_ratio - 1.0) > 0.02:
            warns.append(
                f"tempo_drift_mismatch: local A/B duration ratio "
                f"{local_ratio:.3f} deviates from global {tempo_ratio:.3f} "
                f"by >2% (structural difference, not just tempo)")
        ea, _ = _ncc(_window(env_a, fa0, win), _window(env_b, fb0, win), lag)
        ex, _ = _ncc(_window(env_a, fa1, win), _window(env_b, fb1, win), lag)
        n_out = int(fa1 - fa0)
        ia, _ = _ncc(_window(env_a, fa0, fa1 - fa0),
                     _window_resampled(env_b, fb0, fb1, n_out), lag)
        score = 0.4 * min(ea, ex) + 0.2 * (ea + ex) / 2 + 0.4 * max(ia, 0.0)
        if warns:
            score *= 0.5  # demote structurally mismatched windows
        candidates.append(dict(
            entry_beat=int(e), exit_beat=int(e + span),
            a_entry_frame=float(fa0), a_exit_frame=float(fa1),
            b_entry_frame=float(fb0), b_exit_frame=float(fb1),
            entry_sim=round(ea, 4), exit_sim=round(ex, 4),
            interior_sim=round(ia, 4), b_beats_in_span=inside,
            score=round(float(score), 4), warnings=warns))
    ranked = sorted(candidates, key=lambda c: -c["score"])
    chosen = []
    min_sep = span // 2
    for c in ranked:
        if all(abs(c["entry_beat"] - o["entry_beat"]) >= min_sep
               for o in chosen):
            chosen.append(c)
        if len(chosen) >= top_k:
            break
    chosen.sort(key=lambda c: c["entry_beat"])
    diagnostics = dict(
        tempo_ratio_a_over_b=round(tempo_ratio, 5),
        median_beat_a_frames=round(med_a, 2),
        median_beat_b_frames=round(med_b, 2),
        n_windows_scored=len(candidates))
    return chosen, candidates, diagnostics
