"""Render audition joins: A-head + B-middle + A-tail (and vice versa)."""

from __future__ import annotations

import numpy as np


def _xfade_join(left, right, sr, fade_ms=15.0):
    """Equal-power crossfade join of two 1-D signals."""
    n = int(sr * fade_ms / 1000.0)
    n = min(n, len(left), len(right))
    if n <= 0:
        return np.concatenate([left, right])
    t = np.linspace(0.0, np.pi / 2.0, n)
    out = np.concatenate([
        left[:-n],
        left[-n:] * np.cos(t) + right[:n] * np.sin(t),
        right[n:],
    ])
    return out


def render_swap(head, head_cut_s, mid, mid_in_s, mid_out_s,
                tail, tail_in_s, sr, fade_ms=15.0):
    """head[:head_cut] + mid[mid_in:mid_out] + tail[tail_in:].
    Returns (joined, joint_times_s) with the two splice centre times.
    """
    h = head[:int(head_cut_s * sr)]
    m = mid[int(mid_in_s * sr):int(mid_out_s * sr)]
    t = tail[int(tail_in_s * sr):]
    n = int(sr * fade_ms / 1000.0)
    joined = _xfade_join(_xfade_join(h, m, sr, fade_ms), t, sr, fade_ms)
    joint1 = (len(h) - n / 2) / sr
    joint2 = (len(h) + len(m) - 1.5 * n) / sr
    return joined, (joint1, joint2)


def joint_clip(x, center_s, sr, pre_s=2.0, post_s=2.0):
    """Short audition excerpt around a splice joint."""
    lo = max(0, int((center_s - pre_s) * sr))
    hi = min(len(x), int((center_s + post_s) * sr))
    return x[lo:hi]
