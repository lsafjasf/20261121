"""Freehand strokes -> soft time-frequency mask.

A stroke is a polyline in (time, frequency) space with a brush radius.
Rasterisation handles:

* strokes spanning many STFT time blocks (per-frame interpolation of the
  polyline centre frequency);
* overlapping strokes, whose weights combine as a probabilistic union
  ``w = 1 - prod(1 - w_i)`` so overlaps never exceed 1;
* head/tail feathering in both time and frequency so mask edges are smooth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Stroke:
    """One freehand gesture on the spectrogram."""
    times: np.ndarray      # seconds, sorted ascending
    freqs: np.ndarray      # Hz, same length as ``times``
    radius_hz: float = 120.0
    depth: float = 1.0     # attenuation depth of this stroke, 0..1

    @classmethod
    def from_points(cls, points: list[tuple[float, float]],
                    radius_hz: float = 120.0,
                    depth: float = 1.0) -> "Stroke":
        pts = sorted(points, key=lambda p: p[0])
        times = np.array([p[0] for p in pts], dtype=np.float64)
        freqs = np.array([p[1] for p in pts], dtype=np.float64)
        return cls(times=times, freqs=freqs,
                   radius_hz=radius_hz, depth=depth)


def _raised_cosine(dist: np.ndarray, radius: float,
                   core_frac: float = 0.5) -> np.ndarray:
    """Flat-top profile: 1 inside the core, cosine taper to ``radius``."""
    core = radius * core_frac
    out = np.zeros_like(dist)
    out[dist <= core] = 1.0
    taper = (dist > core) & (dist <= radius)
    span = max(radius - core, 1e-12)
    out[taper] = 0.5 * (1.0 + np.cos(np.pi * (dist[taper] - core) / span))
    return out


def rasterize_stroke(stroke: Stroke, n_frames: int, n_bins: int,
                     frame_sec: float, bin_hz: float,
                     feather_frames: int = 2) -> np.ndarray:
    """Rasterise one stroke to a weight matrix (n_frames, n_bins) in [0, 1]."""
    weights = np.zeros((n_frames, n_bins), dtype=np.float64)
    if len(stroke.times) == 0:
        return weights

    frame_times = (np.arange(n_frames) + 0.5) * frame_sec
    t0, t1 = stroke.times[0], stroke.times[-1]
    if len(stroke.times) == 1:  # single click: give it one frame of extent
        t0 -= frame_sec * 0.5
        t1 += frame_sec * 0.5
    margin = feather_frames * frame_sec
    active = np.where((frame_times >= t0 - margin)
                      & (frame_times <= t1 + margin))[0]
    if active.size == 0:
        return weights

    # centre frequency per active frame, interpolated across time blocks
    if len(stroke.times) == 1:
        centre = np.full(active.size, stroke.freqs[0])
    else:
        centre = np.interp(frame_times[active], stroke.times, stroke.freqs)

    # head/tail time feather: ramps weight 0 -> 1 -> 0 at stroke ends
    t_rel_start = (frame_times[active] - (t0 - margin)) / max(margin, 1e-12)
    t_rel_end = ((t1 + margin) - frame_times[active]) / max(margin, 1e-12)
    t_weight = np.clip(np.minimum(t_rel_start, t_rel_end), 0.0, 1.0)
    t_weight = 0.5 * (1.0 - np.cos(np.pi * t_weight))

    radius_bins = stroke.radius_hz / bin_hz
    bin_index = np.arange(n_bins, dtype=np.float64)
    for row, frame_idx in enumerate(active):
        dist = np.abs(bin_index - centre[row] / bin_hz)
        weights[frame_idx] = t_weight[row] * _raised_cosine(dist, radius_bins)
    return weights


def combine_strokes(strokes: list[Stroke], n_frames: int, n_bins: int,
                    frame_sec: float, bin_hz: float,
                    feather_frames: int = 2) -> np.ndarray:
    """Union of all stroke weights; overlaps saturate at 1."""
    if not strokes:
        return np.zeros((n_frames, n_bins), dtype=np.float64)
    complement = np.ones((n_frames, n_bins), dtype=np.float64)
    for stroke in strokes:
        w = rasterize_stroke(stroke, n_frames, n_bins,
                             frame_sec, bin_hz, feather_frames)
        complement *= 1.0 - np.clip(w, 0.0, 1.0) * stroke.depth
    return 1.0 - complement
