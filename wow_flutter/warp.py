"""Estimate the tape time-base warp from tracked pilot observations.

The recorded pilot phase differs from its nominal phase by the integral of
the machine speed error.  We turn that into a smooth, monotonic sample-map
``tau(n)``: output sample ``n`` (correct time base) is synthesised from input
sample ``tau(n)`` (warped time base).

Pilot dropouts are handled without feeding garbage into the map: the speed
error is extrapolated from its last reliable value and local trend, and the
extrapolation is hard-bounded so a long dropout cannot run away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .pilot import PilotTrack


@dataclass
class WarpEstimate:
    """Sample-accurate correction map plus diagnostics."""

    input_map: np.ndarray
    sample_rate: float
    speed_error: np.ndarray
    frame_times: np.ndarray
    frame_speed_error: np.ndarray
    confidence: np.ndarray
    reliable: np.ndarray
    extrapolated: np.ndarray


def _extrapolate_bounded(values: np.ndarray, mask: np.ndarray, bound: float) -> np.ndarray:
    """Fill gaps with a bounded linear extrapolation of the speed error.

    ``mask`` marks reliable frames.  Interior gaps are filled by linear
    interpolation between the bracketing reliable frames; edge gaps and any
    gap longer than available context use slope extrapolation clipped to
    ``[-bound, bound]``.  Returns a copy; never mutates ``values``.
    """
    filled = values.copy()
    n = len(values)
    reliable_idx = np.flatnonzero(mask)
    if reliable_idx.size == 0:
        return np.zeros(n), np.ones(n, dtype=bool)

    # Interior gaps: interpolate between reliable anchors.
    gaps = np.flatnonzero(~mask)
    if gaps.size:
        filled[gaps] = np.interp(gaps, reliable_idx, values[reliable_idx])

    extrapolated = ~mask.copy()
    # Edge gaps are true extrapolations; bound them explicitly.
    first = reliable_idx[0]
    last = reliable_idx[-1]
    if first > 0:
        slope = _local_slope(values, reliable_idx, first)
        for i in range(first - 1, -1, -1):
            filled[i] = np.clip(filled[i + 1] - slope, -bound, bound)
    if last < n - 1:
        slope = _local_slope(values, reliable_idx, last)
        for i in range(last + 1, n):
            filled[i] = np.clip(filled[last] + slope * (i - last), -bound, bound)
    return filled, extrapolated


def _local_slope(values: np.ndarray, idx: np.ndarray, anchor: int, span: int = 8) -> float:
    """Robust recent slope (per frame) of the speed error at an anchor."""
    pos = idx.searchsorted(anchor)
    lo = idx[max(0, pos - span)]
    hi = anchor
    if hi == lo:
        return 0.0
    return (values[hi] - values[lo]) / (hi - lo)


def estimate_warp(
    track: PilotTrack,
    fs: float,
    *,
    drift_cutoff_hz: float = 0.5,
    smooth_cutoff_hz: float = 20.0,
    max_speed_error: float = 0.04,
) -> WarpEstimate:
    """Build the bounded correction map from a :class:`PilotTrack`.

    Parameters
    ----------
    track:
        Output of :func:`wow_flutter.pilot.track_pilot`.
    fs:
        Program-audio sample rate.
    drift_cutoff_hz:
        Split between slow drift and short-term wow/flutter.
    smooth_cutoff_hz:
        Low-pass cutoff applied to the frame-rate speed estimate; suppresses
        tracking jitter without erasing flutter.
    max_speed_error:
        Hard bound (fractional) on the speed error, including extrapolation.
    """
    frame_rate = track.frame_rate
    speed_error = (track.freq - track.nominal_freq) / track.nominal_freq

    filled, extrapolated = _extrapolate_bounded(
        speed_error, track.reliable, max_speed_error
    )
    filled = np.clip(filled, -max_speed_error, max_speed_error)

    # Zero-phase smoothing of the frame-rate estimate (split drift/flutter).
    cutoff = min(smooth_cutoff_hz, frame_rate / 2.0 * 0.9)
    sos = signal.butter(4, cutoff / (frame_rate / 2.0), output="sos")
    smoothed = signal.sosfiltfilt(sos, filled)
    smoothed = np.clip(smoothed, -max_speed_error, max_speed_error)

    # Integrate speed error to get the tape time offset on frames.  A positive
    # speed error means the tape ran fast (time compressed), so restoring the
    # correct time base stretches playback: output n reads input n - offset.
    tape_offset = np.concatenate(([0.0], np.cumsum(smoothed[:-1]) / frame_rate))

    # Upsample the offset map to audio rate with a monotone-preserving shape.
    # Extend the last observed offset so the full program length is covered.
    n_out = int(round(track.frame_times[-1] * fs)) + 1
    out_index = np.arange(n_out)
    frame_index = out_index / fs * frame_rate
    offset_samples = np.interp(
        frame_index,
        np.arange(len(tape_offset)),
        tape_offset,
        right=tape_offset[-1],
    ) * fs
    input_map = out_index - offset_samples
    input_map = np.clip(input_map, 0.0, None)
    input_map = np.maximum.accumulate(input_map)

    return WarpEstimate(
        input_map=input_map,
        sample_rate=fs,
        speed_error=smoothed,
        frame_times=track.frame_times,
        frame_speed_error=smoothed,
        confidence=track.confidence,
        reliable=track.reliable,
        extrapolated=extrapolated,
    )
