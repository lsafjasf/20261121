"""Place clip boundaries at natural low-energy points.

Each merged group gets context padding (``context_pre`` / ``context_post``).
The nominal padded boundary is then shifted inside a small search window to
the frame whose RMS energy is lowest, which lands on natural pauses/breaths.

To guarantee no word is clipped, every labelled speech interval is a
protected range (plus a small guard); low-energy search is forbidden inside
protected ranges.  This also protects quiet, weakly pronounced final
syllables that would otherwise look like attractive cut points.
"""

from __future__ import annotations

import math

from .merge import UtteranceGroup
from .model import BuildConfig

GUARD_SECONDS = 0.01


def rms(samples: list[float], start: int, end: int) -> float:
    start = max(0, start)
    end = min(len(samples), end)
    if end <= start:
        return 0.0
    total = 0.0
    for index in range(start, end):
        total += samples[index] * samples[index]
    return math.sqrt(total / (end - start))


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    merged: list[tuple[int, int]] = []
    for low, high in sorted(ranges):
        if merged and low <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], high))
        else:
            merged.append((low, high))
    return merged


def _protected_ranges(
    groups: list[UtteranceGroup], sample_rate: int, length: int
) -> list[tuple[int, int]]:
    guard = max(1, round(GUARD_SECONDS * sample_rate))
    ranges: list[tuple[int, int]] = []
    for group in groups:
        for start, end in group.intervals:
            low = max(0, round(start * sample_rate) - guard)
            high = min(length, round(end * sample_rate) + guard)
            ranges.append((low, high))
    return _merge_ranges(ranges)


def _is_protected(
    start: int, end: int, ranges: list[tuple[int, int]]
) -> bool:
    for low, high in ranges:
        if start < high and end > low:
            return True
        if low >= end:
            break
    return False


def _lowest_energy_cut(
    samples: list[float],
    nominal: int,
    search: int,
    window: int,
    protected: list[tuple[int, int]],
) -> int:
    length = len(samples)
    nominal_frame = (
        min(nominal, length - window),
        min(nominal, length - window) + window,
    )
    if _is_protected(nominal_frame[0], nominal_frame[1], protected):
        # The nominal padded boundary already sits inside labelled speech
        # (very short context or unusually placed labels); keep it as a
        # last-resort fallback instead of searching for a deeper cut.
        return max(0, min(length, nominal))

    low_bound = max(0, nominal - search)
    high_bound = min(length - window, nominal + search)
    step = max(1, window // 2)

    best = nominal
    best_energy: float | None = None
    candidate = low_bound
    while candidate <= high_bound:
        frame_end = candidate + window
        if frame_end <= length and not _is_protected(
            candidate, frame_end, protected
        ):
            energy = rms(samples, candidate, frame_end)
            distance = abs(candidate - nominal)
            if (
                best_energy is None
                or energy < best_energy
                or (energy == best_energy and distance < abs(best - nominal))
            ):
                best_energy = energy
                best = candidate
        candidate += step
    if best_energy is None:
        # Every candidate frame was protected: never cut inside speech.
        return max(0, min(length, nominal))
    return max(0, min(length, best))


def resolve_cut_points(
    samples: list[float],
    sample_rate: int,
    groups: list[UtteranceGroup],
    config: BuildConfig,
) -> list[tuple[int, int]]:
    """Return inclusive-exclusive ``(start_sample, end_sample)`` per group."""

    config.validate()
    length = len(samples)
    window = max(1, round(config.energy_window * sample_rate))
    search = round(config.energy_search * sample_rate)
    protected = _protected_ranges(groups, sample_rate, length)

    nominal_starts = [
        max(0, round((group.start - config.context_pre) * sample_rate))
        for group in groups
    ]
    nominal_ends = [
        min(length, round((group.end + config.context_post) * sample_rate))
        for group in groups
    ]

    cuts: list[tuple[int, int]] = []
    for index, group in enumerate(groups):
        start_cut = _lowest_energy_cut(
            samples, nominal_starts[index], search, window, protected
        )
        end_cut = _lowest_energy_cut(
            samples, nominal_ends[index], search, window, protected
        )
        if end_cut <= start_cut:
            start_cut, end_cut = nominal_starts[index], nominal_ends[index]
        cuts.append((start_cut, end_cut))
    return cuts
