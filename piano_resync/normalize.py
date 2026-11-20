from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence


def _quantile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def robust_unit(values: Sequence[float]) -> tuple[list[float], float]:
    """Normalize values to 0..1 using 5th/95th percentiles.

    Returns the normalized values and their robust spread. A zero spread means
    that dynamics carry no useful matching information.
    """

    if not values:
        return [], 0.0
    ordered = sorted(values)
    low = _quantile(ordered, 0.05)
    high = _quantile(ordered, 0.95)
    span = high - low
    if span <= 1e-12:
        return [0.0] * len(values), 0.0
    return ([min(1.0, max(0.0, (value - low) / span)) for value in values], span)


def index_window(
    times: Sequence[float], low: float, high: float
) -> tuple[int, int]:
    """Return the half-open index range of times in [low, high]."""

    return bisect_left(times, low), bisect_left(times, high)

