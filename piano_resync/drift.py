from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from statistics import median
from typing import Sequence

from .config import SyncConfig


@dataclass(frozen=True)
class AffineTransform:
    """Map a source (MIDI) time to the target (audio) timeline."""

    rate: float
    offset: float

    def __post_init__(self) -> None:
        if self.rate <= 0:
            raise ValueError("rate must be positive")

    def apply(self, time: float) -> float:
        return self.rate * time + self.offset

    def to_dict(self) -> dict[str, float | str]:
        return {
            "rate": self.rate,
            "offset_seconds": self.offset,
            "drift_ppm": (self.rate - 1.0) * 1_000_000,
            "formula": "audio_time = rate * midi_time + offset_seconds",
        }


@dataclass(frozen=True)
class Fit:
    transform: AffineTransform
    residuals: tuple[float, ...]
    inliers: tuple[bool, ...]
    threshold: float


def _weighted_affine(
    pairs: Sequence[tuple[float, float, float]]
) -> AffineTransform:
    weight_sum = fsum(weight for _, _, weight in pairs)
    if weight_sum <= 0:
        return AffineTransform(1.0, 0.0)
    mean_x = fsum(x * weight for x, _, weight in pairs) / weight_sum
    mean_y = fsum(y * weight for _, y, weight in pairs) / weight_sum
    variance = fsum(weight * (x - mean_x) ** 2 for x, _, weight in pairs)
    if variance <= 1e-15:
        return AffineTransform(1.0, mean_y - mean_x)
    covariance = fsum(
        weight * (x - mean_x) * (y - mean_y) for x, y, weight in pairs
    )
    rate = covariance / variance
    rate = min(1.5, max(0.5, rate))
    return AffineTransform(rate, mean_y - rate * mean_x)


def robust_fit(
    source_times: Sequence[float],
    target_times: Sequence[float],
    initial: AffineTransform,
    config: SyncConfig,
    *,
    weights: Sequence[float] | None = None,
) -> Fit:
    """Fit a drift line and identify residual outliers.

    The initial slope is estimated with Theil-Sen (sampled for long files), and
    the final line is a weighted least-squares fit on residual inliers.
    """

    count = min(len(source_times), len(target_times))
    if count == 0:
        return Fit(initial, (), (), 0.0)
    if count == 1:
        residual = initial.apply(source_times[0]) - target_times[0]
        return Fit(initial, (residual,), (True,), 0.0)

    if weights is None:
        weights = [1.0] * count
    else:
        weights = list(weights[:count])

    span = max(source_times[:count]) - min(source_times[:count])
    if span <= max(config.timing_tolerance * 2.0, 1e-6):
        offset = median(
            target_times[index] - initial.rate * source_times[index]
            for index in range(count)
        )
        transform = AffineTransform(initial.rate, offset)
    else:
        sample_indices = list(range(count))
        if count > 240:
            stride = max(1, count // 240)
            sample_indices = sample_indices[::stride][:240]
        slopes = []
        for left_pos, left_index in enumerate(sample_indices):
            for right_index in sample_indices[left_pos + 1:]:
                dx = source_times[right_index] - source_times[left_index]
                if dx > config.timing_tolerance * 2.0:
                    slopes.append(
                        (target_times[right_index] - target_times[left_index]) / dx
                    )
        if slopes:
            rate = min(1.5, max(0.5, median(slopes)))
        else:
            rate = initial.rate
        intercepts = [
            target_times[index] - rate * source_times[index]
            for index in range(count)
        ]
        transform = AffineTransform(rate, median(intercepts))

    residuals = tuple(
        transform.apply(source_times[index]) - target_times[index]
        for index in range(count)
    )
    center = median(residuals)
    mad = median(abs(residual - center) for residual in residuals)
    robust_sigma = 1.4826 * mad
    threshold = max(
        config.timing_tolerance * 1.5, config.outlier_sigma * robust_sigma
    )
    inliers = tuple(abs(residual - center) <= threshold for residual in residuals)
    inlier_indices = [index for index, is_inlier in enumerate(inliers) if is_inlier]
    if len(inlier_indices) >= 2:
        transform = _weighted_affine(
            [
                (source_times[index], target_times[index], weights[index])
                for index in inlier_indices
            ]
        )
        residuals = tuple(
            transform.apply(source_times[index]) - target_times[index]
            for index in range(count)
        )
        center = median(residuals)
        threshold = max(
            config.timing_tolerance * 1.5,
            config.outlier_sigma
            * 1.4826
            * median(abs(value - center) for value in residuals),
        )
        inliers = tuple(abs(residual - center) <= threshold for residual in residuals)
    return Fit(transform, residuals, inliers, threshold)
