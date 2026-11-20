from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import LabConfig
from .model import ResponseParams


@dataclass(slots=True)
class RestorationResult:
    frames: np.ndarray
    raw: np.ndarray
    gain: np.ndarray
    unstable: np.ndarray
    saturated: np.ndarray
    direction: np.ndarray
    unstable_frequency: np.ndarray
    saturated_frequency: np.ndarray
    valid_count: np.ndarray


def restore_sequence(
    captured: np.ndarray,
    params: ResponseParams,
    config: LabConfig,
) -> RestorationResult:
    """Invert the switched first-order response while recording uncertainty.

    The temporal branch is selected from the observed signed change.  This is
    exact for a switched first-order panel and becomes ill-conditioned when
    the active response coefficient is small.  Out-of-range source estimates
    are clamped for display but retained in the diagnostic masks.
    """
    if captured.ndim != 4 or captured.shape[-1] != 3:
        raise ValueError("captured must have shape (T, H, W, 3)")
    frame_count, height, width, _ = captured.shape
    restored = np.empty_like(captured, dtype=np.float32)
    raw = np.empty_like(captured, dtype=np.float32)
    gain = np.empty_like(captured, dtype=np.float32)
    unstable = np.zeros((frame_count, height, width), dtype=bool)
    saturated = np.zeros((frame_count, height, width), dtype=bool)
    direction = np.zeros((frame_count, height, width), dtype=np.int8)

    restored[0] = captured[0]
    raw[0] = captured[0]
    gain[0] = 1.0
    for frame_index in range(1, frame_count):
        previous = captured[frame_index - 1]
        observed = captured[frame_index]
        delta = observed - previous
        rising = delta >= 0.0
        alpha = np.where(
            rising,
            params.alpha_rise.reshape(1, 1, 3),
            params.alpha_fall.reshape(1, 1, 3),
        ).astype(np.float32)
        alpha = np.maximum(alpha, 1e-4)
        local_gain = 1.0 / alpha
        estimate = previous + delta / alpha

        gain[frame_index] = local_gain
        raw[frame_index] = estimate
        direction[frame_index] = np.where(rising.any(axis=-1), 1, -1)
        meaningful_change = np.abs(delta) > 0.025
        unstable[frame_index] = (
            local_gain.max(axis=-1) > config.instability_gain
        ) & meaningful_change.any(axis=-1)
        out_of_range = (estimate < -0.012) | (estimate > 1.012)
        saturated[frame_index] = out_of_range.any(axis=-1)
        restored[frame_index] = np.clip(estimate, 0.0, 1.0)

    valid_count = np.full((height, width), frame_count, dtype=np.float32)
    unstable_frequency = unstable.sum(axis=0) / valid_count
    saturated_frequency = saturated.sum(axis=0) / valid_count
    return RestorationResult(
        frames=restored,
        raw=raw,
        gain=gain,
        unstable=unstable,
        saturated=saturated,
        direction=direction,
        unstable_frequency=unstable_frequency,
        saturated_frequency=saturated_frequency,
        valid_count=valid_count,
    )
