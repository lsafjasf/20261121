from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ResponseParams:
    alpha_rise: np.ndarray
    alpha_fall: np.ndarray

    @property
    def tau_rise(self) -> np.ndarray:
        return -1.0 / np.log1p(-np.clip(self.alpha_rise, 1e-5, 1 - 1e-5))

    @property
    def tau_fall(self) -> np.ndarray:
        return -1.0 / np.log1p(-np.clip(self.alpha_fall, 1e-5, 1 - 1e-5))


def simulate_response(
    source: np.ndarray,
    alpha_rise: np.ndarray,
    alpha_fall: np.ndarray,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    """Apply an asymmetric, switched first-order panel response.

    ``source`` is a float RGB sequence in [0, 1] with shape (T, H, W, 3).
    On every frame the panel moves toward the stimulus using either the
    rising or falling coefficient, whichever matches the signed error.
    """
    if source.ndim != 4 or source.shape[-1] != 3:
        raise ValueError("source must have shape (T, H, W, 3)")
    frame_count = source.shape[0]
    if initial is None:
        state = np.zeros(source.shape[1:], dtype=np.float32)
    else:
        state = np.array(initial, dtype=np.float32, copy=True)

    alpha_r = np.asarray(alpha_rise, dtype=np.float32).reshape(1, 1, 3)
    alpha_f = np.asarray(alpha_fall, dtype=np.float32).reshape(1, 1, 3)
    output = np.empty_like(source, dtype=np.float32)
    for index in range(frame_count):
        target = source[index]
        error = target - state
        alpha = np.where(error >= 0.0, alpha_r, alpha_f)
        state = state + alpha * error
        output[index] = state
    return np.clip(output, 0.0, 1.0)


def directional_parameters(
    base_alpha: np.ndarray,
    bias: tuple[float, float, float],
) -> np.ndarray:
    """Convert a temporal bias into a valid asymmetry coefficient."""
    base_alpha = np.asarray(base_alpha, dtype=np.float64)
    tau = -1.0 / np.log1p(-np.clip(base_alpha, 1e-6, 1 - 1e-6))
    biased_tau = tau * np.asarray(bias, dtype=np.float64)
    return np.clip(1.0 - np.exp(-1.0 / biased_tau), 1e-4, 0.999)
