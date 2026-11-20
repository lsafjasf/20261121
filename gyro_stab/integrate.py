"""SO(3)-correct gyroscope integration utilities.

Rotating through ``w1`` then ``w2`` is *not* ``w1 + w2`` in general:
rotation vectors do not add. All integration in this project composes
per-sample rotation matrices instead, so fast jitter does not lose
amplitude the way midpoint vector summation does.
"""

from __future__ import annotations

import numpy as np

from .rotations import matrix_to_rotvec, rotvec_to_matrix


def integrate_interval(
    stamps: np.ndarray,
    rates_cam: np.ndarray,
    t_start: float,
    t_end: float,
) -> np.ndarray:
    """Compose gyro samples inside ``[t_start, t_end]`` into one rotvec."""
    sel = np.where((stamps >= t_start) & (stamps <= t_end))[0]
    if len(sel) < 2:
        lo = np.searchsorted(stamps, t_start)
        hi = np.searchsorted(stamps, t_end)
        sel = np.arange(max(0, lo - 1), min(len(stamps), hi + 1))
        sel = sel[(stamps[sel] >= t_start - 0.1) & (stamps[sel] <= t_end + 0.1)]
        if len(sel) < 2:
            return np.zeros(3)
    total = np.eye(3)
    for k in range(1, len(sel)):
        dt = stamps[sel[k]] - stamps[sel[k - 1]]
        mid = 0.5 * (rates_cam[sel[k]] + rates_cam[sel[k - 1]])
        total = rotvec_to_matrix(mid * dt) @ total
    return matrix_to_rotvec(total)


def integrate_frame_rotvecs(
    stamps: np.ndarray,
    rates_cam: np.ndarray,
    frame_times: np.ndarray,
) -> np.ndarray:
    """Per-frame relative rotation vectors, one row per frame (row 0 = 0)."""
    out = np.zeros((len(frame_times), 3))
    for i in range(1, len(frame_times)):
        out[i] = integrate_interval(
            stamps, rates_cam, frame_times[i - 1], frame_times[i]
        )
    return out


def integrate_absolute_quaternions(
    stamps: np.ndarray,
    rates_cam: np.ndarray,
    frame_times: np.ndarray,
) -> np.ndarray:
    """Chain per-frame rotations into camera-to-world quaternions."""
    from .rotations import matrix_to_quat, quat_multiply, quat_normalize

    relative = integrate_frame_rotvecs(stamps, rates_cam, frame_times)
    quats = [np.array([1.0, 0.0, 0.0, 0.0])]
    for rv in relative[1:]:
        delta = matrix_to_quat(rotvec_to_matrix(rv))
        quats.append(quat_normalize(quat_multiply(quats[-1], delta)))
    return np.array(quats)
