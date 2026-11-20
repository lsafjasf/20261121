"""Loose gyro/vision fusion for robust absolute orientations.

Gyro integration is smooth and high-rate but accumulates error, especially
across the saturated whip where rotation is physically unobservable to the
IMU. Feature tracking is drift-free while alive but fails outright during
the whip and is biased by parallax during translation.

The clip is partitioned into contiguous *healthy* blocks (pure rotation,
good tracks). Inside a block the fused pose follows vision increments
(drift-free) blended with gyro increments (smooth). Across an unhealthy
block the gyro alone carries the pose; the next healthy block is re-anchored
on whatever pose the gyro reached, so a broken feature track can never inject
a jump. Any attitude error accumulated while the gyro was saturated is an
irreducible physical limit and is quantified separately in the report.
"""

from __future__ import annotations

import numpy as np

from .rotations import (
    matrix_to_quat,
    quat_inverse,
    quat_multiply,
    quat_normalize,
    rotvec_to_matrix,
    slerp,
)


def fuse_orientations(
    gyro_quats: np.ndarray,
    vision_rotvec: np.ndarray,
    healthy_mask: np.ndarray,
    vision_weight: float = 0.85,
) -> np.ndarray:
    """Return fused camera-to-world quaternions (same convention as gyro).

    See module docstring. ``vision_weight`` controls the per-frame blend
    between the gyro increment and the (drift-free) vision increment inside
    healthy blocks.
    """
    fused = np.empty_like(gyro_quats)
    fused[0] = gyro_quats[0]
    for i in range(1, len(gyro_quats)):
        gyro_step = quat_multiply(
            quat_inverse(gyro_quats[i - 1]), gyro_quats[i]
        )
        if not healthy_mask[i]:
            fused[i] = quat_normalize(quat_multiply(fused[i - 1], gyro_step))
            continue

        vision_step = matrix_to_quat(rotvec_to_matrix(vision_rotvec[i]))
        blended_step = slerp(gyro_step, vision_step, vision_weight)
        fused[i] = quat_normalize(
            quat_multiply(fused[i - 1], blended_step)
        )
    return fused
