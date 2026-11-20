"""SO(3) rotation utilities used across calibration and stabilization.

Conventions
-----------
* Rotation vectors use the Rodrigues form ``angle * unit_axis`` (radians).
* Quaternions are scalar-first ``[w, x, y, z]`` and always kept unit length.
* All matrices map points *from body/camera frame to world frame* unless
  documented otherwise at the call site.
"""

from __future__ import annotations

import numpy as np


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Rodrigues: 3-vector -> 3x3 rotation matrix."""
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.eye(3)
    axis = rotvec / angle
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return (
        np.eye(3)
        + np.sin(angle) * k
        + (1.0 - np.cos(angle)) * (k @ k)
    )


def matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    """Inverse Rodrigues, robust for both small and near-pi angles."""
    matrix = np.asarray(matrix, dtype=np.float64)
    _, axis, angle = _cv_svd_rvec(matrix)
    return axis * angle


def _cv_svd_rvec(matrix: np.ndarray):
    cos_angle = (np.trace(matrix) - 1.0) / 2.0
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    angle = np.arccos(cos_angle)
    if angle < 1e-9:
        return angle, np.array([0.0, 0.0, 1.0]), 0.0
    if np.pi - angle < 1e-9:
        # Recover axis from the symmetric part.
        eigvals, eigvecs = np.linalg.eigh(matrix + matrix.T)
        axis = eigvecs[:, np.argmax(eigvals)]
        return angle, axis, angle
    axis = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ]
    ) / (2.0 * np.sin(angle))
    return angle, axis, angle


def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = np.trace(matrix)
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = np.array(
            [
                0.25 * s,
                (matrix[2, 1] - matrix[1, 2]) / s,
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[1, 0] - matrix[0, 1]) / s,
            ]
        )
    else:
        i = int(np.argmax(np.diag(matrix)))
        if i == 0:
            s = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            q = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / s,
                    0.25 * s,
                    (matrix[0, 1] + matrix[1, 0]) / s,
                    (matrix[0, 2] + matrix[2, 0]) / s,
                ]
            )
        elif i == 1:
            s = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            q = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / s,
                    (matrix[0, 1] + matrix[1, 0]) / s,
                    0.25 * s,
                    (matrix[1, 2] + matrix[2, 1]) / s,
                ]
            )
        else:
            s = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            q = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / s,
                    (matrix[0, 2] + matrix[2, 0]) / s,
                    (matrix[1, 2] + matrix[2, 1]) / s,
                    0.25 * s,
                ]
            )
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_inverse(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]]) / np.dot(q, q)


def quat_normalize(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    return quat / np.linalg.norm(quat)


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation at scalar fraction ``t`` in [0, 1]."""
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return quat_normalize(q0 + t * (q1 - q0))
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    return (
        np.sin((1.0 - t) * theta) / sin_theta * q0
        + np.sin(t * theta) / sin_theta * q1
    )


def so3_log(matrix: np.ndarray) -> np.ndarray:
    """SO(3) logarithm: rotation matrix -> rotation vector."""
    return matrix_to_rotvec(matrix)


def so3_exp(rotvec: np.ndarray) -> np.ndarray:
    """SO(3) exponential: rotation vector -> rotation matrix."""
    return rotvec_to_matrix(rotvec)


def nearest_rotation(matrix: np.ndarray) -> np.ndarray:
    """Project an arbitrary 3x3 matrix onto SO(3) via SVD."""
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def quat_series_smooth(quats: np.ndarray, sigma_sec: float, dt: float) -> np.ndarray:
    """Gaussian smoothing of an orientation trajectory on SO(3).

    Each output quaternion is the intrinsic weighted mean of neighbours,
    computed in the tangent space around the central orientation so the
    result stays on the manifold even for large inter-frame rotations.
    """
    n = len(quats)
    if n == 0 or sigma_sec <= 0:
        return quats.copy()
    radius = max(1, int(round(3.0 * sigma_sec / dt)))
    kernel_t = np.arange(-radius, radius + 1) * dt
    kernel = np.exp(-0.5 * (kernel_t / sigma_sec) ** 2)
    kernel /= kernel.sum()

    smoothed = np.empty_like(quats)
    for i in range(n):
        center = quats[i]
        tangent = np.zeros(3)
        for k, weight in zip(kernel_t, kernel):
            j = int(np.clip(i + round(k / dt), 0, n - 1))
            delta = quat_to_matrix(
                quat_multiply(quat_inverse(center), quats[j])
            )
            tangent += weight * so3_log(delta)
        smoothed[i] = quat_normalize(
            quat_multiply(center, matrix_to_quat(so3_exp(tangent)))
        )
    return smoothed


def roll_from_matrix(matrix: np.ndarray) -> float:
    """Image-plane roll (rotation about the camera viewing axis), radians.

    Camera looks down +Z, image X right, image Y down. The projected world-up
    direction in the image gives the horizon angle.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    world_up = np.array([0.0, 1.0, 0.0])
    up_cam = matrix.T @ world_up
    return float(np.arctan2(-up_cam[0], up_cam[1]))
