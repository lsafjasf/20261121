"""Synthetic gyroscope generation from the ground-truth camera trajectory.

The IMU is a separate device from the camera: it samples at its own rate,
uses a rotated axis convention (e.g. a phone mounted in a gimbal frame),
carries bias + noise, and its clock is shifted relative to the camera clock.
Fast whips exceed the gyro full-scale range and are clipped, exactly like a
real MPU when saturation is configured.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rotations import (
    matrix_to_rotvec,
    quat_to_matrix,
)


@dataclass
class ImuConfig:
    rate_hz: float = 400.0
    time_offset_s: float = 0.120        # t_camera = t_imu_clock + offset
    bias_dps: tuple[float, float, float] = (-1.2, 0.8, 0.5)
    noise_std_dps: float = 0.35
    saturation_dps: float = 500.0       # ~8.73 rad/s; the whip exceeds it
    seed: int = 23
    trajectory_seed: int = 11


def default_axis_misalignment() -> np.ndarray:
    """Fixed rotation from camera frame to IMU device frame (radians)."""
    rotvec = np.deg2rad([4.0, -6.0, 88.0])  # near-90 deg yaw + small tilts
    from .rotations import rotvec_to_matrix

    return rotvec_to_matrix(rotvec)


def _angular_velocity_camera(
    times: np.ndarray, quats_cw: np.ndarray
) -> np.ndarray:
    """Standard body-frame angular velocity from camera-to-world quats.

    For a camera-to-world rotation ``R`` the body angular velocity is
    ``log(R_lo^T R_hi) / dt``. The vision pipeline instead estimates the
    inverse (world moving relative to the body), so the sign flip that maps
    gyro output onto vision rotvecs is applied in :func:`generate_imu`.
    """
    n = len(times)
    omega = np.zeros((n, 3))
    for i in range(n):
        lo = max(0, i - 1)
        hi = min(n - 1, i + 1)
        r_lo = quat_to_matrix(quats_cw[lo])
        r_hi = quat_to_matrix(quats_cw[hi])
        d_rot = r_lo.T @ r_hi
        rotvec = matrix_to_rotvec(d_rot)
        dt = max(times[hi] - times[lo], 1e-6)
        omega[i] = rotvec / dt
    return omega


def generate_imu(
    frame_times: np.ndarray,
    quats_cw: np.ndarray,
    config: ImuConfig,
    r_imu_cam: np.ndarray,
) -> dict:
    """Generate gyro samples in IMU-clock time and IMU device axes."""
    rng = np.random.default_rng(config.seed)
    duration = frame_times[-1] + 1.0 / 30.0
    n_imu = int(duration * config.rate_hz) + 1

    # Sample on the IMU clock; convert each stamp to camera time via the
    # (hidden) true offset so the correct trajectory instant is evaluated.
    imu_clock_times = np.arange(n_imu) / config.rate_hz
    camera_time_at_sample = imu_clock_times + config.time_offset_s

    # Evaluate the analytic trajectory at IMU instants: frame-to-frame
    # SLERP would erase the intra-frame handheld jitter.
    from .motion import orientation_at

    interp_quats = np.array(
        [
            orientation_at(t, seed=config.trajectory_seed)
            for t in camera_time_at_sample
        ]
    )
    omega_cam = _angular_velocity_camera(camera_time_at_sample, interp_quats)
    # Standard gyro output: body angular velocity expressed in IMU axes.
    omega_imu = (r_imu_cam @ omega_cam.T).T

    bias = np.deg2rad(np.array(config.bias_dps))
    noise = np.deg2rad(config.noise_std_dps) * rng.standard_normal(omega_imu.shape)
    samples = omega_imu + bias + noise

    sat = np.deg2rad(config.saturation_dps)
    clipped = np.abs(samples) > sat
    samples = np.clip(samples, -sat, sat)

    return {
        "times_imu_clock": imu_clock_times,
        "gyro_rad_s": samples,
        "saturation_mask": clipped,
        "rate_hz": config.rate_hz,
        "true_offset_s": config.time_offset_s,
        "true_r_imu_cam": r_imu_cam,
        "bias_rad_s": bias,
    }
