"""Gyro-driven stabilization and adaptive crop trajectory.

Pipeline per frame
------------------
1. Integrate calibrated gyro (offset + axis alignment + bias removed) into
   absolute world-to-camera orientations.
2. Smooth orientations on SO(3) with a Gaussian kernel: high-frequency
   handheld jitter is removed while deliberate pans/tilts are retained.
3. The per-frame virtual-camera rotation is the smoothed orientation.
4. Warp each frame, then shrink the crop window so no undefined border is
   visible; smooth the crop scale over time to avoid breathing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .rotations import (
    quat_inverse,
    quat_series_smooth,
    quat_to_matrix,
)


@dataclass
class StabilizeResult:
    stabilized_video: str
    cropped_video: str
    trajectory_csv: str
    orientations: np.ndarray
    smoothed: np.ndarray
    crop_scale: np.ndarray


def integrate_gyro_orientations(
    gyro: dict,
    frame_times: np.ndarray,
    offset: float,
    r_imu_cam: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Return camera-to-world quaternion at each frame from gyro only.

    SO(3)-correct matrix composition avoids the amplitude loss of summing
    rotation vectors during handheld jitter.
    """
    from .integrate import integrate_absolute_quaternions

    times = gyro["times_imu_clock"] + offset
    samples_cam = (r_imu_cam.T @ (gyro["gyro_rad_s"] - bias).T).T
    return integrate_absolute_quaternions(times, samples_cam, frame_times)


def stabilization_rotations(
    orientations: np.ndarray, smoothed: np.ndarray
) -> list[np.ndarray]:
    """Warp matrix mapping the original frame into the stabilized view.

    A point with world-to-camera orientation Q is re-rendered as if captured
    with smoothed orientation S: image warp uses R = S Q^-1 applied to rays.
    """
    warps = []
    for q, s in zip(orientations, smoothed):
        r_rel = quat_to_matrix(s) @ quat_to_matrix(quat_inverse(q))
        warps.append(r_rel)
    return warps


def _homography_from_rotation(
    rotation: np.ndarray, camera_matrix: np.ndarray
) -> np.ndarray:
    return camera_matrix @ rotation @ np.linalg.inv(camera_matrix)


def _required_crop_scale(
    homography: np.ndarray, width: int, height: int
) -> float:
    """Smallest uniform scale keeping warped corners inside the frame."""
    corners = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )
    ones = np.ones((4, 1), dtype=np.float32)
    warped = (homography @ np.hstack([corners, ones]).T).T
    warped = warped[:, :2] / warped[:, 2:3]
    min_x, min_y = warped.min(axis=0)
    max_x, max_y = warped.max(axis=0)
    need_x = width / max(max_x - min_x, 1e-6)
    need_y = height / max(max_y - min_y, 1e-6)
    return float(min(need_x, need_y))


def _smooth_scale(scale: np.ndarray, fps: float, sigma_sec: float = 0.4) -> np.ndarray:
    radius = max(1, int(round(3 * sigma_sec * fps)))
    kernel_t = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (kernel_t / max(sigma_sec * fps, 1e-6)) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(scale, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def render_stabilized(
    video_path: str,
    camera_matrix: np.ndarray,
    frame_times: np.ndarray,
    orientations: np.ndarray,
    smoothed: np.ndarray,
    out_dir: str,
    smoothing_sigma_s: float = 0.35,
) -> StabilizeResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    rel_rotations = stabilization_rotations(orientations, smoothed)
    homographies = [
        _homography_from_rotation(r, camera_matrix) for r in rel_rotations
    ]
    raw_scale = np.array(
        [_required_crop_scale(h, width, height) for h in homographies]
    )
    # Smooth the raw geometric limit, then clamp each frame to its own
    # geometric boundary (with a small safety margin) so a neighboring frame
    # can never force an over-aggressive zoom-out.
    smoothed_scale = _smooth_scale(np.minimum(raw_scale, 1.0), fps)
    crop_scale = np.minimum(smoothed_scale, raw_scale * 0.97)

    stab_path = out_dir / "stabilized.mp4"
    crop_path = out_dir / "stabilized_cropped.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_stab = cv2.VideoWriter(str(stab_path), fourcc, fps, (width, height), False)
    writer_crop = cv2.VideoWriter(str(crop_path), fourcc, fps, (width, height), False)

    rows = [
        "frame_index,time_s,raw_crop_scale,smooth_crop_scale,"
        "warp_rx_deg,warp_ry_deg,warp_rz_deg,area_loss_pct"
    ]
    idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        warped = cv2.warpPerspective(
            frame,
            homographies[idx],
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        writer_stab.write(warped)

        scale = crop_scale[idx]
        crop_h = _crop_homography(homographies[idx], scale, width, height)
        cropped = cv2.warpPerspective(
            frame,
            crop_h,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        writer_crop.write(cropped)

        from .rotations import matrix_to_rotvec

        rv = np.rad2deg(matrix_to_rotvec(rel_rotations[idx]))
        rows.append(
            f"{idx},{frame_times[idx]:.4f},{raw_scale[idx]:.4f},"
            f"{scale:.4f},{rv[0]:.3f},{rv[1]:.3f},{rv[2]:.3f},"
            f"{(1 - scale * scale) * 100:.2f}"
        )

    cap.release()
    writer_stab.release()
    writer_crop.release()
    csv_path = out_dir / "stabilization_trajectory.csv"
    csv_path.write_text("\n".join(rows) + "\n")
    return StabilizeResult(
        str(stab_path), str(crop_path), str(csv_path),
        orientations, smoothed, crop_scale,
    )


def _crop_homography(
    warp: np.ndarray, scale: float, width: int, height: int
) -> np.ndarray:
    """Compose the stabilization warp with a centered zoom of ``scale``."""
    center = np.array([width / 2.0, height / 2.0])
    zoom = np.array(
        [
            [1.0 / scale, 0.0, center[0] * (1 - 1.0 / scale)],
            [0.0, 1.0 / scale, center[1] * (1 - 1.0 / scale)],
            [0.0, 0.0, 1.0],
        ]
    )
    return zoom @ warp


def stabilize_pipeline(
    video_path: str,
    gyro: dict,
    frame_times: np.ndarray,
    camera_matrix: np.ndarray,
    calibration: dict,
    out_dir: str,
    vision_table: dict | None = None,
    smoothing_sigma_s: float = 0.35,
) -> StabilizeResult:
    orientations = integrate_gyro_orientations(
        gyro,
        frame_times,
        calibration["offset_s"],
        calibration["r_imu_from_cam"],
        calibration["bias_rad_s"],
    )
    if vision_table is not None:
        from .fusion import fuse_orientations

        orientations = fuse_orientations(
            orientations,
            vision_table["rotvec"],
            calibration["usable_mask"],
        )
    fps = 1.0 / np.median(np.diff(frame_times))
    smoothed = quat_series_smooth(orientations, smoothing_sigma_s, 1.0 / fps)
    return render_stabilized(
        video_path, camera_matrix, frame_times, orientations, smoothed, out_dir
    )
