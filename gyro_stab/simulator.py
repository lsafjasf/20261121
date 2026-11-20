"""End-to-end synthetic dataset generation (video + gyro + ground truth)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .imu import ImuConfig, default_axis_misalignment, generate_imu
from .motion import DEFAULT_SEGMENTS, generate_poses
from .scene import CameraIntrinsics, build_default_scene, render_frame


def make_intrinsics(width: int = 960, height: int = 540) -> CameraIntrinsics:
    fx = fy = width * 1.25
    return CameraIntrinsics(width, height, fx, fy, width / 2.0, height / 2.0)


def _apply_motion_blur(
    frame: np.ndarray, omega_deg_s: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Approximate exposure blur from the dominant angular speed."""
    exposure = 1.0 / 60.0
    blur_deg = np.linalg.norm(omega_deg_s) * exposure
    if blur_deg < 0.05:
        return frame
    kernel_size = int(np.clip(blur_deg * 1.6, 3, 31))
    if kernel_size % 2 == 0:
        kernel_size += 1
    angle = np.degrees(np.arctan2(omega_deg_s[1], omega_deg_s[0]))
    kernel = np.zeros((kernel_size, kernel_size), np.float32)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    kernel = _rotate_kernel(kernel, angle)
    blurred = cv2.filter2D(frame, -1, kernel, borderType=cv2.BORDER_REFLECT)
    # Blur also reduces contrast, as real motion blur does.
    mix = np.clip(blur_deg / 8.0, 0.0, 0.85)
    out = (1.0 - mix) * frame.astype(np.float32) + mix * blurred
    if rng.random() < 0.15:  # occasional read noise spike
        out += rng.normal(0, 4, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def _rotate_kernel(kernel: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = kernel.shape
    center = (w / 2.0 - 0.5, h / 2.0 - 0.5)
    rot = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(kernel, rot, (w, h))


def simulate_dataset(out_dir: str | Path, seed: int = 11) -> dict:
    """Render the full synthetic clip and persist all inputs to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + 5)

    intr = make_intrinsics()
    texture, plane_z, props = build_default_scene(intr, seed=7)
    poses = generate_poses(seed=seed)
    r_imu_cam = default_axis_misalignment()
    imu = generate_imu(
        poses["frame_times"],
        poses["quats_cw"],
        ImuConfig(seed=seed + 12, trajectory_seed=seed),
        r_imu_cam,
    )

    video_path = out_dir / "input.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(video_path), fourcc, poses["fps"], (intr.width, intr.height), isColor=False
    )
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        video_path = out_dir / "input.avi"
        writer = cv2.VideoWriter(
            str(video_path), fourcc, poses["fps"], (intr.width, intr.height), False
        )

    for i, t in enumerate(poses["frame_times"]):
        frame = render_frame(
            intr,
            poses["rotations_cw"][i],
            poses["translations_cw"][i],
            texture,
            plane_z,
            props,
            rng,
        )
        omega = _nearest_gyro(imu, t)
        frame = _apply_motion_blur(frame, np.rad2deg(omega), rng)
        writer.write(frame)
    writer.release()

    _write_imu_csv(out_dir / "gyro.csv", imu)
    ground_truth = _write_ground_truth(out_dir, poses, imu, intr)
    _write_segments_json(out_dir / "segments_truth.json")
    return {
        "video": str(video_path),
        "gyro": str(out_dir / "gyro.csv"),
        "ground_truth": ground_truth,
        "intrinsics": intr,
        "segments": DEFAULT_SEGMENTS,
    }


def _nearest_gyro(imu: dict, camera_time: float) -> np.ndarray:
    target = camera_time - imu["true_offset_s"]
    idx = int(np.argmin(np.abs(imu["times_imu_clock"] - target)))
    return imu["gyro_rad_s"][idx]


def _write_imu_csv(path: Path, imu: dict) -> None:
    lines = ["timestamp_s,gx_radps,gy_radps,gz_radps,saturated"]
    for t, g, sat in zip(
        imu["times_imu_clock"], imu["gyro_rad_s"], imu["saturation_mask"].any(axis=1)
    ):
        lines.append(f"{t:.6f},{g[0]:.6f},{g[1]:.6f},{g[2]:.6f},{int(sat)}")
    path.write_text("\n".join(lines) + "\n")


def _write_ground_truth(out_dir: Path, poses: dict, imu: dict, intr) -> str:
    path = out_dir / "ground_truth.json"
    data = {
        "fps": poses["fps"],
        "time_offset_s": imu["true_offset_s"],
        "r_imu_from_cam": imu["true_r_imu_cam"].tolist(),
        "true_bias_rad_s": imu["bias_rad_s"].tolist(),
        "camera_matrix": intr.matrix.tolist(),
        "width": intr.width,
        "height": intr.height,
        "frames": [
            {
                "index": i,
                "time": float(t),
                "quat_cw": poses["quats_cw"][i].tolist(),
                "translation_cw": poses["translations_cw"][i].tolist(),
                "label": str(poses["labels"][i]),
            }
            for i, t in enumerate(poses["frame_times"])
        ],
    }
    path.write_text(json.dumps(data, indent=1))
    return str(path)


def _write_segments_json(path: Path) -> None:
    data = [
        {
            "start_s": s.start,
            "end_s": s.end,
            "kind": s.kind,
            "valid_for_calibration": s.valid_for_calibration,
            "reason": s.reason,
        }
        for s in DEFAULT_SEGMENTS
    ]
    path.write_text(json.dumps(data, indent=1))
