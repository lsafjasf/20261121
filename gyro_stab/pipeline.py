"""Command-line entry point for the full offline gyro-stabilization pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .calibrate import run_calibration
from .evaluate import write_report
from .plotting import plot_all
from .simulator import simulate_dataset
from .stabilize import stabilize_pipeline
from .vision import estimate_frame_motions, motions_to_table


def load_gyro_csv(path: str | Path) -> dict:
    raw = np.genfromtxt(path, delimiter=",", names=True)
    saturated = raw["saturated"].astype(bool)
    return {
        "times_imu_clock": raw["timestamp_s"],
        "gyro_rad_s": np.column_stack(
            [raw["gx_radps"], raw["gy_radps"], raw["gz_radps"]]
        ),
        "saturation_mask": np.column_stack([saturated] * 3),
        "rate_hz": 1.0 / np.median(np.diff(raw["timestamp_s"])),
    }


def run_pipeline(out_dir: str | Path, seed: int = 11, sigma_s: float = 0.35) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] simulating video + IMU dataset ...")
    sim = simulate_dataset(out_dir, seed=seed)
    ground_truth = json.loads(Path(sim["ground_truth"]).read_text())
    camera_matrix = np.array(ground_truth["camera_matrix"])
    frame_times = np.array([f["time"] for f in ground_truth["frames"]])
    labels = np.array([f["label"] for f in ground_truth["frames"]])
    truth_quats = np.array([f["quat_cw"] for f in ground_truth["frames"]])

    print("[2/5] tracking features and estimating frame rotations ...")
    motions = estimate_frame_motions(sim["video"], camera_matrix)
    vision_table = motions_to_table(motions)
    np.savez(
        out_dir / "vision_diagnostics.npz",
        **vision_table,
    )

    print("[3/5] calibrating time offset and axis alignment ...")
    gyro = load_gyro_csv(sim["gyro"])
    calibration = run_calibration(gyro, frame_times, vision_table)
    _persist_calibration(out_dir / "calibration.json", calibration, gyro)

    print("[4/5] gyro integration, smoothing and crop rendering ...")
    result = stabilize_pipeline(
        video_path=sim["video"],
        gyro=gyro,
        frame_times=frame_times,
        camera_matrix=camera_matrix,
        calibration=calibration,
        out_dir=out_dir,
        vision_table=vision_table,
        smoothing_sigma_s=sigma_s,
    )

    print("[5/5] evaluating against ground truth and plotting ...")
    report = write_report(
        out_dir,
        calibration,
        ground_truth,
        frame_times,
        labels,
        truth_quats,
        result.orientations,
        result.smoothed,
        result.crop_scale,
        result.trajectory_csv,
    )
    plot_all(
        out_dir,
        frame_times,
        vision_table,
        calibration,
        truth_quats,
        result.orientations,
        result.smoothed,
        result.crop_scale,
    )
    return report


def _persist_calibration(path: Path, calibration: dict, gyro: dict) -> None:
    payload = {
        "offset_s": calibration["offset_s"],
        "bias_rad_s": calibration["bias_rad_s"].tolist(),
        "r_imu_from_cam": calibration["r_imu_from_cam"].tolist(),
        "axis_residual_deg": calibration["axis_residual_deg"],
        "windows": [
            {
                "start_s": w.start_s,
                "end_s": w.end_s,
                "usable": w.usable,
                "reason": w.reason,
                "inlier_ratio": w.inlier_ratio,
                "reproj_error_px": w.reproj_error_px,
                "median_flow_px": w.median_flow_px,
                "n_tracks": w.n_tracks,
                "parallax_spread_px": w.parallax_spread_px,
                "gyro_saturated": w.gyro_saturated,
                "gyro_norm_dps": w.gyro_norm_dps,
                "vision_rot_deg": w.vision_rot_deg,
                "gyro_rot_deg": w.gyro_rot_deg,
                "amplitude_ratio": w.amplitude_ratio,
                "axis_disagree_deg": w.axis_disagree_deg,
            }
            for w in calibration["windows"]
        ],
    }
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gyro-assisted offline video stabilization on synthetic data"
    )
    parser.add_argument("--out", default="outputs/run", help="output directory")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--smoothing-sigma",
        type=float,
        default=0.35,
        help="SO(3) Gaussian smoothing strength in seconds",
    )
    args = parser.parse_args()
    report = run_pipeline(args.out, seed=args.seed, sigma_s=args.smoothing_sigma)
    print("\n=== summary ===")
    print(json.dumps(report, indent=2)[:4000])


if __name__ == "__main__":
    main()
