"""Diagnostic figures rendered with the headless matplotlib backend."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .motion import DEFAULT_SEGMENTS
from .rotations import quat_inverse, quat_to_matrix, roll_from_matrix


def _segment_spans(ax) -> None:
    colors = {True: "#2ca02c22", False: "#d6272822"}
    for seg in DEFAULT_SEGMENTS:
        ax.axvspan(
            seg.start,
            seg.end,
            color=colors[seg.valid_for_calibration],
            lw=0,
        )


def plot_all(
    out_dir: str | Path,
    frame_times: np.ndarray,
    vision_table: dict,
    calibration: dict,
    truth_quats: np.ndarray,
    raw_quats: np.ndarray,
    smooth_quats: np.ndarray,
    crop_scale: np.ndarray,
) -> None:
    out_dir = Path(out_dir)
    _plot_offset_curve(out_dir, calibration)
    _roll(out_dir, frame_times, truth_quats, raw_quats, smooth_quats)
    _plot_crop(out_dir, frame_times, crop_scale)
    _plot_diagnostics(out_dir, frame_times, vision_table, calibration)


def _plot_offset_curve(out_dir: Path, calibration: dict) -> None:
    curve = calibration["score_curve"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(curve[:, 0] * 1000, curve[:, 1], "-o", ms=3)
    ax.axvline(
        calibration["offset_s"] * 1000,
        color="r",
        ls="--",
        label=f"estimated {calibration['offset_s']*1000:.0f} ms",
    )
    ax.set_xlabel("time offset (ms)")
    ax.set_ylabel("gyro-vision correlation score")
    ax.set_title("Temporal offset search")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_offset_search.png", dpi=130)
    plt.close(fig)


def _roll(
    out_dir: Path,
    frame_times: np.ndarray,
    truth_quats: np.ndarray,
    raw_quats: np.ndarray,
    smooth_quats: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    _segment_spans(ax)
    series = [
        (truth_quats, "ground truth", "k"),
        (raw_quats, "gyro integrated (raw)", "tab:orange"),
        (smooth_quats, "stabilized target", "tab:blue"),
    ]
    for quats, label, color in series:
        roll = np.rad2deg(
            [roll_from_matrix(quat_to_matrix(quat_inverse(q))) for q in quats]
        )
        ax.plot(frame_times, roll, label=label, color=color, lw=1.4)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("horizon roll (deg)")
    ax.set_title("Horizon roll: green spans are calibration-valid")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_horizon_roll.png", dpi=130)
    plt.close(fig)


def _plot_crop(out_dir: Path, frame_times: np.ndarray, crop_scale: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    _segment_spans(ax)
    ax.plot(frame_times, crop_scale, label="smooth crop scale", color="tab:purple")
    ax.plot(
        frame_times,
        crop_scale**2,
        label="area retained",
        color="tab:green",
        ls="--",
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("fraction of original FOV")
    ax.set_title("Crop trajectory and field-of-view loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_crop_trajectory.png", dpi=130)
    plt.close(fig)


def _plot_diagnostics(
    out_dir: Path,
    frame_times: np.ndarray,
    vision_table: dict,
    calibration: dict,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    for ax in axes:
        _segment_spans(ax)
    axes[0].plot(frame_times, vision_table["inlier_ratio"], color="tab:blue")
    axes[0].set_ylabel("RANSAC inlier")
    axes[1].plot(frame_times, vision_table["median_flow_px"], color="tab:orange")
    axes[1].set_ylabel("median flow (px)")
    axes[2].plot(
        frame_times,
        vision_table["parallax_spread_px"],
        color="tab:red",
    )
    axes[2].set_ylabel("parallax spread (px)")
    axes[3].plot(frame_times, vision_table["n_tracks"], color="tab:green")
    axes[3].set_ylabel("track count")
    usable = calibration["usable_mask"].astype(float)
    usable[usable == 0] = np.nan
    axes[3].plot(
        frame_times,
        usable * vision_table["n_tracks"].max() * 0.9,
        "k|",
        label="accepted frames",
    )
    axes[3].legend(loc="lower right")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Vision diagnostics and calibration acceptance")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_diagnostics.png", dpi=130)
    plt.close(fig)
