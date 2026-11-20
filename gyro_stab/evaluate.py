"""Quantitative evaluation against simulation ground truth.

Metrics
-------
* calibration: time-offset error (ms), axis alignment angular error (deg),
  gyro bias error (deg/s);
* horizon: roll amplitude before/after stabilization, plus pan/tilt
  residuals, measured against the known true trajectory;
* field of view: per-frame crop scale, area loss and the worst-case loss;
* segments: which ground-truth segments were accepted/rejected by the
  data-driven window classifier and the reason for each rejection.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .motion import DEFAULT_SEGMENTS
from .rotations import (
    matrix_to_rotvec,
    quat_inverse,
    quat_multiply,
    quat_to_matrix,
    roll_from_matrix,
)


def calibration_metrics(calibration: dict, ground_truth: dict) -> dict:
    offset_err_ms = abs(
        calibration["offset_s"] - ground_truth["time_offset_s"]
    ) * 1000.0
    r_est = calibration["r_imu_from_cam"]
    r_true = np.array(ground_truth["r_imu_from_cam"])
    axis_err_deg = np.rad2deg(
        np.abs(matrix_to_rotvec(r_est @ r_true.T)).max()
    )
    metrics = {
        "time_offset_error_ms": float(offset_err_ms),
        "axis_alignment_error_deg": float(axis_err_deg),
        "axis_residual_fit_deg": calibration["axis_residual_deg"],
        "usable_frame_count": int(calibration["usable_mask"].sum()),
        "usable_frame_pct": float(calibration["usable_mask"].mean() * 100),
    }
    if "true_bias_rad_s" in ground_truth:
        bias_err = np.rad2deg(
            np.linalg.norm(
                calibration["bias_rad_s"] - np.array(ground_truth["true_bias_rad_s"])
            )
        )
        metrics["bias_error_dps"] = float(bias_err)
    return metrics


def _roll_series(quats_w2c: np.ndarray) -> np.ndarray:
    return np.array(
        [roll_from_matrix(quat_to_matrix(q)) for q in quats_w2c]
    )


def horizon_metrics(
    frame_times: np.ndarray,
    truth_quats: np.ndarray,
    raw_quats: np.ndarray,
    smooth_quats: np.ndarray,
    healthy_mask: np.ndarray | None = None,
    whip_mask: np.ndarray | None = None,
) -> dict:
    """Roll/pan/tilt amplitude of truth vs raw vs stabilized trajectories."""
    # All stored quaternions are camera-to-world; the roll helper and the
    # residual algebra below work in the world-to-camera convention.
    truth_quats = np.array([quat_inverse(q) for q in truth_quats])
    raw_quats = np.array([quat_inverse(q) for q in raw_quats])
    smooth_quats = np.array([quat_inverse(q) for q in smooth_quats])
    truth_roll = np.rad2deg(_roll_series(truth_quats))
    raw_roll = np.rad2deg(_roll_series(raw_quats))
    smooth_roll = np.rad2deg(_roll_series(smooth_quats))

    # Compare orientations in the same reference frame by aligning the first
    # quaternion of each series to the truth series.
    raw_aligned = _align_reference(truth_quats, raw_quats)
    smooth_aligned = _align_reference(truth_quats, smooth_quats)

    def residual_euler_deg(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
        out = np.zeros((len(query), 3))
        for i in range(len(query)):
            rel = quat_multiply(
                quat_inverse(query[i]), reference[i]
            )
            out[i] = np.rad2deg(matrix_to_rotvec(quat_to_matrix(rel)))
        return out

    raw_res = residual_euler_deg(truth_quats, raw_aligned)
    smooth_res = residual_euler_deg(truth_quats, smooth_aligned)

    # Deliberate low-frequency horizon motion of the true trajectory.
    from scipy.ndimage import gaussian_filter1d

    fps = 1.0 / np.median(np.diff(frame_times))
    truth_roll_smooth = gaussian_filter1d(truth_roll, sigma=1.0 * fps)
    metrics = {
        "truth_roll_peak_deg": float(np.max(np.abs(truth_roll_smooth))),
        "raw_roll_peak_deg": float(np.max(np.abs(raw_roll))),
        "stabilized_roll_peak_deg": float(np.max(np.abs(smooth_roll))),
        "raw_orientation_residual_rms_deg": float(
            np.sqrt((raw_res**2).mean())
        ),
        "stabilized_orientation_residual_rms_deg": float(
            np.sqrt((smooth_res**2).mean())
        ),
        "stabilized_jitter_rms_deg": _jitter_rms(smooth_aligned, fps),
        "raw_jitter_rms_deg": _jitter_rms(raw_aligned, fps),
    }
    if healthy_mask is not None:
        healthy = np.asarray(healthy_mask, dtype=bool)
        pre_whip = frame_times < 7.0
        healthy_pre = healthy & pre_whip
        metrics["healthy"] = {
            "frame_count": int(healthy.sum()),
            "raw_orientation_residual_rms_deg": float(
                np.sqrt((raw_res[healthy] ** 2).mean())
            ),
            "stabilized_orientation_residual_rms_deg": float(
                np.sqrt((smooth_res[healthy] ** 2).mean())
            ),
            "stabilized_jitter_rms_deg": _jitter_rms(
                smooth_aligned[healthy], fps
            ),
            "raw_jitter_rms_deg": _jitter_rms(raw_aligned[healthy], fps),
            "stabilized_roll_peak_deg": float(
                np.max(np.abs(smooth_roll[healthy]))
            ),
        }
        metrics["healthy_pre_whip"] = {
            "frame_count": int(healthy_pre.sum()),
            "raw_orientation_residual_rms_deg": float(
                np.sqrt((raw_res[healthy_pre] ** 2).mean())
            ),
            "stabilized_orientation_residual_rms_deg": float(
                np.sqrt((smooth_res[healthy_pre] ** 2).mean())
            ),
            "raw_jitter_rms_deg": _jitter_rms(
                raw_aligned[healthy_pre], fps
            ),
            "stabilized_jitter_rms_deg": _jitter_rms(
                smooth_aligned[healthy_pre], fps
            ),
        }
    if whip_mask is not None and np.asarray(whip_mask).any():
        whip = np.asarray(whip_mask, dtype=bool)
        metrics["whip_region"] = {
            "frame_count": int(whip.sum()),
            "raw_orientation_residual_rms_deg": float(
                np.sqrt((raw_res[whip] ** 2).mean())
            ),
            "stabilized_orientation_residual_rms_deg": float(
                np.sqrt((smooth_res[whip] ** 2).mean())
            ),
            "note": "gyro saturation makes absolute yaw physically unrecoverable",
        }
    return metrics


def _align_reference(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    offset = quat_multiply(reference[0], quat_inverse(query[0]))
    return np.array(
        [quat_multiply(offset, q) for q in query]
    )


def _jitter_rms(quats: np.ndarray, fps: float) -> float:
    """RMS of the high-pass filtered orientation (the handheld jitter)."""
    from scipy.ndimage import gaussian_filter1d

    euler = np.array(
        [np.rad2deg(matrix_to_rotvec(quat_to_matrix(q))) for q in quats]
    )
    low = gaussian_filter1d(euler, sigma=0.6 * fps, axis=0)
    high = euler - low
    return float(np.sqrt((high**2).sum(axis=1).mean()))


def fov_metrics(crop_scale: np.ndarray) -> dict:
    area_keep = crop_scale**2
    return {
        "mean_crop_scale": float(crop_scale.mean()),
        "worst_crop_scale": float(crop_scale.min()),
        "mean_area_loss_pct": float((1 - area_keep).mean() * 100),
        "worst_area_loss_pct": float((1 - area_keep.min()) * 100),
        "p95_area_loss_pct": float((1 - np.percentile(area_keep, 5)) * 100),
    }


def fov_metrics_masked(
    crop_scale: np.ndarray, mask: np.ndarray
) -> dict:
    """FOV loss restricted to calibration-valid (non-whip) frames."""
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() < 5:
        return {}
    keep = crop_scale[mask] ** 2
    return {
        "healthy_mean_crop_scale": float(crop_scale[mask].mean()),
        "healthy_mean_area_loss_pct": float((1 - keep).mean() * 100),
        "healthy_worst_area_loss_pct": float((1 - keep.min()) * 100),
    }


def segment_acceptance(
    frame_times: np.ndarray,
    labels: np.ndarray,
    usable_mask: np.ndarray,
    windows: list,
) -> list[dict]:
    rows = []
    for seg in DEFAULT_SEGMENTS:
        in_seg = (frame_times >= seg.start) & (frame_times < seg.end)
        if not in_seg.any():
            continue
        accepted = usable_mask[in_seg].mean()
        # Windows whose center lies in this segment; only rejected ones
        # explain why frames inside the segment were excluded.
        window_rows = [
            w
            for w in windows
            if seg.start <= 0.5 * (w.start_s + w.end_s) < seg.end
        ]
        rejected = [w for w in window_rows if not w.usable]
        reasons = sorted({w.reason for w in rejected})
        rows.append(
            {
                "segment": seg.kind,
                "start_s": seg.start,
                "end_s": seg.end,
                "ground_truth_usable": seg.valid_for_calibration,
                "calibration_role": _segment_role(seg, window_rows),
                "frames_in_segment": int(in_seg.sum()),
                "pct_frames_accepted": float(accepted * 100),
                "rejection_reasons": reasons,
                "truth_reason": seg.reason,
            }
        )
    return rows


def _segment_role(seg, window_rows: list) -> str:
    if not seg.valid_for_calibration:
        return "excluded"
    if seg.kind == "static_rest":
        return "bias_only"
    return "axis_and_offset"


def write_report(
    out_dir: str | Path,
    calibration: dict,
    ground_truth: dict,
    frame_times: np.ndarray,
    labels: np.ndarray,
    truth_quats: np.ndarray,
    raw_quats: np.ndarray,
    smooth_quats: np.ndarray,
    crop_scale: np.ndarray,
    trajectory_csv: str,
) -> dict:
    out_dir = Path(out_dir)
    cal = calibration_metrics(calibration, ground_truth)
    whip_mask = _whip_region_mask(frame_times)
    horizon = horizon_metrics(
        frame_times,
        truth_quats,
        raw_quats,
        smooth_quats,
        healthy_mask=calibration["usable_mask"],
        whip_mask=whip_mask,
    )
    fov = fov_metrics(crop_scale)
    fov.update(
        fov_metrics_masked(crop_scale, calibration["usable_mask"])
    )
    segments = segment_acceptance(
        frame_times, labels, calibration["usable_mask"], calibration["windows"]
    )
    report = {
        "calibration": cal,
        "horizon": horizon,
        "field_of_view": fov,
        "segment_acceptance": segments,
        "trajectory_csv": trajectory_csv,
        "whip_region_mask_frames": np.where(whip_mask)[0].tolist(),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    _write_markdown(out_dir / "report.md", report, calibration)
    return report


def _whip_region_mask(frame_times: np.ndarray) -> np.ndarray:
    """Frames inside the whip plus its post-whip settle/cooldown region."""
    mask = np.zeros(len(frame_times), dtype=bool)
    for seg in DEFAULT_SEGMENTS:
        if seg.kind in ("fast_whip", "whip_settle"):
            mask |= (frame_times >= seg.start) & (
                frame_times < seg.end + 0.6
            )
    return mask


def _write_markdown(path: Path, report: dict, calibration: dict) -> None:
    cal, horizon, fov = report["calibration"], report["horizon"], report["field_of_view"]
    lines = [
        "# 陀螺仪辅助视频防抖 — 评估报告",
        "",
        "## 标定精度（对照仿真真值）",
        "",
        f"- 时间偏移误差: **{cal['time_offset_error_ms']:.1f} ms**",
        f"- 轴对齐角度误差: **{cal['axis_alignment_error_deg']:.2f}°**",
        f"- 轴拟合残差: {cal['axis_residual_fit_deg']:.2f}°",
        f"- 可用帧占比: {cal['usable_frame_pct']:.1f}% ({cal['usable_frame_count']} 帧)",
        "",
        "## 水平线 / 姿态稳定",
        "",
        f"- 真值水平线旋转峰值: {horizon['truth_roll_peak_deg']:.2f}°",
        f"- 原始画面 roll 峰值: {horizon['raw_roll_peak_deg']:.2f}°",
        f"- 防抖后 roll 峰值: **{horizon['stabilized_roll_peak_deg']:.2f}°**",
        f"- 原始姿态残差 RMS: {horizon['raw_orientation_residual_rms_deg']:.2f}°",
        f"- 防抖后姿态残差 RMS: **{horizon['stabilized_orientation_residual_rms_deg']:.2f}°**",
        f"- 抖动 RMS: {horizon['raw_jitter_rms_deg']:.2f}° -> "
        f"**{horizon['stabilized_jitter_rms_deg']:.2f}°**",
        "",
        "### 仅健康（纯旋转、跟踪可靠）帧",
        "",
        f"- 抖动 RMS（甩动前健康段）: {horizon['healthy_pre_whip']['raw_jitter_rms_deg']:.2f}° -> "
        f"**{horizon['healthy_pre_whip']['stabilized_jitter_rms_deg']:.2f}°**",
        f"- 姿态残差 RMS（甩动前健康段）: {horizon['healthy_pre_whip']['raw_orientation_residual_rms_deg']:.2f}° -> "
        f"**{horizon['healthy_pre_whip']['stabilized_orientation_residual_rms_deg']:.2f}°**",
        f"- 防抖后 roll 峰值: **{horizon['healthy']['stabilized_roll_peak_deg']:.2f}°**",
        "",
        "### 甩动区域（陀螺饱和）",
        "",
        f"- 姿态残差 RMS: {horizon['whip_region']['stabilized_orientation_residual_rms_deg']:.2f}°",
        "- 陀螺满量程削顶，绝对偏航在物理上不可由 IMU/视觉恢复，仅做相对平滑",
        "",
        "## 视野损失（裁切轨迹）",
        "",
        f"- 健康帧平均面积损失: **{fov['healthy_mean_area_loss_pct']:.1f}%**，"
        f"健康帧最差: {fov['healthy_worst_area_loss_pct']:.1f}%",
        f"- 平均裁切系数: {fov['mean_crop_scale']:.3f}",
        f"- 最差裁切系数: {fov['worst_crop_scale']:.3f}",
        f"- 平均面积损失: **{fov['mean_area_loss_pct']:.1f}%**",
        f"- P95 面积损失: {fov['p95_area_loss_pct']:.1f}%",
        f"- 最差面积损失: **{fov['worst_area_loss_pct']:.1f}%**（甩动段）",
        "",
        "## 分段标定适用性判定",
        "",
        "| 分段 | 时间(s) | 标定角色 | 接受率 | 判定依据 |",
        "|---|---|---|---|---|",
    ]
    role_label = {
        "axis_and_offset": "轴向/偏移标定",
        "bias_only": "仅零偏估计",
        "excluded": "不可用",
    }
    for row in report["segment_acceptance"]:
        reasons = "; ".join(row["rejection_reasons"]) or row["truth_reason"]
        lines.append(
            f"| {row['segment']} | {row['start_s']:.1f}-{row['end_s']:.1f} | "
            f"{role_label.get(row['calibration_role'], row['calibration_role'])} | "
            f"{row['pct_frames_accepted']:.0f}% | {reasons} |"
        )
    lines += [
        "",
        "## 标定参数",
        "",
        f"- 估计时间偏移: {calibration['offset_s']*1000:.1f} ms",
        f"- 估计陀螺零偏 (deg/s): "
        f"{np.rad2deg(calibration['bias_rad_s']).round(3).tolist()}",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")
