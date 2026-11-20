"""Estimate camera-IMU temporal offset and axis alignment.

Two-stage, translation-aware calibration
----------------------------------------
Stage 1 searches the temporal offset with the *norm* of the angular rate,
which is invariant to axis convention, so no alignment matrix is needed.
Frames with saturated gyro or failed vision are excluded.

Stage 2 classifies 1 s windows by cross-checking vision and gyro:
* whip / blur windows -> saturation, extreme rate or failed feature tracks;
* translation windows -> the homography-derived rotation cannot match the
  gyro-integrated rotation (parallax biases the vision estimate), so the
  per-window amplitude ratio and orientation disagreement reject them;
* static windows carry almost no rotation and are used only for bias.

Only surviving dynamic pure-rotation windows feed the Wahba axis fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rotations import (
    matrix_to_rotvec,
    nearest_rotation,
    rotvec_to_matrix,
)


@dataclass
class WindowReport:
    start_frame: int
    end_frame: int
    start_s: float
    end_s: float
    usable: bool
    reason: str
    inlier_ratio: float
    reproj_error_px: float
    median_flow_px: float
    n_tracks: int
    parallax_spread_px: float
    gyro_saturated: bool
    gyro_norm_dps: float
    vision_rot_deg: float
    gyro_rot_deg: float
    amplitude_ratio: float
    axis_disagree_deg: float


def classify_windows(
    vision_table: dict,
    frame_times: np.ndarray,
    gyro: dict,
    offset: float,
    r_imu_cam: np.ndarray | None = None,
    bias: np.ndarray | None = None,
    window_sec: float = 1.0,
) -> list[WindowReport]:
    fps = 1.0 / np.median(np.diff(frame_times))
    step = max(1, int(round(window_sec * fps / 2)))  # 50% overlap
    win = max(2, int(round(window_sec * fps)))
    reports = []
    n = len(frame_times)
    for start in range(0, n - 1, step):
        end = min(n, start + win)
        reports.append(
            _classify_one(
                start, end, vision_table, frame_times, gyro,
                offset, r_imu_cam, bias,
            )
        )
    return reports


def _classify_one(
    start: int,
    end: int,
    table: dict,
    frame_times: np.ndarray,
    gyro: dict,
    offset: float,
    r_imu_cam: np.ndarray | None,
    bias: np.ndarray | None,
) -> WindowReport:
    sl = slice(start + 1, end)  # frame 0 of each window has no predecessor
    inlier = float(np.median(table["inlier_ratio"][sl]))
    reproj = float(
        np.median(np.nan_to_num(table["reproj_error_px"][sl], nan=99.0))
    )
    flow = float(np.median(table["median_flow_px"][sl]))
    tracks = int(np.median(table["n_tracks"][sl]))
    spread = float(np.median(table.get("parallax_spread_px", np.zeros(len(table)))[sl]))

    t0, t1 = frame_times[start], frame_times[min(end - 1, len(frame_times) - 1)]
    cam_stamps = gyro["times_imu_clock"] + offset
    g_mask = (cam_stamps >= t0) & (cam_stamps <= t1)
    g = gyro["gyro_rad_s"][g_mask] if g_mask.any() else np.zeros((1, 3))
    gyro_norm = float(np.rad2deg(np.linalg.norm(g, axis=1).max()))
    saturated = bool(
        gyro["saturation_mask"][g_mask].any() if g_mask.any() else False
    )
    # Motion blur outlives the whip itself: veto a cooldown window after any
    # saturated sample, even if tracking later recovers. Normal fast pans
    # (well below the gyro full-scale limit) must not trigger the cooldown.
    cooldown_s = 0.9
    extreme = (
        gyro["saturation_mask"].any(axis=1)
    )
    extreme_times = gyro["times_imu_clock"][extreme] + offset
    # Only an extreme event inside the window or in its *preceding* cooldown
    # span counts. A future whip must never veto an earlier healthy window.
    post_whip = bool(
        len(extreme_times)
        and np.any(
            (extreme_times >= max(0.0, t0 - cooldown_s))
            & (extreme_times <= t1)
        )
        and float(extreme_times[extreme_times <= t1].max()) >= t0 - cooldown_s
    )

    # Cumulative window rotation from vision and from aligned gyro.
    vis_delta = _window_vision_rotation(table["rotvec"], start, end)
    gyro_delta = _window_gyro_rotation(
        gyro, frame_times, start, end, offset, r_imu_cam, bias
    )
    vis_deg = float(np.rad2deg(np.linalg.norm(vis_delta)))
    gyro_deg = float(np.rad2deg(np.linalg.norm(gyro_delta)))
    ratio = vis_deg / max(gyro_deg, 1e-6)
    disagree = float(
        np.rad2deg(
            np.linalg.norm(
                _unit(vis_delta) - _unit(gyro_delta)
            )
        )
    ) if vis_deg > 1.0 and gyro_deg > 1.0 else 0.0

    usable, reason = _decision(
        inlier, reproj, tracks, gyro_norm, saturated, post_whip,
        vis_deg, gyro_deg, ratio, disagree,
    )
    return WindowReport(
        start, end, float(t0), float(t1), usable, reason,
        inlier, reproj, flow, tracks, spread, saturated, gyro_norm,
        vis_deg, gyro_deg, ratio, disagree,
    )


def _decision(
    inlier: float,
    reproj: float,
    tracks: int,
    gyro_norm_dps: float,
    saturated: bool,
    post_whip: bool,
    vis_deg: float,
    gyro_deg: float,
    ratio: float,
    disagree_deg: float,
) -> tuple[bool, str]:
    if saturated:
        return False, "gyro saturation during fast whip"
    if post_whip:
        return False, "post-whip cooldown: residual motion blur"
    if gyro_norm_dps > 700.0:
        return False, "angular rate too high for reliable vision"
    if tracks < 25:
        return False, "too few tracked features (motion blur / whip)"
    if inlier < 0.55:
        return False, "low homography inlier ratio (parallax or blur)"
    if reproj > 2.2:
        return False, "homography reprojection error too large"
    if gyro_deg < 1.5:
        return False, "near-static window: bias only, no alignment constraint"
    if not 0.55 <= ratio <= 1.85:
        return False, "vision/gyro rotation amplitude mismatch (translation parallax)"
    if disagree_deg > 25.0:
        return False, "vision/gyro rotation axis mismatch (translation parallax)"
    return True, "pure rotation with reliable tracks"


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-9 else np.zeros(3)


def _window_vision_rotation(
    rotvecs: np.ndarray, start: int, end: int
) -> np.ndarray:
    total = np.eye(3)
    for rv in rotvecs[start + 1 : end]:
        total = rotvec_to_matrix(rv) @ total
    return matrix_to_rotvec(total)


def _window_gyro_rotation(
    gyro: dict,
    frame_times: np.ndarray,
    start: int,
    end: int,
    offset: float,
    r_imu_cam: np.ndarray | None,
    bias: np.ndarray | None,
) -> np.ndarray:
    from .integrate import integrate_interval

    times = gyro["times_imu_clock"] + offset
    samples = gyro["gyro_rad_s"]
    if bias is not None:
        samples = samples - bias
    if r_imu_cam is not None:
        samples = (r_imu_cam.T @ samples.T).T
    t0, t1 = frame_times[start], frame_times[min(end - 1, len(frame_times) - 1)]
    return integrate_interval(times, samples, t0, t1)


def usable_frame_mask(
    reports: list[WindowReport], n_frames: int
) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=bool)
    votes = np.zeros(n_frames, dtype=int)
    for report in reports:
        if report.usable:
            mask[report.start_frame : report.end_frame] = True
            votes[report.start_frame : report.end_frame] += 1
    return mask


def gyro_integrated_rotvec(
    gyro: dict,
    frame_times: np.ndarray,
    offset: float,
    r_imu_cam: np.ndarray | None = None,
) -> np.ndarray:
    """Integrate gyro over each frame interval, optionally rotate to cam axes."""
    from .integrate import integrate_frame_rotvecs

    times = gyro["times_imu_clock"]
    samples = gyro["gyro_rad_s"]
    if r_imu_cam is not None:
        samples = (r_imu_cam.T @ samples.T).T
    return integrate_frame_rotvecs(times + offset, samples, frame_times)


def estimate_time_offset(
    gyro: dict,
    frame_times: np.ndarray,
    vision_rotvec: np.ndarray,
    usable_mask: np.ndarray,
    search_sec: float = 0.4,
    step_ms: float = 5.0,
) -> tuple[float, np.ndarray]:
    """Grid-search the offset maximizing gyro-vision angular-rate correlation."""
    dt_frame = np.median(np.diff(frame_times))
    vision_rate = vision_rotvec / dt_frame
    offsets = np.arange(-search_sec, search_sec + 1e-9, step_ms / 1000.0)
    scores = np.zeros(len(offsets))
    for k, offset in enumerate(offsets):
        gyro_rv = gyro_integrated_rotvec(gyro, frame_times, offset)
        gyro_rate = gyro_rv / dt_frame
        a = vision_rate[usable_mask]
        b = gyro_rate[usable_mask]
        if len(a) < 10:
            scores[k] = -np.inf
            continue
        # Axis-agnostic score: norms plus per-axis best permutation correlation.
        norm_corr = _zero_mean_corr(np.linalg.norm(a, axis=1),
                                    np.linalg.norm(b, axis=1))
        best_axis = max(
            _zero_mean_corr(a[:, j], b[:, j]) for j in range(3)
        )
        scores[k] = 0.7 * norm_corr + 0.3 * best_axis
    best = int(np.argmax(scores))
    return float(offsets[best]), np.column_stack([offsets, scores])


def _zero_mean_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def estimate_bias(
    gyro: dict,
    frame_times: np.ndarray,
    offset: float,
    usable_mask: np.ndarray,
) -> np.ndarray:
    """Estimate IMU bias from the slowest-moving usable samples.

    Bias is a constant offset in IMU axes, so it is read directly from the
    raw gyro samples on near-static frames (no axis alignment needed).
    """
    cam_stamps = gyro["times_imu_clock"] + offset
    rates = gyro["gyro_rad_s"]
    rate_norm = np.linalg.norm(rates, axis=1)
    # The lowest-rate samples correspond to the at-rest periods; bias is the
    # median raw gyro reading over that stationary subset.
    frame_motion = np.zeros(len(frame_times))
    for i in range(1, len(frame_times)):
        sel = (cam_stamps >= frame_times[i - 1]) & (
            cam_stamps <= frame_times[i]
        )
        frame_motion[i] = rate_norm[sel].mean() if sel.any() else np.inf
    rate_threshold = np.percentile(rate_norm, 15)
    static_samples = rate_norm <= max(rate_threshold, np.deg2rad(6.0))
    if static_samples.sum() < 20:
        static_samples = rate_norm <= np.percentile(rate_norm, 10)
    if static_samples.sum() < 10:
        return np.zeros(3)
    return np.median(rates[static_samples], axis=0)


def estimate_axis_alignment(
    gyro: dict,
    frame_times: np.ndarray,
    offset: float,
    bias: np.ndarray,
    vision_rotvec: np.ndarray,
    usable_mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Wahba/SVD solution for R_imu_from_cam on usable frame intervals."""
    times = gyro["times_imu_clock"]
    samples = gyro["gyro_rad_s"] - bias
    cam_times = times + offset

    vision_axes = []
    imu_axes = []
    for i in np.where(usable_mask)[0]:
        if i == 0:
            continue
        t0, t1 = frame_times[i - 1], frame_times[i]
        from .integrate import integrate_interval

        imu_rv = integrate_interval(cam_times, samples, t0, t1)
        vis_rv = vision_rotvec[i]
        if np.linalg.norm(vis_rv) < np.deg2rad(0.15) or np.linalg.norm(imu_rv) < np.deg2rad(
            0.15
        ):
            continue
        vision_axes.append(vis_rv / np.linalg.norm(vis_rv))
        imu_axes.append(imu_rv / np.linalg.norm(imu_rv))

    vision_axes = np.array(vision_axes)
    imu_axes = np.array(imu_axes)
    # Solve R such that imu_axis = R @ vision_axis (R = R_imu_from_cam).
    # With H = V_vis^T V_imu = U S V^T the minimizer of |V_imu - R V_vis|
    # is R = V U^T (with the determinant reflection guard).
    h_mat = vision_axes.T @ imu_axes
    u, _, vt = np.linalg.svd(h_mat)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(u @ vt))
    rotation = nearest_rotation(vt.T @ correction @ u.T)

    residual = np.rad2deg(
        np.mean(
            [
                np.linalg.norm(
                    matrix_to_rotvec(
                        rotvec_to_matrix(imu_axes[k])
                        @ rotvec_to_matrix(rotation @ vision_axes[k]).T
                    )
                )
                for k in range(len(vision_axes))
            ]
        )
    )
    return rotation, float(residual)


def run_calibration(
    gyro: dict,
    frame_times: np.ndarray,
    vision_table: dict,
) -> dict:
    gyro = dict(gyro)
    # Stage 1: temporal offset from axis-agnostic angular-rate norm. Only
    # exclude frames where vision or the saturated gyro carry no signal.
    dt = np.median(np.diff(frame_times))
    rate_norm = np.linalg.norm(vision_table["rotvec"] / dt, axis=1)
    healthy_vision = (
        (vision_table["n_tracks"] >= 25)
        & (vision_table["inlier_ratio"] >= 0.45)
        & (rate_norm < np.deg2rad(600.0))
    )
    offset, score_curve = estimate_time_offset(
        gyro, frame_times, vision_table["rotvec"], healthy_vision
    )

    # Coarse axis fit on all healthy frames, purely to enable window checks.
    coarse_bias = estimate_bias(gyro, frame_times, offset, healthy_vision)
    r_coarse, _ = estimate_axis_alignment(
        gyro, frame_times, offset, coarse_bias,
        vision_table["rotvec"], healthy_vision,
    )

    # Stage 2: data-driven window veto of translation and whip segments.
    reports = classify_windows(
        vision_table, frame_times, gyro,
        offset=offset, r_imu_cam=r_coarse, bias=coarse_bias,
    )
    usable = usable_frame_mask(reports, len(frame_times))
    if usable.sum() < 30:  # fall back rather than emit an empty calibration
        usable = healthy_vision

    bias = estimate_bias(gyro, frame_times, offset, usable)
    r_imu_cam, residual_deg = estimate_axis_alignment(
        gyro, frame_times, offset, bias, vision_table["rotvec"], usable
    )
    return {
        "offset_s": offset,
        "bias_rad_s": bias,
        "r_imu_from_cam": r_imu_cam,
        "axis_residual_deg": residual_deg,
        "usable_mask": usable,
        "windows": reports,
        "score_curve": score_curve,
        "healthy_vision_mask": healthy_vision,
    }
