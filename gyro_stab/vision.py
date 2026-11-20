"""Frame-to-frame rotation estimation from video (feature track + homography).

For each pair of consecutive frames we:
1. detect Shi-Tomasi corners and track them with Lucas-Kanade optical flow;
2. fit a homography with RANSAC, recording inlier ratio and reprojection error;
3. extract the closest rotation matrix (SVD projection of K^-1 H K).

The diagnostics (inlier ratio, median flow, reprojection error) are the raw
signals the calibration stage uses to flag translation and whip segments.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .rotations import matrix_to_rotvec, nearest_rotation, rotvec_to_matrix


@dataclass
class FrameMotion:
    index: int
    rotvec: np.ndarray           # relative rotation frame i-1 -> i, camera axes
    inlier_ratio: float
    reproj_error_px: float
    median_flow_px: float
    n_tracks: int
    homography_condition: float
    parallax_spread_px: float


def estimate_frame_motions(
    video_path: str, camera_matrix: np.ndarray
) -> list[FrameMotion]:
    cap = cv2.VideoCapture(video_path)
    motions: list[FrameMotion] = []
    prev_gray = None
    idx = -1
    k_inv = np.linalg.inv(camera_matrix)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if prev_gray is None:
            motions.append(
                FrameMotion(idx, np.zeros(3), 1.0, 0.0, 0.0, 0, 1.0, 0.0)
            )
            prev_gray = gray
            continue

        motion = _pair_motion(idx, prev_gray, gray, camera_matrix, k_inv)
        motions.append(motion)
        prev_gray = gray

    cap.release()
    return motions


def _pair_motion(
    idx: int,
    prev: np.ndarray,
    curr: np.ndarray,
    camera_matrix: np.ndarray,
    k_inv: np.ndarray,
) -> FrameMotion:
    points = cv2.goodFeaturesToTrack(
        prev,
        maxCorners=800,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7,
    )
    if points is None or len(points) < 12:
        return FrameMotion(idx, np.zeros(3), 0.0, np.inf, 0.0, 0, np.inf, 0.0)

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev,
        curr,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if next_pts is None:
        return FrameMotion(idx, np.zeros(3), 0.0, np.inf, 0.0, 0, np.inf, 0.0)

    good_prev = points[status.flatten() == 1].reshape(-1, 2)
    good_next = next_pts[status.flatten() == 1].reshape(-1, 2)
    n_tracks = len(good_prev)
    if n_tracks < 12:
        return FrameMotion(
            idx, np.zeros(3), 0.0, np.inf, 0.0, n_tracks, np.inf, 0.0
        )

    flow = np.linalg.norm(good_next - good_prev, axis=1)
    median_flow = float(np.median(flow))

    homography, inliers = cv2.findHomography(
        good_prev, good_next, cv2.RANSAC, ransacReprojThreshold=2.5, maxIters=2000
    )
    if homography is None:
        return FrameMotion(
            idx, np.zeros(3), 0.0, np.inf, median_flow, n_tracks, np.inf, 0.0
        )

    inlier_mask = inliers.flatten().astype(bool)
    inlier_ratio = float(inlier_mask.mean())
    reproj = _reprojection_error(homography, good_prev, good_next)
    condition = float(np.linalg.cond(homography))
    parallax_spread = _parallax_spread(
        good_prev, good_next, inlier_mask, median_flow
    )

    # The homography maps previous-frame pixels to the current frame. Its
    # nearest rotation is the world-relative frame rotation; negate it to
    # obtain the standard body-frame increment used by gyro integration.
    similarity = k_inv @ homography @ camera_matrix
    rotation = nearest_rotation(similarity)
    rotvec = -matrix_to_rotvec(rotation)
    return FrameMotion(
        idx, rotvec, inlier_ratio, float(reproj), median_flow, n_tracks,
        condition, parallax_spread,
    )


def _parallax_spread(
    src: np.ndarray,
    dst: np.ndarray,
    inlier_mask: np.ndarray,
    median_flow: float,
) -> float:
    """Translation evidence beyond the single-homography motion model.

    A second homography with a relaxed 6 px RANSAC threshold absorbs ordinary
    tracking noise. Under pure rotation every depth obeys one homography, so
    the relaxed outlier set stays tiny. With lateral translation the
    foreground props systematically violate the background-plane homography:
    both the outlier fraction and their excess residual rise.
    """
    if len(src) < 20 or median_flow < 0.5:
        return 0.0
    relaxed_h, relaxed_inliers = cv2.findHomography(
        src, dst, cv2.RANSAC, ransacReprojThreshold=6.0, maxIters=1000
    )
    if relaxed_h is None:
        return 0.0
    ones = np.ones((len(src), 1))
    proj = (relaxed_h @ np.hstack([src, ones]).T).T
    proj = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-9)
    residual = np.linalg.norm(dst - proj, axis=1)
    relaxed_mask = relaxed_inliers.flatten().astype(bool)
    bg = residual[relaxed_mask]
    fg = residual[~relaxed_mask]
    if len(bg) < 12 or len(fg) < 4:
        return 0.0
    fg_fraction = len(fg) / len(src)
    excess = max(np.median(fg) - np.median(bg), 0.0)
    # Isolated noise gives a few outliers with small weight; a translated
    # foreground layer gives a large fraction of coherent high residuals.
    return float(excess * min(fg_fraction / 0.15, 1.0))


def _reprojection_error(
    homography: np.ndarray, src: np.ndarray, dst: np.ndarray
) -> float:
    ones = np.ones((len(src), 1))
    proj = (homography @ np.hstack([src, ones]).T).T
    proj = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-9)
    return float(np.median(np.linalg.norm(proj - dst, axis=1)))


def cumulative_orientations(motions: list[FrameMotion]) -> np.ndarray:
    """Chain body-frame increments into absolute camera-to-world quaternions."""
    from .rotations import matrix_to_quat, quat_multiply, quat_normalize

    quats = [np.array([1.0, 0.0, 0.0, 0.0])]
    for motion in motions[1:]:
        delta = matrix_to_quat(rotvec_to_matrix(motion.rotvec))
        quats.append(quat_normalize(quat_multiply(quats[-1], delta)))
    return np.array(quats)


def motions_to_table(motions: list[FrameMotion]) -> dict:
    """Pack per-frame diagnostics into numpy arrays for the calibrator."""
    return {
        "index": np.array([m.index for m in motions]),
        "rotvec": np.array([m.rotvec for m in motions]),
        "inlier_ratio": np.array([m.inlier_ratio for m in motions]),
        "reproj_error_px": np.array([m.reproj_error_px for m in motions]),
        "median_flow_px": np.array([m.median_flow_px for m in motions]),
        "n_tracks": np.array([m.n_tracks for m in motions]),
        "parallax_spread_px": np.array(
            [m.parallax_spread_px for m in motions]
        ),
    }
