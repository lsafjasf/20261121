"""Ground-truth camera trajectory with labeled motion segments.

Timeline of the default 12 s clip (30 fps -> 360 frames)
--------------------------------------------------------
seg 0  0.0-2.5 s  static + handheld jitter              valid for calibration
seg 1  2.5-5.0 s  smooth pan (pure rotation)            valid
seg 2  5.0-7.0 s  lateral translation (parallax)        INVALID
seg 3  7.0-7.3 s  fast whip, gyro saturates             INVALID
seg 4  7.3-8.2 s  whip settle (blur residue)            INVALID
seg 5  8.2-10.5 s smooth tilt + roll (pure rotation)    valid
seg 6  10.5-12.0 s static + jitter                      valid
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rotations import (
    matrix_to_quat,
    quat_multiply,
    quat_normalize,
    quat_to_matrix,
    rotvec_to_matrix,
    slerp,
)


@dataclass
class Segment:
    start: float
    end: float
    kind: str
    valid_for_calibration: bool
    reason: str

    def contains(self, t: float) -> bool:
        return self.start <= t < self.end


DEFAULT_SEGMENTS = [
    Segment(0.0, 1.2, "static_rest", True, "camera at rest: gyro bias observable"),
    Segment(1.2, 2.5, "static_jitter", True, "pure rotation, trackable"),
    Segment(2.5, 5.0, "smooth_pan", True, "pure rotation, trackable"),
    Segment(5.0, 7.0, "lateral_translation", False, "parallax: rotation estimate biased"),
    Segment(7.0, 7.3, "fast_whip", False, "angular rate beyond vision/gyro tracking limits"),
    Segment(7.3, 8.2, "whip_settle", False, "motion blur residue from the whip"),
    Segment(8.2, 10.5, "smooth_tilt_roll", True, "pure rotation, trackable"),
    Segment(10.5, 11.2, "static_jitter", True, "pure rotation, trackable"),
    Segment(11.2, 12.0, "static_rest", True, "camera at rest: gyro bias observable"),
]

# Time ranges where the handheld tremor is switched off entirely.
REST_RANGES = ((0.0, 1.2), (11.2, 12.0))


def _ease_in_out(u: float) -> float:
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(u, 0.0, 1.0))


def _keyframe_orientations() -> list[tuple[float, np.ndarray]]:
    """Absolute orientation keyframes (camera-to-world), rotvec in radians."""
    keys = [
        (0.0, np.zeros(3)),
        (2.5, np.zeros(3)),
        (5.0, np.array([0.0, -0.35, 0.0])),           # ~20 deg yaw, ~8 deg/s
        (7.0, np.array([0.0, -0.35, 0.0])),           # hold during translation
        (7.3, np.array([0.0, -3.50, 0.0])),           # whip: ~180 deg in 0.3 s
        (8.2, np.array([0.0, -3.60, 0.0])),           # settle
        (10.5, np.array([0.32, -3.60, -0.22])),       # tilt + horizon roll
        (12.0, np.array([0.32, -3.60, -0.22])),
    ]
    return [(t, matrix_to_quat(rotvec_to_matrix(rv))) for t, rv in keys]


def _base_quat(t: float, keys: list[tuple[float, np.ndarray]]) -> np.ndarray:
    for idx in range(len(keys) - 1):
        t0, q0 = keys[idx]
        t1, q1 = keys[idx + 1]
        if t0 <= t <= t1:
            u = _ease_in_out((t - t0) / max(t1 - t0, 1e-9))
            return slerp(q0, q1, u)
    return keys[-1][1]


def _handheld_jitter(t: float, rng: np.random.Generator) -> np.ndarray:
    """Band-limited micro rotation (~1-3 deg) emulating handheld tremor.

    Frequencies stay well below the 200 Hz gyro Nyquist rate so the IMU can
    resolve them; the residual integration gap mirrors a real device.
    """
    amp = np.array([0.035, 0.030, 0.020])
    phases = _jitter_phases(rng)
    freqs = np.array([3.3, 2.7, 4.1])
    return amp * np.sin(2 * np.pi * freqs * t + phases) + 0.4 * amp * np.sin(
        2 * np.pi * freqs * 2.1 * t + phases * 1.7
    )


_JITTER_PHASES: np.ndarray | None = None


def _jitter_phases(rng: np.random.Generator) -> np.ndarray:
    global _JITTER_PHASES
    if _JITTER_PHASES is None:
        _JITTER_PHASES = rng.uniform(0, 2 * np.pi, 3)
    return _JITTER_PHASES


def _translation_xyz(t: float) -> np.ndarray:
    """Lateral + vertical translation only inside the 5-7 s segment."""
    if t < 5.0 or t > 7.0:
        return np.zeros(3)
    u = _ease_in_out((t - 5.0) / 2.0)
    return np.array([1.6 * u, 0.5 * np.sin(np.pi * u), 0.0])


def generate_poses(
    duration: float = 12.0,
    fps: float = 30.0,
    seed: int = 11,
) -> dict:
    """Sample the true camera trajectory at frame exposure instants.

    Returns a dict with frame timestamps, camera-to-world R/t, quaternions
    and the per-frame segment label.
    """
    n_frames = int(round(duration * fps))
    frame_times = np.arange(n_frames) / fps

    quats = []
    rotations = []
    translations = []
    labels = []
    for t in frame_times:
        q = orientation_at(t, seed)
        r_cw = quat_to_matrix(q).T  # quat stores camera-to-world
        quats.append(q)
        rotations.append(r_cw)
        translations.append(-r_cw @ _translation_xyz(t))
        labels.append(_label_at(t))

    return {
        "frame_times": frame_times,
        "fps": fps,
        "quats_cw": np.array(quats),
        "rotations_cw": np.array(rotations),
        "translations_cw": np.array(translations),
        "labels": np.array(labels),
    }


def orientation_at(t: float, seed: int = 11) -> np.ndarray:
    """Analytic camera-to-world quaternion at any continuous time.

    Evaluating the same deterministic jitter function at IMU sample times
    keeps the high-frequency handheld tremor inside every camera frame,
    instead of smoothing it away by interpolating frame orientations.
    """
    rng = np.random.default_rng(seed)
    base_q = _base_quat(t, _keyframe_orientations())
    jitter = (
        np.zeros(3)
        if any(a <= t <= b for a, b in REST_RANGES)
        else _handheld_jitter(t, rng)
    )
    return quat_normalize(
        quat_multiply(base_q, matrix_to_quat(rotvec_to_matrix(jitter)))
    )


def _label_at(t: float) -> str:
    for seg in DEFAULT_SEGMENTS:
        if seg.contains(t):
            return seg.kind
    return DEFAULT_SEGMENTS[-1].kind


def segments_frame_mask(frame_times: np.ndarray) -> np.ndarray:
    """Boolean mask: frames whose segment is usable for gyro calibration."""
    mask = np.zeros(len(frame_times), dtype=bool)
    for i, t in enumerate(frame_times):
        for seg in DEFAULT_SEGMENTS:
            if seg.contains(t) and seg.valid_for_calibration:
                mask[i] = True
                break
    return mask
