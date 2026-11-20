from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .stabilizer import StabilizationResult
from .synthetic import SyntheticTruth
from .types import apply_matrix


@dataclass
class VerificationReport:
    frames: int
    measured_both: int
    measured_single: int
    filled: int
    gate_alignment_rms_px: float
    gate_alignment_max_px: float
    gate_rotation_rms_deg: float
    actor_motion_range_px: float
    gate_motion_range_px: float
    splice_frames: list[int]
    passed: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else 0.0


def verify_against_truth(
    result: StabilizationResult,
    truth: list[SyntheticTruth],
    max_rms_px: float = 1.5,
    max_point_px: float = 3.0,
    max_rotation_rms_deg: float = 0.45,
) -> VerificationReport:
    if len(result.records) != len(truth):
        raise ValueError("track and truth have different frame counts")
    left_stable = []
    right_stable = []
    stabilized_angles = []
    for record, expected in zip(result.records, truth):
        left = apply_matrix(record.matrix, [expected.left])[0]
        right = apply_matrix(record.matrix, [expected.right])[0]
        left_stable.append(left)
        right_stable.append(right)
        stabilized_angles.append(
            np.arctan2(right[1] - left[1], right[0] - left[0])
        )

    left_stable = np.asarray(left_stable)
    right_stable = np.asarray(right_stable)
    left_median = np.median(left_stable, axis=0)
    right_median = np.median(right_stable, axis=0)
    left_error = np.linalg.norm(left_stable - left_median, axis=1)
    right_error = np.linalg.norm(right_stable - right_median, axis=1)
    all_error = np.concatenate([left_error, right_error])
    angle = np.unwrap(np.asarray(stabilized_angles))
    angle_error_deg = np.degrees(angle - np.median(angle))

    actor = np.asarray([item.actor for item in truth])
    gate = np.asarray([(item.gate_tx, item.gate_ty) for item in truth])
    actor_range = float(np.max(np.linalg.norm(actor - actor.mean(axis=0), axis=1)))
    gate_range = float(np.max(np.linalg.norm(gate - gate.mean(axis=0), axis=1)))

    measured_both = sum(record.confidence == 2 for record in result.records)
    measured_single = sum(record.confidence == 1 for record in result.records)
    filled = len(result.records) - measured_both - measured_single
    rms = _rms(all_error)
    maximum = float(np.max(all_error)) if len(all_error) else 0.0
    rotation_rms = _rms(angle_error_deg)
    return VerificationReport(
        frames=len(result.records),
        measured_both=measured_both,
        measured_single=measured_single,
        filled=filled,
        gate_alignment_rms_px=rms,
        gate_alignment_max_px=maximum,
        gate_rotation_rms_deg=rotation_rms,
        actor_motion_range_px=actor_range,
        gate_motion_range_px=gate_range,
        splice_frames=list(result.splice_frames),
        passed=(
            rms <= max_rms_px
            and maximum <= max_point_px
            and rotation_rms <= max_rotation_rms_deg
        ),
    )
