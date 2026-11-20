from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import (
    FrameRecord,
    Perforation,
    apply_matrix,
    rigid_matrix,
)


@dataclass
class TrackerConfig:
    spike_jump_px: float = 4.0
    splice_jump_px: float = 6.0
    splice_angle_deg: float = 0.85
    max_pair_scale_error: float = 0.22


@dataclass
class TrackResult:
    records: list[FrameRecord]
    reference_left: tuple[float, float] | None
    reference_right: tuple[float, float] | None
    reference_angle_deg: float
    splice_frames: list[int] = field(default_factory=list)


def _wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _selected_candidates(
    records: list[FrameRecord],
) -> tuple[list[Perforation | None], list[Perforation | None]]:
    medians: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        points = [
            (item.x, item.y)
            for record in records
            for item in record.detections
            if item.side == side
        ]
        if points:
            medians[side] = np.median(np.asarray(points, dtype=np.float64), axis=0)

    selected: dict[str, list[Perforation | None]] = {"left": [], "right": []}
    for side in ("left", "right"):
        target = medians.get(side)
        for record in records:
            candidates = [item for item in record.detections if item.side == side]
            if not candidates:
                selected[side].append(None)
            elif target is None:
                selected[side].append(max(candidates, key=lambda item: item.score))
            else:
                selected[side].append(
                    min(
                        candidates,
                        key=lambda item: np.hypot(item.x - target[0], item.y - target[1]),
                    )
                )
    return selected["left"], selected["right"]


def _reference_anchors(
    left: list[Perforation | None], right: list[Perforation | None]
) -> tuple[tuple[float, float] | None, tuple[float, float] | None, float]:
    both = [
        (li, ri)
        for li, ri in zip(left, right)
        if li is not None and ri is not None
    ]
    if both:
        left_ref = np.median(
            np.asarray([(item.x, item.y) for item, _ in both]), axis=0
        )
        right_ref = np.median(
            np.asarray([(item.x, item.y) for _, item in both]), axis=0
        )
    else:
        left_points = [(item.x, item.y) for item in left if item is not None]
        right_points = [(item.x, item.y) for item in right if item is not None]
        left_ref = (
            np.median(np.asarray(left_points), axis=0) if left_points else None
        )
        right_ref = (
            np.median(np.asarray(right_points), axis=0) if right_points else None
        )

    ltuple = None if left_ref is None else (float(left_ref[0]), float(left_ref[1]))
    rtuple = None if right_ref is None else (float(right_ref[0]), float(right_ref[1]))
    if ltuple is not None and rtuple is not None:
        angle = float(
            np.degrees(
                np.arctan2(rtuple[1] - ltuple[1], rtuple[0] - ltuple[0])
            )
        )
    else:
        angle = 0.0
    return ltuple, rtuple, angle


def _choose_for_side(
    record: FrameRecord,
    side: str,
    reference: tuple[float, float] | None,
    expected_current: np.ndarray | None,
) -> Perforation | None:
    candidates = [item for item in record.detections if item.side == side]
    if not candidates:
        return None
    target = expected_current
    if target is None and reference is not None:
        target = np.asarray(reference)
    if target is None:
        return max(candidates, key=lambda item: item.score)
    return min(
        candidates,
        key=lambda item: np.hypot(item.x - target[0], item.y - target[1]),
    )


def _build_measurement(
    left: Perforation | None,
    right: Perforation | None,
    reference_left: tuple[float, float] | None,
    reference_right: tuple[float, float] | None,
    predicted_angle: float,
    max_scale_error: float,
) -> tuple[np.ndarray | None, int, bool]:
    valid_pair = left is not None and right is not None
    if valid_pair:
        ref_dist = float(
            np.hypot(
                reference_right[0] - reference_left[0],
                reference_right[1] - reference_left[1],
            )
        )
        cur_dist = float(np.hypot(right.x - left.x, right.y - left.y))
        scale_error = abs(np.log(max(cur_dist, 1e-6) / max(ref_dist, 1e-6)))
        if scale_error > max_scale_error:
            if left.score >= right.score:
                right = None
            else:
                left = None
            valid_pair = False

    if valid_pair:
        ref_angle = np.arctan2(
            reference_right[1] - reference_left[1],
            reference_right[0] - reference_left[0],
        )
        cur_angle = np.arctan2(right.y - left.y, right.x - left.x)
        angle = float(_wrap_angle(ref_angle - cur_angle))
        ref_center = np.asarray(
            (
                (reference_left[0] + reference_right[0]) * 0.5,
                (reference_left[1] + reference_right[1]) * 0.5,
            )
        )
        cur_center = np.asarray(
            ((left.x + right.x) * 0.5, (left.y + right.y) * 0.5)
        )
        rotation = rigid_matrix(angle, 0.0, 0.0)[:2, :2]
        translation = ref_center - rotation @ cur_center
        return rigid_matrix(angle, translation[0], translation[1]), 2, False

    if left is not None and reference_left is not None:
        item, reference = left, reference_left
    elif right is not None and reference_right is not None:
        item, reference = right, reference_right
    else:
        return None, 0, False
    rotation = rigid_matrix(predicted_angle, 0.0, 0.0)[:2, :2]
    ref_point = np.asarray(reference, dtype=np.float64)
    cur_point = np.asarray((item.x, item.y), dtype=np.float64)
    translation = ref_point - rotation @ cur_point
    return rigid_matrix(predicted_angle, translation[0], translation[1]), 1, True


def _matrix_delta(a: np.ndarray, b: np.ndarray, radius: float) -> tuple[float, float]:
    dx = float(a[0, 2] - b[0, 2])
    dy = float(a[1, 2] - b[1, 2])
    angle = abs(float(_wrap_angle(np.arctan2(a[1, 0], a[0, 0]) - np.arctan2(b[1, 0], b[0, 0]))))
    return float(np.hypot(dx, dy) + angle * radius), float(np.degrees(angle))


def _angle_of(matrix: np.ndarray) -> float:
    return float(np.arctan2(matrix[1, 0], matrix[0, 0]))


def _interpolate(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    angle_a = _angle_of(a)
    angle_b = angle_a + float(_wrap_angle(_angle_of(b) - angle_a))
    angle = angle_a + alpha * (angle_b - angle_a)
    tx = (1.0 - alpha) * a[0, 2] + alpha * b[0, 2]
    ty = (1.0 - alpha) * a[1, 2] + alpha * b[1, 2]
    return rigid_matrix(angle, float(tx), float(ty))


def _repair_isolated_spikes(
    records: list[FrameRecord], config: TrackerConfig, radius: float
) -> None:
    measured = [record.matrix for record in records]
    for index, matrix in enumerate(measured):
        if matrix is None or records[index].confidence not in (1, 2):
            continue
        prev_index = next(
            (i for i in range(index - 1, -1, -1) if measured[i] is not None),
            None,
        )
        next_index = next(
            (i for i in range(index + 1, len(measured)) if measured[i] is not None),
            None,
        )
        if prev_index is None or next_index is None:
            continue
        prev_matrix = measured[prev_index]
        next_matrix = measured[next_index]
        neighbor_distance, _ = _matrix_delta(prev_matrix, next_matrix, radius)
        if neighbor_distance > config.spike_jump_px:
            continue
        expected = _interpolate(
            prev_matrix,
            next_matrix,
            (index - prev_index) / float(next_index - prev_index),
        )
        spike_distance, _ = _matrix_delta(matrix, expected, radius)
        if spike_distance > config.spike_jump_px:
            records[index].matrix = None
            records[index].confidence = 0
            records[index].status = "rejected_spike"


def _refine_single_hole_angles(
    records: list[FrameRecord],
    left_ref: tuple[float, float] | None,
    right_ref: tuple[float, float] | None,
    config: TrackerConfig,
) -> None:
    strong = [
        index
        for index, record in enumerate(records)
        if record.confidence == 2 and record.matrix is not None
    ]
    for index, record in enumerate(records):
        if record.confidence != 1 or record.matrix is None:
            continue
        previous = max((item for item in strong if item < index), default=None)
        following = min((item for item in strong if item > index), default=None)
        if previous is None and following is None:
            continue
        if previous is not None and following is not None:
            angle_previous = _angle_of(records[previous].matrix)
            angle_following = _angle_of(records[following].matrix)
            angle_delta = float(_wrap_angle(angle_following - angle_previous))
            if abs(angle_delta) > np.radians(config.splice_angle_deg):
                angle = (
                    angle_previous
                    if index - previous <= following - index
                    else angle_following
                )
            else:
                alpha = (index - previous) / float(following - previous)
                angle = angle_previous + alpha * angle_delta
        elif previous is not None:
            angle = _angle_of(records[previous].matrix)
        else:
            angle = _angle_of(records[following].matrix)

        if record.left_anchor is not None and left_ref is not None:
            point = np.asarray(record.left_anchor, dtype=np.float64)
            reference = np.asarray(left_ref, dtype=np.float64)
        elif record.right_anchor is not None and right_ref is not None:
            point = np.asarray(record.right_anchor, dtype=np.float64)
            reference = np.asarray(right_ref, dtype=np.float64)
        else:
            continue
        rotation = rigid_matrix(angle, 0.0, 0.0)[:2, :2]
        translation = reference - rotation @ point
        record.matrix = rigid_matrix(angle, translation[0], translation[1])


def _fill_gaps(records: list[FrameRecord]) -> None:
    known = [i for i, record in enumerate(records) if record.matrix is not None]
    for index, record in enumerate(records):
        if record.matrix is not None:
            continue
        prev = max((i for i in known if i < index), default=None)
        nxt = min((i for i in known if i > index), default=None)
        if prev is not None and nxt is not None:
            alpha = (index - prev) / float(nxt - prev)
            record.matrix = _interpolate(
                records[prev].matrix, records[nxt].matrix, alpha
            )
            if record.status != "rejected_spike":
                record.status = "interpolated"
        elif prev is not None:
            before = max((i for i in known if i < prev), default=None)
            if before is None:
                record.matrix = records[prev].matrix.copy()
            else:
                span = float(prev - before)
                velocity = (
                    records[prev].matrix[:2, 2] - records[before].matrix[:2, 2]
                ) / span
                angle_velocity = float(
                    _wrap_angle(
                        _angle_of(records[prev].matrix)
                        - _angle_of(records[before].matrix)
                    )
                ) / span
                step = index - prev
                tx = records[prev].matrix[0, 2] + velocity[0] * step
                ty = records[prev].matrix[1, 2] + velocity[1] * step
                angle = _angle_of(records[prev].matrix) + angle_velocity * step
                record.matrix = rigid_matrix(angle, float(tx), float(ty))
            if record.status != "rejected_spike":
                record.status = "extrapolated"
        elif nxt is not None:
            after = min((i for i in known if i > nxt), default=None)
            if after is None:
                record.matrix = records[nxt].matrix.copy()
            else:
                span = float(after - nxt)
                velocity = (
                    records[after].matrix[:2, 2] - records[nxt].matrix[:2, 2]
                ) / span
                angle_velocity = float(
                    _wrap_angle(
                        _angle_of(records[after].matrix)
                        - _angle_of(records[nxt].matrix)
                    )
                ) / span
                step = index - nxt
                tx = records[nxt].matrix[0, 2] + velocity[0] * step
                ty = records[nxt].matrix[1, 2] + velocity[1] * step
                angle = _angle_of(records[nxt].matrix) + angle_velocity * step
                record.matrix = rigid_matrix(angle, float(tx), float(ty))
            if record.status != "rejected_spike":
                record.status = "extrapolated"
        else:
            record.matrix = np.eye(3, dtype=np.float64)
            record.status = "identity_no_reference"


def track_records(
    records: list[FrameRecord], config: TrackerConfig | None = None
) -> TrackResult:
    config = config or TrackerConfig()
    left, right = _selected_candidates(records)
    left_ref, right_ref, ref_angle_deg = _reference_anchors(left, right)
    if left_ref is None and right_ref is None:
        for record in records:
            record.matrix = np.eye(3)
            record.status = "identity_no_reference"
        return TrackResult(records, None, None, 0.0)

    predicted_angle = float(np.radians(ref_angle_deg))
    last_matrix: np.ndarray | None = None
    anchor_radius = 0.0
    if left_ref is not None and right_ref is not None:
        anchor_radius = float(np.hypot(right_ref[0] - left_ref[0], right_ref[1] - left_ref[1]))

    for record, left_item, right_item in zip(records, left, right):
        if last_matrix is not None:
            inverse = np.linalg.inv(last_matrix)
            predicted_left = (
                apply_matrix(inverse, [left_ref])[0] if left_ref is not None else None
            )
            predicted_right = (
                apply_matrix(inverse, [right_ref])[0] if right_ref is not None else None
            )
        else:
            predicted_left = predicted_right = None
        chosen_left = _choose_for_side(record, "left", left_ref, predicted_left)
        chosen_right = _choose_for_side(record, "right", right_ref, predicted_right)
        matrix, confidence, _ = _build_measurement(
            chosen_left,
            chosen_right,
            left_ref,
            right_ref,
            predicted_angle,
            config.max_pair_scale_error,
        )
        record.left_observed = chosen_left is not None
        record.right_observed = chosen_right is not None
        record.left_anchor = (
            None if chosen_left is None else (chosen_left.x, chosen_left.y)
        )
        record.right_anchor = (
            None if chosen_right is None else (chosen_right.x, chosen_right.y)
        )
        if matrix is None:
            record.confidence = 0
            record.status = "missing_both"
            continue
        record.matrix = matrix
        record.confidence = confidence
        record.status = "measured_both" if confidence == 2 else "measured_single"
        last_matrix = matrix
        predicted_angle = _angle_of(matrix)

    _refine_single_hole_angles(records, left_ref, right_ref, config)
    _repair_isolated_spikes(records, config, max(anchor_radius, 1.0))
    _fill_gaps(records)

    splice_frames: list[int] = []
    strong_indices = [
        index
        for index, record in enumerate(records)
        if record.confidence == 2 and record.matrix is not None
    ]
    previous_strong: int | None = None
    for index in strong_indices:
        if previous_strong is None:
            previous_strong = index
            continue
        distance, angle_delta = _matrix_delta(
            records[index].matrix,
            records[previous_strong].matrix,
            max(anchor_radius, 1.0),
        )
        gap = index - previous_strong
        distance_limit = config.splice_jump_px + 1.5 * (gap - 1)
        angle_limit = config.splice_angle_deg * gap
        if (
            distance > distance_limit
            or angle_delta > angle_limit
        ):
            records[index].is_splice = True
            splice_frames.append(records[index].original_frame)
        previous_strong = index

    return TrackResult(
        records=records,
        reference_left=left_ref,
        reference_right=right_ref,
        reference_angle_deg=ref_angle_deg,
        splice_frames=splice_frames,
    )
