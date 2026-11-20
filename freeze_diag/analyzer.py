from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import DiagnosticConfig
from .features import FeatureExtractor
from .models import FrameFeatures, Interval
from .timestamps import extract_decoder_timestamps, summarize_timestamps


def analyze_video(path: str | Path, config: DiagnosticConfig | None = None) -> dict[str, Any]:
    path = Path(path)
    config = config or DiagnosticConfig()
    pts_ms = extract_decoder_timestamps(path)
    features, video_info = _extract_features(path, pts_ms, config)
    if not features:
        raise ValueError(f"no decodable video frames: {path}")

    frame_count = len(features)
    fps = video_info["fps"] or _fps_from_pts(pts_ms)
    timestamp_summary = summarize_timestamps(
        pts_ms[:frame_count],
        frame_count,
        config.timestamp_tolerance_ms,
        config.timestamp_tolerance_ratio,
    )
    baseline_noise = _baseline_noise(features, config.noise_floor)
    intervals = _build_intervals(features, timestamp_summary, baseline_noise, config, fps)
    freeze_intervals = [item for item in intervals if item.status == "frozen"]
    black_intervals = [item for item in intervals if item.status == "blackout_indeterminate"]
    verdict = _verdict(intervals, features, black_intervals)
    noise_frames = sum(
        item.local_noise is not None
        and not item.black
        and item.local_noise > config.noise_floor
        for item in features
    )

    return {
        "schema_version": "1.0",
        "input": str(path),
        "video": video_info,
        "timestamps": timestamp_summary.to_dict(),
        "metrics": {
            "baseline_local_noise": _round_optional(baseline_noise),
            "duplicate_fraction": round(
                sum(item.duplicate for item in features) / max(frame_count - 1, 1), 6
            ),
            "black_fraction": round(sum(item.black for item in features) / frame_count, 6),
            "noise_above_floor_fraction": round(
                noise_frames / max(frame_count - 1, 1), 6
            ),
            "median_local_noise": _round_optional(
                _median_optional([item.local_noise for item in features])
            ),
        },
        "config": config.to_dict(),
        "verdict": verdict,
        "diagnosis": _diagnosis(verdict, config),
        "intervals": [item.to_dict() for item in intervals],
        "freeze_intervals": [item.to_dict() for item in freeze_intervals],
        "black_intervals": [item.to_dict() for item in black_intervals],
    }


def _extract_features(
    path: Path,
    pts_ms: list[float | None],
    config: DiagnosticConfig,
) -> tuple[list[FrameFeatures], dict[str, Any]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"OpenCV cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    extractor = FeatureExtractor(config)
    features: list[FrameFeatures] = []
    ok, frame = capture.read()
    height = width = 0
    while ok:
        height, width = frame.shape[:2]
        index = len(features)
        pts = pts_ms[index] if index < len(pts_ms) else None
        features.append(extractor.process(frame, index, pts))
        ok, frame = capture.read()
    capture.release()

    if len(pts_ms) != len(features):
        raise RuntimeError(
            "frame/timestamp count mismatch after decode: "
            f"OpenCV={len(features)}, PyAV={len(pts_ms)}"
        )
    duration_ms = None
    valid_pts = [item for item in pts_ms if item is not None]
    if len(valid_pts) >= 2:
        duration_ms = valid_pts[-1] - valid_pts[0]
    return features, {
        "fps": round(fps, 6) if fps > 0 else None,
        "width": width,
        "height": height,
        "frame_count": len(features),
        "duration_ms": _round_optional(duration_ms),
        "timestamp_source": "PyAV decoded AVFrame.pts",
        "pixel_source": "OpenCV VideoCapture",
    }


def _build_intervals(
    features: list[FrameFeatures],
    timestamps,
    baseline_noise: float | None,
    config: DiagnosticConfig,
    fps: float | None,
) -> list[Interval]:
    intervals: list[Interval] = []
    duplicate_mask = _fill_short_gaps(
        [item.duplicate and not item.black for item in features],
        config.max_duplicate_gap,
    )
    for start, end in _mask_runs(duplicate_mask, include_previous=True):
        if end - start < 2:
            continue
        slice_features = features[start : end + 1]
        black_ratio = sum(item.black for item in slice_features) / len(slice_features)
        if black_ratio > 0.5:
            continue
        local_noise = _median_optional([item.local_noise for item in slice_features])
        if local_noise is None or local_noise > config.noise_floor:
            continue
        interval = _make_interval(
            "frozen", start, end, features, timestamps, baseline_noise, config, fps
        )
        if interval.duration_ms >= config.min_freeze_ms:
            intervals.append(interval)

    black_mask = [item.black for item in features]
    for start, end in _mask_runs(black_mask, include_previous=False):
        interval = _make_interval(
            "blackout_indeterminate",
            start,
            end,
            features,
            timestamps,
            baseline_noise,
            config,
            fps,
        )
        if interval.duration_ms >= config.min_freeze_ms:
            intervals.append(interval)
    return sorted(intervals, key=lambda item: item.start_frame)


def _make_interval(
    status: str,
    start: int,
    end: int,
    features: list[FrameFeatures],
    timestamps,
    baseline_noise: float | None,
    config: DiagnosticConfig,
    fps: float | None,
) -> Interval:
    selected = features[start : end + 1]
    frame_period = timestamps.nominal_delta_ms or (1000.0 / fps if fps else 0.0)
    start_ms = _frame_time(features, start, frame_period)
    end_ms = _frame_time(features, end, frame_period)
    if start_ms is not None and end_ms is not None and end_ms >= start_ms:
        duration = end_ms - start_ms + frame_period
    else:
        duration = len(selected) * frame_period

    local_pts = [item.pts_ms for item in selected]
    local_timestamps = summarize_timestamps(
        local_pts,
        len(local_pts),
        config.timestamp_tolerance_ms,
        config.timestamp_tolerance_ratio,
    )
    noise_median = _median_optional([item.local_noise for item in selected])
    duplicate_transitions = sum(item.duplicate for item in selected[1:])
    noise_ratio = None
    if noise_median is not None and baseline_noise and baseline_noise > 1e-9:
        noise_ratio = noise_median / baseline_noise

    evidence = {
        "frame_count": len(selected),
        "duplicate_transitions": duplicate_transitions,
        "duplicate_ratio": round(duplicate_transitions / max(len(selected) - 1, 1), 6),
        "local_noise_median": _round_optional(noise_median),
        "baseline_local_noise": _round_optional(baseline_noise),
        "local_noise_ratio": round(noise_ratio, 6) if noise_ratio is not None else None,
        "noise_floor": config.noise_floor,
        "noise_below_floor": noise_median is not None and noise_median <= config.noise_floor,
        "luma_mean_median": round(float(np.median([item.luma_mean for item in selected])), 4),
        "detail_median": round(float(np.median([item.detail for item in selected])), 4),
        "timestamp_continuity": local_timestamps.continuity,
        "timestamp_regular_ratio": round(local_timestamps.regular_ratio, 6),
        "timestamp_gap_count": local_timestamps.gap_count,
        "timestamp_backstep_count": local_timestamps.backstep_count,
        "nominal_delta_ms": _round_optional(local_timestamps.nominal_delta_ms),
    }
    if status == "frozen":
        evidence["determinable"] = True
        evidence["cause_hint"] = (
            "upstream_repeated_frame"
            if local_timestamps.continuity == "continuous"
            else "repeat_with_timestamp_anomaly"
        )
        evidence["reason_codes"] = [
            "repeated_decoded_pixels",
            "local_temporal_noise_absent",
            "non_black_content",
            f"timestamps_{local_timestamps.continuity}",
        ]
    else:
        evidence["determinable"] = False
        evidence["cause_hint"] = "insufficient_image_information"
        evidence["reason_codes"] = [
            "luma_below_black_floor",
            "texture_below_black_floor",
            "liveness_cannot_be_proven",
            f"timestamps_{local_timestamps.continuity}",
        ]
    return Interval(
        status=status,
        start_frame=start,
        end_frame=end,
        start_ms=_round_optional(start_ms),
        end_ms=_round_optional(end_ms),
        duration_ms=duration,
        evidence=evidence,
    )


def _mask_runs(mask: list[bool], include_previous: bool):
    index = 0
    while index < len(mask):
        if not mask[index]:
            index += 1
            continue
        run_start = index
        while index + 1 < len(mask) and mask[index + 1]:
            index += 1
        run_end = index
        start = run_start - 1 if include_previous and run_start > 0 else run_start
        yield start, run_end
        index += 1


def _fill_short_gaps(mask: list[bool], max_gap: int) -> list[bool]:
    if max_gap <= 0:
        return mask
    filled = mask.copy()
    index = 0
    while index < len(mask):
        if mask[index]:
            index += 1
            continue
        start = index
        while index < len(mask) and not mask[index]:
            index += 1
        if start > 0 and index < len(mask) and index - start <= max_gap:
            filled[start:index] = [True] * (index - start)
    return filled


def _baseline_noise(
    features: list[FrameFeatures], noise_floor: float
) -> float | None:
    values = [
        item.local_noise
        for item in features
        if not item.black and item.local_noise is not None
    ]
    if not values:
        return None
    baseline = float(np.percentile(values, 90))
    return baseline if baseline > noise_floor else None


def _verdict(
    intervals: list[Interval],
    features: list[FrameFeatures],
    black_intervals: list[Interval],
) -> str:
    has_frozen = any(item.status == "frozen" for item in intervals)
    has_black = bool(black_intervals)
    if has_frozen and has_black:
        return "mixed_frozen_and_indeterminate_black"
    if has_frozen:
        return "frozen_repeat_detected"
    black_fraction = sum(item.black for item in features) / len(features)
    if has_black and black_fraction >= 0.5:
        return "indeterminate_blackout"
    return "no_freeze_detected"


def _diagnosis(verdict: str, config: DiagnosticConfig) -> str:
    if verdict == "frozen_repeat_detected":
        return (
            "检测到非黑画面连续重复，且相邻帧局部噪声低于 "
            f"{config.noise_floor:g}；PTS 连续性决定这是上游重复还是时间轴异常。"
        )
    if verdict == "indeterminate_blackout":
        return (
            "画面无亮度、纹理和局部噪声，PTS 推进只能证明时间戳在走，"
            "不能证明图像仍在实时采集，因此不判冻结。"
        )
    if verdict == "mixed_frozen_and_indeterminate_black":
        return "同时存在可判定的重复帧冻结和无法判定的全黑区间。"
    return (
        f"未发现连续重复且无噪声的区间；局部噪声高于 {config.noise_floor:g} 时，"
        "固定机位画面静止也不判冻结。"
    )


def _fps_from_pts(pts_ms: list[float | None]) -> float | None:
    valid = [item for item in pts_ms if item is not None]
    if len(valid) < 2:
        return None
    positive = np.diff(valid)
    positive = positive[positive > 1e-6]
    if positive.size == 0:
        return None
    return float(1000.0 / np.median(positive))


def _frame_time(
    features: list[FrameFeatures],
    index: int,
    frame_period_ms: float,
) -> float | None:
    if features[index].pts_ms is not None:
        return features[index].pts_ms
    first_pts = next(
        (item.pts_ms for item in features if item.pts_ms is not None), None
    )
    if first_pts is not None and frame_period_ms > 0:
        return first_pts + index * frame_period_ms
    return None


def _median_optional(values: list[float | None]) -> float | None:
    valid = [item for item in values if item is not None and math.isfinite(item)]
    return float(np.median(valid)) if valid else None


def _round_optional(value: float | None) -> float | None:
    return round(float(value), 6) if value is not None else None
