from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import LabConfig
from .restoration import RestorationResult
from .synth import DemoClip


@dataclass(slots=True)
class ErrorSummary:
    rms_8bit: float
    mean_abs_8bit: float
    peak_8bit: float
    per_channel_rms_8bit: list[float]


@dataclass(slots=True)
class OvershootSummary:
    rms_excursion_8bit: float
    mean_excursion_8bit: float
    affected_fraction: float
    peak_excursion_8bit: float


def _motion_overlap_masks(motion_masks: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    overlap = np.zeros_like(motion_masks, dtype=bool)
    for frame_index in range(1, len(motion_masks)):
        previous = cv2.dilate(
            motion_masks[frame_index - 1].astype(np.uint8), kernel
        ).astype(bool)
        current = cv2.dilate(
            motion_masks[frame_index].astype(np.uint8), kernel
        ).astype(bool)
        overlap[frame_index] = previous & current
    return overlap


def _error_summary(
    truth: np.ndarray,
    observed: np.ndarray,
    roi: np.ndarray,
    start_frame: int = 2,
) -> ErrorSummary:
    squared: list[np.ndarray] = []
    absolute: list[np.ndarray] = []
    peak = 0.0
    for frame_index in range(start_frame, len(truth)):
        selected = roi[frame_index]
        if not np.any(selected):
            continue
        error = observed[frame_index][selected] - truth[frame_index][selected]
        squared.append(error * error)
        absolute.append(np.abs(error))
        peak = max(peak, float(np.max(np.abs(error))))
    if not squared:
        return ErrorSummary(0.0, 0.0, 0.0, [0.0, 0.0, 0.0])
    merged = np.concatenate(squared, axis=0)
    merged_abs = np.concatenate(absolute, axis=0)
    return ErrorSummary(
        rms_8bit=float(np.sqrt(np.mean(merged)) * 255.0),
        mean_abs_8bit=float(np.mean(merged_abs) * 255.0),
        peak_8bit=float(peak * 255.0),
        per_channel_rms_8bit=[
            float(np.sqrt(np.mean(merged[:, channel])) * 255.0)
            for channel in range(3)
        ],
    )


def _overshoot_summary(
    truth: np.ndarray,
    observed: np.ndarray,
    roi: np.ndarray,
    tolerance: float = 0.035,
) -> OvershootSummary:
    excursion_values: list[np.ndarray] = []
    total = 0
    for frame_index in range(len(truth)):
        selected = roi[frame_index]
        if not np.any(selected):
            continue
        target = truth[frame_index][selected]
        error = observed[frame_index][selected] - target
        low_target = target <= 0.10
        high_target = target >= 0.90
        excursion = np.zeros_like(error)
        excursion[low_target] = np.maximum(error[low_target] - tolerance, 0.0)
        excursion[high_target] = np.maximum(-error[high_target] - tolerance, 0.0)
        excursion_values.append(excursion.reshape(-1))
        total += int(excursion.size)
    if not excursion_values:
        return OvershootSummary(0.0, 0.0, 0.0, 0.0)
    merged = np.concatenate(excursion_values)
    return OvershootSummary(
        rms_excursion_8bit=float(np.sqrt(np.mean(merged * merged)) * 255.0),
        mean_excursion_8bit=float(np.mean(merged) * 255.0),
        affected_fraction=float(np.count_nonzero(merged > 0.0) / max(1, total)),
        peak_excursion_8bit=float(np.max(merged) * 255.0),
    )


def _checker_roi_by_frame(
    clip: DemoClip,
    event_frame: int = 72,
    settle_frames: int = 18,
) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    roi = np.zeros_like(clip.motion_masks, dtype=bool)
    for frame_index in range(event_frame, min(len(roi), event_frame + settle_frames)):
        blocked = cv2.dilate(
            clip.motion_masks[frame_index].astype(np.uint8), kernel
        ).astype(bool)
        if frame_index > 0:
            blocked |= cv2.dilate(
                clip.motion_masks[frame_index - 1].astype(np.uint8), kernel
            ).astype(bool)
        roi[frame_index] = clip.checker_roi & ~blocked
    return roi


def compute_metrics(
    clip: DemoClip,
    restoration: RestorationResult,
    config: LabConfig,
) -> dict[str, Any]:
    motion_roi = _motion_overlap_masks(clip.motion_masks)
    checker_roi = _checker_roi_by_frame(clip)
    capture_ghost = _error_summary(clip.source, clip.captured, motion_roi)
    restored_ghost = _error_summary(clip.source, restoration.frames, motion_roi)
    capture_over = _overshoot_summary(clip.source, clip.captured, checker_roi)
    restored_over = _overshoot_summary(clip.source, restoration.frames, checker_roi)
    capture_all = _error_summary(
        clip.source, clip.captured, np.ones_like(motion_roi), start_frame=1
    )
    restored_all = _error_summary(
        clip.source, restoration.frames, np.ones_like(motion_roi), start_frame=1
    )
    ghost_db = 20.0 * np.log10(
        max(capture_ghost.rms_8bit, 1e-6) / max(restored_ghost.rms_8bit, 1e-6)
    )
    overshoot_db = 20.0 * np.log10(
        max(capture_over.rms_excursion_8bit, 1e-6)
        / max(restored_over.rms_excursion_8bit, 1e-6)
    )
    unstable_fraction = float(np.mean(restoration.unstable_frequency))
    saturated_fraction = float(np.mean(restoration.saturated_frequency))
    return {
        "video": {
            "width": config.width,
            "height": config.height,
            "fps": config.fps,
            "frames": int(len(clip.source)),
            "duration_seconds": float(len(clip.source) / config.fps),
        },
        "ghost": {
            "roi": "overlap of consecutive dilated motion contours",
            "captured": asdict(capture_ghost),
            "restored": asdict(restored_ghost),
            "improvement_db": float(ghost_db),
        },
        "overshoot": {
            "roi": "stationary checkerboard after phase inversion at frame 72",
            "tolerance_8bit": 0.035 * 255.0,
            "captured": asdict(capture_over),
            "restored": asdict(restored_over),
            "improvement_db": float(overshoot_db),
        },
        "overall": {
            "captured": asdict(capture_all),
            "restored": asdict(restored_all),
            "improvement_db": float(
                20.0
                * np.log10(
                    max(capture_all.rms_8bit, 1e-6)
                    / max(restored_all.rms_8bit, 1e-6)
                )
            ),
        },
        "diagnostics": {
            "unstable_pixel_fraction": unstable_fraction,
            "saturated_pixel_fraction": saturated_fraction,
            "unstable_gain_threshold": config.instability_gain,
            "unstable_frame_pixels": restoration.unstable.sum(axis=(1, 2)).tolist(),
            "saturated_frame_pixels": restoration.saturated.sum(axis=(1, 2)).tolist(),
        },
    }


def save_metrics(metrics: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
