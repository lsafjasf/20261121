from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .config import LabConfig
from .model import directional_parameters, simulate_response


@dataclass(slots=True)
class Segment:
    name: str
    start: int
    end: int
    kind: str
    direction: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start_frame": self.start,
            "end_frame": self.end,
            "kind": self.kind,
            "direction": self.direction,
        }


@dataclass(slots=True)
class CalibrationClip:
    source: np.ndarray
    captured: np.ndarray
    segments: list[Segment]


@dataclass(slots=True)
class DemoClip:
    source: np.ndarray
    captured: np.ndarray
    motion_masks: np.ndarray
    checker_roi: np.ndarray
    zoom_centers: np.ndarray
    object_centers: np.ndarray


def _draw_float_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[float, float, float],
    thickness: int,
) -> None:
    canvas = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    rgb = tuple(int(round(component * 255.0)) for component in color)
    cv2.putText(
        canvas,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        rgb,
        thickness,
        cv2.LINE_AA,
    )
    frame[:] = canvas.astype(np.float32) / 255.0


def _calibration_segments() -> list[Segment]:
    specs = [
        ("black_reset_0", 8, "reset", None),
        ("flash_rise_0", 20, "flash", None),
        ("flash_fall_0", 18, "flash", None),
        ("flash_rise_1", 20, "flash", None),
        ("flash_fall_1", 18, "flash", None),
        ("sweep_right", 22, "sweep", "right"),
        ("sweep_right_reset", 4, "reset", None),
        ("sweep_left", 22, "sweep", "left"),
        ("sweep_left_reset", 4, "reset", None),
        ("sweep_down", 22, "sweep", "down"),
        ("sweep_down_reset", 4, "reset", None),
        ("sweep_up", 22, "sweep", "up"),
        ("sweep_up_reset", 6, "reset", None),
    ]
    segments: list[Segment] = []
    cursor = 0
    for name, length, kind, direction in specs:
        segments.append(Segment(name, cursor, cursor + length, kind, direction))
        cursor += length
    return segments


def _sweep_bar(
    frame: np.ndarray,
    progress: float,
    direction: str,
) -> None:
    height, width = frame.shape[:2]
    if direction == "right":
        start = int(round(-width * 0.34 + progress * width * 1.34))
        frame[:, max(0, start) : max(0, start + width // 3)] = 1.0
    elif direction == "left":
        start = int(round(width - progress * width * 1.34))
        frame[:, max(0, start) : max(0, start + width // 3)] = 1.0
    elif direction == "down":
        start = int(round(-height * 0.34 + progress * height * 1.34))
        frame[max(0, start) : max(0, start + height // 3), :] = 1.0
    elif direction == "up":
        start = int(round(height - progress * height * 1.34))
        frame[max(0, start) : max(0, start + height // 3), :] = 1.0
    else:
        raise ValueError(f"unknown sweep direction: {direction}")


def build_calibration_clip(config: LabConfig) -> CalibrationClip:
    segments = _calibration_segments()
    frame_count = segments[-1].end
    source = np.zeros((frame_count, config.height, config.width, 3), dtype=np.float32)
    for segment in segments:
        for frame_index in range(segment.start, segment.end):
            frame = source[frame_index]
            if segment.kind == "flash":
                if "rise" in segment.name:
                    frame[:] = 1.0
            elif segment.kind == "sweep" and segment.direction:
                progress = (frame_index - segment.start) / max(1, segment.end - segment.start - 1)
                _sweep_bar(frame, progress, segment.direction)

    captured = np.empty_like(source)
    state = np.zeros((config.height, config.width, 3), dtype=np.float32)
    rng = np.random.default_rng(config.seed)
    for frame_index, segment in enumerate(_frame_segments(segments, frame_count)):
        bias = config.direction_tau_bias.get(segment.direction or "")
        if bias is None:
            alpha_rise = config.alpha_rise
            alpha_fall = config.alpha_fall
        else:
            alpha_rise = directional_parameters(config.alpha_rise, bias)
            alpha_fall = directional_parameters(config.alpha_fall, bias)
        step_source = source[frame_index : frame_index + 1]
        state = simulate_response(
            step_source, alpha_rise, alpha_fall, initial=state
        )[0]
        captured[frame_index] = state

    captured += rng.normal(0.0, config.noise_sigma, captured.shape).astype(np.float32)
    captured = np.clip(captured, 0.0, 1.0)
    return CalibrationClip(source=source, captured=captured, segments=segments)


def _frame_segments(segments: list[Segment], frame_count: int) -> list[Segment]:
    lookup: list[Segment] = []
    for frame_index in range(frame_count):
        match = next(
            segment for segment in segments if segment.start <= frame_index < segment.end
        )
        lookup.append(match)
    return lookup


def _draw_demo_frame(
    config: LabConfig,
    frame_index: int,
    checker_roi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = config.height, config.width
    frame = np.empty((height, width, 3), dtype=np.float32)
    frame[:] = (0.025, 0.033, 0.055)

    for x in range(0, width, 40):
        frame[:, x] = np.maximum(frame[:, x], (0.055, 0.067, 0.094))
    for y in range(0, height, 40):
        frame[y, :] = np.maximum(frame[y, :], (0.055, 0.067, 0.094))

    checker_x, checker_y = 294, 48
    cell = 16
    rows, cols = 8, 10
    phase_flip = 1 if frame_index >= 72 else 0
    for row in range(rows):
        for col in range(cols):
            y0 = checker_y + row * cell
            x0 = checker_x + col * cell
            value = 0.94 if (row + col + phase_flip) % 2 == 0 else 0.035
            frame[y0 : y0 + cell, x0 : x0 + cell] = (value, value, value)
            checker_roi[y0 : y0 + cell, x0 : x0 + cell] = True
    cv2.rectangle(
        frame,
        (checker_x - 3, checker_y - 3),
        (checker_x + cols * cell + 2, checker_y + rows * cell + 2),
        (0.15, 0.55, 0.68),
        1,
    )

    _draw_float_text(
        frame,
        "LCD RESPONSE LAB",
        (18, 31),
        0.62,
        (0.78, 0.86, 0.92),
        1,
    )
    _draw_float_text(
        frame,
        "STATIC CHECKER / MOVING CONTOURS",
        (18, 52),
        0.34,
        (0.35, 0.48, 0.60),
        1,
    )

    time_value = frame_index / config.fps
    box_x = int(round(-70 + (time_value * 73.0) % (width + 140)))
    box_y = int(round(248 + 26 * np.sin(time_value * 2.3)))
    box_w, box_h = 116, 64
    cv2.rectangle(
        frame,
        (box_x, box_y),
        (box_x + box_w, box_y + box_h),
        (0.055, 0.11, 0.15),
        -1,
    )
    cv2.rectangle(
        frame,
        (box_x, box_y),
        (box_x + box_w, box_y + box_h),
        (0.88, 0.94, 0.80),
        3,
        cv2.LINE_AA,
    )
    cv2.line(
        frame,
        (box_x + 14, box_y + box_h // 2),
        (box_x + box_w - 14, box_y + box_h // 2),
        (0.18, 0.53, 0.45),
        2,
        cv2.LINE_AA,
    )
    mask_box = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(
        mask_box,
        (box_x, box_y),
        (box_x + box_w, box_y + box_h),
        255,
        -1,
    )

    circle_x = int(round(240 + 150 * np.cos(time_value * 1.05 + 0.2)))
    circle_y = int(round(214 + 86 * np.sin(time_value * 1.35)))
    radius = 54
    cv2.circle(frame, (circle_x, circle_y), radius, (0.045, 0.075, 0.11), -1)
    cv2.circle(
        frame,
        (circle_x, circle_y),
        radius,
        (0.90, 0.32, 0.72),
        3,
        cv2.LINE_AA,
    )
    cv2.circle(
        frame,
        (circle_x, circle_y),
        radius - 15,
        (0.18, 0.86, 0.89),
        2,
        cv2.LINE_AA,
    )
    cv2.line(
        frame,
        (circle_x - radius + 8, circle_y),
        (circle_x + radius - 8, circle_y),
        (0.88, 0.42, 0.70),
        2,
        cv2.LINE_AA,
    )
    mask_circle = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask_circle, (circle_x, circle_y), radius, 255, -1)

    dot_size = 26
    phase = (frame_index * 8.2) % (width + dot_size)
    dot_x = int(round(phase - dot_size))
    dot_y = int(round(118 + 58 * np.sin(time_value * 2.9 + 0.6)))
    cv2.rectangle(
        frame,
        (dot_x, dot_y),
        (dot_x + dot_size, dot_y + dot_size),
        (0.97, 0.98, 0.90),
        -1,
    )
    mask_dot = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(
        mask_dot,
        (dot_x, dot_y),
        (dot_x + dot_size, dot_y + dot_size),
        255,
        -1,
    )

    masks = np.zeros((height, width, 3), dtype=bool)
    masks[..., 0] = mask_box > 0
    masks[..., 1] = mask_circle > 0
    masks[..., 2] = mask_dot > 0
    return np.clip(frame, 0.0, 1.0), masks, np.array([circle_x, circle_y])


def build_demo_clip(config: LabConfig, frame_count: int = 144) -> DemoClip:
    source = np.empty(
        (frame_count, config.height, config.width, 3), dtype=np.float32
    )
    masks = np.empty(
        (frame_count, config.height, config.width, 3), dtype=bool
    )
    centers = np.empty((frame_count, 2), dtype=np.float32)
    checker_roi = np.zeros((config.height, config.width), dtype=bool)
    for frame_index in range(frame_count):
        source[frame_index], masks[frame_index], centers[frame_index] = (
            _draw_demo_frame(config, frame_index, checker_roi)
        )

    captured = simulate_response(
        source,
        config.alpha_rise,
        config.alpha_fall,
        initial=source[0],
    )
    rng = np.random.default_rng(config.seed + 101)
    captured += rng.normal(0.0, config.noise_sigma, captured.shape).astype(np.float32)
    captured = np.clip(captured, 0.0, 1.0)

    motion_masks = masks.any(axis=-1)
    return DemoClip(
        source=source,
        captured=captured,
        motion_masks=motion_masks,
        checker_roi=checker_roi,
        zoom_centers=centers,
        object_centers=centers,
    )
