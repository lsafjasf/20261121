"""Residual maps and truth-vs-estimate comparison plots (OpenCV only)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _color_map(gray: np.ndarray) -> np.ndarray:
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_TURBO)


def residual_map(
    observed: np.ndarray,
    model: np.ndarray,
    out_path: str | Path,
    title: str = "row profile residual",
) -> np.ndarray:
    """Write a 3-panel residual image: observed, model, signed residual."""
    observed = np.asarray(observed, dtype=np.float64)
    model = np.asarray(model, dtype=np.float64)
    residual = observed - model
    scale = max(float(np.nanmax(np.abs(residual))), 1e-9)
    signed = np.clip(128.0 + 127.0 * residual / scale, 0, 255).astype(np.uint8)

    panels = [
        _color_map(observed),
        _color_map(model),
        cv2.applyColorMap(signed, cv2.COLORMAP_BONE),
    ]
    labels = ("observed", "fitted model", "residual (signed)")
    annotated = []
    for panel, label in zip(panels, labels):
        canvas = cv2.copyMakeBorder(
            panel, 28, 4, 4, 4, cv2.BORDER_CONSTANT, value=(20, 20, 20)
        )
        cv2.putText(canvas, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        annotated.append(canvas)
    figure = np.hstack(annotated)
    cv2.putText(figure, title, (8, figure.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1, cv2.LINE_AA)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), figure)
    return figure


def profile_comparison(
    observed: np.ndarray,
    model: np.ndarray,
    out_path: str | Path,
) -> np.ndarray:
    """Line plot of measured vs fitted row profile and their residual."""
    observed = np.asarray(observed, dtype=np.float64)
    model = np.asarray(model, dtype=np.float64)
    rows = np.arange(observed.size)
    height, width = 360, 900
    margin = 40
    canvas = np.full((height, width, 3), 255, np.uint8)

    def to_xy(values: np.ndarray) -> np.ndarray:
        vmin, vmax = np.min(observed), np.max(observed)
        span = max(vmax - vmin, 1e-9)
        x = margin + rows / max(rows.size - 1, 1) * (width - 2 * margin)
        y = height - margin - (values - vmin) / span * (height - 2 * margin)
        return np.column_stack([x, y]).astype(np.int32)

    cv2.polylines(canvas, [to_xy(observed)], False, (40, 40, 220), 1, cv2.LINE_AA)
    cv2.polylines(canvas, [to_xy(model)], False, (30, 160, 60), 1, cv2.LINE_AA)
    cv2.putText(canvas, "observed (red) vs model (green)", (margin, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return canvas


def truth_comparison(
    truth: dict, estimate: dict, out_path: str | Path
) -> np.ndarray:
    """Bar-style truth-vs-estimate chart for the three scalar outputs."""
    metrics = [
        ("line time (us)", truth["t_line_us"], estimate["t_line_us"]),
        ("exposure (us)", truth["t_exp_us"], estimate["t_exp_us"]),
    ]
    height, width = 300, 800
    canvas = np.full((height, width, 3), 255, np.uint8)
    group_w = width // len(metrics)
    max_value = max(max(abs(t), abs(e)) for _, t, e in metrics)
    for i, (name, true_v, est_v) in enumerate(metrics):
        base_x = int(group_w * (i + 0.5))
        for j, (value, color) in enumerate(
            ((true_v, (180, 90, 40)), (est_v, (40, 140, 60)))
        ):
            bar_h = int(abs(value) / max(max_value, 1e-9) * 180)
            x = base_x - 55 + j * 65
            cv2.rectangle(canvas, (x, 240 - bar_h), (x + 45, 240), color, -1)
            if bar_h >= 28:
                label_y = 240 - bar_h + 18
                label_color = (255, 255, 255)
            else:
                label_y = 240 - bar_h - 6
                label_color = (0, 0, 0)
            cv2.putText(canvas, f"{value:.2f}", (x - 2, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1, cv2.LINE_AA)
        cv2.putText(canvas, name, (base_x - 70, 285),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    direction_text = (
        f"scan direction  truth={truth['scan_direction']} "
        f"estimate={estimate['scan_direction']}"
    )
    cv2.putText(canvas, direction_text, (40, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return canvas
