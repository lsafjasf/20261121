from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .color import heatmap_image, image_chroma
from .imageio import GrayImage, RGBImage, read_pgm, read_ppm


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def _error_summary(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": round(sum(values) / len(values), 4),
        "p95": round(_percentile(values, 0.95), 4),
        "max": round(max(values), 4),
    }


def _edge_errors(
    image: RGBImage,
    truth: RGBImage,
    edge_mask: GrayImage,
    visible_mask: GrayImage,
) -> tuple[list[float], list[float], list[float], int]:
    width = image.width
    height = image.height
    chroma = image_chroma(image)
    truth_chroma = image_chroma(truth)
    errors: list[float] = [0.0] * (width * height)
    values: list[float] = []
    visible_values: list[float] = []
    edge_count = 0
    for index in range(width * height):
        if edge_mask.pixels[index] == 0:
            continue
        edge_count += 1
        current_cb, current_cr = chroma[index]
        truth_cb, truth_cr = truth_chroma[index]
        error = math.hypot(current_cb - truth_cb, current_cr - truth_cr)
        errors[index] = error
        values.append(error)
        if visible_mask.pixels[index]:
            visible_values.append(error)
    return errors, values, visible_values, edge_count


@dataclass
class GroupEvaluation:
    group_id: int
    before: dict
    after: dict
    improvement_percent: float
    edge_pixels: int
    visible_edge_pixels: int
    before_heatmap: RGBImage
    after_heatmap: RGBImage

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "edge_pixels": self.edge_pixels,
            "visible_edge_pixels": self.visible_edge_pixels,
            "before": self.before,
            "after": self.after,
            "improvement_percent": round(self.improvement_percent, 2),
        }


def evaluate_group(
    group_id: int,
    before: RGBImage,
    after: RGBImage,
    truth: RGBImage,
    edge_mask: GrayImage,
    visible_mask: GrayImage,
) -> GroupEvaluation:
    before_heat, before_all, before_visible, edge_count = _edge_errors(
        before, truth, edge_mask, visible_mask
    )
    after_heat, after_all, after_visible, _ = _edge_errors(
        after, truth, edge_mask, visible_mask
    )
    before_mean = sum(before_visible) / max(1, len(before_visible))
    after_mean = sum(after_visible) / max(1, len(after_visible))
    if before_mean > 1e-9:
        improvement = (before_mean - after_mean) / before_mean * 100.0
    else:
        improvement = 0.0
    before_summary = {
        "visible_edge_chroma_error": _error_summary(before_visible),
        "all_edge_chroma_error": _error_summary(before_all),
    }
    after_summary = {
        "visible_edge_chroma_error": _error_summary(after_visible),
        "all_edge_chroma_error": _error_summary(after_all),
    }
    before_heatmap = heatmap_image(
        before_heat, before.width, before.height, gain=3.0
    )
    after_heatmap = heatmap_image(after_heat, after.width, after.height, gain=3.0)
    return GroupEvaluation(
        group_id=group_id,
        before=before_summary,
        after=after_summary,
        improvement_percent=improvement,
        edge_pixels=edge_count,
        visible_edge_pixels=len(after_visible),
        before_heatmap=before_heatmap,
        after_heatmap=after_heatmap,
    )


def load_truth(path: str | Path) -> dict:
    truth_path = Path(path)
    metadata = json.loads(truth_path.read_text(encoding="utf-8"))
    metadata["_root"] = str(truth_path.parent)
    return metadata


def truth_group_index(truth: dict) -> dict[int, dict]:
    return {int(group["group_id"]): group for group in truth.get("groups", [])}


def load_truth_assets(
    root: str | Path, group: dict
) -> tuple[RGBImage, GrayImage, GrayImage]:
    root_path = Path(root)
    truth = read_ppm(root_path / group["truth_image"])
    visible = read_pgm(root_path / group["visible_mask"])
    edges = read_pgm(root_path / group["edge_mask"])
    return truth, visible, edges


def aggregate_summary(evaluations: list[GroupEvaluation]) -> dict:
    before_means = [
        evaluation.before["visible_edge_chroma_error"]["mean"]
        for evaluation in evaluations
        if evaluation.visible_edge_pixels
    ]
    after_means = [
        evaluation.after["visible_edge_chroma_error"]["mean"]
        for evaluation in evaluations
        if evaluation.visible_edge_pixels
    ]
    before = sum(before_means) / max(1, len(before_means))
    after = sum(after_means) / max(1, len(after_means))
    improvement = (before - after) / before * 100.0 if before > 1e-9 else 0.0
    return {
        "evaluated_group_count": len(before_means),
        "metric": "mean visible-edge chroma error in CbCr units (0..255 scale)",
        "before_mean": round(before, 4),
        "after_mean": round(after, 4),
        "improvement_percent": round(improvement, 2),
    }
