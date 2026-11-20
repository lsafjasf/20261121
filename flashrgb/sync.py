from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .imageio import GrayImage


CHANNELS = ("R", "G", "B")
IMAGE_SUFFIXES = {".pgm", ".ppm", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class FrameRecord:
    path: Path
    order: int
    channel: str | None
    confidence: float
    cell_means: tuple[float, float, float]

    def to_dict(self, group_id: int | None = None) -> dict:
        payload = {
            "order": self.order,
            "file": self.path.name,
            "channel": self.channel,
            "flash_confidence": round(self.confidence, 4),
            "flash_cell_means": [round(value, 2) for value in self.cell_means],
        }
        if group_id is not None:
            payload["group_id"] = group_id
        return payload


@dataclass
class ExposureGroup:
    frames: dict[str, FrameRecord]
    group_id: int = -1
    warnings: list[str] = field(default_factory=list)

    @property
    def missing(self) -> list[str]:
        return [channel for channel in CHANNELS if channel not in self.frames]

    @property
    def status(self) -> str:
        if not self.missing:
            return "complete"
        return f"missing_{len(self.missing)}:" + "".join(self.missing)

    @property
    def first_order(self) -> int:
        return min(record.order for record in self.frames.values())

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "status": self.status,
            "missing": self.missing,
            "frames": {
                channel: record.to_dict(self.group_id)
                for channel, record in sorted(
                    self.frames.items(), key=lambda item: CHANNELS.index(item[0])
                )
            },
            "warnings": self.warnings,
        }


def natural_key(path: Path) -> tuple:
    parts = re.split(r"(\d+)", path.name.lower())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def discover_frames(path_value: str) -> list[Path]:
    path = Path(path_value)
    if path.is_dir():
        paths = [
            item
            for item in path.iterdir()
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        ]
    elif path.is_file():
        paths = [path]
    else:
        import glob

        paths = [Path(item) for item in glob.glob(path_value) if Path(item).is_file()]
    paths.sort(key=natural_key)
    if not paths:
        raise ValueError(f"no image frames found at {path_value!r}")
    return paths


def parse_roi(value: str | None, width: int, height: int) -> tuple[int, int, int, int]:
    if value is None or value.lower() == "auto":
        return (0, 0, min(width, 96), min(height, 32))
    fields = [int(part.strip()) for part in value.split(",")]
    if len(fields) != 4:
        raise ValueError("flash ROI must be X,Y,WIDTH,HEIGHT")
    x, y, roi_width, roi_height = fields
    if roi_width < 3 or roi_height < 1:
        raise ValueError("flash ROI is too small")
    if x < 0 or y < 0 or x + roi_width > width or y + roi_height > height:
        raise ValueError("flash ROI lies outside the frame")
    return x, y, roi_width, roi_height


def _cell_mean(image: GrayImage, x0: int, y0: int, x1: int, y1: int) -> float:
    total = 0
    count = 0
    for y in range(y0, y1):
        start = y * image.width + x0
        total += sum(image.pixels[start : start + (x1 - x0)])
        count += x1 - x0
    return total / max(1, count)


def decode_flash(
    image: GrayImage,
    roi: tuple[int, int, int, int],
    min_confidence: float = 0.025,
) -> tuple[str | None, float, tuple[float, float, float]]:
    x, y, width, height = roi
    boundaries = [
        (x + (width * index) // 3, x + (width * (index + 1)) // 3)
        for index in range(3)
    ]
    means = tuple(_cell_mean(image, left, y, right, y + height) for left, right in boundaries)
    ranked = sorted(range(3), key=lambda index: means[index], reverse=True)
    strongest = means[ranked[0]]
    second = means[ranked[1]]
    confidence = (strongest - second) / max(1.0, strongest)
    if strongest < 8.0 or confidence < min_confidence:
        return None, confidence, means
    return CHANNELS[ranked[0]], confidence, means


def _records_to_group(records: list[FrameRecord]) -> ExposureGroup:
    selected: dict[str, FrameRecord] = {}
    warnings: list[str] = []
    for record in records:
        channel = record.channel
        if channel is None:
            continue
        if channel in selected:
            previous = selected[channel]
            warnings.append(
                f"duplicate {channel} exposure; kept {previous.path.name}, ignored {record.path.name}"
            )
            if record.confidence > previous.confidence:
                selected[channel] = record
        else:
            selected[channel] = record
    return ExposureGroup(frames=selected, warnings=warnings)


def group_exposures(records: list[FrameRecord]) -> list[ExposureGroup]:
    groups: list[ExposureGroup] = []
    pending: list[FrameRecord] = []
    for record in records:
        if record.channel is None:
            continue
        if record.channel == "R":
            if pending:
                groups.append(_records_to_group(pending))
                pending = []
            pending.append(record)
            continue
        pending.append(record)
        if record.channel == "B":
            groups.append(_records_to_group(pending))
            pending = []
    if pending:
        groups.append(_records_to_group(pending))
    for group_id, group in enumerate(groups):
        group.group_id = group_id
    return groups


def build_sync_report(
    records: list[FrameRecord],
    groups: list[ExposureGroup],
    roi: tuple[int, int, int, int],
) -> dict:
    frame_groups: dict[int, int] = {}
    for group in groups:
        for record in group.frames.values():
            frame_groups[record.order] = group.group_id
    return {
        "flash_roi": list(roi),
        "flash_encoding": "spatial 3-cell code: left=R, center=G, right=B",
        "total_frames": len(records),
        "decoded_frames": sum(record.channel is not None for record in records),
        "groups": [group.to_dict() for group in groups],
        "frames": [
            record.to_dict(frame_groups.get(record.order)) for record in records
        ],
    }


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
