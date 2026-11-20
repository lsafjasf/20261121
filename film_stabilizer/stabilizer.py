from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import cv2
import numpy as np

from .crop import Rect, common_valid_mask, largest_rectangle, valid_mask
from .detector import DetectorConfig, PerforationDetector
from .io import FrameSource
from .tracker import TrackResult, TrackerConfig, track_records
from .types import FrameRecord, SequenceInfo


@dataclass
class StabilizerConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)


@dataclass
class StabilizationResult:
    info: SequenceInfo
    records: list[FrameRecord]
    track: TrackResult
    common_crop: Rect

    @property
    def splice_frames(self) -> list[int]:
        return self.track.splice_frames


def analyze_records(
    info: SequenceInfo,
    records: list[FrameRecord],
    config: StabilizerConfig | None = None,
) -> StabilizationResult:
    config = config or StabilizerConfig()
    track = track_records(records, config.tracker)
    size = (info.width, info.height)
    for record in records:
        mask = valid_mask(record.matrix, size)
        record.crop = largest_rectangle(mask)
    common_mask = common_valid_mask([record.matrix for record in records], size)
    return StabilizationResult(
        info=info,
        records=records,
        track=track,
        common_crop=largest_rectangle(common_mask),
    )


def stabilize(
    source: FrameSource, config: StabilizerConfig | None = None
) -> StabilizationResult:
    config = config or StabilizerConfig()
    detector = PerforationDetector(config.detector)
    records: list[FrameRecord] = []
    for index, source_name, original_frame, frame in source.iter_frames():
        records.append(
            FrameRecord(
                index=index,
                source_name=source_name,
                original_frame=original_frame,
                detections=detector.detect(frame),
            )
        )
    if not records:
        raise ValueError("input contains no frames")
    return analyze_records(source.info, records, config)


def render_frames(
    source: FrameSource, result: StabilizationResult
) -> Iterator[np.ndarray]:
    size = (result.info.width, result.info.height)
    for record, (_, _, _, frame) in zip(result.records, source.iter_frames()):
        if record.matrix is None:
            raise RuntimeError("stabilization matrix missing")
        yield cv2.warpAffine(
            frame,
            record.matrix[:2, :],
            size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )


def render_cropped_frames(
    source: FrameSource, result: StabilizationResult, crop: Rect
) -> Iterator[np.ndarray]:
    x, y, width, height = crop
    if width <= 0 or height <= 0:
        raise ValueError("crop rectangle is empty")
    for frame in render_frames(source, result):
        yield frame[y : y + height, x : x + width]
