from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .types import SequenceInfo

IMAGE_SUFFIXES = {
    ".bmp",
    ".dib",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def natural_key(value: str) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]


class FrameSource:
    """Uniform reader for an image sequence or a decoded video."""

    info: SequenceInfo

    def iter_frames(self) -> Iterator[tuple[int, str, int, np.ndarray]]:
        """Yield ``(index, source_name, original_frame, bgr_frame)``."""
        raise NotImplementedError


class ImageSequenceSource(FrameSource):
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.paths = sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ),
            key=lambda path: natural_key(path.name),
        )
        if not self.paths:
            raise ValueError(f"no supported images found in {directory}")
        first = cv2.imread(str(self.paths[0]), cv2.IMREAD_COLOR)
        if first is None:
            raise ValueError(f"cannot read image: {self.paths[0]}")
        height, width = first.shape[:2]
        self.info = SequenceInfo(
            width=width,
            height=height,
            fps=24.0,
            frame_count=len(self.paths),
            source=str(directory),
        )
        self._numbers = self._frame_numbers()

    def _frame_numbers(self) -> list[int]:
        suffixes = []
        for path in self.paths:
            match = re.search(r"(\d+)$", path.stem)
            suffixes.append(int(match.group(1)) if match else None)
        if all(value is not None for value in suffixes):
            values = [int(value) for value in suffixes]
            if len(set(values)) == len(values):
                return values
        return list(range(len(self.paths)))

    def iter_frames(self) -> Iterator[tuple[int, str, int, np.ndarray]]:
        for index, path in enumerate(self.paths):
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"cannot read image: {path}")
            if frame.shape[1] != self.info.width or frame.shape[0] != self.info.height:
                raise ValueError(
                    f"frame size changed at {path}: "
                    f"{frame.shape[1]}x{frame.shape[0]}"
                )
            yield index, path.name, self._numbers[index], frame


class VideoSource(FrameSource):
    def __init__(self, path: Path, frame_number_offset: int = 0) -> None:
        self.path = path
        self.frame_number_offset = frame_number_offset
        capture = cv2.VideoCapture(str(path))
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            raise ValueError(f"cannot decode video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            fps = 24.0
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        height, width = frame.shape[:2]
        capture.release()
        self.info = SequenceInfo(
            width=width,
            height=height,
            fps=fps,
            frame_count=count,
            source=str(path),
        )

    def iter_frames(self) -> Iterator[tuple[int, str, int, np.ndarray]]:
        capture = cv2.VideoCapture(str(self.path))
        try:
            index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame.shape[1] != self.info.width or frame.shape[0] != self.info.height:
                    raise ValueError("video frame dimensions changed mid-stream")
                original = index + self.frame_number_offset
                yield index, f"frame_{original:06d}", original, frame
                index += 1
        finally:
            capture.release()


def open_source(path: str | Path, frame_number_offset: int = 0) -> FrameSource:
    resolved = Path(path)
    if resolved.is_dir():
        return ImageSequenceSource(resolved)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return VideoSource(resolved, frame_number_offset=frame_number_offset)


def write_video(
    path: Path,
    frames: Iterator[np.ndarray],
    fps: float,
    size: tuple[int, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer for {path}")
    try:
        for frame in frames:
            if (frame.shape[1], frame.shape[0]) != size:
                raise ValueError("frame dimensions do not match video size")
            writer.write(frame)
    finally:
        writer.release()


def write_manifest(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
