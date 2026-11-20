"""Stream readers/writers for PGM sequences and raw gray8 streams."""

from __future__ import annotations

import glob
import os

from .imageio import VideoFrame


def _read_pgm(path: str) -> tuple[bytes, int, int]:
    with open(path, "rb") as fh:
        magic = fh.readline().strip()
        if magic != b"P5":
            raise ValueError(f"{path}: only binary P5 PGM is supported")
        header = []
        while len(header) < 3:
            line = fh.readline()
            if not line:
                raise ValueError(f"{path}: truncated PGM header")
            for token in line.split():
                header.append(token)
                if len(header) == 3:
                    break
        width, height, _maxval = (int(x) for x in header)
        data = fh.read(width * height)
    if len(data) != width * height:
        raise ValueError(f"{path}: truncated pixel data")
    return data, width, height


def _iter_pgm_stream(path: str):
    """Yield (pixels, width, height) for every PGM frame concatenated in one
    file (e.g. ffmpeg ``-f image2pipe -vcodec pgmyuv`` style output)."""
    with open(path, "rb") as fh:
        while True:
            magic = fh.readline()
            if not magic:
                return
            if magic.strip() != b"P5":
                raise ValueError(f"{path}: bad PGM magic {magic!r}")
            header: list[bytes] = []
            while len(header) < 3:
                line = fh.readline()
                if not line:
                    raise ValueError(f"{path}: truncated PGM header")
                for token in line.split():
                    header.append(token)
                    if len(header) == 3:
                        break
            width, height, _maxval = (int(x) for x in header)
            data = fh.read(width * height)
            if len(data) != width * height:
                raise ValueError(f"{path}: truncated pixel data")
            yield data, width, height


def load_frames(path: str, width: int | None = None,
                height: int | None = None) -> list[VideoFrame]:
    """Load an input clip as full-height gray8 frames.

    ``path`` may be a glob of ``.pgm`` frames, a single concatenated PGM
    stream, or a raw gray8 file (geometry required).
    """
    frames: list[VideoFrame] = []
    if any(ch in path for ch in "*?["):
        names = sorted(glob.glob(path))
        if not names:
            raise FileNotFoundError(path)
        for i, name in enumerate(names):
            data, w, h = _read_pgm(name)
            frames.append(VideoFrame(i, w, h, data, {"input": name}))
        return frames

    ext = os.path.splitext(path)[1].lower()
    if ext in (".pgm", ".pnm"):
        for i, (data, w, h) in enumerate(_iter_pgm_stream(path)):
            frames.append(VideoFrame(i, w, h, data, {"input": path, "offset": i}))
        return frames

    if ext in (".gray", ".raw", ".yuv", ".y"):
        if not width or not height:
            raise ValueError("raw gray input requires --width/--height")
        frame_size = width * height
        with open(path, "rb") as fh:
            blob = fh.read()
        if len(blob) % frame_size:
            raise ValueError("raw input size is not a whole number of frames")
        for i in range(len(blob) // frame_size):
            chunk = blob[i * frame_size:(i + 1) * frame_size]
            frames.append(VideoFrame(i, width, height, chunk,
                                     {"input": path, "offset": i}))
        return frames

    raise ValueError(f"unsupported input extension: {ext}")


def write_pgm(frame: VideoFrame, path: str) -> None:
    header = f"P5\n{frame.width} {frame.height}\n255\n".encode()
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(frame.buf)


def write_raw(frames: list[VideoFrame], path: str) -> None:
    with open(path, "wb") as fh:
        for frame in frames:
            fh.write(frame.buf)
