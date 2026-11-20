from __future__ import annotations

import re
import struct
import subprocess
import zlib
from pathlib import Path


class ImageError(RuntimeError):
    pass


class GrayImage:
    __slots__ = ("width", "height", "pixels")

    def __init__(self, width: int, height: int, pixels: bytes | bytearray):
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        if len(pixels) != width * height:
            raise ValueError("pixel count does not match dimensions")
        self.width = width
        self.height = height
        self.pixels = bytearray(pixels)

    def get(self, x: int, y: int) -> int:
        return self.pixels[y * self.width + x]

    def copy(self) -> "GrayImage":
        return GrayImage(self.width, self.height, self.pixels)


class RGBImage:
    __slots__ = ("width", "height", "pixels")

    def __init__(self, width: int, height: int, pixels: bytes | bytearray):
        if len(pixels) != width * height * 3:
            raise ValueError("pixel count does not match dimensions")
        self.width = width
        self.height = height
        self.pixels = bytearray(pixels)


def _next_token(data: bytes, pos: int) -> tuple[bytes, int]:
    n = len(data)
    while pos < n:
        if data[pos] in b" \t\r\n":
            pos += 1
        elif data[pos] == 35:
            end = data.find(b"\n", pos)
            pos = n if end < 0 else end + 1
        else:
            break
    start = pos
    while pos < n and data[pos] not in b" \t\r\n":
        pos += 1
    if start == pos:
        raise ImageError("truncated PNM header")
    return data[start:pos], pos


def read_pgm(path: str | Path) -> GrayImage:
    data = Path(path).read_bytes()
    if not data.startswith((b"P2", b"P5")):
        raise ImageError(f"not a PGM file: {path}")
    magic, pos = _next_token(data, 0)
    width_token, pos = _next_token(data, pos)
    height_token, pos = _next_token(data, pos)
    max_token, pos = _next_token(data, pos)
    width = int(width_token)
    height = int(height_token)
    max_value = int(max_token)
    if width <= 0 or height <= 0 or max_value <= 0 or max_value > 65535:
        raise ImageError(f"invalid PGM header: {path}")
    count = width * height
    if magic == b"P5":
        if pos < len(data) and data[pos] in b" \t\r\n":
            pos += 1
        bytes_per_sample = 1 if max_value < 256 else 2
        raw = data[pos : pos + count * bytes_per_sample]
        if len(raw) != count * bytes_per_sample:
            raise ImageError(f"truncated PGM payload: {path}")
        if bytes_per_sample == 1:
            pixels = raw if max_value == 255 else bytes(
                (value * 255 + max_value // 2) // max_value for value in raw
            )
        else:
            samples = struct.unpack(f">{count}H", raw)
            pixels = bytes((value * 255 + max_value // 2) // max_value for value in samples)
    else:
        values: list[int] = []
        while len(values) < count:
            token, pos = _next_token(data, pos)
            if not token:
                raise ImageError(f"truncated plain PGM: {path}")
            values.append(int(token))
        pixels = bytes((value * 255 + max_value // 2) // max_value for value in values)
    return GrayImage(width, height, pixels)


def write_pgm(path: str | Path, image: GrayImage) -> None:
    header = f"P5\n{image.width} {image.height}\n255\n".encode("ascii")
    Path(path).write_bytes(header + bytes(image.pixels))


def write_ppm(path: str | Path, image: RGBImage) -> None:
    header = f"P6\n{image.width} {image.height}\n255\n".encode("ascii")
    Path(path).write_bytes(header + bytes(image.pixels))


def read_ppm(path: str | Path) -> RGBImage:
    data = Path(path).read_bytes()
    if not data.startswith(b"P6"):
        raise ImageError(f"not a binary PPM file: {path}")
    magic, pos = _next_token(data, 0)
    width_token, pos = _next_token(data, pos)
    height_token, pos = _next_token(data, pos)
    max_token, pos = _next_token(data, pos)
    width = int(width_token)
    height = int(height_token)
    max_value = int(max_token)
    if magic != b"P6" or max_value != 255:
        raise ImageError(f"unsupported PPM format: {path}")
    if pos < len(data) and data[pos] in b" \t\r\n":
        pos += 1
    count = width * height * 3
    pixels = data[pos : pos + count]
    if len(pixels) != count:
        raise ImageError(f"truncated PPM payload: {path}")
    return RGBImage(width, height, pixels)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def write_png(path: str | Path, image: RGBImage) -> None:
    stride = image.width * 3
    rows = bytearray()
    for y in range(image.height):
        start = y * stride
        rows.append(0)
        rows.extend(image.pixels[start : start + stride])
    header = struct.pack(">IIBBBBB", image.width, image.height, 8, 2, 0, 0, 0)
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(bytes(rows), 6))
        + _png_chunk(b"IEND", b"")
    )
    Path(path).write_bytes(payload)


def probe_size(path: str | Path) -> tuple[int, int]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise ImageError(result.stderr.strip() or f"cannot probe {path}")
    match = re.search(r"(\d+)x(\d+)", result.stdout)
    if not match:
        raise ImageError(f"cannot determine image size: {path}")
    return int(match.group(1)), int(match.group(2))


def read_gray(path: str | Path) -> GrayImage:
    source = Path(path)
    if source.suffix.lower() == ".pgm":
        return read_pgm(source)
    width, height = probe_size(source)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        raise ImageError(result.stderr.decode("utf-8", "replace").strip())
    return GrayImage(width, height, result.stdout)


class VideoWriter:
    def __init__(self, path: str | Path, width: int, height: int, fps: float):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(self.path),
        ]
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.width = width
        self.height = height

    def write(self, image: RGBImage) -> None:
        if image.width != self.width or image.height != self.height:
            raise ImageError("frame size changed while writing video")
        if self._process.stdin is None:
            raise ImageError("video encoder is closed")
        self._process.stdin.write(image.pixels)

    def close(self) -> None:
        process = self._process
        if process.stdin is not None:
            process.stdin.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        if process.stderr is not None:
            process.stderr.close()
        code = process.wait()
        if code != 0:
            raise ImageError(stderr.decode("utf-8", "replace").strip() or "ffmpeg failed")
