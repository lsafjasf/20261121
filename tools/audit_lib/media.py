from __future__ import annotations

import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Iterable


class MediaError(RuntimeError):
    pass


def require_binary(name: str) -> str:
    from shutil import which

    path = which(name)
    if not path:
        raise MediaError(f"找不到 {name}，请先安装 FFmpeg。")
    return path


def run(args: list[str], input_bytes: bytes | None = None) -> bytes:
    process = subprocess.run(
        args,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode:
        message = process.stderr.decode("utf-8", "replace").strip()
        raise MediaError(f"命令失败：{' '.join(args)}\n{message}")
    return process.stdout


def probe_video(path: Path) -> dict:
    require_binary("ffprobe")
    output = run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames:format=duration",
        "-of", "json", str(path),
    ])
    data = json.loads(output)
    streams = data.get("streams") or []
    if not streams:
        raise MediaError(f"{path} 中没有视频流。")
    stream = streams[0]
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "24/1"
    fps = float(Fraction(rate))
    if not math.isfinite(fps) or fps <= 0:
        raise MediaError(f"{path} 的帧率无效：{rate}")
    duration = float((data.get("format") or {}).get("duration") or 0)
    if duration <= 0:
        duration = float(stream.get("duration") or 0)
    if duration <= 0:
        raise MediaError(f"{path} 的时长无效。")
    frame_count = int(stream.get("nb_frames") or 0)
    if frame_count <= 0:
        frame_count = int(round(duration * fps))
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "duration": duration,
        "frame_count": frame_count,
    }


def read_gray(path: Path, width: int, height: int) -> bytes:
    data = run([
        "ffmpeg", "-v", "error", "-i", str(path),
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ])
    expected = width * height
    if len(data) != expected:
        raise MediaError(
            f"掩模 {path} 解码后为 {len(data)} 字节，预期 {expected} 字节。"
        )
    return data


def dilate_binary(mask: bytes, width: int, height: int, radius: int) -> bytes:
    if radius < 0:
        raise ValueError("radius 不能为负数")
    source = bytearray(1 if value >= 128 else 0 for value in mask)
    if radius == 0:
        return bytes(255 if value else 0 for value in source)

    horizontal = bytearray(width * height)
    for y in range(height):
        offset = y * width
        prefix = [0] * (width + 1)
        for x in range(width):
            prefix[x + 1] = prefix[x] + source[offset + x]
        for x in range(width):
            left = max(0, x - radius)
            right = min(width, x + radius + 1)
            horizontal[offset + x] = 1 if prefix[right] - prefix[left] else 0

    dilated = bytearray(width * height)
    for x in range(width):
        prefix = [0] * (height + 1)
        for y in range(height):
            prefix[y + 1] = prefix[y] + horizontal[y * width + x]
        for y in range(height):
            top = max(0, y - radius)
            bottom = min(height, y + radius + 1)
            dilated[y * width + x] = 1 if prefix[bottom] - prefix[top] else 0
    return bytes(255 if value else 0 for value in dilated)


def make_rgba_layer(
    alpha: bytes,
    width: int,
    height: int,
    color: tuple[int, int, int],
) -> bytes:
    if len(alpha) != width * height:
        raise ValueError("alpha 尺寸不匹配")
    red, green, blue = color
    rgba = bytearray(width * height * 4)
    offset = 0
    for value in alpha:
        rgba[offset] = red
        rgba[offset + 1] = green
        rgba[offset + 2] = blue
        rgba[offset + 3] = value
        offset += 4
    return bytes(rgba)


def save_rgba_png(data: bytes, width: int, height: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{width}x{height}",
        "-i", "-", "-frames:v", "1", str(path),
    ], data)


def compose_videos(
    base: Path,
    fps: float,
    duration: float,
    cue_layers: Iterable[dict],
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    inputs = ["ffmpeg", "-v", "error", "-y", "-i", str(base)]
    filters: list[str] = []
    current = "0:v"
    index = 1
    for cue in cue_layers:
        for layer in cue["layers"]:
            inputs.extend([
                "-loop", "1", "-framerate", f"{fps:g}", "-i", str(layer["path"]),
            ])
            x, y = layer["x"], layer["y"]
            enable = f"between(t,{cue['start']:.6f},{cue['end']:.6f})"
            target = f"v{index}"
            filters.append(
                f"[{current}][{index}:v]overlay=x={x}:y={y}:"
                f"enable='{enable}':format=auto[{target}]"
            )
            current = target
            index += 1
    if not filters:
        raise MediaError("没有可合成的字幕层。")
    command = inputs + [
        "-filter_complex", ";".join(filters), "-map", f"[{current}]",
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-t", f"{duration:.6f}", str(output),
    ]
    run(command)


def iter_crop_frames(
    video: Path,
    bbox: tuple[int, int, int, int],
    margin: int,
    frame_count: int,
    video_width: int,
    video_height: int,
) -> Iterable[bytes]:
    x, y, width, height = bbox
    crop_x = max(0, x - margin)
    crop_y = max(0, y - margin)
    crop_right = min(video_width, x + width + margin)
    crop_bottom = min(video_height, y + height + margin)
    crop_width = crop_right - crop_x
    crop_height = crop_bottom - crop_y
    frame_size = video_width * video_height * 3
    process = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-i", str(video),
            "-fps_mode", "passthrough",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    try:
        for _ in range(frame_count):
            frame = process.stdout.read(frame_size)
            if len(frame) != frame_size:
                break
            rows = (
                frame[(row * video_width + crop_x) * 3:
                      (row * video_width + crop_x + crop_width) * 3]
                for row in range(crop_y, crop_y + crop_height)
            )
            yield b"".join(rows)
    finally:
        process.stdout.close()
        process.wait()
        if process.returncode not in (0, -13, -15):
            error = process.stderr.read().decode("utf-8", "replace")
            if error.strip() and "Broken pipe" not in error:
                raise MediaError(error.strip())


def global_luma_series(video: Path, frame_count: int) -> list[float]:
    info = probe_video(video)
    frame_size = info["width"] * info["height"] * 3
    values = []
    process = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-i", str(video),
            "-fps_mode", "passthrough",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    try:
        for _ in range(frame_count):
            frame = process.stdout.read(frame_size)
            if len(frame) != frame_size:
                break
            sample = frame[::96]
            values.append(sum(sample) / (len(sample) * 255.0))
    finally:
        process.stdout.close()
        process.wait()
        if process.returncode:
            error = process.stderr.read().decode("utf-8", "replace")
            if error.strip():
                raise MediaError(error.strip())
    return values


def extract_frame(video: Path, time: float, output: Path, width: int = 800) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, time):.6f}",
        "-i", str(video), "-frames:v", "1",
        "-vf", f"scale={width}:-2:flags=lanczos",
        "-q:v", "3", str(output),
    ])
