"""Constant-frame-rate MP4 encoding through the system ffmpeg binary."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Protocol

from .model import Timeline


class FrameSource(Protocol):
    def render(self, frame_index: int) -> bytes: ...


def require_ffmpeg() -> str:
    binary = shutil.which("ffmpeg")
    if not binary:
        raise RuntimeError(
            "ffmpeg was not found on PATH; install ffmpeg to encode MP4 output"
        )
    return binary


def encode_mp4(
    output_path: str | Path,
    wav_path: str | Path,
    renderer: FrameSource,
    timeline: Timeline,
    *,
    width: int,
    height: int,
    supersample: int,
    crf: int = 18,
    preset: str = "medium",
    progress: Callable[[int, int], None] | None = None,
) -> None:
    """Encode every frame as CFR video and mux the cue-tone WAV."""

    ffmpeg = require_ffmpeg()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    internal_width, internal_height = width * supersample, height * supersample
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{internal_width}x{internal_height}",
        "-framerate",
        str(timeline.fps),
        "-i",
        "pipe:0",
        "-i",
        str(wav_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-vf",
        f"scale={width}:{height}:flags=lanczos,format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(timeline.fps),
        "-fps_mode",
        "cfr",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-ac",
        "1",
        "-movflags",
        "+faststart",
        str(output),
    ]
    with tempfile.TemporaryFile() as error_log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=error_log,
            stdout=subprocess.DEVNULL,
        )
        assert process.stdin is not None
        try:
            for frame_index in range(timeline.frame_count):
                process.stdin.write(renderer.render(frame_index))
                if progress:
                    progress(frame_index + 1, timeline.frame_count)
            process.stdin.close()
        except BrokenPipeError:
            pass
        return_code = process.wait()
        if return_code != 0:
            error_log.seek(0)
            details = error_log.read().decode("utf-8", "replace").strip()
            raise RuntimeError(f"ffmpeg failed ({return_code}): {details}")
