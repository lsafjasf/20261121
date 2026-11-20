from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import LangcutError


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    video_duration: float
    source_fps: float
    source_nb_frames: int | None

    def frame_count(self, target_fps: int) -> int:
        same_rate = abs(self.source_fps - target_fps) < 0.001
        if same_rate and self.source_nb_frames is not None:
            return max(1, self.source_nb_frames)
        return max(1, round(self.video_duration * target_fps))


@dataclass(frozen=True)
class AudioInfo:
    path: Path
    duration: float
    sample_rate: int
    channels: int


def require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise LangcutError("FFMPEG_MISSING", f"{name} was not found on PATH")
    return path


def _parse_rate(raw: str | None) -> float:
    if not raw or raw in {"0/0", "N/A"}:
        return 0.0
    parts = raw.split("/", 1)
    try:
        numerator = float(parts[0])
        denominator = float(parts[1]) if len(parts) == 2 else 1.0
    except ValueError:
        return 0.0
    return numerator / denominator if denominator else 0.0


def _float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _probe(path: Path, entries: str, stream: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise LangcutError("ASSET_MISSING", f"asset not found: {path}", path=str(path))
    ffprobe = require_binary("ffprobe")
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        entries,
        "-of",
        "json",
    ]
    if stream:
        command.extend(["-select_streams", stream])
    command.append(str(path))
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        message = completed.stderr.strip() or "ffprobe failed"
        raise LangcutError(
            "MEDIA_INVALID", f"cannot read media {path}: {message}", path=str(path)
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise LangcutError(
            "MEDIA_INVALID", f"ffprobe returned invalid JSON for {path}", path=str(path)
        ) from exc
    return payload


def probe_video(path: Path) -> MediaInfo:
    payload = _probe(
        path,
        (
            "format=duration:stream=codec_type,avg_frame_rate,"
            "r_frame_rate,duration,nb_frames"
        ),
        "v:0",
    )
    streams = payload.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise LangcutError(
            "MEDIA_INVALID", f"no video stream in {path}", path=str(path)
        )
    format_duration = _float(payload.get("format", {}).get("duration"))
    video_duration = _float(video.get("duration")) or format_duration
    if video_duration is None:
        raise LangcutError(
            "MEDIA_INVALID", f"video stream has no duration: {path}", path=str(path)
        )
    source_fps = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(
        video.get("r_frame_rate")
    )
    if source_fps <= 0:
        raise LangcutError(
            "MEDIA_INVALID", f"video stream has no frame rate: {path}", path=str(path)
        )
    return MediaInfo(
        path=path,
        video_duration=video_duration,
        source_fps=source_fps,
        source_nb_frames=_int(video.get("nb_frames")),
    )


def probe_audio(path: Path) -> AudioInfo:
    payload = _probe(
        path,
        "format=duration:stream=codec_type,duration,sample_rate,channels",
        "a:0",
    )
    streams = payload.get("streams", [])
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if audio is None:
        raise LangcutError(
            "MEDIA_INVALID", f"no audio stream in {path}", path=str(path)
        )
    duration = _float(audio.get("duration")) or _float(
        payload.get("format", {}).get("duration")
    )
    if duration is None:
        raise LangcutError(
            "MEDIA_INVALID", f"audio stream has no duration: {path}", path=str(path)
        )
    sample_rate = _int(audio.get("sample_rate"))
    channels = _int(audio.get("channels"))
    if sample_rate is None or channels is None:
        raise LangcutError(
            "MEDIA_INVALID", f"audio stream has no sample layout: {path}", path=str(path)
        )
    return AudioInfo(
        path=path, duration=duration, sample_rate=sample_rate, channels=channels
    )
