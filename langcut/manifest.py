from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import LangcutError


_LANGUAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_EXTENSION_RE = re.compile(r"^[A-Za-z0-9]{1,10}$")


@dataclass(frozen=True)
class MediaAssets:
    video: Path
    audio: Path


@dataclass(frozen=True)
class Notice:
    language: str
    label: str
    assets: MediaAssets


@dataclass(frozen=True)
class Chapter:
    id: str
    title: str
    program_frame: int


@dataclass(frozen=True)
class OutputOptions:
    filename: str = "{name}.{language}"
    extension: str = "mp4"
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    pixel_format: str = "yuv420p"
    crf: int = 18
    audio_sample_rate: int = 48000


@dataclass(frozen=True)
class Manifest:
    path: Path
    name: str
    fps: int
    width: int
    height: int
    opening_leader_frames: int
    program: MediaAssets
    notices: dict[str, Notice]
    chapters: tuple[Chapter, ...]
    cut_points: tuple[int, ...]
    output: OutputOptions


def _fail(message: str) -> None:
    raise LangcutError("MANIFEST_INVALID", message)


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{where} must be an object")
    return value


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{where} must be a non-empty string")
    return value.strip()


def _integer(value: Any, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{where} must be an integer >= {minimum}")
    return value


def _path(value: Any, where: str, base: Path) -> Path:
    raw = Path(_string(value, where)).expanduser()
    return raw if raw.is_absolute() else (base / raw).resolve()


def _assets(value: Any, where: str, base: Path) -> MediaAssets:
    if isinstance(value, str):
        path = _path(value, where, base)
        return MediaAssets(video=path, audio=path)
    data = _object(value, where)
    if "video" not in data:
        _fail(f"{where}.video is required")
    video = _path(data["video"], f"{where}.video", base)
    audio = _path(data.get("audio", data["video"]), f"{where}.audio", base)
    return MediaAssets(video=video, audio=audio)


def _dimensions(data: dict[str, Any]) -> tuple[int, int]:
    canvas = data.get("canvas")
    if canvas is None and isinstance(data.get("size"), str):
        match = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", data["size"])
        if not match:
            _fail("size must look like WIDTHxHEIGHT")
        return int(match.group(1)), int(match.group(2))
    canvas = _object(canvas, "canvas")
    width = _integer(canvas.get("width"), "canvas.width", 2)
    height = _integer(canvas.get("height"), "canvas.height", 2)
    if width % 2 or height % 2:
        _fail("canvas width and height must be even")
    return width, height


def _output_options(data: dict[str, Any]) -> OutputOptions:
    raw = _object(data.get("output", {}), "output")
    extension = _string(raw.get("extension", "mp4"), "output.extension").lstrip(".")
    if not _EXTENSION_RE.fullmatch(extension):
        _fail("output.extension must contain only letters and digits")
    template = _string(raw.get("filename", "{name}.{language}"), "output.filename")
    if "{" not in template:
        _fail("output.filename must contain {name} and/or {language}")
    audio_rate = _integer(
        raw.get("audio_sample_rate", 48000), "output.audio_sample_rate", 8000
    )
    crf = _integer(raw.get("crf", 18), "output.crf")
    if crf > 51:
        _fail("output.crf must be <= 51")
    return OutputOptions(
        filename=template,
        extension=extension,
        video_codec=_string(raw.get("video_codec", "libx264"), "output.video_codec"),
        audio_codec=_string(raw.get("audio_codec", "aac"), "output.audio_codec"),
        pixel_format=_string(raw.get("pixel_format", "yuv420p"), "output.pixel_format"),
        crf=crf,
        audio_sample_rate=audio_rate,
    )


def _chapters(data: dict[str, Any]) -> tuple[Chapter, ...]:
    raw = data.get("chapters", [])
    if not isinstance(raw, list):
        _fail("chapters must be an array")
    chapters: list[Chapter] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        where = f"chapters[{index}]"
        record = _object(item, where)
        chapter_id = _string(record.get("id"), f"{where}.id")
        if chapter_id in seen:
            _fail(f"duplicate chapter id: {chapter_id}")
        seen.add(chapter_id)
        frame = _integer(record.get("program_frame"), f"{where}.program_frame")
        chapters.append(
            Chapter(
                id=chapter_id,
                title=_string(record.get("title", chapter_id), f"{where}.title"),
                program_frame=frame,
            )
        )
    return tuple(chapters)


def _cut_points(data: dict[str, Any]) -> tuple[int, ...]:
    raw = data.get("cut_points", [])
    if not isinstance(raw, list):
        _fail("cut_points must be an array")
    points = tuple(_integer(item, f"cut_points[{index}]") for index, item in enumerate(raw))
    if len(points) != len(set(points)):
        _fail("cut_points must not contain duplicates")
    return points


def load_manifest(path: str | Path) -> Manifest:
    manifest_path = Path(path).expanduser().resolve()
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LangcutError(
            "MANIFEST_MISSING", f"manifest not found: {manifest_path}", path=str(manifest_path)
        ) from exc
    except json.JSONDecodeError as exc:
        raise LangcutError(
            "MANIFEST_INVALID",
            f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
            path=str(manifest_path),
        ) from exc
    root = _object(data, "manifest")
    version = _integer(root.get("version", 1), "version", 1)
    if version != 1:
        _fail(f"unsupported manifest version: {version}")
    fps = _integer(root.get("fps"), "fps", 1)
    if fps > 120:
        _fail("fps must be <= 120")
    width, height = _dimensions(root)
    base = manifest_path.parent
    if "program" not in root:
        _fail("program is required")
    program = _assets(root["program"], "program", base)
    notices_raw = _object(root.get("notices"), "notices")
    if not notices_raw:
        _fail("notices must contain at least one language")
    notices: dict[str, Notice] = {}
    for language, item in notices_raw.items():
        if not _LANGUAGE_RE.fullmatch(language):
            _fail(f"invalid language id: {language!r}")
        record = _object(item, f"notices.{language}")
        label = _string(record.get("label", language), f"notices.{language}.label")
        if "video" not in record:
            _fail(f"notices.{language}.video is required")
        notices[language] = Notice(
            language=language,
            label=label,
            assets=_assets(record, f"notices.{language}", base),
        )
    name = _string(root.get("name", manifest_path.stem), "name")
    chapters = _chapters(root)
    cut_points = _cut_points(root)
    return Manifest(
        path=manifest_path,
        name=name,
        fps=fps,
        width=width,
        height=height,
        opening_leader_frames=_integer(
            root.get("opening_leader_frames"), "opening_leader_frames"
        ),
        program=program,
        notices=notices,
        chapters=chapters,
        cut_points=cut_points,
        output=_output_options(root),
    )
