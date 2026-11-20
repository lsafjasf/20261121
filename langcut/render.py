from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .errors import LangcutError
from .manifest import Manifest, Notice, OutputOptions
from .media import MediaInfo, probe_audio, probe_video, require_binary
from .timeline import Timeline, timecode


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._\-\u4e00-\u9fff]+", "_", value).strip("._")
    return cleaned or "program"


def _filename_stem(
    output: OutputOptions, name: str, language: str, report_names: set[str]
) -> str:
    try:
        rendered = output.filename.format(
            name=_safe_name(name), language=language, lang=language
        )
    except (KeyError, ValueError) as exc:
        raise LangcutError(
            "MANIFEST_INVALID",
            "output.filename may only use {name}, {language}, and {lang}",
        ) from exc
    if not rendered or Path(rendered).name != rendered or rendered in {".", ".."}:
        raise LangcutError(
            "MANIFEST_INVALID", "output.filename must produce a plain file name"
        )
    if f"{rendered}.{output.extension}" in report_names:
        raise LangcutError(
            "MANIFEST_INVALID",
            f"output filename collides with a report file: {rendered}",
        )
    return rendered


def _preflight(manifest: Manifest, notices: list[Notice]) -> None:
    paths = [manifest.program.video, manifest.program.audio]
    for notice in notices:
        paths.extend([notice.assets.video, notice.assets.audio])
    missing = sorted({path for path in paths if not path.is_file()})
    if missing:
        raise LangcutError(
            "ASSET_MISSING",
            "missing asset(s): " + ", ".join(str(path) for path in missing),
            paths=[str(path) for path in missing],
        )


def _probe_inputs(
    manifest: Manifest, notices: list[Notice]
) -> tuple[MediaInfo, dict[str, MediaInfo]]:
    video_cache: dict[Path, MediaInfo] = {}
    audio_cache: dict[Path, AudioInfo] = {}

    def video(path: Path) -> MediaInfo:
        if path not in video_cache:
            video_cache[path] = probe_video(path)
        return video_cache[path]

    def audio(path: Path) -> AudioInfo:
        if path not in audio_cache:
            audio_cache[path] = probe_audio(path)
        return audio_cache[path]

    program_video = video(manifest.program.video)
    audio(manifest.program.audio)
    probed: dict[str, MediaInfo] = {}
    for notice in notices:
        probed[notice.language] = video(notice.assets.video)
        audio(notice.assets.audio)
    return program_video, probed


def _frames_to_samples(frames: int, fps: int, sample_rate: int) -> int:
    return round(frames * sample_rate / fps)


def _filter_graph(
    manifest: Manifest,
    timeline: Timeline,
    output: OutputOptions,
) -> str:
    width, height = manifest.width, manifest.height
    fps = manifest.fps
    notice_frames = timeline.notice_frames
    program_frames = timeline.program_frames
    start_frames = timeline.program_start_frame
    gap_frames = timeline.trailing_blank_frames
    total_frames = timeline.total_frames
    sample_rate = output.audio_sample_rate
    intro_samples = _frames_to_samples(start_frames, fps, sample_rate)
    program_samples = _frames_to_samples(program_frames, fps, sample_rate)
    total_samples = intro_samples + program_samples

    def video_chain(source: str, frames: int) -> str:
        return (
            f"{source}fps=fps={fps}:round=near,"
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"setsar=1,format={output.pixel_format},"
            f"tpad=stop={frames}:stop_mode=clone,"
            f"trim=end_frame={frames}"
        )

    notice_video = video_chain("[0:v:0]", notice_frames)
    if gap_frames:
        notice_video += (
            f",tpad=stop={gap_frames}:stop_mode=add:color=black"
        )
    notice_video += (
        f",setpts=N/({fps}*TB),settb=1/{fps}[notice_v]"
    )
    notice_audio = (
        f"[1:a:0]aresample={sample_rate},"
        "aformat=sample_fmts=fltp:channel_layouts=stereo,"
        "asetpts=PTS-STARTPTS,"
        f"apad=whole_len={intro_samples},"
        f"atrim=end_sample={intro_samples},"
        "asetpts=N/SR/TB[notice_a]"
    )
    program_video = video_chain("[2:v:0]", program_frames)
    program_video += (
        f",setpts=N/({fps}*TB),settb=1/{fps}[program_v]"
    )
    program_audio = (
        f"[3:a:0]aresample={sample_rate},"
        "aformat=sample_fmts=fltp:channel_layouts=stereo,"
        "asetpts=PTS-STARTPTS,"
        f"apad=whole_len={program_samples},"
        f"atrim=end_sample={program_samples},"
        "asetpts=N/SR/TB[program_a]"
    )
    join = (
        "[notice_v][notice_a][program_v][program_a]"
        "concat=n=2:v=1:a=1[joined_v][joined_a]"
    )
    finish = (
        f"[joined_v]trim=end_frame={total_frames},setpts=N/({fps}*TB),"
        f"settb=1/{fps}[vout];"
        f"[joined_a]atrim=end_sample={total_samples},"
        "asetpts=N/SR/TB[aout]"
    )
    return ";".join([notice_video, notice_audio, program_video, program_audio, join, finish])


def _encoder_args(output: OutputOptions, target: Path, fps: int) -> list[str]:
    args = [
        "-map",
        "[vout]",
        "-map",
        "[aout]",
        "-c:v",
        output.video_codec,
        "-pix_fmt",
        output.pixel_format,
        "-c:a",
        output.audio_codec,
        "-map_metadata",
        "-1",
        "-fps_mode",
        "cfr",
        "-r",
        str(fps),
    ]
    if output.video_codec in {"libx264", "libx265"}:
        args.extend(["-crf", str(output.crf)])
    if output.audio_codec in {"aac", "libmp3lame", "libopus"}:
        args.extend(["-b:a", "192k"])
    if target.suffix.lower() in {".mp4", ".mov", ".m4v"}:
        args.extend(["-movflags", "+faststart"])
    return args


def _render_one(
    manifest: Manifest,
    notice: Notice,
    timeline: Timeline,
    output: OutputOptions,
    target: Path,
) -> None:
    ffmpeg = require_binary("ffmpeg")
    command = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(notice.assets.video),
        "-i",
        str(notice.assets.audio),
        "-i",
        str(manifest.program.video),
        "-i",
        str(manifest.program.audio),
        "-filter_complex",
        _filter_graph(manifest, timeline, output),
        *_encoder_args(output, target, manifest.fps),
        "-frames:v",
        str(timeline.total_frames),
        str(target),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        detail = completed.stderr.strip() or "ffmpeg failed"
        raise LangcutError(
            "RENDER_FAILED",
            f"ffmpeg failed for language {notice.language}: {detail[-8000:]}",
            language=notice.language,
        )


def _timeline_dict(timeline: Timeline) -> dict[str, int | float | str]:
    return {
        "opening_leader_frames": timeline.opening_leader_frames,
        "notice_frames": timeline.notice_frames,
        "program_frames": timeline.program_frames,
        "program_start_frame": timeline.program_start_frame,
        "extension_frames": timeline.extension_frames,
        "trailing_blank_frames": timeline.trailing_blank_frames,
        "shift_frames": timeline.shift_frames,
        "total_frames": timeline.total_frames,
        "program_start_timecode": timecode(timeline.program_start_frame, timeline.fps),
    }


def _mapping_payload(
    manifest: Manifest,
    timelines: dict[str, Timeline],
    targets: dict[str, Path],
    notices: dict[str, Notice],
) -> dict[str, Any]:
    languages: dict[str, Any] = {}
    for language, timeline in timelines.items():
        languages[language] = {
            "label": notices[language].label,
            "output_file": targets[language].name,
            "notice_frames": timeline.notice_frames,
            "old_program_start_frame": timeline.opening_leader_frames,
            "new_program_start_frame": timeline.program_start_frame,
            "extension_frames": timeline.extension_frames,
            "trailing_blank_frames": timeline.trailing_blank_frames,
            "shift_frames": timeline.shift_frames,
            "shift_seconds": timeline.shift_frames / timeline.fps,
            "chapters": timeline.map_chapters(manifest.chapters),
            "cut_points": timeline.map_cut_points(manifest.cut_points),
        }
    return {
        "schema_version": 1,
        "fps": manifest.fps,
        "frame_origin": "old version starts at program frame 0 after the original leader",
        "languages": languages,
    }


def _report_payload(
    manifest: Manifest,
    program_video: MediaInfo,
    timeline_by_language: dict[str, Timeline],
    targets: dict[str, Path],
    notices: dict[str, Notice],
) -> dict[str, Any]:
    languages: dict[str, Any] = {}
    for language, timeline in timeline_by_language.items():
        notice = notices[language]
        languages[language] = {
            "label": notice.label,
            "file": str(targets[language]),
            "notice_video": str(notice.assets.video),
            "notice_audio": str(notice.assets.audio),
            "status": "ok",
            "timeline": _timeline_dict(timeline),
        }
    return {
        "schema_version": 1,
        "renderer": "langcut",
        "manifest": str(manifest.path),
        "name": manifest.name,
        "fps": manifest.fps,
        "canvas": {"width": manifest.width, "height": manifest.height},
        "opening_leader_frames": manifest.opening_leader_frames,
        "program": {
            "video": str(manifest.program.video),
            "audio": str(manifest.program.audio),
            "frames": program_video.frame_count(manifest.fps),
            "duration_seconds": round(
                program_video.frame_count(manifest.fps) / manifest.fps, 6
            ),
            "source_fps": round(program_video.source_fps, 6),
        },
        "languages": languages,
        "missing_assets": [],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, mapping: dict[str, Any]) -> None:
    fields = [
        "language",
        "kind",
        "id",
        "title",
        "program_frame",
        "old_frame",
        "old_seconds",
        "old_timecode",
        "new_frame",
        "new_seconds",
        "new_timecode",
        "delta_frames",
        "delta_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for language, language_map in mapping["languages"].items():
            for kind in ("chapters", "cut_points"):
                for item in language_map[kind]:
                    writer.writerow(
                        {
                            "language": language,
                            "kind": "chapter" if kind == "chapters" else "cut",
                            "id": item["id"],
                            "title": item.get("title", ""),
                            "program_frame": item["program_frame"],
                            "old_frame": item["old"]["frame"],
                            "old_seconds": item["old"]["seconds"],
                            "old_timecode": item["old"]["timecode"],
                            "new_frame": item["new"]["frame"],
                            "new_seconds": item["new"]["seconds"],
                            "new_timecode": item["new"]["timecode"],
                            "delta_frames": item["delta_frames"],
                            "delta_seconds": item["delta_seconds"],
                        }
                    )


def render(
    manifest: Manifest,
    out_dir: str | Path,
    *,
    languages: list[str] | None = None,
    output: OutputOptions | None = None,
) -> dict[str, Any]:
    selected = list(dict.fromkeys(languages)) if languages else list(manifest.notices)
    unknown = sorted(set(selected) - set(manifest.notices))
    if unknown:
        raise LangcutError(
            "UNKNOWN_LANGUAGE",
            "unknown language id(s): " + ", ".join(unknown),
            languages=unknown,
        )
    notices = {language: manifest.notices[language] for language in selected}
    _preflight(manifest, list(notices.values()))
    program_video, probed = _probe_inputs(manifest, list(notices.values()))
    program_frames = program_video.frame_count(manifest.fps)
    timeline_by_language = {
        language: Timeline(
            fps=manifest.fps,
            opening_leader_frames=manifest.opening_leader_frames,
            notice_frames=probed[language].frame_count(manifest.fps),
            program_frames=program_frames,
        )
        for language in selected
    }
    for timeline in timeline_by_language.values():
        timeline.map_chapters(manifest.chapters)
        timeline.map_cut_points(manifest.cut_points)

    options = output or manifest.output
    reserved = {"render-report.json", "chapter-map.json", "chapter-map.csv"}
    stems = {
        language: _filename_stem(options, manifest.name, language, reserved)
        for language in selected
    }
    targets = {
        language: (Path(out_dir) / f"{stem}.{options.extension}").resolve()
        for language, stem in stems.items()
    }
    if len(set(targets.values())) != len(targets):
        raise LangcutError(
            "MANIFEST_INVALID", "output.filename produces duplicate file names"
        )

    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=".langcut-", dir=output_dir))
    try:
        staged: dict[str, Path] = {}
        for language in selected:
            staged[language] = temp_dir / targets[language].name
            _render_one(
                manifest,
                notices[language],
                timeline_by_language[language],
                options,
                staged[language],
            )
        mapping = _mapping_payload(
            manifest, timeline_by_language, targets, notices
        )
        report = _report_payload(
            manifest, program_video, timeline_by_language, targets, notices
        )
        report["output_directory"] = str(output_dir)
        report["chapter_map"] = str(output_dir / "chapter-map.json")
        _write_json(temp_dir / "chapter-map.json", mapping)
        _write_json(temp_dir / "render-report.json", report)
        _write_csv(temp_dir / "chapter-map.csv", mapping)
        for language in selected:
            os.replace(staged[language], targets[language])
        for name in ("chapter-map.json", "render-report.json", "chapter-map.csv"):
            os.replace(temp_dir / name, output_dir / name)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return report
