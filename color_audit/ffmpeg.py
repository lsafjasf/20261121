from __future__ import annotations

import json
import os
import pwd
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model import LayerTags

TRACE_FIELDS = {
    "primaries": ("colour_primaries", "color_primaries", "primaries"),
    "transfer": (
        "transfer_characteristics",
        "transfer_characteristic",
        "transfer",
    ),
    "matrix": ("matrix_coefficients", "matrix_coefficient", "matrix"),
    "range": ("video_full_range_flag", "color_range", "range"),
}
TRACE_LINE = re.compile(
    r"\[trace_headers @ [^\]]+\]\s+\d+\s+(\w+)\s+.*?=\s*(\d+)\s*$"
)


@dataclass(frozen=True)
class Tools:
    ffmpeg: str
    ffprobe: str
    ffmpeg_version: str
    ffprobe_version: str

    def to_dict(self) -> dict[str, str]:
        return {
            "ffmpeg": self.ffmpeg,
            "ffprobe": self.ffprobe,
            "ffmpeg_version": self.ffmpeg_version,
            "ffprobe_version": self.ffprobe_version,
        }


def _works(executable: Path) -> bool:
    if not executable.is_file():
        return False
    try:
        completed = subprocess.run(
            [str(executable), "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0 and b"version" in completed.stdout.lower()


def _candidates(name: str) -> list[Path]:
    candidates: list[Path] = []
    env_name = f"{name.upper()}_BIN"
    if os.environ.get(env_name):
        candidates.append(Path(os.environ[env_name]).expanduser())
    found = shutil.which(name)
    if found:
        candidates.append(Path(found))
    try:
        home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    except KeyError:
        home = Path.home()
    candidates.extend(sorted((home / ".local" / "ffstatic").glob(f"*/{name}"), reverse=True))
    candidates.extend(
        [
            Path(f"/usr/local/bin/{name}"),
            Path(f"/usr/bin/{name}"),
            Path(f"/opt/homebrew/bin/{name}"),
        ]
    )
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def _find_tool(name: str) -> str:
    for candidate in _candidates(name):
        if _works(candidate):
            return str(candidate.resolve())
    tried = ", ".join(str(item) for item in _candidates(name))
    raise RuntimeError(
        f"could not execute {name}; set {name.upper()}_BIN or install FFmpeg "
        f"(tried: {tried})"
    )


def _version_line(executable: str) -> str:
    completed = subprocess.run(
        [executable, "-version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=10,
        check=True,
        text=True,
    )
    return completed.stdout.splitlines()[0]


def discover_tools() -> Tools:
    ffmpeg = _find_tool("ffmpeg")
    sibling = Path(ffmpeg).with_name("ffprobe")
    if _works(sibling):
        ffprobe = str(sibling)
    else:
        ffprobe = _find_tool("ffprobe")
    return Tools(
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        ffmpeg_version=_version_line(ffmpeg),
        ffprobe_version=_version_line(ffprobe),
    )


def run_capture(
    args: list[str],
    *,
    input_data: bytes | None = None,
    timeout: int = 120,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        args,
        input=input_data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def probe_media(tools: Tools, path: str | Path) -> dict[str, Any]:
    completed = run_capture(
        [
            tools.ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        timeout=60,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"ffprobe failed for {path}: {detail}")
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError(f"no video stream found in {path}")
    return payload


def _trace_colors(tools: Tools, path: str | Path) -> tuple[dict[str, list[int]], list[str]]:
    command = [
        tools.ffmpeg,
        "-hide_banner",
        "-loglevel",
        "trace",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-frames:v",
        "1",
        "-c",
        "copy",
        "-bsf:v",
        "trace_headers",
        "-f",
        "null",
        "-",
    ]
    completed = run_capture(command, timeout=60)
    trace = completed.stderr.decode("utf-8", "replace")
    if completed.returncode != 0 and "Bitstream filter" not in trace:
        detail = trace.strip().splitlines()[-1:] or ["trace_headers failed"]
        raise RuntimeError(detail[0])
    by_name: dict[str, list[int]] = {}
    evidence: list[str] = []
    for line in trace.splitlines():
        match = TRACE_LINE.search(line)
        if not match:
            continue
        field_name, raw_value = match.groups()
        by_name.setdefault(field_name, []).append(int(raw_value))
        if field_name in {alias for aliases in TRACE_FIELDS.values() for alias in aliases}:
            evidence.append(line.split("] ", 1)[-1].strip())
    return by_name, evidence


def read_codec_tags(tools: Tools, path: str | Path) -> LayerTags:
    try:
        by_name, evidence = _trace_colors(tools, path)
    except RuntimeError as exc:
        return LayerTags(
            layer="codec",
            status="unsupported",
            evidence=[f"codec bitstream parse failed: {exc}"],
        )
    codes: dict[str, int | None] = {}
    ambiguities: list[str] = []
    for field_name, aliases in TRACE_FIELDS.items():
        values: list[int] = []
        for alias in aliases:
            values.extend(by_name.get(alias, []))
        unique = sorted(set(values))
        codes[field_name] = values[0] if values else None
        if len(unique) > 1:
            ambiguities.append(f"{field_name} occurs with values {unique}")
    if not evidence:
        return LayerTags(
            layer="codec",
            status="absent",
            evidence=["trace_headers found no VUI colour description"],
        )
    from .model import tags_from_codes

    status = "ambiguous" if ambiguities else "parsed"
    return LayerTags(
        layer="codec",
        values=tags_from_codes(codes),
        evidence=evidence + ambiguities,
        status=status,
    )


def extract_frame_yuv444(
    tools: Tools, path: str | Path, width: int, height: int
) -> bytes:
    command = [
        tools.ffmpeg,
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-frames:v",
        "1",
        "-vf",
        "format=yuv444p",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv444p",
        "pipe:1",
    ]
    completed = run_capture(command, timeout=60)
    expected = width * height * 3
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"sample extraction failed: {detail}")
    if len(completed.stdout) != expected:
        raise RuntimeError(
            f"sample extraction returned {len(completed.stdout)} bytes, expected {expected}"
        )
    return completed.stdout


def transcode(
    tools: Tools,
    source: str | Path,
    destination: str | Path,
    filter_graph: str,
    target: dict[str, int],
) -> list[str]:
    command = [
        tools.ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        filter_graph,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-qp",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-color_primaries",
        str(target["primaries"]),
        "-color_trc",
        str(target["transfer"]),
        "-colorspace",
        str(target["matrix"]),
        "-color_range",
        "pc" if target["range"] == 1 else "tv",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    completed = run_capture(command, timeout=300)
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"verification transcode failed: {detail}")
    return command
