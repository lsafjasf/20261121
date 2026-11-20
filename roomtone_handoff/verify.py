"""Independently verify a rendered WAV against its replacement manifest."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .errors import ManifestError
from .manifest import MANIFEST_SCHEMA, build_manifest, sha256_file
from .model import LEFT, RIGHT, Region, Version
from .render import render_version
from .wav import WavAudio, read_wav


@dataclass(frozen=True)
class VerifyReport:
    version: str
    output_path: Path
    outside_pcm_bytes_checked: int
    replacement_count: int


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{where} must be an object")
    return value


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{where} must be a non-empty string")
    return value


def _integer(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{where} must be an integer")
    return value


def _load_document(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid manifest JSON in {path}: {exc}") from exc
    return _mapping(value, "manifest")


def _parse_rules(document: dict[str, Any]) -> tuple[Region, ...]:
    raw_rules = document.get("rules")
    if not isinstance(raw_rules, list):
        raise ManifestError("manifest.rules must be an array")
    regions = []
    for index, raw_rule in enumerate(raw_rules):
        where = f"manifest.rules[{index}]"
        rule = _mapping(raw_rule, where)
        channel = _text(rule.get("channel"), f"{where}.channel")
        if channel not in (LEFT, RIGHT):
            raise ManifestError(f"{where}.channel must be left or right")
        region = Region(
            rule_id=_text(rule.get("id"), f"{where}.id"),
            channel=channel,
            start_frame=_integer(rule.get("start_frame"), f"{where}.start_frame"),
            end_frame=_integer(rule.get("end_frame"), f"{where}.end_frame"),
            priority=_integer(rule.get("priority"), f"{where}.priority"),
            noise_offset=_integer(
                rule.get("noise_offset_frames"), f"{where}.noise_offset_frames"
            ),
        )
        if region.start_frame < 0 or region.end_frame <= region.start_frame:
            raise ManifestError(f"{where} has an invalid sample interval")
        if region.noise_offset < 0:
            raise ManifestError(f"{where}.noise_offset_frames must be non-negative")
        regions.append(region)
    return tuple(regions)


def _first_difference(actual: Any, expected: Any, path: str = "manifest") -> str | None:
    if type(actual) is not type(expected):
        return f"{path} has type {type(actual).__name__}, expected {type(expected).__name__}"
    if isinstance(actual, dict):
        if set(actual) != set(expected):
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            return f"{path} keys differ (missing={missing}, extra={extra})"
        for key in actual:
            difference = _first_difference(actual[key], expected[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(actual, list):
        if len(actual) != len(expected):
            return f"{path} has {len(actual)} items, expected {len(expected)}"
        for index, (left, right) in enumerate(zip(actual, expected)):
            difference = _first_difference(left, right, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if actual != expected:
        return f"{path} is {actual!r}, expected {expected!r}"
    return None


def _merged(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    result = [sorted(intervals)[0]]
    for start, end in sorted(intervals)[1:]:
        previous_start, previous_end = result[-1]
        if start <= previous_end:
            result[-1] = (previous_start, max(previous_end, end))
        else:
            result.append((start, end))
    return result


def _outside_segments(frame_count: int, excluded: list[tuple[int, int]]) -> list[tuple[int, int]]:
    segments = []
    cursor = 0
    for start, end in _merged(excluded):
        start = max(0, min(frame_count, start))
        end = max(0, min(frame_count, end))
        if start > cursor:
            segments.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < frame_count:
        segments.append((cursor, frame_count))
    return segments


def _channel_bytes(audio: WavAudio, channel_index: int, start: int, end: int) -> bytes:
    width = audio.format.sample_width
    step = audio.format.block_align
    data = audio.data
    result = bytearray((end - start) * width)
    source_pos = start * step + channel_index * width
    target_pos = 0
    for _ in range(end - start):
        result[target_pos : target_pos + width] = data[source_pos : source_pos + width]
        source_pos += step
        target_pos += width
    return bytes(result)


def _check_outside_pcm(
    source: WavAudio,
    output: WavAudio,
    rules: tuple[Region, ...],
) -> int:
    if output.data_size != source.data_size:
        raise ManifestError("output PCM data size differs from the source")
    checked = 0
    for channel, channel_index in ((LEFT, 0), (RIGHT, 1)):
        excluded = [
            (region.start_frame, region.end_frame)
            for region in rules
            if region.channel == channel
        ]
        for start, end in _outside_segments(source.frame_count, excluded):
            expected = _channel_bytes(source, channel_index, start, end)
            actual = _channel_bytes(output, channel_index, start, end)
            if actual != expected:
                raise ManifestError(
                    f"{channel} PCM differs outside requested intervals "
                    f"in frames [{start}, {end})"
                )
            checked += len(expected)
    return checked


def verify_manifest(manifest_path: Path) -> VerifyReport:
    document = _load_document(manifest_path)
    if document.get("schema") != MANIFEST_SCHEMA:
        raise ManifestError(f"unsupported manifest schema: {document.get('schema')!r}")
    base = manifest_path.resolve().parent
    source_info = _mapping(document.get("source"), "manifest.source")
    tone_info = _mapping(document.get("room_tone"), "manifest.room_tone")
    output_info = _mapping(document.get("output"), "manifest.output")
    source_path = (base / _text(source_info.get("path"), "source.path")).resolve()
    tone_path = (base / _text(tone_info.get("path"), "room_tone.path")).resolve()
    output_name = _text(output_info.get("path"), "output.path")
    if Path(output_name).name != output_name:
        raise ManifestError("output.path must be a basename beside the manifest")
    output_path = base / output_name
    try:
        source_hash = sha256_file(source_path)
        tone_hash = sha256_file(tone_path)
        output_hash = sha256_file(output_path)
    except OSError as exc:
        raise ManifestError(str(exc)) from exc
    if source_hash != source_info.get("sha256"):
        raise ManifestError("source WAV hash does not match the manifest")
    if tone_hash != tone_info.get("sha256"):
        raise ManifestError("room-tone WAV hash does not match the manifest")
    if output_hash != output_info.get("sha256"):
        raise ManifestError("output WAV hash does not match the manifest")

    source = read_wav(source_path)
    room_tone = read_wav(tone_path)
    output = read_wav(output_path)
    rules = _parse_rules(document)
    version_name = _text(document.get("version"), "manifest.version")
    expected = render_version(source, room_tone, Version(version_name, rules))
    if expected.output_bytes != output.raw:
        raise ManifestError("re-rendering the manifest rules does not reproduce the output WAV")
    expected_document = build_manifest(expected, base, source_path, tone_path)
    difference = _first_difference(document, expected_document)
    if difference:
        raise ManifestError(f"manifest does not match its files and rules: {difference}")
    outside_bytes = _check_outside_pcm(source, output, rules)
    return VerifyReport(
        version=version_name,
        output_path=output_path,
        outside_pcm_bytes_checked=outside_bytes,
        replacement_count=len(expected.replacements),
    )
