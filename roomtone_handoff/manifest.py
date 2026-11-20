"""Build deterministic, independently reviewable replacement manifests."""

from __future__ import annotations

from csv import writer
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

from . import __version__
from .model import LEFT
from .render import RenderResult, Replacement


MANIFEST_SCHEMA = "roomtone-handoff.manifest.v1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise OSError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _relative(path: Path, directory: Path) -> str:
    return os.path.relpath(path.resolve(), directory.resolve())


def _format_dict(audio_format: Any) -> dict[str, int | str]:
    return dict(audio_format.as_dict())


def _replacement_dict(replacement: Replacement, result: RenderResult) -> dict[str, Any]:
    source = result.source
    span = replacement.span
    channel_index = 0 if span.channel == LEFT else 1
    byte_offset = (
        span.start_frame * source.format.block_align
        + channel_index * source.format.sample_width
    )
    return {
        "channel": span.channel,
        "start_frame": span.start_frame,
        "end_frame": span.end_frame,
        "frame_count": span.frame_count,
        "rule_id": span.region.rule_id,
        "priority": span.region.priority,
        "noise_channel_index": replacement.noise_channel_index,
        "effective_noise_offset_frames": replacement.effective_noise_offset,
        "noise_mappings": [
            {
                "output_start_frame": mapping.output_start_frame,
                "output_end_frame": mapping.output_end_frame,
                "noise_start_frame": mapping.noise_start_frame,
                "noise_end_frame": mapping.noise_end_frame,
            }
            for mapping in replacement.mappings
        ],
        "channel_pcm_sha256": replacement.channel_sha256,
        "pcm_location": {
            "data_chunk_relative_byte_offset": byte_offset,
            "file_byte_offset": source.data_offset + byte_offset,
            "sample_width_bytes": source.format.sample_width,
            "block_align_bytes": source.format.block_align,
        },
    }


def build_manifest(
    result: RenderResult,
    output_dir: Path,
    source_path: Path,
    room_tone_path: Path,
) -> dict[str, Any]:
    source = result.source
    room_tone = result.room_tone
    output_name = f"{result.version.name}.wav"
    rules = []
    for region in sorted(
        result.version.regions,
        key=lambda item: (item.channel != LEFT, item.start_frame, item.end_frame, item.rule_id),
    ):
        plan = result.plans[region.channel]
        effective = [
            {"start_frame": span.start_frame, "end_frame": span.end_frame}
            for span in plan.spans
            if span.region == region
        ]
        shadowed = [
            {"start_frame": start, "end_frame": end}
            for start, end in plan.shadowed.get(region.rule_id, ())
        ]
        rules.append({**region.as_dict(), "effective": effective, "shadowed": shadowed})

    channel_edits = {
        channel: sum(span.frame_count for span in plan.spans)
        for channel, plan in result.plans.items()
    }
    return {
        "schema": MANIFEST_SCHEMA,
        "tool": {"name": "roomtone-handoff", "version": __version__},
        "version": result.version.name,
        "source": {
            "path": _relative(source_path, output_dir),
            "sha256": sha256_bytes(source.raw),
            "file_bytes": len(source.raw),
            "frames": source.frame_count,
            "format": _format_dict(source.format),
            "pcm_sha256": sha256_bytes(source.data),
        },
        "room_tone": {
            "path": _relative(room_tone_path, output_dir),
            "sha256": sha256_bytes(room_tone.raw),
            "file_bytes": len(room_tone.raw),
            "frames": room_tone.frame_count,
            "format": _format_dict(room_tone.format),
            "pcm_sha256": sha256_bytes(room_tone.data),
        },
        "output": {
            "path": output_name,
            "sha256": result.output_sha256,
            "file_bytes": len(result.output_bytes),
            "frames": source.frame_count,
        },
        "precedence": {
            "higher_priority_first": True,
            "tie_break_order": [
                "start_frame_desc",
                "end_frame_desc",
                "rule_id_desc",
                "noise_offset_frames_desc",
            ],
        },
        "summary": {
            "rule_count": len(result.version.regions),
            "effective_span_count": len(result.replacements),
            "replaced_frames_by_channel": channel_edits,
        },
        "rules": rules,
        "effective_replacements": [
            _replacement_dict(replacement, result)
            for replacement in result.replacements
        ],
    }


def manifest_json(document: dict[str, Any]) -> str:
    return json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def manifest_csv(document: dict[str, Any]) -> str:
    buffer = io.StringIO(newline="")
    rows = writer(buffer, lineterminator="\n")
    rows.writerow(
        [
            "version",
            "source_sha256",
            "room_tone_sha256",
            "output_sha256",
            "channel",
            "start_frame",
            "end_frame",
            "priority",
            "rule_id",
            "noise_channel_index",
            "effective_noise_offset_frames",
            "data_chunk_byte_offset",
            "file_byte_offset",
            "sample_width_bytes",
            "block_align_bytes",
            "channel_pcm_sha256",
            "noise_mappings",
        ]
    )
    for item in document["effective_replacements"]:
        mappings = ";".join(
            f"{mapping['output_start_frame']}-{mapping['output_end_frame']}"
            f"<-{mapping['noise_start_frame']}-{mapping['noise_end_frame']}"
            for mapping in item["noise_mappings"]
        )
        rows.writerow(
            [
                document["version"],
                document["source"]["sha256"],
                document["room_tone"]["sha256"],
                document["output"]["sha256"],
                item["channel"],
                item["start_frame"],
                item["end_frame"],
                item["priority"],
                item["rule_id"],
                item["noise_channel_index"],
                item["effective_noise_offset_frames"],
                item["pcm_location"]["data_chunk_relative_byte_offset"],
                item["pcm_location"]["file_byte_offset"],
                item["pcm_location"]["sample_width_bytes"],
                item["pcm_location"]["block_align_bytes"],
                item["channel_pcm_sha256"],
                mappings,
            ]
        )
    return buffer.getvalue()
