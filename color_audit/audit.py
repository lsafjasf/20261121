from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .ffmpeg import (
    Tools,
    extract_frame_yuv444,
    probe_media,
    read_codec_tags,
    transcode,
)
from .container import read_container_tags
from .model import (
    FIELDS,
    ZSCALE_NAMES,
    Conflict,
    LayerTags,
    find_conflicts,
    resolve_tags,
)

SCHEMA = "ffmpeg-color-audit/v1"
DEFAULT_TARGET = {"primaries": 1, "transfer": 1, "matrix": 1, "range": 0}


class AuditError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_media(
    tools: Tools, path: str | Path
) -> dict[str, Any]:
    media_path = Path(path).resolve()
    if not media_path.is_file():
        raise AuditError(f"input file does not exist: {path}")
    probe = probe_media(tools, media_path)
    container = read_container_tags(media_path)
    codec = read_codec_tags(tools, media_path)
    conflicts = find_conflicts(container, codec)
    stream = probe["streams"][0]
    return {
        "path": str(media_path),
        "sha256": sha256_file(media_path),
        "size": media_path.stat().st_size,
        "probe": probe,
        "resolved_ffprobe_tags": {
            "primaries": stream.get("color_primaries"),
            "transfer": stream.get("color_transfer"),
            "matrix": stream.get("color_space"),
            "range": stream.get("color_range"),
        },
        "container": container,
        "codec": codec,
        "conflicts": conflicts,
    }


def build_filter_graph(source: dict[str, int], target: dict[str, int]) -> str:
    for field_name in FIELDS:
        if source[field_name] not in ZSCALE_NAMES[field_name]:
            raise AuditError(
                f"source {field_name}={source[field_name]} is not supported by zscale"
            )
        if target[field_name] not in ZSCALE_NAMES[field_name]:
            raise AuditError(
                f"target {field_name}={target[field_name]} is not supported by zscale"
            )
    if target["transfer"] in {16, 18}:
        raise AuditError(
            "PQ/HLG verification targets require a 10-bit output path; "
            "this tool verifies 8-bit SDR targets"
        )
    options = [
        f"primariesin={source['primaries']}",
        f"transferin={source['transfer']}",
        f"matrixin={source['matrix']}",
        f"rangein={source['range']}",
        f"primaries={target['primaries']}",
        f"transfer={target['transfer']}",
        f"matrix={target['matrix']}",
        f"range={target['range']}",
        "dither=none",
    ]
    return "zscale=" + ":".join(options) + ",format=yuv420p"


def _sample_at(raw: bytes, width: int, height: int, x: int, y: int) -> dict[str, int]:
    if not (0 <= x < width and 0 <= y < height):
        raise AuditError(f"sample coordinate ({x},{y}) is outside {width}x{height}")
    index = y * width + x
    plane = width * height
    return {
        "y": raw[index],
        "u": raw[plane + index],
        "v": raw[2 * plane + index],
    }


def yuv709_limited_to_rgb(values: dict[str, int]) -> dict[str, int]:
    y = (values["y"] - 16) / 219.0
    cb = (values["u"] - 128) / 224.0
    cr = (values["v"] - 128) / 224.0
    red = y + 1.5748 * cr
    green = y - 0.1873 * cb - 0.4681 * cr
    blue = y + 1.8556 * cb

    def quantize(value: float) -> int:
        return max(0, min(255, round(value * 255)))

    return {"r": quantize(red), "g": quantize(green), "b": quantize(blue)}


def _delta(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {name: left[name] - right[name] for name in ("y", "u", "v")}


def _decide_interactively(conflicts: list[Conflict]) -> dict[str, str]:
    if not conflicts or not sys.stdin.isatty():
        return {}
    decisions: dict[str, str] = {}
    for conflict in conflicts:
        while True:
            answer = input(
                f"{conflict.field}: container={conflict.container.name}"
                f"({conflict.container.code}) / codec={conflict.codec.name}"
                f"({conflict.codec.code}) -- use [c]ontainer or [k]odec? "
            ).strip().lower()
            if answer in {"c", "container"}:
                decisions[conflict.field] = "container"
                break
            if answer in {"k", "codec"}:
                decisions[conflict.field] = "codec"
                break
    return decisions


def resolve_interpretation(
    container: LayerTags,
    codec: LayerTags,
    explicit: dict[str, str],
    assume_layer: str | None,
    interactive: bool,
) -> tuple[dict[str, dict[str, Any]], str, dict[str, str]]:
    conflicts = find_conflicts(container, codec)
    decisions = dict(explicit)
    if assume_layer:
        if assume_layer not in {"container", "codec"}:
            raise AuditError("--assume-layer must be container or codec")
        for conflict in conflicts:
            decisions.setdefault(conflict.field, assume_layer)
    method = "explicit" if decisions else ("no-conflict" if not conflicts else "unresolved")
    if interactive:
        prompted = _decide_interactively(conflicts)
        if prompted:
            decisions.update(prompted)
            method = "interactive"
    resolved, errors = resolve_tags(container, codec, decisions)
    if errors:
        lines = ["conflicting or incomplete colour tags require confirmation:"]
        lines.extend(f"  - {error}" for error in errors)
        lines.append("  use --assume-layer codec or --assume field=container|codec")
        raise AuditError("\n".join(lines))
    result = {name: item.to_dict() for name, item in resolved.items()}
    return result, method, decisions


def _source_codes(interpretation: dict[str, dict[str, Any]]) -> dict[str, int]:
    codes: dict[str, int] = {}
    for name in FIELDS:
        value = interpretation[name]["value"]["code"]
        if value is None:
            raise AuditError(f"resolved {name} has no numeric code")
        codes[name] = int(value)
    return codes


def _load_manifest(path: str | Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, None
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise AuditError(f"manifest does not exist: {path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return payload, sha256_file(manifest_path)


def _layer_codes(layer: LayerTags) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for name in FIELDS:
        value = layer.values.get(name)
        result[name] = value.code if value else None
    return result


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "status": "pass" if passed else "fail", "detail": detail}


def _layer_health_check(layer: LayerTags) -> dict[str, Any]:
    usable = layer.status in {"parsed", "absent"}
    return _check(
        f"{layer.layer}_layer_health",
        usable,
        f"status={layer.status}; {'usable' if usable else 'manual review required'}",
    )


def _collect_sample_rows(
    input_raw: bytes,
    output_raw: bytes,
    width: int,
    height: int,
    samples: list[dict[str, Any]],
    tolerance: int,
    compare_output_expected: bool,
    compare_change: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        x = int(sample["x"])
        y = int(sample["y"])
        input_yuv = _sample_at(input_raw, width, height, x, y)
        output_yuv = _sample_at(output_raw, width, height, x, y)
        expected = sample.get("expected_yuv")
        row: dict[str, Any] = {
            "id": sample["id"],
            "kind": sample.get("kind", "sample"),
            "x": x,
            "y": y,
            "input_yuv": input_yuv,
            "output_yuv": output_yuv,
            "change_yuv": _delta(output_yuv, input_yuv),
            "input_rgb_bt709": yuv709_limited_to_rgb(input_yuv),
            "output_rgb_bt709": yuv709_limited_to_rgb(output_yuv),
        }
        if expected:
            expected_yuv = {key: int(expected[key]) for key in ("y", "u", "v")}
            input_delta = _delta(input_yuv, expected_yuv)
            output_delta = _delta(output_yuv, expected_yuv)
            row["expected_yuv"] = expected_yuv
            row["input_delta"] = input_delta
            row["output_delta"] = output_delta
            row["input_match"] = max(abs(value) for value in input_delta.values()) <= tolerance
            row["output_match"] = max(abs(value) for value in output_delta.values()) <= tolerance
            if not compare_output_expected:
                row["output_match"] = None
            if compare_change:
                row["change_match"] = (
                    max(abs(value) for value in row["change_yuv"].values()) <= tolerance
                )
        rows.append(row)
    return rows


def verify(
    tools: Tools,
    source: str | Path,
    output_dir: str | Path,
    explicit_decisions: dict[str, str],
    assume_layer: str | None,
    target: dict[str, int] | None,
    manifest_path: str | Path | None,
    tolerance: int,
    interactive: bool = True,
) -> dict[str, Any]:
    target = dict(target or DEFAULT_TARGET)
    if tolerance < 0:
        raise AuditError("tolerance must be non-negative")
    inspection = inspect_media(tools, source)
    interpretation, method, decisions = resolve_interpretation(
        inspection["container"],
        inspection["codec"],
        explicit_decisions,
        assume_layer,
        interactive,
    )
    source_codes = _source_codes(interpretation)
    output_dir_path = Path(output_dir).resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)
    verified_path = output_dir_path / "verified.mp4"
    filter_graph = build_filter_graph(source_codes, target)
    command = transcode(tools, inspection["path"], verified_path, filter_graph, target)

    output_probe = probe_media(tools, verified_path)
    output_container = read_container_tags(verified_path)
    output_codec = read_codec_tags(tools, verified_path)
    output_container_codes = _layer_codes(output_container)
    output_codec_codes = _layer_codes(output_codec)

    manifest, manifest_sha = _load_manifest(manifest_path)
    stream = inspection["probe"]["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])
    manifest_samples: list[dict[str, Any]] = []
    if manifest:
        manifest_samples = list(manifest.get("samples") or [])
        if manifest.get("width") and int(manifest["width"]) != width:
            raise AuditError("manifest width does not match input video")
        if manifest.get("height") and int(manifest["height"]) != height:
            raise AuditError("manifest height does not match input video")

    input_raw = extract_frame_yuv444(tools, inspection["path"], width, height)
    output_raw = extract_frame_yuv444(tools, verified_path, width, height)
    manifest_source = dict((manifest or {}).get("source_tags") or {})
    if manifest_source:
        compare_expected = all(
            int(manifest_source.get(name, -1)) == target[name] for name in FIELDS
        )
    else:
        compare_expected = source_codes == target
    compare_change = source_codes == target
    rows = _collect_sample_rows(
        input_raw,
        output_raw,
        width,
        height,
        manifest_samples,
        tolerance,
        compare_expected,
        compare_change,
    )

    tags_match = {name: output_container_codes.get(name) == target[name] for name in FIELDS}
    codec_match = {name: output_codec_codes.get(name) == target[name] for name in FIELDS}
    checks = [
        _layer_health_check(inspection["container"]),
        _layer_health_check(inspection["codec"]),
        _check(
            "output_container_tags",
            all(tags_match.values()),
            f"observed={output_container_codes}, target={target}",
        ),
        _check(
            "output_codec_tags",
            all(codec_match.values()),
            f"observed={output_codec_codes}, target={target}",
        ),
    ]
    if rows:
        input_match = all(row.get("input_match", True) for row in rows)
        output_match = all(row.get("output_match", True) for row in rows)
        change_match = all(row.get("change_match", True) for row in rows)
        checks.append(
            _check(
                "input_samples_match_manifest",
                bool(input_match),
                f"max allowed component delta={tolerance}",
            )
        )
        if compare_expected:
            checks.append(
                _check(
                    "output_samples_match_manifest",
                    bool(output_match),
                    f"max allowed component delta={tolerance}",
                )
            )
        if compare_change:
            checks.append(
                _check(
                    "input_output_stability",
                    bool(change_match),
                    f"max allowed component delta={tolerance}",
                )
            )
        if compare_expected:
            for sample_id, level, label in (
                ("gray_16", 16, "black_level"),
                ("black", 16, "black_patch"),
                ("gray_235", 235, "white_level"),
                ("white", 235, "white_patch"),
            ):
                row = next((item for item in rows if item["id"] == sample_id), None)
                if row:
                    value = row["output_yuv"]["y"]
                    checks.append(
                        _check(
                            label,
                            abs(value - level) <= tolerance,
                            f"{sample_id}: output Y={value}, expected {level}+/-{tolerance}",
                        )
                    )
    else:
        checks.append(
            {
                "name": "sample_comparison",
                "status": "skip",
                "detail": "no --manifest samples supplied; only metadata checks ran",
            }
        )

    failures = [check for check in checks if check["status"] == "fail"]
    audit = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result": {
            "status": "fail" if failures else "pass",
            "failures": [item["name"] for item in failures],
        },
        "tools": tools.to_dict(),
        "input": {
            "path": inspection["path"],
            "size": inspection["size"],
            "sha256": inspection["sha256"],
            "ffprobe": inspection["probe"],
            "resolved_ffprobe_tags": inspection["resolved_ffprobe_tags"],
        },
        "metadata": {
            "container": inspection["container"].to_dict(),
            "codec": inspection["codec"].to_dict(),
            "conflicts": [item.to_dict() for item in inspection["conflicts"]],
        },
        "interpretation": {
            "fields": interpretation,
            "resolved_codes": source_codes,
            "confirmation": {
                "method": method,
                "decisions": decisions,
                "assume_layer": assume_layer,
            },
        },
        "conversion": {
            "target_codes": target,
            "filter_graph": filter_graph,
            "ffmpeg_command": command,
            "expected_comparison": compare_expected,
            "tolerance_per_component": tolerance,
        },
        "manifest": {
            "path": str(Path(manifest_path).resolve()) if manifest_path else None,
            "sha256": manifest_sha,
        },
        "output": {
            "path": str(verified_path),
            "size": verified_path.stat().st_size,
            "sha256": sha256_file(verified_path),
            "ffprobe": output_probe,
            "container": output_container.to_dict(),
            "codec": output_codec.to_dict(),
        },
        "samples": rows,
        "checks": checks,
    }
    return audit
