from __future__ import annotations

import csv
import json
import shlex
from pathlib import Path
from typing import Any

from .model import FIELDS, MAPPINGS


def _name(field_name: str, code: int | None) -> str:
    if code is None:
        return "-"
    return MAPPINGS[field_name].get(code, f"unknown({code})")


def _tag_cell(value: Any) -> str:
    if value is None:
        return "-"
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if value.get("code") is None:
        return "-"
    return f"{value['name']} ({value['code']})"


def write_artifacts(audit: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "audit.json"
    json_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    csv_path = output / "samples.csv"
    fieldnames = [
        "id",
        "kind",
        "x",
        "y",
        "expected_y",
        "expected_u",
        "expected_v",
        "input_y",
        "input_u",
        "input_v",
        "output_y",
        "output_u",
        "output_v",
        "output_delta_y",
        "output_delta_u",
        "output_delta_v",
        "change_y",
        "change_u",
        "change_v",
        "input_rgb_r",
        "input_rgb_g",
        "input_rgb_b",
        "output_rgb_r",
        "output_rgb_g",
        "output_rgb_b",
        "output_match",
        "change_match",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in audit["samples"]:
            expected = row.get("expected_yuv") or {}
            output_delta = row.get("output_delta") or {}
            writer.writerow(
                {
                    "id": row["id"],
                    "kind": row["kind"],
                    "x": row["x"],
                    "y": row["y"],
                    "expected_y": expected.get("y", ""),
                    "expected_u": expected.get("u", ""),
                    "expected_v": expected.get("v", ""),
                    "input_y": row["input_yuv"]["y"],
                    "input_u": row["input_yuv"]["u"],
                    "input_v": row["input_yuv"]["v"],
                    "output_y": row["output_yuv"]["y"],
                    "output_u": row["output_yuv"]["u"],
                    "output_v": row["output_yuv"]["v"],
                    "output_delta_y": output_delta.get("y", ""),
                    "output_delta_u": output_delta.get("u", ""),
                    "output_delta_v": output_delta.get("v", ""),
                    "change_y": row["change_yuv"]["y"],
                    "change_u": row["change_yuv"]["u"],
                    "change_v": row["change_yuv"]["v"],
                    "input_rgb_r": row["input_rgb_bt709"]["r"],
                    "input_rgb_g": row["input_rgb_bt709"]["g"],
                    "input_rgb_b": row["input_rgb_bt709"]["b"],
                    "output_rgb_r": row["output_rgb_bt709"]["r"],
                    "output_rgb_g": row["output_rgb_bt709"]["g"],
                    "output_rgb_b": row["output_rgb_bt709"]["b"],
                    "output_match": row.get("output_match", ""),
                    "change_match": row.get("change_match", ""),
                }
            )

    markdown_path = output / "report.md"
    markdown_path.write_text(render_markdown(audit), encoding="utf-8")
    return {"audit_json": json_path, "samples_csv": csv_path, "report_md": markdown_path}


def render_markdown(audit: dict[str, Any]) -> str:
    result = audit["result"]
    metadata = audit["metadata"]
    container_values = metadata["container"]["values"]
    codec_values = metadata["codec"]["values"]
    conflict_fields = {item["field"] for item in metadata["conflicts"]}
    lines = [
        "# FFmpeg color-tag verification",
        "",
        f"**Result:** `{result['status'].upper()}`",
        "",
        "## Layer declarations",
        "",
        "| Field | Container | Codec | Conflict |",
        "|---|---|---|---|",
    ]
    for field_name in FIELDS:
        left = _tag_cell(container_values.get(field_name))
        right = _tag_cell(codec_values.get(field_name))
        lines.append(
            f"| {field_name} | {left} | {right} | "
            f"{'YES' if field_name in conflict_fields else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"Container evidence: `{'; '.join(metadata['container']['evidence'])}`",
            "",
            f"Codec evidence: `{'; '.join(metadata['codec']['evidence'])}`",
            "",
            "## Confirmed interpretation",
            "",
            f"Confirmation method: `{audit['interpretation']['confirmation']['method']}`",
            "",
            "| Field | Value | Origin | Reason |",
            "|---|---|---|---|",
        ]
    )
    for field_name in FIELDS:
        item = audit["interpretation"]["fields"][field_name]
        lines.append(
            f"| {field_name} | {_tag_cell(item['value'])} | "
            f"{item['origin']} | {item['reason']} |"
        )
    conversion = audit["conversion"]
    lines.extend(
        [
            "",
            "## Conversion",
            "",
            f"Expected comparison: `{conversion['expected_comparison']}`  ",
            f"Per-component tolerance: `{conversion['tolerance_per_component']}`",
            "",
            "```text",
            conversion["filter_graph"],
            "```",
            "",
            "```bash",
            shlex.join(conversion["ffmpeg_command"]),
            "```",
            "",
            "## Output tags",
            "",
            "| Field | Container | Codec | Target |",
            "|---|---|---|---|",
        ]
    )
    output = audit["output"]
    for field_name in FIELDS:
        target = conversion["target_codes"][field_name]
        lines.append(
            f"| {field_name} | {_tag_cell(output['container']['values'].get(field_name))} "
            f"| {_tag_cell(output['codec']['values'].get(field_name))} "
            f"| {_name(field_name, target)} ({target}) |"
        )
    lines.extend(
        [
            "",
            "## Sample values",
            "",
            "YUV values are raw 8-bit decoded samples at the listed coordinates.",
            "",
            "| Sample | Kind | XY | Expected YUV | Input YUV | Output YUV | "
            "Output delta | Change | Status |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in audit["samples"]:
        expected = row.get("expected_yuv")
        input_yuv = row["input_yuv"]
        output_yuv = row["output_yuv"]
        delta = row.get("output_delta")
        change = row["change_yuv"]
        status = row.get("output_match")
        lines.append(
            f"| {row['id']} | {row['kind']} | {row['x']},{row['y']} "
            f"| {_yuv(expected)} | {_yuv(input_yuv)} | {_yuv(output_yuv)} "
            f"| {_yuv(delta)} | {_yuv(change)} "
            f"| {'pass' if status else ('skip' if status is None else 'fail')} |"
        )
    lines.extend(["", "## Checks", "", "| Check | Status | Detail |", "|---|---|---|"])
    for check in audit["checks"]:
        lines.append(
            f"| {check['name']} | {check['status']} | {check['detail'].replace('|', '/')} |"
        )
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- Input: `{audit['input']['path']}`",
            f"- Input SHA-256: `{audit['input']['sha256']}`",
            f"- Output: `{audit['output']['path']}`",
            f"- Output SHA-256: `{audit['output']['sha256']}`",
            f"- FFmpeg: `{audit['tools']['ffmpeg_version']}`",
            f"- FFprobe: `{audit['tools']['ffprobe_version']}`",
            f"- Manifest SHA-256: `{audit['manifest']['sha256'] or '-'}`",
            "",
        ]
    )
    return "\n".join(lines)


def _yuv(values: dict[str, int] | None) -> str:
    if not values:
        return "-"
    return f"{values.get('y', '-')},{values.get('u', '-')},{values.get('v', '-')}"


def print_inspection(inspection: dict[str, Any]) -> None:
    print(f"Input: {inspection['path']}")
    print(f"SHA-256: {inspection['sha256']}")
    print("")
    print(f"{'Field':<12} {'Container':<24} {'Codec':<24} Conflict")
    print(f"{'-' * 12} {'-' * 24} {'-' * 24} {'-' * 8}")
    conflicts = {item.field for item in inspection["conflicts"]}
    for field_name in FIELDS:
        left = _tag_cell(inspection["container"].values.get(field_name))
        right = _tag_cell(inspection["codec"].values.get(field_name))
        print(
            f"{field_name:<12} {left:<24} {right:<24} "
            f"{'YES' if field_name in conflicts else 'no'}"
        )
    print("")
    if inspection["conflicts"]:
        print("Conflicts require an explicit interpretation.")
        for conflict in inspection["conflicts"]:
            print(
                f"  {conflict.field}: container={conflict.container.name}"
                f"({conflict.container.code}) codec={conflict.codec.name}"
                f"({conflict.codec.code})"
            )
    else:
        print("No conflicts between container and codec declarations.")
    print("")
    print("ffprobe resolved (informational only):")
    print(json.dumps(inspection["resolved_ffprobe_tags"], ensure_ascii=False))


def print_verification_summary(audit: dict[str, Any], artifacts: dict[str, Path]) -> None:
    print(f"Result: {audit['result']['status'].upper()}")
    print(
        f"Tag conflicts: {len(audit['metadata']['conflicts'])}; "
        f"confirmation: {audit['interpretation']['confirmation']['method']}"
    )
    for check in audit["checks"]:
        print(f"  [{check['status'].upper():<4}] {check['name']}: {check['detail']}")
    if audit["samples"]:
        print("")
        print("Selected samples (raw Y,U,V):")
        for row in audit["samples"]:
            input_yuv = row["input_yuv"]
            output_yuv = row["output_yuv"]
            print(
                f"  {row['id']:<10} input="
                f"{input_yuv['y']},{input_yuv['u']},{input_yuv['v']} "
                f"output={output_yuv['y']},{output_yuv['u']},{output_yuv['v']}"
            )
    print("")
    for label, path in artifacts.items():
        print(f"{label}: {path}")
