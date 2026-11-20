"""Timeline rendering: JSON report and human-readable text timeline."""

from __future__ import annotations

import json

from .detector import Analysis


def build_report(analysis: Analysis, fps_nominal: float = 30.0 / 1.001) -> dict:
    field_rate = 2 * fps_nominal
    return {
        "format": "ivtc-report-v1",
        "frame_count": len(analysis.fields) // 2,
        "field_count": len(analysis.fields),
        "residue_threshold": round(analysis.threshold, 4),
        "segments": [
            {
                "kind": seg.kind,
                "field_start": seg.field_start,
                "field_end": seg.field_end,
                "frame_start": seg.frame_start,
                "frame_end": seg.frame_end,
                "start_seconds": round(seg.field_start / field_rate, 4),
                "end_seconds": round(seg.field_end / field_rate, 4),
                "confidence": seg.confidence,
                "cadence_phase": seg.phase,
                "anchor_fields": seg.anchors,
                "note": seg.note,
            }
            for seg in analysis.segments
        ],
        "cuts": [
            {
                "frame_index": cut.frame_index,
                "field_index": cut.field_index,
                "seconds": round(cut.field_index / field_rate, 4),
                "frame_diff": round(cut.frame_diff, 4),
                "residue_left": None if cut.residue_left is None
                else round(cut.residue_left, 4),
                "residue_right": None if cut.residue_right is None
                else round(cut.residue_right, 4),
                "reason": cut.reason,
            }
            for cut in analysis.cuts
        ],
    }


def render_text(analysis: Analysis, width: int = 64) -> str:
    total = len(analysis.fields)
    lines = [
        f"fields={total} frames={total // 2} "
        f"residue_threshold={analysis.threshold:.3f}",
        "",
        "TIMELINE  (<> pulldown, :: native 60i, .. uncertain, | cut)",
    ]
    bar = [" "] * width
    kind_glyph = {"pulldown": "<>", "native": "::", "uncertain": ".."}
    for seg in analysis.segments:
        x0 = int(seg.field_start / total * width)
        x1 = max(x0 + 1, int(seg.field_end / total * width))
        glyph = kind_glyph[seg.kind]
        for x in range(x0, min(x1, width)):
            bar[x] = glyph[x % 2]
    for cut in analysis.cuts:
        x = min(width - 1, int(cut.field_index / total * width))
        bar[x] = "|"
    lines.append("|" + "".join(bar) + "|")
    lines.append("")
    for seg in analysis.segments:
        lines.append(
            f"  {seg.kind:<9} fields {seg.field_start:>3}-{seg.field_end:<3} "
            f"frames {seg.frame_start:>2}-{seg.frame_end:<2} "
            f"conf={seg.confidence:.2f} "
            f"phase={'-' if seg.phase is None else seg.phase}  {seg.note}")
    if analysis.cuts:
        lines.append("")
        for cut in analysis.cuts:
            lines.append(
                f"  CUT       frame {cut.frame_index:>2} (field "
                f"{cut.field_index:>2}) diff={cut.frame_diff:6.2f}  "
                f"{cut.reason}")
    else:
        lines.append("\n  no cuts detected")
    return "\n".join(lines)


def write_json(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
