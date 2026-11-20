"""Ground-truth alignment and reproducible report generation."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from .detector import analyze_file
from .synth import build_sample

MATCH_TOLERANCE_SECONDS = 0.30


def _nearest_truth(detection: dict, truth: list[dict]) -> dict | None:
    when = detection["marker_seconds"]
    nearest = None
    distance = MATCH_TOLERANCE_SECONDS
    for item in truth:
        if detection["kind"] == "hit" and detection["asset_id"] != item["asset_id"]:
            continue
        candidate_distance = abs(when - item["expected_seconds"])
        if candidate_distance <= distance:
            nearest = item
            distance = candidate_distance
    return nearest


def attach_ground_truth(report: dict, manifest: dict) -> dict:
    truth = manifest.get("truth", [])
    matched_truth: set[int] = set()
    for detection in report["detections"]:
        match = _nearest_truth(detection, truth)
        if match is None:
            detection["false_positive"] = True
            continue
        detection["matched_truth_index"] = match["index"]
        detection["timing_offset_ms"] = round(
            (detection["marker_seconds"] - match["expected_seconds"]) * 1000.0, 1
        )
        detection["truth_kind"] = match["expected_kind"]
        matched_truth.add(match["index"])

    hits = [item for item in report["detections"] if item["kind"] == "hit"]
    candidates = [item for item in report["detections"] if item["kind"] == "candidate"]
    matched_hits = [item for item in hits if item["matched_truth_index"] is not None]
    false_positive_hits = [item for item in hits if item["false_positive"]]
    unmatched_candidates = [
        item
        for item in candidates
        if item["matched_truth_index"] is None
    ]
    missed = [item for item in truth if item["index"] not in matched_truth]
    offsets = [item["timing_offset_ms"] for item in matched_hits]
    mae = sum(abs(value) for value in offsets) / len(offsets) if offsets else None
    maximum = max((abs(value) for value in offsets), default=None)
    rms = (
        math.sqrt(sum(value * value for value in offsets) / len(offsets))
        if offsets
        else None
    )

    report["summary"].update(
        {
            "hits": len(hits),
            "candidates": len(candidates),
            "false_positives": len(false_positive_hits) + len(unmatched_candidates),
            "false_positive_hits": len(false_positive_hits),
            "unmatched_candidates": len(unmatched_candidates),
            "timing_mae_ms": round(mae, 1) if mae is not None else None,
            "timing_max_abs_ms": round(maximum, 1) if maximum is not None else None,
            "timing_rms_ms": round(rms, 1) if rms is not None else None,
            "missed": len(missed),
            "truth_total": len(truth),
            "confirmed_hits": len(matched_hits),
            "damaged_candidates": sum(
                1
                for item in candidates
                if item["matched_truth_index"] is not None
                and item.get("truth_kind") == "candidate"
            ),
        }
    )
    report["false_positive_detail"] = [
        {
            "marker_seconds": item["marker_seconds"],
            "kind": item["kind"],
            "asset_label": item["asset_label"],
            "reason": item.get("candidate_reason") or "无真值对应",
        }
        for item in report["detections"]
        if item["false_positive"]
    ]
    report["evaluation"] = {
        "tolerance_ms": MATCH_TOLERANCE_SECONDS * 1000.0,
        "ground_truth": truth,
        "matched_hit_indices": [item["matched_truth_index"] for item in matched_hits],
        "matched_candidate_indices": [
            item["matched_truth_index"]
            for item in candidates
            if item["matched_truth_index"] is not None
        ],
        "missed_truth_indices": [item["index"] for item in missed],
        "false_positive_indices": [
            item["marker_seconds"]
            for item in report["detections"]
            if item["false_positive"]
        ],
    }
    report["damage_chain"] = {
        "downsample_from_hz": manifest.get("downsample_from"),
        "output_rate_hz": manifest.get("sample_rate"),
        "speed_factor": manifest.get("speed_factor"),
        "reverb_wet": manifest.get("reverb_wet"),
        "snr_db": manifest.get("snr_db"),
    }
    return report


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def generate_demo(root: str | Path = ".") -> tuple[dict, dict]:
    root = Path(root)
    audio_path = root / "samples" / "museum_sample.wav"
    manifest_path = root / "samples" / "museum_sample_manifest.json"
    report_path = root / "reports" / "museum_sample_report.json"

    manifest = build_sample(audio_path)
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    save_json(manifest_path, manifest)
    report = analyze_file(str(audio_path), "museum_sample.wav")
    report = attach_ground_truth(report, manifest)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["audio_url"] = "/samples/museum_sample.wav"
    save_json(report_path, report)
    static_path = root / "web" / "sample-report.js"
    static_path.parent.mkdir(parents=True, exist_ok=True)
    static_path.write_text(
        "window.__SAMPLE_REPORT__ = "
        + json.dumps(report, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    manifest["report_file"] = str(report_path)
    save_json(manifest_path, manifest)
    return report, manifest
