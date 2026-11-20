"""Verify incremental renders against full re-renders.

Comparison uses two kinds of probes:

  - fixed probes: deterministic, content-independent segments anchored at
    the head, the tail, and evenly spaced fractions of the file;
  - boundary probes: windows straddling every splice boundary recorded in
    the version history.

For each probe the report counts mismatched samples and the max absolute
delta; boundary probes additionally list the per-sample differences around
the boundary (frame, channel, full value, incremental value, delta).
"""
from __future__ import annotations

import json
import os

from .bwf import read_bwf
from .render import render_full
from .store import ProjectStore

FIXED_PROBE_COUNT = 6
FIXED_PROBE_LEN = 64
BOUNDARY_HALF_WINDOW = 8
MAX_LISTED_DIFFS = 32


def fixed_probes(
    frame_count: int,
    count: int = FIXED_PROBE_COUNT,
    length: int = FIXED_PROBE_LEN,
) -> list:
    """Deterministic fixed segments: head, tail, and evenly spaced middles."""
    if frame_count <= length:
        return [(0, frame_count)]
    starts = {0, frame_count - length}
    for i in range(count):
        starts.add((frame_count - length) * (2 * i + 1) // (2 * count))
    return [(s, s + length) for s in sorted(starts)]


def boundary_probes(
    boundaries: list, frame_count: int, half: int = BOUNDARY_HALF_WINDOW
) -> list:
    """Windows [b-half, b+half) around each splice boundary frame."""
    probes = []
    for boundary in sorted(set(boundaries)):
        if boundary <= 0 or boundary >= frame_count:
            continue  # file edges are already covered by fixed probes
        start = max(0, boundary - half)
        end = min(frame_count, boundary + half)
        probes.append((boundary, start, end))
    return probes


def _diff_probe(a_samples, b_samples, start, end, channels, list_diffs):
    mismatches = 0
    max_abs = 0
    diffs = []
    for frame in range(start, end):
        for ch in range(channels):
            idx = frame * channels + ch
            delta = b_samples[idx] - a_samples[idx]
            if delta:
                mismatches += 1
                max_abs = max(max_abs, abs(delta))
                if list_diffs and len(diffs) < MAX_LISTED_DIFFS:
                    diffs.append(
                        {
                            "frame": frame,
                            "channel": ch,
                            "full": a_samples[idx],
                            "incremental": b_samples[idx],
                            "delta": delta,
                        }
                    )
    return mismatches, max_abs, diffs


def compare(full, incr, boundaries: list) -> dict:
    """Compare two rendered masters on fixed and boundary probes."""
    if full.frame_count != incr.frame_count or full.channels != incr.channels:
        raise ValueError("master geometry mismatch")
    frames, channels = full.frame_count, full.channels
    probes = []
    total_mismatched = 0
    for start, end in fixed_probes(frames):
        mismatches, max_abs, _ = _diff_probe(
            full.samples, incr.samples, start, end, channels, False
        )
        total_mismatched += mismatches
        probes.append(
            {
                "kind": "fixed",
                "start": start,
                "end": end,
                "mismatched_samples": mismatches,
                "max_abs_delta": max_abs,
            }
        )
    for boundary, start, end in boundary_probes(boundaries, frames):
        mismatches, max_abs, diffs = _diff_probe(
            full.samples, incr.samples, start, end, channels, True
        )
        total_mismatched += mismatches
        probes.append(
            {
                "kind": "boundary",
                "boundary_frame": boundary,
                "start": start,
                "end": end,
                "mismatched_samples": mismatches,
                "max_abs_delta": max_abs,
                "sample_diffs": diffs,
            }
        )
    return {
        "frames": frames,
        "channels": channels,
        "equal": total_mismatched == 0,
        "total_mismatched_samples": total_mismatched,
        "probes": probes,
    }


def verify_file(store: ProjectStore, stem: str) -> dict:
    """Full re-render vs. stored (incrementally built) master for one file."""
    entry = store.project["files"][stem]
    src = read_bwf(entry["source"])
    env = store.load_envelope(stem)
    full = render_full(src, env)
    incr = store.load_master(stem)
    boundaries = []
    for meta in store.iter_version_meta(stem):
        affected = meta.get("affected_frames")
        if affected:
            boundaries.extend([affected["start"], affected["end"]])
    report = compare(full, incr, boundaries)
    report["file"] = stem
    report["current_version"] = entry["current_version"]
    report_path = os.path.join(store.stem_dir(stem), "verify_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return report


def verify_all(store: ProjectStore) -> dict:
    reports = {stem: verify_file(store, stem) for stem in store.project["files"]}
    return {
        "equal": all(r["equal"] for r in reports.values()),
        "files": reports,
    }
