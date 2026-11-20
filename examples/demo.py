#!/usr/bin/env python3
"""Generate and inspect three deterministic MIDI/audio resync examples."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from piano_resync import SyncConfig, resynchronize  # noqa: E402


def _notes(count: int, step: float, velocities: list[int]) -> list[dict[str, object]]:
    rows = []
    for index in range(count):
        onset = 0.180 + index * step
        rows.append(
            {
                "note": 60 + (index * 4) % 13,
                "onset": round(onset, 6),
                "duration": 0.18 + (index % 3) * 0.03,
                "velocity": velocities[index % len(velocities)],
            }
        )
    return rows


def _taps(
    notes: list[dict[str, object]],
    rate: float,
    offset: float,
    jitter_ms: list[float],
) -> list[dict[str, object]]:
    taps = []
    for index, note in enumerate(notes):
        jitter = jitter_ms[index % len(jitter_ms)] / 1000.0
        taps.append(
            {
                "time": round(rate * float(note["onset"]) + offset + jitter, 6),
                "strength": round(float(note["velocity"]) / 127.0, 6),
            }
        )
    return taps


def _pedals() -> list[dict[str, object]]:
    return [
        {"controller": 64, "start": 0.30, "end": 2.20, "value": 1.0},
        {"controller": 67, "start": 2.35, "end": 3.05, "value": 0.75},
    ]


def scenarios() -> list[dict[str, object]]:
    running_notes = _notes(16, 0.265, [32, 48, 70, 96, 116, 58])
    running_taps = _taps(
        running_notes, 1.0019, 0.034, [1.6, -1.2, 0.7, -2.0, 1.1]
    )

    edge_notes = _notes(13, 0.300, [38, 62, 91, 112, 51, 79])
    edge_taps = _taps(
        edge_notes, 0.9986, 0.061, [12.0, -1.5, 1.0, 0.5, -2.0, 1.2, -1.1]
    )
    edge_taps[0] = {
        "time": round(edge_taps[0]["time"] + 0.210, 6),
        "strength": 0.18,
    }
    edge_taps.insert(0, {"time": 0.012, "strength": 0.22})
    edge_taps.append(
        {
            "time": round(edge_taps[-1]["time"] + 0.240, 6),
            "strength": 0.27,
        }
    )

    dropout_notes = _notes(18, 0.235, [29, 72, 108, 44, 126, 66, 83])
    dropout_taps = _taps(
        dropout_notes, 1.0012, 0.027, [0.8, -1.7, 1.3, -0.6, 2.1]
    )
    for index in sorted((6, 7, 13), reverse=True):
        del dropout_taps[index]

    return [
        {
            "name": "running-speed",
            "description": "1.0019 clock rate, short accelerating run",
            "true_rate": 1.0019,
            "true_offset": 0.034,
            "payload": {
                "notes": running_notes,
                "pedals": _pedals(),
                "taps": running_taps,
            },
        },
        {
            "name": "edge-mismatch",
            "description": "extra head/tail transients and a displaced first tap",
            "true_rate": 0.9986,
            "true_offset": 0.061,
            "payload": {
                "notes": edge_notes,
                "pedals": _pedals(),
                "taps": edge_taps,
            },
        },
        {
            "name": "brief-dropouts",
            "description": "three missing taps in two short bursts",
            "true_rate": 1.0012,
            "true_offset": 0.027,
            "payload": {
                "notes": dropout_notes,
                "pedals": _pedals(),
                "taps": dropout_taps,
            },
        },
    ]


def _print_scenario(scenario: dict[str, object]) -> None:
    payload = scenario["payload"]
    result = resynchronize(payload, SyncConfig())
    true_rate = float(scenario["true_rate"])
    true_offset = float(scenario["true_offset"])
    print(f"\n== {scenario['name']} ==")
    print(scenario["description"])
    print(
        f"true: rate={true_rate:.6f}, offset={true_offset * 1000:+.2f} ms | "
        f"estimated: rate={result.transform.rate:.6f}, "
        f"offset={result.transform.offset * 1000:+.2f} ms"
    )
    stats = result.statistics
    print(
        f"anchors: {stats['inlier_count']}/{stats['anchor_count']} inliers, "
        f"median|err|={stats['median_abs_error_ms']:.2f} ms, "
        f"p95|err|={stats['p95_abs_error_ms']:.2f} ms, "
        f"velocity r={stats['velocity_strength_correlation']:.3f}"
    )
    print("midi_onset  velocity  resynthesized  true_tap  error_ms")
    for comparison in result.comparisons:
        print(
            f"{float(comparison['source_onset']):9.4f}  "
            f"{int(comparison['velocity']):8d}  "
            f"{float(comparison['resynthesized_onset']):13.4f}  "
            f"{float(comparison['true_tap']):8.4f}  "
            f"{float(comparison['error_ms']):+8.2f}"
        )
    if result.unmatched_notes:
        print("unmatched MIDI onsets (missing or extra transient nearby):")
        for row in result.unmatched_notes:
            print(
                f"  note={row['note']} midi={row['source_onset']:.4f} "
                f"mapped={row['mapped_onset']:.4f}"
            )
    if result.warnings:
        print("warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")


def main() -> int:
    for scenario in scenarios():
        _print_scenario(scenario)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
