"""Deterministic teaching demo: high-frequency carrier and short transients."""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

from .model import Event, TraceData
from .pipeline import render_trace_video
from .verify import verify_mp4

EVENTS = [
    Event(1.1373, "A", cue_hz=1180.0),
    Event(4.3179, "B", cue_hz=1480.0),
    Event(6.4021, "C", cue_hz=1760.0),
]
EXPLICIT_UNKNOWN = (5.35, 5.49)
MISSING_GAP = (2.40, 2.82)


def build_demo_trace() -> TraceData:
    rng = random.Random(20261121)
    times: list[float] = []
    voltages: list[float | None] = []
    t = 0.0
    base_dt = 1.0 / 24000.0
    while t < 7.2:
        t += base_dt * (0.72 + 0.56 * rng.random())
        if MISSING_GAP[0] <= t <= MISSING_GAP[1]:
            continue
        carrier = (
            0.31
            * (0.86 + 0.14 * math.sin(2.0 * math.pi * 0.31 * t))
            * math.sin(2.0 * math.pi * 8750.0 * t + 0.42 * math.sin(2.0 * math.pi * 1.7 * t))
        )
        drift = 0.11 * math.sin(2.0 * math.pi * 0.63 * t)
        noise = rng.uniform(-0.012, 0.012)
        transient = _transient(t)
        voltage = 0.08 + carrier + drift + noise + transient
        if EXPLICIT_UNKNOWN[0] <= t <= EXPLICIT_UNKNOWN[1]:
            voltages.append(None)
        else:
            voltages.append(voltage)
        times.append(t)
    return TraceData(
        times_s=times,
        voltages=voltages,
        unit="V",
        events=list(EVENTS),
        unknown_intervals=[EXPLICIT_UNKNOWN],
        source="synthetic high-frequency carrier + transients",
        gap_factor=8.0,
        metadata={"carrier_hz": "8750", "nominal_sample_rate_hz": "24000"},
    )


def _transient(time_s: float) -> float:
    values = 0.0
    for event, amplitude, width in zip(EVENTS, (0.92, -0.78, 1.05), (0.00046, 0.00058, 0.00038)):
        delta = time_s - event.time_s
        if abs(delta) > 0.007:
            continue
        pulse = amplitude * math.exp(-((delta / width) ** 2))
        ring = 0.18 * amplitude * math.exp(-abs(delta) / 0.0015) * math.sin(
            2.0 * math.pi * 3100.0 * delta
        )
        values += pulse + ring
    return values


def write_demo_inputs(directory: Path, data: TraceData) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "trace.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "voltage", "unit"])
        for time_s, voltage in zip(data.times_s, data.voltages):
            writer.writerow([f"{time_s:.9f}", "" if voltage is None else f"{voltage:.9f}", data.unit])
    events_path = directory / "events.json"
    events_payload = {
        "events": [
            {"time_s": event.time_s, "label": event.label, "cue_hz": event.cue_hz}
            for event in data.events
        ]
    }
    events_path.write_text(json.dumps(events_payload, indent=2) + "\n", encoding="utf-8")
    return csv_path, events_path


def run_demo(
    directory: str | Path,
    *,
    width: int = 960,
    height: int = 540,
    fps: int = 30,
    progress: bool = True,
) -> dict:
    root = Path(directory)
    data = build_demo_trace()
    csv_path, events_path = write_demo_inputs(root, data)
    output = root / "oscilloscope_demo.mp4"
    manifest = render_trace_video(
        data,
        output,
        fps=fps,
        window_seconds=0.75,
        width=width,
        height=height,
        supersample=2,
        title="OSCILLOSCOPE TRACE",
        subtitle="8750 Hz carrier / 24 kSa/s uneven sampling / transients",
        progress=progress,
    )
    report = verify_mp4(output, output.with_suffix(".manifest.json"))
    (root / "verification.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "manifest": manifest,
        "verification": report,
        "csv": csv_path,
        "events": events_path,
        "video": output,
    }
