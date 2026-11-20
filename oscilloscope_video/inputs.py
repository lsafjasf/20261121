"""CSV/JSON ingestion for traces, units, events, and unknown intervals."""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .model import Event, TraceData

TIME_KEYS = {"time_s", "time", "t", "seconds", "timestamp_s", "timestamp"}
VALUE_KEYS = {
    "voltage",
    "voltage_v",
    "volts",
    "value",
    "amplitude",
    "v",
    "sample",
}
TRIGGER_KEYS = {"trigger", "pulse", "event", "trig"}
UNIT_KEYS = {"unit", "units", "voltage_unit"}
NULL_TEXT = {"", "nan", "na", "null", "none", "?", "unknown"}


def load_input(
    path: str | Path,
    *,
    unit: str | None = None,
    events_path: str | Path | None = None,
    unknown: Iterable[str] = (),
    gap_factor: float = 8.0,
) -> TraceData:
    """Load a trace and merge optional events and unknown interval strings."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".json":
        data = _load_json(source)
    elif suffix in {".csv", ".tsv", ".txt"}:
        data = _load_csv(source)
    else:
        raise ValueError(f"unsupported input extension: {source.suffix}")

    file_unit = data.pop("unit", None)
    if file_unit is not None and unit is not None and str(file_unit) != unit:
        raise ValueError(
            f"unit conflict: input declares {file_unit!r}, CLI declares {unit!r}; "
            "refusing to relabel data"
        )
    resolved_unit = str(file_unit if file_unit is not None else unit or "").strip()
    if not resolved_unit:
        raise ValueError("a voltage unit is required (input metadata or --unit)")

    events = [_event_from_value(value) for value in data.pop("events", [])]
    if events_path:
        events.extend(_load_events(Path(events_path)))
    events.extend(_edge_events(data.pop("trigger_values", []), data["times_s"]))
    events.extend(_event_from_value(value, index=index) for index, value in enumerate(data.pop("edge_events", [])))

    intervals = [
        _interval_from_value(value)
        for value in data.pop("unknown_intervals", [])
    ]
    intervals.extend(parse_unknown_specs(unknown))

    return TraceData(
        times_s=data["times_s"],
        voltages=data["voltages"],
        unit=resolved_unit,
        events=events,
        unknown_intervals=intervals,
        source=str(source),
        gap_factor=gap_factor,
        metadata={str(key): str(value) for key, value in data.get("metadata", {}).items()},
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("JSON input must be an object")

    result: dict[str, Any] = {
        "unit": raw.get("unit"),
        "events": [],
        "unknown_intervals": list(raw.get("unknown_intervals", [])),
        "metadata": dict(raw.get("metadata", {})),
    }
    if "samples" in raw:
        pairs = []
        for value in raw["samples"]:
            if isinstance(value, dict):
                pairs.append((value.get("time_s", value.get("time")), value.get("voltage", value.get("value"))))
            elif isinstance(value, (list, tuple)) and len(value) == 2:
                pairs.append((value[0], value[1]))
            else:
                raise ValueError(f"bad JSON sample: {value!r}")
        result["times_s"] = [float(pair[0]) for pair in pairs]
        result["voltages"] = [_optional_float(pair[1]) for pair in pairs]
    elif "times_s" in raw and "voltages" in raw:
        if len(raw["times_s"]) != len(raw["voltages"]):
            raise ValueError("JSON times_s and voltages have different lengths")
        result["times_s"] = [float(value) for value in raw["times_s"]]
        result["voltages"] = [_optional_float(value) for value in raw["voltages"]]
    else:
        raise ValueError("JSON needs 'samples' or 'times_s' plus 'voltages'")
    result["events"] = list(raw.get("events", []))
    return result


def _load_csv(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        raise ValueError("CSV input is empty")
    delimiter = _detect_delimiter(lines[0])
    rows = list(csv.reader(io.StringIO("\n".join(lines)), delimiter=delimiter))
    header = [cell.strip().lower().replace(" ", "_") for cell in rows[0]]
    time_col = _find_column(header, TIME_KEYS, "time")
    value_col = _find_column(header, VALUE_KEYS, "voltage")
    trigger_col = _find_optional_column(header, TRIGGER_KEYS)
    unit_col = _find_optional_column(header, UNIT_KEYS)

    times: list[float] = []
    voltages: list[float | None] = []
    triggers: list[bool] = []
    units: set[str] = set()
    for row_number, row in enumerate(rows[1:], start=2):
        if not row or all(not cell.strip() for cell in row):
            continue
        try:
            time_s = float(row[time_col].strip())
            voltage = _optional_float(row[value_col])
        except (IndexError, ValueError) as error:
            raise ValueError(f"{path}:{row_number}: bad sample row") from error
        times.append(time_s)
        voltages.append(voltage)
        if trigger_col is not None:
            triggers.append(_truthy(row[trigger_col] if trigger_col < len(row) else ""))
        if unit_col is not None and unit_col < len(row) and row[unit_col].strip():
            units.add(row[unit_col].strip())
    if len(units) > 1:
        raise ValueError(f"CSV contains multiple units: {sorted(units)}")
    return {
        "unit": next(iter(units)) if units else None,
        "times_s": times,
        "voltages": voltages,
        "trigger_values": triggers,
        "events": [],
        "unknown_intervals": [],
        "metadata": {},
    }


def _load_events(path: Path) -> list[Event]:
    with path.open("r", encoding="utf-8-sig") as handle:
        raw = json.load(handle)
    if isinstance(raw, dict):
        raw = raw.get("events", [])
    if not isinstance(raw, list):
        raise ValueError("event JSON must be a list or an object with 'events'")
    return [_event_from_value(value, index=index) for index, value in enumerate(raw)]


def _event_from_value(value: Any, index: int | None = None) -> Event:
    if isinstance(value, (int, float)):
        return Event(float(value), label=f"TRIG {index + 1}" if index is not None else "TRIG")
    if not isinstance(value, dict):
        raise ValueError(f"bad event: {value!r}")
    time_s = value.get("time_s", value.get("time", value.get("t")))
    if time_s is None:
        raise ValueError(f"event has no time: {value!r}")
    label = value.get("label", value.get("name", "TRIG"))
    cue_hz = value.get("cue_hz", value.get("beep_hz"))
    return Event(
        time_s=float(time_s),
        label=str(label),
        cue_hz=None if cue_hz is None else float(cue_hz),
    )


def _interval_from_value(value: Any) -> tuple[float, float]:
    if isinstance(value, dict):
        start = value.get("start_s", value.get("start"))
        end = value.get("end_s", value.get("end"))
    else:
        start, end = value
    return float(start), float(end)


def _edge_events(states: list[bool], times: list[float]) -> list[Event]:
    events: list[Event] = []
    was_high = False
    for state, time_s in zip(states, times):
        if state and not was_high:
            events.append(Event(float(time_s), label=f"TRIG {len(events) + 1}"))
        was_high = state
    return events


def parse_unknown_specs(specs: Iterable[str]) -> list[tuple[float, float]]:
    """Parse ``start:end`` strings (comma separated values may also be used)."""

    intervals: list[tuple[float, float]] = []
    for spec in specs:
        for part in str(spec).split(","):
            if not part.strip():
                continue
            if ":" not in part:
                raise ValueError(f"unknown interval must be START:END: {part!r}")
            left, right = part.split(":", 1)
            intervals.append((float(left), float(right)))
    return intervals


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in NULL_TEXT:
            return None
        value = stripped
    result = float(value)
    return result if math.isfinite(result) else None


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "high", "yes", "on"}:
            return True
        if normalized in {"0", "false", "low", "no", "off", "", "nan"}:
            return False
        return float(normalized) >= 0.5
    return bool(value)


def _detect_delimiter(header: str) -> str:
    candidates = {",": header.count(","), ";": header.count(";"), "\t": header.count("\t")}
    delimiter, count = max(candidates.items(), key=lambda item: item[1])
    if count == 0:
        raise ValueError("could not detect CSV delimiter")
    return delimiter


def _find_column(header: list[str], names: set[str], kind: str) -> int:
    column = _find_optional_column(header, names)
    if column is None:
        raise ValueError(f"CSV has no {kind} column; headers: {header}")
    return column


def _find_optional_column(header: list[str], names: set[str]) -> int | None:
    for index, name in enumerate(header):
        if name in names:
            return index
    return None
