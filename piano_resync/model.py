from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    return float(value)


def _first(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


@dataclass(frozen=True)
class Note:
    note: int
    start: float
    end: float
    velocity: int = 64

    def __post_init__(self) -> None:
        if not 0 <= self.note <= 127:
            raise ValueError("note must be in the MIDI range 0..127")
        if self.start < 0 or self.end < self.start:
            raise ValueError("note requires 0 <= start <= end")
        if not 1 <= self.velocity <= 127:
            raise ValueError("velocity must be in 1..127")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Note":
        note = _first(raw, "note", "key", "note_number", "pitch")
        start = _first(raw, "start", "onset", "on", "time")
        end = _first(raw, "end", "offset", "off")
        duration = _first(raw, "duration", "length")
        if note is None or start is None:
            raise ValueError("each note needs note/key and start/onset")
        if end is None:
            if duration is None:
                raise ValueError("each note needs end/offset or duration")
            end = _number(start, "start") + _number(duration, "duration")
        velocity = _first(raw, "velocity", "vel", default=64)
        velocity_value = _number(velocity, "velocity")
        if 0 < velocity_value <= 1:
            velocity_value *= 127
        velocity_int = max(1, min(127, round(velocity_value)))
        return cls(
            note=round(_number(note, "note")),
            start=_number(start, "start"),
            end=_number(end, "end"),
            velocity=velocity_int,
        )


_PEDAL_NAMES = {64: "sustain", 66: "sostenuto", 67: "soft"}


@dataclass(frozen=True)
class PedalInterval:
    kind: str
    start: float
    end: float
    value: float = 1.0

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("pedal kind must not be empty")
        if self.start < 0 or self.end < self.start:
            raise ValueError("pedal requires 0 <= start <= end")
        if self.value < 0 or self.value > 1:
            raise ValueError("pedal value must be between 0 and 1")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PedalInterval":
        start = _first(raw, "start", "onset", "on", "time")
        end = _first(raw, "end", "offset", "off")
        if start is None or end is None:
            raise ValueError("each pedal interval needs start and end")
        pedal = _first(raw, "kind", "pedal", "controller", "cc", default=64)
        if isinstance(pedal, str):
            kind = pedal
        elif isinstance(pedal, int):
            kind = _PEDAL_NAMES.get(pedal, f"cc{pedal}")
        else:
            raise ValueError("pedal kind must be a string or controller number")
        return cls(
            kind=kind,
            start=_number(start, "pedal start"),
            end=_number(end, "pedal end"),
            value=_number(_first(raw, "value", default=1.0), "pedal value"),
        )


@dataclass(frozen=True)
class Tap:
    time: float
    strength: float = 1.0

    def __post_init__(self) -> None:
        if self.time < 0:
            raise ValueError("tap time must be non-negative")
        if self.strength < 0:
            raise ValueError("tap strength must be non-negative")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Tap":
        time = _first(raw, "time", "onset", "start")
        if time is None:
            raise ValueError("each tap needs time/onset")
        return cls(
            time=_number(time, "tap time"),
            strength=_number(
                _first(raw, "strength", "amplitude", "velocity", default=1.0),
                "tap strength",
            ),
        )


@dataclass(frozen=True)
class Recording:
    notes: tuple[Note, ...]
    pedals: tuple[PedalInterval, ...]
    taps: tuple[Tap, ...]

    def __post_init__(self) -> None:
        if any(a.start > b.start for a, b in zip(self.notes, self.notes[1:])):
            raise ValueError("notes must be sorted by start time")
        if any(a.time > b.time for a, b in zip(self.taps, self.taps[1:])):
            raise ValueError("taps must be sorted by time")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Recording":
        midi = raw.get("midi", {})
        audio = raw.get("audio", {})
        note_rows = _first(raw, "notes", default=None)
        if note_rows is None:
            note_rows = _first(midi, "notes", default=[])
        pedal_rows = _first(raw, "pedals", default=None)
        if pedal_rows is None:
            pedal_rows = _first(midi, "pedals", default=[])
        tap_rows = _first(raw, "taps", "onsets", default=None)
        if tap_rows is None:
            tap_rows = _first(audio, "taps", "onsets", default=[])
        if not isinstance(note_rows, Iterable) or isinstance(note_rows, (str, bytes)):
            raise ValueError("notes must be a list")
        if not isinstance(pedal_rows, Iterable) or isinstance(pedal_rows, (str, bytes)):
            raise ValueError("pedals must be a list")
        if not isinstance(tap_rows, Iterable) or isinstance(tap_rows, (str, bytes)):
            raise ValueError("taps must be a list")
        notes = sorted((Note.from_dict(row) for row in note_rows), key=lambda item: item.start)
        pedals = tuple(PedalInterval.from_dict(row) for row in pedal_rows)
        taps = sorted((Tap.from_dict(row) for row in tap_rows), key=lambda item: item.time)
        return cls(notes=tuple(notes), pedals=pedals, taps=taps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "notes": [
                {
                    "note": note.note,
                    "start": note.start,
                    "end": note.end,
                    "velocity": note.velocity,
                }
                for note in self.notes
            ],
            "pedals": [
                {
                    "kind": pedal.kind,
                    "start": pedal.start,
                    "end": pedal.end,
                    "value": pedal.value,
                }
                for pedal in self.pedals
            ],
            "taps": [
                {"time": tap.time, "strength": tap.strength} for tap in self.taps
            ],
        }
