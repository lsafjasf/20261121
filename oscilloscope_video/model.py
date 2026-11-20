"""Core data model and timeline rules.

All public times are seconds in the capture clock.  Rendering never assumes
that samples are evenly spaced; samples are only binned when they are mapped
to screen columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, isfinite
from statistics import median
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Event:
    """A real pulse-trigger event in capture time."""

    time_s: float
    label: str = "TRIG"
    cue_hz: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.time_s, (int, float)) or not isfinite(self.time_s):
            raise ValueError("event time must be a finite number")
        if self.cue_hz is not None and not 200.0 <= self.cue_hz <= 6000.0:
            raise ValueError("event cue_hz must be between 200 and 6000 Hz")

    @property
    def safe_label(self) -> str:
        return " ".join(str(self.label).split())[:24].upper() or "TRIG"


@dataclass
class TraceData:
    """Voltage samples plus trigger events.

    ``voltages`` may contain ``None`` for a missing/untrusted sample.  The
    original unit string is deliberately retained instead of converting units.
    """

    times_s: Sequence[float]
    voltages: Sequence[float | None]
    unit: str
    events: list[Event] = field(default_factory=list)
    unknown_intervals: list[tuple[float, float]] = field(default_factory=list)
    source: str = "<memory>"
    gap_factor: float = 8.0
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.times_s) != len(self.voltages):
            raise ValueError("time and voltage arrays have different lengths")
        if not self.unit.strip():
            raise ValueError("unit must not be empty")
        previous = float("-inf")
        for index, value in enumerate(self.times_s):
            current = float(value)
            if current != current or current in (float("inf"), float("-inf")):
                raise ValueError(f"sample {index}: time is not finite")
            if current <= previous:
                raise ValueError(
                    f"sample {index}: times must be strictly increasing "
                    f"({current} follows {previous})"
                )
            previous = current
        if self.gap_factor < 1.0:
            raise ValueError("gap_factor must be at least 1")
        self.events = sorted(self.events, key=lambda event: event.time_s)
        self.unknown_intervals = merge_intervals(self.unknown_intervals)

    @property
    def valid_times(self) -> list[float]:
        return [
            float(time_s)
            for time_s, voltage in zip(self.times_s, self.voltages)
            if voltage is not None and float(voltage) == float(voltage)
        ]

    @property
    def valid_voltages(self) -> list[float]:
        return [
            float(voltage)
            for voltage in self.voltages
            if voltage is not None and float(voltage) == float(voltage)
        ]

    def gap_threshold(self) -> float:
        """Largest ordinary sample spacing before a gap is marked unknown."""

        times = [float(value) for value in self.times_s]
        deltas = [b - a for a, b in zip(times, times[1:]) if b > a]
        if not deltas:
            return 0.0
        return median(deltas) * self.gap_factor

    def effective_unknown_intervals(self) -> list[tuple[float, float]]:
        """Explicit gaps plus gaps caused by sparse rows or invalid samples."""

        found = list(self.unknown_intervals)
        threshold = self.gap_threshold()
        active_start: float | None = None
        active_end: float | None = None
        if threshold:
            for index, (left, right) in enumerate(zip(self.times_s, self.times_s[1:])):
                invalid_left = self.voltages[index] is None
                invalid_right = self.voltages[index + 1] is None
                if float(right) - float(left) > threshold or invalid_left or invalid_right:
                    if active_start is None:
                        active_start = float(left)
                    active_end = float(right)
                elif active_start is not None and active_end is not None:
                    found.append((active_start, active_end))
                    active_start = active_end = None
        if active_start is not None and active_end is not None:
            found.append((active_start, active_end))
        return merge_intervals(found)

    def sample_runs(self) -> list[int]:
        """Assign each sample to a contiguous drawable run.

        Unknown intervals, invalid voltages, and large time gaps split the
        trace.  No line is interpolated across a run boundary.
        """

        gaps = self.effective_unknown_intervals()
        threshold = self.gap_threshold()
        gap_index = 0
        run = 0
        runs: list[int] = []
        previous_invalid = True
        for index, time_s in enumerate(self.times_s):
            t = float(time_s)
            while gap_index < len(gaps) and gaps[gap_index][1] <= t:
                gap_index += 1
            inside_gap = (
                gap_index < len(gaps)
                and gaps[gap_index][0] < t < gaps[gap_index][1]
            )
            invalid = self.voltages[index] is None or inside_gap
            if index == 0:
                runs.append(run)
                previous_invalid = invalid
                continue
            previous = float(self.times_s[index - 1])
            if not invalid and (
                previous_invalid or (threshold and t - previous > threshold)
            ):
                run += 1
            runs.append(run)
            previous_invalid = invalid
        return runs


@dataclass(frozen=True)
class Timeline:
    """Mapping from capture time to constant-frame-rate playback time."""

    capture_start_s: float
    capture_end_s: float
    window_span_s: float
    duration_s: float
    fps: int
    frame_count: int

    def __post_init__(self) -> None:
        if self.capture_end_s < self.capture_start_s:
            raise ValueError("capture end precedes capture start")
        if self.window_span_s <= 0:
            raise ValueError("window span must be positive")
        if self.duration_s <= 0:
            raise ValueError("duration must be positive")
        if self.fps <= 0:
            raise ValueError("fps must be a positive integer")
        if self.frame_count <= 0:
            raise ValueError("frame_count must be positive")

    @property
    def capture_span_s(self) -> float:
        return self.capture_end_s - self.capture_start_s

    @property
    def time_scale(self) -> float:
        if self.capture_span_s <= 0:
            return 1.0
        return self.capture_span_s / self.duration_s

    def video_time(self, capture_time_s: float) -> float:
        return (capture_time_s - self.capture_start_s) / self.time_scale

    def first_frame(self, capture_time_s: float) -> int:
        """First frame whose presentation time is at or after an event."""

        frame = ceil(self.video_time(capture_time_s) * self.fps - 1e-9)
        return max(0, min(self.frame_count - 1, frame))

    def window_center(self, frame_index: int) -> float:
        return self.capture_start_s + self.time_scale * frame_index / self.fps

    def window_start(self, frame_index: int) -> float:
        return self.window_center(frame_index) - self.window_span_s / 2.0

    @classmethod
    def fit(
        cls,
        capture_start_s: float,
        capture_end_s: float,
        window_span_s: float,
        fps: int,
        duration_s: float | None = None,
    ) -> "Timeline":
        span = max(0.0, capture_end_s - capture_start_s)
        nominal_duration = max(span, 1.0 / fps)
        playback_duration = nominal_duration if duration_s is None else duration_s
        if playback_duration <= 0:
            raise ValueError("duration must be positive")
        # Keep a frame at (or after) the capture end so an event exactly at the
        # end still has a well-defined first frame >= its presentation time.
        frame_count = max(1, ceil(playback_duration * fps - 1e-9) + 1)
        return cls(
            capture_start_s=capture_start_s,
            capture_end_s=capture_end_s,
            window_span_s=window_span_s,
            duration_s=playback_duration,
            fps=fps,
            frame_count=frame_count,
        )


def merge_intervals(
    intervals: Iterable[tuple[float, float]],
) -> list[tuple[float, float]]:
    cleaned: list[tuple[float, float]] = []
    for raw_start, raw_end in intervals:
        start, end = float(raw_start), float(raw_end)
        if start != start or end != end:
            raise ValueError("unknown interval endpoints must be finite")
        if end <= start:
            raise ValueError("unknown interval end must exceed its start")
        cleaned.append((start, end))
    cleaned.sort()
    merged: list[tuple[float, float]] = []
    for start, end in cleaned:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def capture_bounds(data: TraceData) -> tuple[float, float]:
    candidates = list(data.times_s)
    candidates.extend(event.time_s for event in data.events)
    for start, end in data.unknown_intervals:
        candidates.extend((start, end))
    if not candidates:
        raise ValueError("trace has no samples or events")
    return min(candidates), max(candidates)
