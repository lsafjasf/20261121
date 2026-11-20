"""Editable gain envelope with explicit overlap-merge semantics.

Model
-----
An envelope is an ordered list of segments. Each segment is a named,
time-sorted list of control points (time in seconds, gain in dB) and is
evaluated by linear interpolation in the dB domain. Control points may lie
before the head (t < 0) or after the tail (t > duration) of the material:

  - Points outside [0, duration] still shape interpolation inside the file
    (e.g. a point at t=-1s pulls the gain at t=0 towards its value).
  - Outside a segment's own point span the segment holds its edge value.
  - Outside every segment's span the merged envelope holds the value at the
    nearest covered edge (constant extrapolation, never a fade to unity).

Overlap-merge semantics (explicit, per envelope)
------------------------------------------------
When several segments cover the same instant t:

  - "last_wins": the segment inserted last defines the gain (default).
  - "multiply":  segment gains are summed in dB (i.e. multiplied linearly).
  - "max":       the highest gain (in dB) wins.

Instants in a gap between segments (covered by none) take the value of the
nearest segment edge; ties resolve to the earlier segment.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

MERGE_LAST_WINS = "last_wins"
MERGE_MULTIPLY = "multiply"
MERGE_MAX = "max"
MERGE_MODES = (MERGE_LAST_WINS, MERGE_MULTIPLY, MERGE_MAX)

EPSILON_DB = 1e-9


@dataclass
class ControlPoint:
    time: float      # seconds; may be < 0 (pre-head) or > duration (post-tail)
    gain_db: float

    def to_dict(self) -> dict:
        return {"time": self.time, "gain_db": self.gain_db}


@dataclass
class Segment:
    name: str
    points: list = field(default_factory=list)  # of ControlPoint

    def __post_init__(self) -> None:
        self.points = [ControlPoint(p.time, p.gain_db) for p in self.points]
        self.points.sort(key=lambda p: p.time)
        if not self.points:
            raise ValueError("segment %r needs at least one point" % self.name)
        self._times = [p.time for p in self.points]

    @property
    def t0(self) -> float:
        return self.points[0].time

    @property
    def t1(self) -> float:
        return self.points[-1].time

    def eval_db(self, t: float) -> float:
        """Gain at t; constant hold outside the segment's own point span."""
        pts = self.points
        if t <= pts[0].time:
            return pts[0].gain_db
        if t >= pts[-1].time:
            return pts[-1].gain_db
        idx = bisect.bisect_right(self._times, t)
        a, b = pts[idx - 1], pts[idx]
        if b.time <= a.time:
            return b.gain_db
        frac = (t - a.time) / (b.time - a.time)
        return a.gain_db + frac * (b.gain_db - a.gain_db)

    def to_dict(self) -> dict:
        return {"name": self.name, "points": [p.to_dict() for p in self.points]}

    @classmethod
    def from_dict(cls, data: dict) -> "Segment":
        return cls(
            name=data["name"],
            points=[ControlPoint(p["time"], p["gain_db"]) for p in data["points"]],
        )


@dataclass
class Envelope:
    segments: list = field(default_factory=list)  # of Segment, insertion order
    merge_mode: str = MERGE_LAST_WINS

    def __post_init__(self) -> None:
        if self.merge_mode not in MERGE_MODES:
            raise ValueError("unknown merge mode: %r" % self.merge_mode)

    # -- evaluation ------------------------------------------------------

    def eval_db(self, t: float) -> float:
        """Merged gain at t under the envelope's explicit merge semantics."""
        if not self.segments:
            return 0.0
        lo = min(s.t0 for s in self.segments)
        hi = max(s.t1 for s in self.segments)
        if t < lo:
            t = lo
        elif t > hi:
            t = hi
        candidates = [s for s in self.segments if s.t0 <= t <= s.t1]
        if not candidates:
            return self._gap_value_db(t)
        if self.merge_mode == MERGE_MULTIPLY:
            return sum(s.eval_db(t) for s in candidates)
        if self.merge_mode == MERGE_MAX:
            return max(s.eval_db(t) for s in candidates)
        return candidates[-1].eval_db(t)  # MERGE_LAST_WINS

    def _gap_value_db(self, t: float) -> float:
        """Nearest-edge hold for instants covered by no segment."""
        best = None
        for seg in self.segments:
            if t < seg.t0:
                dist, value = seg.t0 - t, seg.eval_db(seg.t0)
            else:
                dist, value = t - seg.t1, seg.eval_db(seg.t1)
            if best is None or dist < best[0]:
                best = (dist, value)
        return best[1]

    def event_times(self) -> list:
        """Sorted union of all control-point times and segment edges."""
        times = set()
        for seg in self.segments:
            times.update(seg._times)
        return sorted(times)

    def breakpoints(self) -> list:
        """All times where the merged gain function may change slope.

        Includes event times plus, for max-merge, pairwise crossings of
        segment curves inside each event interval (both are linear there,
        so at most one crossing per pair per interval).
        """
        times = sorted(set(self.event_times()))
        extra = []
        for a, b in zip(times, times[1:]):
            mid = (a + b) / 2.0
            active = [s for s in self.segments if s.t0 <= mid <= s.t1]
            for i in range(len(active)):
                for j in range(i + 1, len(active)):
                    fa0, fa1 = active[i].eval_db(a), active[i].eval_db(b)
                    fb0, fb1 = active[j].eval_db(a), active[j].eval_db(b)
                    denom = (fa1 - fa0) - (fb1 - fb0)
                    if abs(denom) < 1e-12:
                        continue
                    frac = (fb0 - fa0) / denom
                    if 0.0 < frac < 1.0:
                        extra.append(a + frac * (b - a))
        return sorted(set(times) | set(extra))

    # -- editing ----------------------------------------------------------

    def find_segment(self, name: str) -> Segment:
        for seg in self.segments:
            if seg.name == name:
                return seg
        raise KeyError("no segment named %r" % name)

    def add_segment(self, segment: Segment) -> None:
        if any(s.name == segment.name for s in self.segments):
            raise ValueError("duplicate segment name: %r" % segment.name)
        self.segments.append(segment)

    def remove_segment(self, name: str) -> None:
        self.segments.remove(self.find_segment(name))

    def add_point(self, segment: str, time: float, gain_db: float) -> None:
        seg = self.find_segment(segment)
        seg.points.append(ControlPoint(time, gain_db))
        seg.__post_init__()

    def move_point(
        self, segment: str, index: int, time: float, gain_db: float
    ) -> None:
        seg = self.find_segment(segment)
        seg.points[index] = ControlPoint(time, gain_db)
        seg.__post_init__()

    def delete_point(self, segment: str, index: int) -> None:
        seg = self.find_segment(segment)
        if len(seg.points) <= 1:
            raise ValueError("segment %r must keep one point" % segment)
        del seg.points[index]
        seg.__post_init__()

    # -- diff / serialization ---------------------------------------------

    def affected_range(self, other: "Envelope", duration: float) -> tuple | None:
        """Time range where self and other differ, clamped to [0, duration].

        Both merged curves are linear on every open interval between
        consecutive breakpoints, but may be discontinuous at the
        breakpoints themselves (last_wins segment switches). Comparing at
        every breakpoint plus two interior probes per interval is exact:
        two distinct lines coincide at most once, so they cannot match at
        both a third and two thirds of an interval. Differing probes are
        expanded to their bounding breakpoints. Returns None when the
        envelopes agree over the whole material.
        """
        events = sorted(
            set(self.breakpoints() + other.breakpoints() + [0.0, duration])
        )
        probes = []  # (time, lo_event_index, hi_event_index)
        for i, t in enumerate(events):
            probes.append((t, i, i))
        for i in range(len(events) - 1):
            a, b = events[i], events[i + 1]
            if b > a:
                probes.append((a + (b - a) / 3.0, i, i + 1))
                probes.append((a + 2.0 * (b - a) / 3.0, i, i + 1))
        lo_index = None
        hi_index = None
        for t, i_lo, i_hi in probes:
            if not 0.0 <= t <= duration:
                continue
            if abs(self.eval_db(t) - other.eval_db(t)) > EPSILON_DB:
                lo_index = i_lo if lo_index is None else min(lo_index, i_lo)
                hi_index = i_hi if hi_index is None else max(hi_index, i_hi)
        if lo_index is None:
            return None
        return (
            max(0.0, events[lo_index]),
            min(duration, events[hi_index]),
        )

    def to_dict(self) -> dict:
        return {
            "merge_mode": self.merge_mode,
            "segments": [s.to_dict() for s in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Envelope":
        return cls(
            segments=[Segment.from_dict(s) for s in data.get("segments", [])],
            merge_mode=data.get("merge_mode", MERGE_LAST_WINS),
        )

    def copy(self) -> "Envelope":
        return Envelope.from_dict(self.to_dict())
