"""Persistent, append-only recording manifest.

All state lives in ``state/events.jsonl``.  Every segment boundary and every
capture-gap transition is appended as one JSON line and fsynced, so completed
segments and their unified timestamps survive a process restart (or crash).
On load, the log is replayed to rebuild in-memory state; segments that were
left open by a dead process are closed as ``interrupted`` at the timestamp of
the last event seen for that track.
"""

import json
import os
import threading
import time


class Manifest:
    def __init__(self, state_dir, read_only=False):
        self.state_dir = state_dir
        self.read_only = read_only
        self.log_path = os.path.join(state_dir, "events.jsonl")
        self.snapshot_path = os.path.join(state_dir, "manifest.json")
        self._lock = threading.Lock()
        self._events = []
        self._segments = {}  # track -> [ {id,file,start_ns,end_ns,interrupted} ]
        self._gaps = {}      # track -> [ {start_ns,end_ns,reason} ]
        self._open_seg = {}  # track -> segment dict not yet closed
        self._open_gap = {}  # track -> gap dict not yet closed
        if os.path.exists(self.log_path):
            with open(self.log_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        self._apply(json.loads(line))
            self._close_dangling()
        self._log = None
        if not read_only:
            os.makedirs(state_dir, exist_ok=True)
            self._log = open(self.log_path, "a", encoding="utf-8")

    # -- replay ----------------------------------------------------------

    def _track(self, mapping, track):
        return mapping.setdefault(int(track), [])

    def _apply(self, ev):
        self._events.append(ev)
        kind = ev["ev"]
        track = ev.get("track")
        if kind == "segment_open":
            seg = {
                "id": ev["seg"],
                "file": ev["file"],
                "start_ns": ev["t"],
                "end_ns": None,
                "interrupted": False,
            }
            self._track(self._segments, track).append(seg)
            self._open_seg[track] = seg
        elif kind == "segment_close":
            seg = self._open_seg.pop(track, None)
            if seg is not None and seg["id"] == ev["seg"]:
                seg["end_ns"] = ev["t"]
        elif kind == "gap_start":
            gap = {"start_ns": ev["t"], "end_ns": None,
                   "reason": ev.get("reason", "capture-lost")}
            self._track(self._gaps, track).append(gap)
            self._open_gap[track] = gap
        elif kind == "gap_end":
            gap = self._open_gap.pop(track, None)
            if gap is not None:
                gap["end_ns"] = ev["t"]

    def _close_dangling(self):
        """Close records left open by a previous, now-dead process."""
        last_t = {}
        for ev in self._events:
            if "track" in ev:
                last_t[ev["track"]] = ev["t"]
        for track, seg in list(self._open_seg.items()):
            seg["end_ns"] = last_t.get(track, seg["start_ns"])
            seg["interrupted"] = True
            del self._open_seg[track]
        for track, gap in list(self._open_gap.items()):
            gap["end_ns"] = last_t.get(track, gap["start_ns"])
            gap["reason"] = gap["reason"] + "+service-restart"
            del self._open_gap[track]

    # -- writing ---------------------------------------------------------

    def emit(self, kind, t_ns, track=None, **fields):
        if self._log is None:
            raise RuntimeError("manifest is read-only")
        ev = {"ev": kind, "t": int(t_ns)}
        if track is not None:
            ev["track"] = int(track)
        ev.update(fields)
        with self._lock:
            self._log.write(json.dumps(ev) + "\n")
            self._log.flush()
            os.fsync(self._log.fileno())
            self._apply(ev)
            self._write_snapshot()
        return ev

    def _write_snapshot(self):
        snap = {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "segments": {str(k): v for k, v in self._segments.items()},
            "gaps": {str(k): v for k, v in self._gaps.items()},
        }
        tmp = self.snapshot_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(snap, fh, indent=2)
        os.replace(tmp, self.snapshot_path)

    # -- reading ---------------------------------------------------------

    def segments(self, track):
        return list(self._segments.get(int(track), []))

    def gaps(self, track):
        return list(self._gaps.get(int(track), []))

    def tracks(self):
        return sorted(set(self._segments) | set(self._gaps))

    def close(self):
        if self._log is not None:
            with self._lock:
                self._log.close()
