"""Batch orchestration: apply versioned edits and render broadcast masters.

Each edit op is one adjustment. The first adjustment for a file renders the
full master; later adjustments re-render only the affected frame range and
splice it into the existing master. Every adjustment is persisted as a
version with the source samples it referenced.
"""
from __future__ import annotations

import os

from .bwf import read_bwf
from .envelope import Envelope, Segment, ControlPoint
from .render import affected_frames, render_full, render_incremental
from .store import ProjectStore


def apply_op(env: Envelope, op: dict) -> None:
    """Apply one edit op to an envelope, in place."""
    kind = op["op"]
    if kind == "add_segment":
        env.add_segment(
            Segment(
                name=op["name"],
                points=[ControlPoint(t, db) for t, db in op["points"]],
            )
        )
    elif kind == "remove_segment":
        env.remove_segment(op["name"])
    elif kind == "add_point":
        env.add_point(op["segment"], op["time"], op["gain_db"])
    elif kind == "move_point":
        env.move_point(op["segment"], op["index"], op["time"], op["gain_db"])
    elif kind == "delete_point":
        env.delete_point(op["segment"], op["index"])
    elif kind == "set_merge_mode":
        if op["mode"] not in ("last_wins", "multiply", "max"):
            raise ValueError("unknown merge mode: %r" % op["mode"])
        env.merge_mode = op["mode"]
    else:
        raise ValueError("unknown op: %r" % kind)


class BatchProcessor:
    def __init__(self, outdir: str):
        self.store = ProjectStore(outdir)

    def process_file(self, source_path: str, edits: list) -> list:
        """Apply all edits to one source file; returns version ids."""
        stem = os.path.splitext(os.path.basename(source_path))[0]
        self.store.register_file(stem, source_path)
        src = read_bwf(source_path)
        versions = []
        for op in edits:
            versions.append(self._apply_one(stem, src, op))
        return versions

    def _apply_one(self, stem: str, src, op: dict) -> str:
        env_before = self.store.load_envelope(stem)
        env_after = env_before.copy()
        apply_op(env_after, op)

        time_range = env_before.affected_range(env_after, src.duration)
        frames = affected_frames(time_range, src) if time_range else None

        has_master = os.path.exists(self.store.master_path(stem))
        if frames is None and has_master:
            render_mode = "none"
            source_ref = None
        elif not has_master or frames == (0, src.frame_count):
            render_mode = "full"
            master = render_full(src, env_after)
            self.store.save_master(stem, master)
            source_ref = src.slice_frames(0, src.frame_count)
            frames = (0, src.frame_count)
        else:
            render_mode = "incremental"
            master = self.store.load_master(stem)
            master = render_incremental(src, env_after, master, frames)
            self.store.save_master(stem, master)
            source_ref = src.slice_frames(*frames)

        self.store.save_envelope(stem, env_after)
        return self.store.add_version(
            stem,
            op=op,
            env_before=env_before,
            env_after=env_after,
            affected_frames=frames,
            source_ref=source_ref,
            render_mode=render_mode,
        )
