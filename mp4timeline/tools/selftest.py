"""Assertion-based checks run against the generated fixture.

Run after `make_fixture`; validates the three-layer model end to end and
cross-checks the parser against ffprobe's packet view.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .timing import parse_movie, presentation_map


def ffprobe_packets(movie: Path) -> list[dict]:
    out = subprocess.run([
        "ffprobe", "-v", "error", "-ignore_editlist", "1",
        "-select_streams", "v:0",
        "-show_packets", "-show_entries", "packet=pts,dts,flags",
        "-of", "json", str(movie),
    ], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)["packets"]


def run_checks(movie: Path) -> None:
    data = movie.read_bytes()
    movie_ts, tracks, _ = parse_movie(data)
    video = next(t for t in tracks if t.is_video)
    audio = next(t for t in tracks if t.handler == "soun")

    assert movie_ts == 1000, f"movie timescale {movie_ts}"
    assert len(video.samples) == 75, "expect 75 coded video samples"

    # Layer 2: B-frame reordering must be present and internally consistent.
    assert video.samples[0].composition_offset > 0, "expected encoder pre-roll"
    assert video.display_order[:4] == [0, 3, 2, 4], video.display_order[:4]
    for pos, sample_index in enumerate(video.display_order):
        sample = video.samples[sample_index]
        assert sample.pts == sample.dts + sample.composition_offset

    # Cross-check PTS/DTS ordering against ffprobe (ignoring edit list units).
    packets = ffprobe_packets(movie)
    probe_pts = [int(p["pts"]) for p in packets]
    our_pts = [s.pts for s in video.samples]
    assert probe_pts == our_pts, "parser PTS sequence disagrees with ffprobe"

    # Layer 3: edit list shape.
    empty = [e for e in video.edits if e.is_empty]
    media = [e for e in video.edits if not e.is_empty]
    assert len(empty) == 3, "pre-roll + two paused gaps"
    assert len(media) == 3, "three recorded segments"
    assert video.edits[0].segment_duration == 700

    # Audio starts one pre-roll + 600 ms delay after movie origin.
    audio_rows = presentation_map(audio)
    assert abs(audio_rows[0]["movie_pts_seconds"] - 1.3) < 1e-6

    # Presentation mapping: three picture segments at 0.7 / 2.1 / 3.5 s.
    rows = presentation_map(video)
    starts = [round(r["movie_pts_seconds"], 3) for r in rows
              if r["display_position"] in (0, 25, 50)]
    assert starts == [0.7, 2.1, 3.5], starts
    assert len(rows) == 75

    # Every displayed frame maps back to a real decode sample.
    assert all(0 <= r["sample_index"] < 75 for r in rows)

    print("self-test OK: 75 samples, reorder",
          video.display_order[:6], "segment starts", starts)


def main() -> None:
    movie = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/fixture.mp4")
    run_checks(movie)


if __name__ == "__main__":
    main()
