"""Acceptance script: align the four recordings on the unified show clock.

Reads the persistent manifest (state/events.jsonl) and reports, for any
moment of the show, which tracks have footage and which are missing frames
(and why).  Without --at it scans the whole timeline and lists every
uncovered interval per track, classifying each as an explicitly marked
capture gap or an unmarked loss.

Exit code: 0 = every missing interval is an explicitly marked gap and all
segment files exist; 2 = unmarked loss or missing/corrupt files found.
"""

import argparse
import json
import os
import subprocess
import sys

from recorder.clockutil import NS_PER_SEC, ShowClock
from recorder.manifest import Manifest


def fmt_t(ns):
    return f"{ns / NS_PER_SEC:10.3f}s"


def coverage_holes(segments):
    """Uncovered intervals inside [first start, last end]."""
    segs = sorted((s for s in segments if s["end_ns"] is not None),
                  key=lambda s: s["start_ns"])
    holes = []
    cursor = None
    for seg in segs:
        if cursor is None:
            cursor = seg["end_ns"]
            continue
        if seg["start_ns"] > cursor:
            holes.append((cursor, seg["start_ns"]))
        cursor = max(cursor, seg["end_ns"])
    return holes


def gap_for(gaps, start, end):
    """Return the marked gap covering most of [start, end], if any."""
    best = None
    for gap in gaps:
        ov = min(end, gap["end_ns"]) - max(start, gap["start_ns"])
        if ov > 0 and (best is None or ov > best[0]):
            best = (ov, gap)
    return best


def frames_between(clock, start, end):
    f0 = clock.frame_of(start)
    f1 = clock.frame_of(end)
    return f0, f1


def probe_file(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path],
            capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            return None
        return float(json.loads(out.stdout)["format"]["duration"])
    except Exception:
        return None


def report_at(manifest, clock, tracks, t_ns, run_dir):
    print(f"--- alignment at show time {fmt_t(t_ns)} "
          f"(epoch {clock.to_epoch_ns(t_ns)}) ---")
    missing = 0
    for track in tracks:
        segs = manifest.segments(track)
        hit = next((s for s in segs
                    if s["start_ns"] <= t_ns < (s["end_ns"] or 0)), None)
        if hit:
            off = (t_ns - hit["start_ns"]) / NS_PER_SEC
            print(f"  track{track}: OK  {hit['file']}  (offset {off:.3f}s)")
            continue
        missing += 1
        gaps = manifest.gaps(track)
        gap = next((g for g in gaps
                    if g["start_ns"] <= t_ns < g["end_ns"]), None)
        if gap:
            reason = (f"capture gap [{fmt_t(gap['start_ns'])} .. "
                      f"{fmt_t(gap['end_ns'])}] ({gap['reason']})")
        elif not segs or t_ns < segs[0]["start_ns"]:
            reason = "track not started yet"
        elif t_ns >= max(s["end_ns"] or 0 for s in segs):
            reason = "track already ended"
        else:
            reason = "UNMARKED LOSS (hole between segments)"
        print(f"  track{track}: MISSING  {reason}")
    return missing


def main():
    parser = argparse.ArgumentParser(
        description="Align four-track recordings on the unified show clock")
    parser.add_argument("--state-dir", default="state")
    parser.add_argument("--run-dir", default="run")
    parser.add_argument("--tracks", type=int, default=4)
    parser.add_argument("--at", action="append", default=[],
                        help="show time in seconds (or 'now'); repeatable")
    parser.add_argument("--probe", action="store_true",
                        help="ffprobe every segment file (duration check)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    clock = ShowClock(args.state_dir)
    manifest = Manifest(args.state_dir, read_only=True)
    tracks = sorted(set(range(args.tracks)) | set(manifest.tracks()))

    failures = []

    for spec in args.at:
        t_ns = (clock.show_time_ns() if spec == "now"
                else int(float(spec) * NS_PER_SEC))
        report_at(manifest, clock, tracks, t_ns, args.run_dir)

    report = {"tracks": {}}
    for track in tracks:
        segs = manifest.segments(track)
        gaps = manifest.gaps(track)
        holes = coverage_holes(segs)
        entry = {"segments": len(segs), "gaps": gaps, "holes": []}
        if segs:
            first = min(s["start_ns"] for s in segs)
            last = max(s["end_ns"] or 0 for s in segs)
            covered = sum((s["end_ns"] or 0) - s["start_ns"] for s in segs)
            entry["span"] = [first, last]
            entry["covered_ns"] = covered
            print(f"track{track}: {len(segs)} segments, span "
                  f"[{fmt_t(first)} .. {fmt_t(last)}], "
                  f"covered {covered / NS_PER_SEC:.3f}s, "
                  f"{len(gaps)} marked gap(s)")
        else:
            print(f"track{track}: no segments")
        for start, end in holes:
            marked = gap_for(gaps, start, end)
            f0, f1 = frames_between(clock, start, end)
            n_frames = max(0, f1 - f0)
            kind = ("marked-gap" if marked and marked[0] >= 0.9 * (end - start)
                    else "UNMARKED-LOSS")
            hole = {"start_ns": start, "end_ns": end, "kind": kind,
                    "frames": [f0, f1]}
            entry["holes"].append(hole)
            print(f"    hole [{fmt_t(start)} .. {fmt_t(end)}] "
                  f"-> {kind}, {n_frames} frame(s) missing "
                  f"(frames {f0}..{f1 - 1})")
            if kind != "marked-gap":
                failures.append(f"track{track} unmarked loss at {fmt_t(start)}")
        for seg in segs:
            path = os.path.join(args.run_dir, seg["file"])
            if not os.path.exists(path):
                failures.append(f"track{track} missing file {seg['file']}")
                print(f"    MISSING FILE {seg['file']}")
            elif args.probe:
                dur = probe_file(path)
                declared = ((seg["end_ns"] or 0) - seg["start_ns"]) / NS_PER_SEC
                if dur is None:
                    failures.append(f"track{track} corrupt file {seg['file']}")
                    print(f"    CORRUPT FILE {seg['file']}")
                elif abs(dur - declared) > 1.0 and not seg["interrupted"]:
                    failures.append(
                        f"track{track} duration mismatch {seg['file']}")
                    print(f"    DURATION MISMATCH {seg['file']}: "
                          f"file {dur:.3f}s vs manifest {declared:.3f}s")
        report["tracks"][str(track)] = entry

    if args.json:
        print(json.dumps(report, indent=2))

    if failures:
        print(f"\nRESULT: FAIL ({len(failures)} problem(s))")
        for f in failures:
            print(f"  - {f}")
        sys.exit(2)
    print("\nRESULT: OK (all missing footage is explicitly marked; "
          "all segment files present)")


if __name__ == "__main__":
    main()
