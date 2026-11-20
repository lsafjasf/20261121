"""Command line interface for traceable BWF batch processing.

Usage:
    python -m bwf_batch run JOB.json        # apply batch edits, render masters
    python -m bwf_batch verify OUTDIR       # full vs. incremental comparison
    python -m bwf_batch show OUTDIR STEM    # dump envelope and version history

Job file format:
    {
      "outdir": "out",
      "files": [
        {"path": "take1.wav", "edits": [
          {"op": "add_segment", "name": "fade",
           "points": [[-0.5, -60.0], [1.0, 0.0]]},
          {"op": "set_merge_mode", "mode": "last_wins"},
          {"op": "add_point", "segment": "fade", "time": 2.0,
           "gain_db": -6.0}
        ]}
      ]
    }
Paths inside the job file resolve relative to the job file's directory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .batch import BatchProcessor
from .store import ProjectStore
from .verify import verify_all


def _cmd_run(args) -> int:
    with open(args.job, "r", encoding="utf-8") as fh:
        job = json.load(fh)
    base = os.path.dirname(os.path.abspath(args.job))
    outdir = job.get("outdir", "out")
    if not os.path.isabs(outdir):
        outdir = os.path.join(base, outdir)
    processor = BatchProcessor(outdir)
    for entry in job.get("files", []):
        path = entry["path"]
        if not os.path.isabs(path):
            path = os.path.join(base, path)
        versions = processor.process_file(path, entry.get("edits", []))
        stem = os.path.splitext(os.path.basename(path))[0]
        print("%s: %d adjustment(s) -> %s" % (stem, len(versions), versions[-1]))
    print("masters written to %s" % outdir)
    return 0


def _cmd_verify(args) -> int:
    store = ProjectStore(args.outdir)
    result = verify_all(store)
    for stem, report in result["files"].items():
        status = "OK" if report["equal"] else "MISMATCH"
        print(
            "%s [%s]: %d probe(s), %d mismatched sample(s)"
            % (stem, status, len(report["probes"]), report["total_mismatched_samples"])
        )
        for probe in report["probes"]:
            if probe["mismatched_samples"]:
                print(
                    "  %s probe [%d, %d): %d mismatch, max |delta|=%s"
                    % (
                        probe["kind"],
                        probe["start"],
                        probe["end"],
                        probe["mismatched_samples"],
                        probe["max_abs_delta"],
                    )
                )
                for diff in probe.get("sample_diffs", []):
                    print(
                        "    frame %(frame)d ch%(channel)d: full=%(full)s "
                        "incremental=%(incremental)s delta=%(delta)s" % diff
                    )
    return 0 if result["equal"] else 1


def _cmd_show(args) -> int:
    store = ProjectStore(args.outdir)
    env = store.load_envelope(args.stem)
    print(json.dumps(env.to_dict(), indent=2))
    for meta in store.iter_version_meta(args.stem):
        affected = meta["affected_frames"]
        span = "none" if not affected else "%d..%d" % (affected["start"], affected["end"])
        print(
            "%s %s render=%s affected=%s source_ref=%s"
            % (meta["version"], meta["op"]["op"], meta["render_mode"], span, meta["source_ref"])
        )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="bwf_batch")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="apply a batch job")
    p_run.add_argument("job")
    p_run.set_defaults(func=_cmd_run)

    p_verify = sub.add_parser("verify", help="compare full vs. incremental renders")
    p_verify.add_argument("outdir")
    p_verify.set_defaults(func=_cmd_verify)

    p_show = sub.add_parser("show", help="dump envelope and version history")
    p_show.add_argument("outdir")
    p_show.add_argument("stem")
    p_show.set_defaults(func=_cmd_show)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
