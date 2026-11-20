"""Command-line entry point.

Examples
--------
./run.sh demo --out-dir out                 # build synthetic stereo sample
./run.sh analyze out/demo_stereo.wav --out-dir out
"""

from __future__ import annotations

import argparse
import json
import os

from .make_demo import build_demo
from .pipeline import analyze_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="windtool",
        description="Offline wind-noise annotation and audition tool.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="generate a synthetic stereo demo file")
    demo.add_argument("--out-dir", default="out")
    demo.add_argument("--seed", type=int, default=20261121)

    analyze = sub.add_parser("analyze", help="annotate and process a stereo WAV")
    analyze.add_argument("wav")
    analyze.add_argument("--out-dir", default="out")
    analyze.add_argument("--title", default=None)

    args = parser.parse_args(argv)

    if args.command == "demo":
        os.makedirs(args.out_dir, exist_ok=True)
        path = os.path.join(args.out_dir, "demo_stereo.wav")
        info = build_demo(path, seed=args.seed)
        print(f"demo written: {path}")
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return 0

    payload = analyze_file(args.wav, args.out_dir, report_title=args.title)
    print(f"segments: {len(payload['segments'])}")
    for seg in payload["segments"]:
        print(
            f"  #{seg['index']} {seg['start_sec']:5.2f}-{seg['end_sec']:5.2f}s "
            f"{seg['label']:<11} target cut {seg['target_attenuation_db']:4.1f} dB"
        )
    print("outputs:")
    for name, path in payload["outputs"].items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
