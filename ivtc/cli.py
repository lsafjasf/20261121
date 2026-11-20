"""Command line interface for the ivtc toolkit."""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import backend
from .detector import analyze
from .fieldmap import dump_cut_neighbourhood, dump_output_provenance
from .reconstruct import manifest, reconstruct
from .report import build_report, render_text, write_json
from .streams import load_frames, write_pgm, write_raw
from .synth import build_mixed_clip, timeline_truth
from .verify import verify_reconstruction


def _cmd_sample(args) -> int:
    frames, _truth = build_mixed_clip()
    os.makedirs(args.outdir, exist_ok=True)
    write_raw(frames, os.path.join(args.outdir, "mixed_2997i.gray"))
    for frame in frames:
        write_pgm(frame, os.path.join(args.outdir, f"frame_{frame.index:03d}.pgm"))
    write_json(timeline_truth(), os.path.join(args.outdir, "ground_truth.json"))
    print(f"wrote {len(frames)} interlaced frames "
          f"({frames[0].width}x{frames[0].height}) to {args.outdir}")
    print("ground truth timeline:")
    for seg in timeline_truth()["segments"]:
        print(f"  {seg['kind']:<9} fields {seg['field_start']:>2}-{seg['field_end']:<2} "
              f"scene {seg['scene']}")
    return 0


def _load(args):
    if args.sample:
        frames, _ = build_mixed_clip()
        return frames
    if not args.input:
        raise SystemExit("error: --input is required unless --sample is used")
    return load_frames(args.input, args.width, args.height)


def _cmd_analyze(args) -> int:
    frames = _load(args)
    analysis = analyze(frames, min_confidence=args.min_confidence)
    print(render_text(analysis))
    if args.report:
        write_json(build_report(analysis), args.report)
        print(f"\nJSON report written to {args.report}")
    return 0


def _cmd_reconstruct(args) -> int:
    frames = _load(args)
    analysis = analyze(frames, min_confidence=args.min_confidence)
    rebuilt = reconstruct(analysis, min_confidence=args.min_confidence)
    os.makedirs(args.outdir, exist_ok=True)
    write_raw([item.frame for item in rebuilt],
              os.path.join(args.outdir, "progressive.gray"))
    for item in rebuilt:
        write_pgm(item.frame,
                  os.path.join(args.outdir, f"prog_{item.frame.index:03d}.pgm"))
    man = manifest(rebuilt)
    write_json(man, os.path.join(args.outdir, "manifest.json"))
    write_json(build_report(analysis), os.path.join(args.outdir, "report.json"))
    print(f"reconstructed {len(rebuilt)} progressive frames -> {args.outdir}")
    print("each manifest entry records top_field_index/bottom_field_index "
          "of the source 29.97i stream")
    return 0


def _cmd_verify(args) -> int:
    frames = _load(args)
    analysis = analyze(frames, min_confidence=args.min_confidence)
    rebuilt = reconstruct(analysis, min_confidence=args.min_confidence)
    result = verify_reconstruction(analysis, rebuilt)
    print(f"backend={backend.backend_name()}  "
          f"reconstructed={len(rebuilt)}  all_passed={result['all_passed']}")
    for row in result["frames"]:
        flag = "ok " if row["passed"] else "FAIL"
        print(f"  [{flag}] out{row['output_frame']:02d} "
              f"T={row['top_field']:<2} B={row['bottom_field']:<2} "
              f"comb_ratio={row['comb_ratio']:<7} "
              f"boundary={row['boundary_mismatch']}")
    ctrl = result["negative_control"]
    if ctrl.get("available"):
        print("  negative control (deliberate field mismatch):")
        print(f"    matched    ratio={ctrl['matched_weave']['comb_ratio']} "
              f"boundary={ctrl['matched_weave']['boundary_mismatch']}")
        print(f"    mismatched ratio={ctrl['mismatched_weave']['comb_ratio']} "
              f"boundary={ctrl['mismatched_weave']['boundary_mismatch']} "
              f"detected={ctrl['mismatch_detected']}")
    if args.report:
        write_json(result, args.report)
    return 0 if result["all_passed"] and ctrl.get("mismatch_detected", True) else 1


def _cmd_fieldmap(args) -> int:
    frames = _load(args)
    analysis = analyze(frames, min_confidence=args.min_confidence)
    rebuilt = reconstruct(analysis, min_confidence=args.min_confidence)
    print(dump_cut_neighbourhood(analysis))
    print(dump_output_provenance(rebuilt))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ivtc",
        description="Detect 3:2 pulldown / native 60i / cut points in "
                    "29.97i archive video and inverse-telecine confident runs")
    parser.add_argument("--input", help="input clip: pgm glob, concatenated "
                        "pgm stream, or raw .gray/.yuv (needs --width/--height)")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--sample", action="store_true",
                        help="use the built-in synthetic mixed-cadence clip")
    parser.add_argument("--min-confidence", type=float, default=0.9)
    sub = parser.add_subparsers(dest="command", required=True)

    p_sample = sub.add_parser("sample", help="generate the mixed sample clip")
    p_sample.add_argument("--outdir", default="artifacts/sample")
    p_sample.set_defaults(func=_cmd_sample)

    p_analyze = sub.add_parser("analyze", help="label cadence segments and cuts")
    p_analyze.add_argument("--report", help="write JSON report here")
    p_analyze.set_defaults(func=_cmd_analyze)

    p_recon = sub.add_parser("reconstruct", help="weave progressive frames "
                             "for confident pulldown segments")
    p_recon.add_argument("--outdir", default="artifacts/reconstructed")
    p_recon.set_defaults(func=_cmd_reconstruct)

    p_verify = sub.add_parser("verify", help="verify provenance and combing")
    p_verify.add_argument("--report", help="write JSON verification report")
    p_verify.set_defaults(func=_cmd_verify)

    p_map = sub.add_parser("fieldmap", help="show field order around cuts "
                           "and output-frame provenance")
    p_map.set_defaults(func=_cmd_fieldmap)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
