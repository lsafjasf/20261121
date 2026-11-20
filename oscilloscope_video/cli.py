"""Command-line interface for scope trace video generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .demo import run_demo
from .inputs import load_input
from .pipeline import render_trace_video
from .verify import verify_mp4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scope-video",
        description="Render unevenly sampled voltage traces as CFR MP4 files.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    render = subparsers.add_parser("render", help="render one trace")
    render.add_argument("input", type=Path, help="CSV or JSON trace")
    render.add_argument("output", type=Path, help="output MP4")
    render.add_argument("--events", type=Path, help="optional event JSON")
    render.add_argument(
        "--unknown",
        action="append",
        default=[],
        metavar="START:END",
        help="explicit unknown interval; repeat or comma-separate",
    )
    render.add_argument("--unit", help="voltage unit when input has no unit")
    render.add_argument("--gap-factor", type=float, default=8.0)
    render.add_argument("--fps", type=int, default=30)
    render.add_argument("--window", type=float, default=0.6, help="visible span in seconds")
    render.add_argument(
        "--duration",
        type=float,
        help="playback seconds (default: real capture duration)",
    )
    render.add_argument("--width", type=int, default=960)
    render.add_argument("--height", type=int, default=540)
    render.add_argument("--supersample", type=int, choices=(1, 2, 3, 4), default=2)
    render.add_argument("--title", default="OSCILLOSCOPE TRACE")
    render.add_argument("--subtitle", default="")
    render.add_argument("--y-min", type=float)
    render.add_argument("--y-max", type=float)
    render.add_argument("--cue-seconds", type=float, default=0.085)
    render.add_argument("--crf", type=int, default=18)
    render.add_argument(
        "--preset",
        choices=("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"),
        default="medium",
    )
    render.add_argument("--verify", action="store_true", help="decode and verify the MP4")
    render.add_argument("--quiet", action="store_true")

    demo = subparsers.add_parser(
        "demo",
        help="write the synthetic carrier/transient demo and its verification",
    )
    demo.add_argument("--output-dir", type=Path, default=Path("artifacts/demo"))
    demo.add_argument("--width", type=int, default=960)
    demo.add_argument("--height", type=int, default=540)
    demo.add_argument("--fps", type=int, default=30)
    demo.add_argument("--quiet", action="store_true")

    verify = subparsers.add_parser("verify", help="decode MP4 and check cues/frames")
    verify.add_argument("video", type=Path)
    verify.add_argument("--manifest", type=Path)
    verify.add_argument("--audio-tolerance-ms", type=float, default=10.0)
    verify.add_argument("--min-correlation", type=float, default=0.62)
    verify.add_argument("--json-out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "render":
            return _command_render(args)
        if args.command == "demo":
            return _command_demo(args)
        if args.command == "verify":
            return _command_verify(args, parser)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    return 2


def _command_render(args: argparse.Namespace) -> int:
    data = load_input(
        args.input,
        unit=args.unit,
        events_path=args.events,
        unknown=args.unknown,
        gap_factor=args.gap_factor,
    )
    manifest = render_trace_video(
        data,
        args.output,
        fps=args.fps,
        window_seconds=args.window,
        duration_seconds=args.duration,
        width=args.width,
        height=args.height,
        supersample=args.supersample,
        title=args.title,
        subtitle=args.subtitle,
        y_min=args.y_min,
        y_max=args.y_max,
        cue_seconds=args.cue_seconds,
        crf=args.crf,
        preset=args.preset,
        progress=not args.quiet,
    )
    manifest_path = args.output.with_suffix(".manifest.json")
    print(
        f"wrote {args.output} ({manifest['video']['frame_count']} frames, "
        f"{manifest['video']['fps']} fps) and {manifest_path}"
    )
    if args.verify:
        report = verify_mp4(args.output, manifest_path)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["passed"] else 1
    return 0


def _command_demo(args: argparse.Namespace) -> int:
    result = run_demo(
        args.output_dir,
        width=args.width,
        height=args.height,
        fps=args.fps,
        progress=not args.quiet,
    )
    report = result["verification"]
    print(
        f"wrote {result['video']}, "
        f"{result['video'].with_suffix('.manifest.json')}, "
        f"and {args.output_dir / 'verification.json'}"
    )
    print(f"verification: {'PASS' if report['passed'] else 'FAIL'}")
    if not report["passed"]:
        for issue in report["issues"]:
            print(f"- {issue}", file=sys.stderr)
    return 0 if report["passed"] else 1


def _command_verify(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    manifest = args.manifest or args.video.with_suffix(".manifest.json")
    if not manifest.exists():
        parser.error(f"manifest not found: {manifest}")
    report = verify_mp4(
        args.video,
        manifest,
        audio_tolerance_ms=args.audio_tolerance_ms,
        min_waveform_correlation=args.min_correlation,
    )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 1
