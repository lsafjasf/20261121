"""Command line entry point: ``python -m local_render``."""

from __future__ import annotations

import argparse
import sys

from local_render.errors import LocalRenderError, MissingHandleError
from local_render.ffmpeg_runner import render
from local_render.planner import build_plan


def _print_handle_error(error: MissingHandleError) -> int:
    print("ERROR: insufficient media handle(s) for transition(s):", file=sys.stderr)
    for detail in error.details:
        print(f"  - {detail}", file=sys.stderr)
    return 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="local_render",
        description="Render an OTIO timeline locally with FFmpeg.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate timeline handles")
    validate.add_argument("timeline")

    render_parser = sub.add_parser("render", help="render timeline to a media file")
    render_parser.add_argument("timeline")
    render_parser.add_argument("output")
    render_parser.add_argument(
        "--workdir",
        default="work/render",
        help="directory for lossless track intermediates and the manifest",
    )

    args = parser.parse_args(argv)
    try:
        plan = build_plan(args.timeline)
        if args.command == "validate":
            transitions = sum(len(track.transitions) for track in plan.video_tracks)
            print(
                f"OK: {plan.total_frames} frames at {plan.timeline_rate} fps, "
                f"{len(plan.video_tracks)} video / {len(plan.audio_tracks)} audio "
                f"track(s), {transitions} video transition(s), all handles present"
            )
            return 0
        result = render(plan, args.output, args.workdir)
        print(f"rendered: {result['output']}")
        for name, path in result["intermediates"].items():
            print(f"  intermediate {name}: {path}")
        print(f"  manifest: {result['manifest']}")
        return 0
    except MissingHandleError as error:
        return _print_handle_error(error)
    except LocalRenderError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
