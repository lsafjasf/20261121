from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .errors import LangcutError
from .manifest import OutputOptions, load_manifest
from .render import render


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="langcut",
        description="Render one frame-accurate program version per language.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("render", help="render all listed languages")
    command.add_argument("manifest", type=Path, help="asset manifest JSON")
    command.add_argument("--out", type=Path, required=True, help="output directory")
    command.add_argument(
        "--language",
        action="append",
        default=None,
        metavar="ID",
        help="render only this language (repeatable)",
    )
    command.add_argument("--extension", help="override output container extension")
    command.add_argument("--video-codec", dest="video_codec")
    command.add_argument("--audio-codec", dest="audio_codec")
    command.add_argument("--pixel-format", dest="pixel_format")
    command.add_argument("--filename", help="output stem template")
    command.add_argument("--crf", type=int)
    command.add_argument("--audio-sample-rate", type=int)
    command.add_argument("--quiet", action="store_true")
    return parser


def _options(args: argparse.Namespace, base: OutputOptions) -> OutputOptions:
    return OutputOptions(
        filename=args.filename or base.filename,
        extension=(args.extension or base.extension).lstrip("."),
        video_codec=args.video_codec or base.video_codec,
        audio_codec=args.audio_codec or base.audio_codec,
        pixel_format=args.pixel_format or base.pixel_format,
        crf=base.crf if args.crf is None else args.crf,
        audio_sample_rate=(
            base.audio_sample_rate
            if args.audio_sample_rate is None
            else args.audio_sample_rate
        ),
    )


def _exit_code(error: LangcutError) -> int:
    if error.code in {"FFMPEG_MISSING"}:
        return 3
    if error.code in {"RENDER_FAILED"}:
        return 1
    return 2


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        report = render(
            manifest,
            args.out,
            languages=args.language,
            output=_options(args, manifest.output),
        )
    except LangcutError as exc:
        print(f"langcut: {exc.code}: {exc.message}", file=sys.stderr)
        return _exit_code(exc)
    if not args.quiet:
        print(f"Rendered {len(report['languages'])} language version(s)")
        for language, item in report["languages"].items():
            timeline = item["timeline"]
            print(
                f"  {language}: {item['file']} | "
                f"program@{timeline['program_start_frame']}f "
                f"({timeline['program_start_timecode']}) | "
                f"preserved cut shift +{timeline['shift_frames']}f"
            )
        print(f"Chapter map: {report['chapter_map']}")
    return 0
