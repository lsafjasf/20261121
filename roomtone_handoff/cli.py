"""Command-line interface for rendering and verifying handoff batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import __version__
from .errors import HandoffError
from .model import load_config
from .service import render_all
from .verify import verify_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="roomtone-handoff",
        description=(
            "Render per-channel room-tone replacements from a pristine WAV "
            "and verify their byte-level manifests."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    render = commands.add_parser("render", help="render all versions in a config")
    render.add_argument("config", type=Path, help="JSON batch configuration")
    render.add_argument(
        "--output-dir",
        type=Path,
        help="override output_dir from the config (relative to the current directory)",
    )
    render.add_argument(
        "--json", action="store_true", help="print a machine-readable summary"
    )

    verify = commands.add_parser("verify", help="verify one or more manifests")
    verify.add_argument("manifests", nargs="+", type=Path)
    verify.add_argument(
        "--json", action="store_true", help="print a machine-readable summary"
    )
    return parser


def _render(args: argparse.Namespace) -> int:
    config = load_config(args.config, args.output_dir)
    outputs = render_all(config)
    if args.json:
        payload = {
            "output_dir": str(config.output_dir),
            "versions": [
                {
                    "name": output.version,
                    "wav": str(output.wav_path),
                    "manifest": str(output.manifest_path),
                    "csv": str(output.csv_path),
                    "sha256": output.result.output_sha256,
                    "effective_replacements": len(output.result.replacements),
                }
                for output in outputs
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    for output in outputs:
        print(
            f"{output.version}: {output.wav_path} "
            f"({len(output.result.replacements)} effective replacements, "
            f"sha256 {output.result.output_sha256})"
        )
        print(f"  manifest: {output.manifest_path}")
        print(f"  csv:      {output.csv_path}")
    return 0


def _verify(args: argparse.Namespace) -> int:
    reports = [verify_manifest(path) for path in args.manifests]
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "version": report.version,
                        "output": str(report.output_path),
                        "outside_pcm_bytes_checked": report.outside_pcm_bytes_checked,
                        "effective_replacements": report.replacement_count,
                    }
                    for report in reports
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    for report in reports:
        print(
            f"OK {report.version}: {report.output_path} "
            f"({report.replacement_count} replacements, "
            f"{report.outside_pcm_bytes_checked} outside PCM bytes checked)"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "render":
            return _render(args)
        return _verify(args)
    except (HandoffError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

