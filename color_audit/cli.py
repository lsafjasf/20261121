from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .audit import (
    AuditError,
    DEFAULT_TARGET,
    inspect_media,
    verify,
)
from .ffmpeg import discover_tools
from .fixtures import generate_fixtures
from .model import FIELDS, parse_decisions, parse_tag_code
from .report import print_inspection, print_verification_summary, write_artifacts


def _inspection_payload(inspection: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": inspection["path"],
        "size": inspection["size"],
        "sha256": inspection["sha256"],
        "resolved_ffprobe_tags": inspection["resolved_ffprobe_tags"],
        "container": inspection["container"].to_dict(),
        "codec": inspection["codec"].to_dict(),
        "conflicts": [item.to_dict() for item in inspection["conflicts"]],
    }


def _target_from_args(args: argparse.Namespace) -> dict[str, int]:
    target = dict(DEFAULT_TARGET)
    for field_name in FIELDS:
        raw = getattr(args, f"target_{field_name}", None)
        if raw is not None:
            target[field_name] = parse_tag_code(field_name, raw)
    return target


def _add_interpretation_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--assume",
        action="append",
        default=[],
        metavar="FIELD=SOURCE",
        help="confirm one conflict, e.g. matrix=codec (repeatable/comma-separated)",
    )
    parser.add_argument(
        "--assume-layer",
        choices=("container", "codec"),
        help="use one layer for every conflicting field",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="never prompt; unresolved conflicts are errors",
    )


def _add_target_options(parser: argparse.ArgumentParser) -> None:
    for field_name in FIELDS:
        parser.add_argument(
            f"--target-{field_name}",
            metavar="VALUE",
            help=f"target {field_name} name or H.273 code",
        )


def _run_verify(args: argparse.Namespace, tools) -> int:
    try:
        decisions = parse_decisions(args.assume)
    except ValueError as exc:
        raise AuditError(str(exc)) from exc
    audit = verify(
        tools,
        source=args.input,
        output_dir=args.out,
        explicit_decisions=decisions,
        assume_layer=args.assume_layer,
        target=_target_from_args(args),
        manifest_path=args.manifest,
        tolerance=args.tolerance,
        interactive=not args.non_interactive,
    )
    artifacts = write_artifacts(audit, args.out)
    if args.json:
        print(json.dumps(audit, indent=2, ensure_ascii=False))
    else:
        print_verification_summary(audit, artifacts)
    return 0 if audit["result"]["status"] == "pass" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="color-audit",
        description="Inspect and verify FFmpeg container/codec colour tags.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="show container and codec declarations separately"
    )
    inspect_parser.add_argument("input")
    inspect_parser.add_argument("--json", action="store_true")

    verify_parser = subparsers.add_parser(
        "verify", help="transcode with confirmed tags and sample the result"
    )
    verify_parser.add_argument("input")
    verify_parser.add_argument(
        "--out", default="color-audit-output", help="artifact directory"
    )
    verify_parser.add_argument("--manifest", help="fixture manifest with sample points")
    verify_parser.add_argument("--tolerance", type=int, default=2)
    verify_parser.add_argument("--json", action="store_true")
    _add_interpretation_options(verify_parser)
    _add_target_options(verify_parser)

    fixtures_parser = subparsers.add_parser(
        "fixtures", help="generate grayscale, colour patches and a mistagged clip"
    )
    fixtures_parser.add_argument("--out", default="artifacts/fixtures")
    fixtures_parser.add_argument("--json", action="store_true")

    demo_parser = subparsers.add_parser(
        "demo", help="generate fixtures and verify the mistagged clip as Rec.709"
    )
    demo_parser.add_argument("--out", default="artifacts/demo")
    demo_parser.add_argument("--tolerance", type=int, default=2)
    demo_parser.add_argument("--json", action="store_true")
    _add_target_options(demo_parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        tools = discover_tools()
        if args.command == "inspect":
            inspection = inspect_media(tools, args.input)
            if args.json:
                print(json.dumps(_inspection_payload(inspection), indent=2))
            else:
                print_inspection(inspection)
            return 0
        if args.command == "fixtures":
            manifest = generate_fixtures(tools, args.out)
            if args.json:
                print(json.dumps(manifest, indent=2))
            else:
                print(f"fixtures written to {Path(args.out).resolve()}")
                print(f"manifest: {Path(args.out).resolve() / 'manifest.json'}")
                print(f"mistagged clip: {manifest['assets']['mistagged_container']}")
            return 0
        if args.command == "demo":
            fixture_dir = Path(args.out).resolve() / "fixtures"
            audit_dir = Path(args.out).resolve() / "audit"
            manifest = generate_fixtures(tools, fixture_dir)
            audit = verify(
                tools,
                source=fixture_dir / manifest["assets"]["mistagged_container"],
                output_dir=audit_dir,
                explicit_decisions={},
                assume_layer="codec",
                target=_target_from_args(args),
                manifest_path=fixture_dir / "manifest.json",
                tolerance=args.tolerance,
                interactive=False,
            )
            artifacts = write_artifacts(audit, audit_dir)
            if args.json:
                print(json.dumps(audit, indent=2, ensure_ascii=False))
            else:
                print(f"Fixtures: {fixture_dir}")
                print_verification_summary(audit, artifacts)
            return 0 if audit["result"]["status"] == "pass" else 2
        if args.command == "verify":
            return _run_verify(args, tools)
    except (AuditError, RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1

