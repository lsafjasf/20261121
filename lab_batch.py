#!/usr/bin/env python3
"""Run the transient pre-echo laboratory and write report/audio artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from audio_lab.metrics import MetricSettings
from audio_lab.runner import run_lab


def _build_settings(args: argparse.Namespace) -> MetricSettings | None:
    overrides = {
        "window_ms": args.window_ms,
        "guard_ms": args.guard_ms,
        "source_visible_dbfs": args.source_visible_dbfs,
        "source_dominance_db": args.source_dominance_db,
    }
    if all(value is None for value in overrides.values()):
        return None
    defaults = MetricSettings()
    return MetricSettings(
        **{
            key: defaults.__dict__[key] if value is None else value
            for key, value in overrides.items()
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Encode transient fixtures with three lossy codecs, "
        "calibrate delay, measure pre-onset leakage, and generate an HTML report."
    )
    parser.add_argument("--fixtures", default="fixtures", help="fixture directory")
    parser.add_argument("--out", default="lab_output", help="output directory")
    parser.add_argument("--window-ms", type=float, default=None)
    parser.add_argument("--guard-ms", type=float, default=None)
    parser.add_argument("--source-visible-dbfs", type=float, default=None)
    parser.add_argument("--source-dominance-db", type=float, default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return a non-zero exit status when an acceptance check fails",
    )
    args = parser.parse_args()
    try:
        result = run_lab(
            Path(args.fixtures),
            Path(args.out),
            metric_settings=_build_settings(args),
        )
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    report = result.output_dir / "report.html"
    print(f"\nreport: {report.resolve()}")
    print(
        f"acceptance: {result.acceptance['passed_cases']}/"
        f"{result.acceptance['total_cases']} cases"
    )
    if args.strict and not result.acceptance["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
