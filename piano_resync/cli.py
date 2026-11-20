from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .config import SyncConfig
from .sync import SyncResult, resynchronize


def _parser() -> argparse.ArgumentParser:
    defaults = SyncConfig()
    parser = argparse.ArgumentParser(
        prog="piano-resync",
        description=(
            "Estimate MIDI-to-audio clock drift from note/tap anchors and "
            "rewrite notes and pedal intervals onto the audio timeline."
        ),
    )
    parser.add_argument("input", help="input JSON file, or - for stdin")
    parser.add_argument("-o", "--output", help="write result JSON here; default is stdout")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="also print a short human-readable summary to stderr",
    )
    parser.add_argument("--pretty", action="store_true", help="indent JSON output")
    parser.add_argument(
        "--timing-tolerance",
        type=float,
        default=defaults.timing_tolerance,
        help="anchor residual scale in seconds (default: 0.025)",
    )
    parser.add_argument(
        "--match-window",
        type=float,
        default=defaults.match_window,
        help="maximum note/tap pairing distance in seconds (default: 0.150)",
    )
    parser.add_argument(
        "--max-rate-error",
        type=float,
        default=defaults.max_rate_error,
        help="coarse clock-rate search half-width (default: 0.020)",
    )
    parser.add_argument(
        "--initial-offset-window",
        type=float,
        default=defaults.initial_offset_window,
        help="coarse offset search half-width in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--min-anchors",
        type=int,
        default=defaults.min_anchors,
        help="warn below this many reliable anchors (default: 3)",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    return parser


def _load(path: str) -> object:
    if path == "-":
        return json.load(sys.stdin)
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _emit(payload: object, output: str | None, pretty: bool) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None)
    if output:
        Path(output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def _summary(result: SyncResult) -> str:
    stats = result.statistics
    transform = result.transform
    return (
        f"rate={transform.rate:.8f} ({(transform.rate - 1) * 1e6:+.1f} ppm) "
        f"offset={transform.offset * 1000:+.2f} ms\n"
        f"anchors={stats['inlier_count']}/{stats['anchor_count']} inliers  "
        f"median|err|={stats['median_abs_error_ms']:.2f} ms  "
        f"p95|err|={stats['p95_abs_error_ms']:.2f} ms\n"
        f"unmatched notes={stats['unmatched_note_count']}  "
        f"unmatched taps={stats['unmatched_tap_count']}  "
        f"warnings={len(result.warnings)}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = SyncConfig(
            timing_tolerance=args.timing_tolerance,
            match_window=args.match_window,
            max_rate_error=args.max_rate_error,
            initial_offset_window=args.initial_offset_window,
            min_anchors=args.min_anchors,
        )
        payload = _load(args.input)
        if not isinstance(payload, dict):
            raise ValueError("input JSON must be an object")
        result = resynchronize(payload, config)
        _emit(result.to_dict(), args.output, args.pretty)
        if args.summary:
            print(_summary(result), file=sys.stderr)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"piano-resync: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

