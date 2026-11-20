"""Command line entry point.

Example::

    python -m relisten \
        --annotations labels.json --output out/ \
        --track A=tracks/a.wav --track B=tracks/b.wav \
        --track C=tracks/c.wav --track D=tracks/d.wav
"""

from __future__ import annotations

import argparse

from .audio import read_wav
from .annotations import load_annotations
from .builder import build_package
from .model import BuildConfig


def _parse_track(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"track must be SPEAKER=PATH, got {value!r}"
        )
    speaker_id, path = value.split("=", 1)
    return speaker_id.strip(), path.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build per-speaker relisten clips with source mapping."
    )
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--track",
        action="append",
        type=_parse_track,
        default=[],
        help="speaker track as SPEAKER=PATH; repeat for each speaker",
    )
    parser.add_argument("--merge-gap", type=float, default=1.2)
    parser.add_argument("--short-utterance", type=float, default=2.0)
    parser.add_argument("--context-pre", type=float, default=0.4)
    parser.add_argument("--context-post", type=float, default=0.4)
    parser.add_argument("--energy-search", type=float, default=0.3)
    parser.add_argument("--energy-window", type=float, default=0.08)
    args = parser.parse_args(argv)

    utterances = load_annotations(args.annotations)
    tracks = {
        speaker_id: read_wav(path, speaker_id)
        for speaker_id, path in args.track
    }
    config = BuildConfig(
        merge_gap=args.merge_gap,
        short_utterance=args.short_utterance,
        context_pre=args.context_pre,
        context_post=args.context_post,
        energy_window=args.energy_window,
        energy_search=args.energy_search,
    )
    manifest = build_package(tracks, utterances, args.output, config)
    print(
        f"wrote {len(manifest['clips'])} clips for "
        f"{len(manifest['speakers'])} speakers to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
