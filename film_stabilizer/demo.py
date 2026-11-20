from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

from .cli import _render_outputs, _write_crop_csv, _write_track_csv
from .io import open_source, write_manifest
from .stabilizer import StabilizerConfig, stabilize
from .synthetic import SyntheticConfig, SyntheticSequence
from .verify import verify_against_truth


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic film, stabilize it, and verify gate tracking"
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts/demo"))
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--splice-frame", type=int, default=41)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)

    synthetic = SyntheticSequence(
        SyntheticConfig(
            width=args.width,
            height=args.height,
            frames=args.frames,
            splice_frame=args.splice_frame,
            seed=args.seed,
        )
    )
    frames_dir = synthetic.write(args.out / "synthetic")
    source = open_source(frames_dir)
    result = stabilize(source, StabilizerConfig())
    render_args = Namespace(
        no_full_video=False,
        no_cropped_video=False,
        write_frames=False,
        burn_frame_number=False,
    )
    _render_outputs(source, result, args.out, render_args)
    _write_crop_csv(args.out / "crop_ranges.csv", result)
    _write_track_csv(args.out / "track.csv", result)
    verification = verify_against_truth(result, synthetic.truth)
    (args.out / "verification.json").write_text(
        json.dumps(verification.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    write_manifest(
        args.out / "manifest.json",
        {
            "tool": "film-stabilizer demo",
            "synthetic_config": {
                "width": args.width,
                "height": args.height,
                "frames": args.frames,
                "splice_frame": args.splice_frame,
                "seed": args.seed,
                "missing_left": sorted(synthetic.missing_left),
                "missing_right": sorted(synthetic.missing_right),
            },
            "common_crop": result.common_crop,
            "splice_frames": result.splice_frames,
            "verification": verification.to_dict(),
        },
    )
    print(json.dumps(verification.to_dict(), indent=2))
    if not verification.passed:
        raise SystemExit("verification failed")


if __name__ == "__main__":
    main()
