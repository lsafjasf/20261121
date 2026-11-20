from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import LabConfig
from .pipeline import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate and restore LCD persistence from a step chart."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="output directory (default: outputs)",
    )
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--skip-comparison-video",
        action="store_true",
        help="skip the large synchronized MP4 (charts and report still render)",
    )
    args = parser.parse_args()
    config = LabConfig(output_dir=args.output, seed=args.seed, fps=args.fps)
    summary = run_experiment(config, make_comparison_video=not args.skip_comparison_video)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
