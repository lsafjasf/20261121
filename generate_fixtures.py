#!/usr/bin/env python3
"""Generate the deterministic laboratory fixtures and manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from audio_lab.fixtures import build_fixtures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="fixtures", help="fixture output directory")
    args = parser.parse_args()
    destination = Path(args.out)
    manifest = build_fixtures(destination)
    print(f"wrote {len(manifest['cases'])} cases to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
