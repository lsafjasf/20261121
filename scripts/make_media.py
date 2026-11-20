"""Generate the short project's synthetic source media with FFmpeg.

Each source is a lossless FFV1/PCM clip with a solid RGB color and a single
sine tone so later verification can identify the contributing source both
visually and spectrally.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

WIDTH = 320
HEIGHT = 180

SOURCES = {
    "red": {
        "fps": 24,
        "frames": 120,
        "color": "0xc81e1e",
        "tone": 220,
    },
    "blue": {
        "fps": 24,
        "frames": 72,
        "color": "0x1e3cdc",
        "tone": 440,
    },
    "green": {
        "fps": 30,
        "frames": 60,
        "color": "0x1ebe3c",
        "tone": 880,
    },
}


def make_one(name: str, spec: dict, out_dir: Path) -> Path:
    duration = spec["frames"] / spec["fps"]
    out_path = out_dir / f"{name}.mkv"
    filtergraph = (
        f"color=c={spec['color']}:s={WIDTH}x{HEIGHT}:r={spec['fps']}:"
        f"d={duration:.6f}[v];"
        f"sine=frequency={spec['tone']}:sample_rate=48000:duration={duration:.6f}"
        f",aformat=channel_layouts=stereo[a]"
    )
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-filter_complex", filtergraph,
        "-map", "[v]", "-map", "[a]",
        "-c:v", "ffv1", "-level", "3", "-coder", "1",
        "-context", "1", "-g", "1", "-slices", "4",
        "-c:a", "pcm_s16le",
        str(out_path),
    ]
    subprocess.run(command, check=True)
    print(f"wrote {out_path} ({spec['fps']} fps, {spec['frames']} frames)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", help="directory for generated media")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in SOURCES.items():
        make_one(name, spec, out_dir)


if __name__ == "__main__":
    main()
