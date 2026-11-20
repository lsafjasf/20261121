#!/usr/bin/env python3
"""Generate a deterministic stereo WAV whose left/right splices are offset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import wave


def _sample(value: int) -> int:
    value &= 0xFFFF
    return value - 0x10000 if value & 0x8000 else value


def _source_frames(frame_count: int) -> list[tuple[int, int]]:
    left_cuts = [0, 13, 29, 47, 71, frame_count]
    right_cuts = [0, 5, 17, 31, 62, frame_count]
    frames = []
    for index in range(frame_count):
        left_part = max(i for i, cut in enumerate(left_cuts[:-1]) if cut <= index)
        right_part = max(i for i, cut in enumerate(right_cuts[:-1]) if cut <= index)
        frames.append(
            (
                _sample(0x1100 + left_part * 0x2100 + index * 19),
                _sample(0x7300 - right_part * 0x1900 + index * 23),
            )
        )
    return frames


def _room_tone() -> list[tuple[int, int]]:
    return [
        (-1001, 1201),
        (-887, 1093),
        (-733, 977),
        (-611, 811),
        (-499, 733),
        (-353, 617),
        (-211, 503),
        (-97, 389),
        (43, 271),
        (181, 149),
        (307, 31),
        (449, -107),
        (577, -239),
    ]


def _write_wav(path: Path, frames: list[tuple[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(
            b"".join(struct.pack("<hh", left, right) for left, right in frames)
        )


def generate(directory: Path, frame_count: int = 96) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "desynced-source.wav"
    room_tone = directory / "room-tone.wav"
    _write_wav(source, _source_frames(frame_count))
    _write_wav(room_tone, _room_tone())
    rules = [
        {
            "id": "left-bed",
            "channel": "left",
            "start_frame": 3,
            "end_frame": 39,
            "priority": 1,
            "noise_offset_frames": 2,
        },
        {
            "id": "left-fix",
            "channel": "left",
            "start_frame": 17,
            "end_frame": 53,
            "priority": 9,
            "noise_offset_frames": 0,
        },
        {
            "id": "right-bed",
            "channel": "right",
            "start_frame": 8,
            "end_frame": 44,
            "priority": 2,
            "noise_offset_frames": 5,
        },
        {
            "id": "right-fix",
            "channel": "right",
            "start_frame": 31,
            "end_frame": 62,
            "priority": 8,
            "noise_offset_frames": 7,
        },
        {
            "id": "right-tie",
            "channel": "right",
            "start_frame": 37,
            "end_frame": 48,
            "priority": 8,
            "noise_offset_frames": 3,
        },
    ]
    config = {
        "schema": "roomtone-handoff.config.v1",
        "source": source.name,
        "room_tone": room_tone.name,
        "output_dir": "renders",
        "versions": [
            {"name": "dialogue-a", "regions": rules},
            {"name": "dialogue-b-reordered", "regions": list(reversed(rules))},
        ],
    }
    config_path = directory / "config.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return config_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(__file__).resolve().parent / "demo",
        help="folder for generated WAVs and config",
    )
    parser.add_argument("--frames", type=int, default=96)
    args = parser.parse_args()
    if args.frames < 72:
        parser.error("--frames must be at least 72")
    config = generate(args.directory, args.frames)
    print(config)


if __name__ == "__main__":
    main()
