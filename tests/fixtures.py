"""Deterministic WAV fixtures with deliberately desynchronized stereo splices."""

from __future__ import annotations

import struct
from pathlib import Path
import wave


def desynchronized_stereo_frames(frame_count: int = 96) -> list[tuple[int, int]]:
    """Build independent L/R takes whose splice boundaries never line up."""
    left_cuts = {0, 13, 29, 47, 71, frame_count}
    right_cuts = {0, 5, 17, 31, 62, frame_count}
    left_cuts = sorted(cut for cut in left_cuts if cut <= frame_count)
    right_cuts = sorted(cut for cut in right_cuts if cut <= frame_count)
    frames: list[tuple[int, int]] = []
    for index in range(frame_count):
        left_segment = max(i for i, cut in enumerate(left_cuts[:-1]) if cut <= index)
        right_segment = max(i for i, cut in enumerate(right_cuts[:-1]) if cut <= index)
        left = 0x1100 + left_segment * 0x2100 + index * 19
        right = 0x7300 - right_segment * 0x1900 + index * 23
        frames.append((_int16(left), _int16(right)))
    return frames


def stereo_room_tone_frames() -> list[tuple[int, int]]:
    """A short, asymmetric two-channel room tone that must loop cleanly."""
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


def _int16(value: int) -> int:
    value &= 0xFFFF
    return value - 0x10000 if value & 0x8000 else value


def write_pcm16_wav(
    path: Path, frames: list[tuple[int, int]], sample_rate: int = 16000
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(struct.pack("<hh", left, right) for left, right in frames)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(payload)


def create_fixture_set(directory: Path) -> tuple[Path, Path]:
    source = directory / "desynced-source.wav"
    room_tone = directory / "room-tone.wav"
    write_pcm16_wav(source, desynchronized_stereo_frames())
    write_pcm16_wav(room_tone, stereo_room_tone_frames())
    return source, room_tone
