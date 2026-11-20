"""Audio loading/saving and report serialization."""

from __future__ import annotations

import json
import os

import numpy as np
import soundfile as sf


def load_stereo(path: str) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=True, dtype="float64")
    if audio.shape[1] != 2:
        raise ValueError(
            f"{path}: expected 2 channels, got {audio.shape[1]}. "
            "The tool is built for two mics on one handheld recorder."
        )
    return audio.T.copy(), sr


def save_stereo(path: str, audio: np.ndarray, sr: int) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    peak = float(np.max(np.abs(audio)))
    if peak > 0.999:
        audio = audio * (0.999 / peak)
    sf.write(path, audio.T, sr, subtype="PCM_16")


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
