"""Minimal WAV I/O (16-bit PCM and float), mono/stereo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Load a wav file as float64 in [-1, 1]; mixes down to mono."""
    sr, data = wavfile.read(str(path))
    if data.dtype == np.int16:
        x = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        x = data.astype(np.float64) / 2147483648.0
    elif data.dtype in (np.float32, np.float64):
        x = data.astype(np.float64)
    else:
        raise ValueError(f"unsupported wav dtype: {data.dtype}")
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x, int(sr)


def save_wav(path: str | Path, x: np.ndarray, sr: int) -> None:
    pcm = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    wavfile.write(str(path), sr, (pcm * 32767.0).astype(np.int16))
