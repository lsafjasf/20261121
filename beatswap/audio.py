"""Audio loading, loudness normalisation and resampling."""

from __future__ import annotations

import numpy as np
from scipy import io, signal

TARGET_SR = 22050


def load_wav(path: str, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Load a wav file, downmix to mono, resample to target_sr."""
    sr, data = io.wavfile.read(path)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.max(np.abs(data)) > 2.0:
        data = data / 32768.0
    if sr != target_sr:
        n_out = int(round(len(data) * target_sr / sr))
        data = signal.resample(data, n_out)
        sr = target_sr
    return data, sr


def rms_db(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x)) + 1e-12))
    return 20.0 * np.log10(rms + 1e-12)


def normalize_loudness(x: np.ndarray, target_db: float = -20.0) -> tuple[np.ndarray, float]:
    """RMS-normalise so both recordings sit at the same loudness.

    Returns (normalised signal, gain applied in dB). This is what makes the
    comparison robust to different recording levels.
    """
    gain_db = target_db - rms_db(x)
    gain = 10.0 ** (gain_db / 20.0)
    return x * gain, gain_db


def save_wav(path: str, x: np.ndarray, sr: int = TARGET_SR) -> None:
    peak = float(np.max(np.abs(x)) + 1e-12)
    if peak > 0.99:
        x = x / peak * 0.99
    io.wavfile.write(path, sr, (x * 32767.0).astype(np.int16))
