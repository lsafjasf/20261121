"""Adaptive segment-driven low-band reduction and reduction measurement."""

from __future__ import annotations

import numpy as np

from .config import AnalysisConfig
from .dsp import FrameFeatures, istft_channels
from .segmentation import Segment


def _frequency_gain_curve(
    freqs: np.ndarray, attenuation_db: float, cfg: AnalysisConfig
) -> np.ndarray:
    """Piecewise-linear attenuation following the configured anchor fractions."""
    if attenuation_db <= 0.0:
        return np.ones_like(freqs)
    anchors = cfg.gain_anchors_hz
    fractions = cfg.gain_anchor_fractions
    fraction = np.interp(freqs, anchors, fractions,
                         left=fractions[0], right=0.0)
    gain_db = -attenuation_db * fraction
    return 10.0 ** (gain_db / 20.0)


def build_gain_envelope(
    features: FrameFeatures,
    segments: list[Segment],
    cfg: AnalysisConfig,
) -> np.ndarray:
    """Per-frame, per-bin gain matrix (T, F) with attack/release smoothing."""
    n_frames = len(features.times)
    target = np.ones((n_frames, len(features.freqs)), dtype=float)
    for seg in segments:
        if seg.attenuation_db <= 0.0:
            continue
        curve = _frequency_gain_curve(features.freqs, seg.attenuation_db, cfg)
        sl = (features.times >= seg.start_sec) & (features.times <= seg.end_sec)
        target[sl] = curve

    hop = float(np.median(np.diff(features.times)))
    attack = np.exp(-hop / max(cfg.attack_sec, 1e-6))
    release = np.exp(-hop / max(cfg.release_sec, 1e-6))
    smoothed = np.ones_like(target)
    for t in range(1, n_frames):
        # Gain reduction ramps in fast (attack) and recovers slowly (release).
        coeff = np.where(target[t] < smoothed[t - 1], attack, release)
        smoothed[t] = coeff * smoothed[t - 1] + (1.0 - coeff) * target[t]
    return smoothed


def apply_reduction(
    features: FrameFeatures,
    gain: np.ndarray,
    n_fft: int,
    hop: int,
    win: np.ndarray,
    sr: int,
    length: int,
) -> np.ndarray:
    processed = features.stft * gain[np.newaxis, :, :]
    return istft_channels(processed, n_fft, hop, win, sr, length)


def band_rms_db(audio: np.ndarray, sr: int, band: tuple[float, float]) -> float:
    """Band-limited RMS in dBFS over a waveform (both channels averaged)."""
    from scipy import signal

    sos = signal.butter(4, band, btype="bandpass", fs=sr, output="sos")
    filtered = signal.sosfiltfilt(sos, audio, axis=-1)
    rms = np.sqrt(np.mean(filtered ** 2) + 1e-20)
    return float(20.0 * np.log10(rms + 1e-20))


def measure_wind_reduction(
    original: np.ndarray,
    processed: np.ndarray,
    sr: int,
    segments: list[Segment],
    cfg: AnalysisConfig,
) -> list[dict]:
    """Actual achieved low-band reduction per wind/mixed/bump segment.

    Measured on rendered waveforms (not on requested gains) so the number
    reflects the real audition result.
    """
    rows = []
    for seg in segments:
        if seg.label not in {"wind", "mixed_wind", "bump"}:
            continue
        a, b = int(seg.start_sec * sr), int(seg.end_sec * sr)
        before = band_rms_db(original[:, a:b], sr, cfg.wind_band_hz)
        after = band_rms_db(processed[:, a:b], sr, cfg.wind_band_hz)
        rows.append(
            {
                "segment": seg.index,
                "label": seg.label,
                "start_sec": seg.start_sec,
                "end_sec": seg.end_sec,
                "requested_db": seg.attenuation_db,
                "measured_low_band_reduction_db": round(before - after, 2),
                "before_dbfs": round(before, 2),
                "after_dbfs": round(after, 2),
            }
        )
    return rows
