"""Measure note-frequency trajectories for quantitative validation."""

from __future__ import annotations

import numpy as np


def note_frequency_track(
    audio: np.ndarray,
    fs: float,
    nominal_freq: float,
    *,
    win_ms: float = 90.0,
    hop_ms: float = 20.0,
    rel_band: float = 0.18,
) -> tuple[np.ndarray, np.ndarray]:
    """Track one note's fundamental via parabolic-interpolated STFT peaks.

    Returns frame centre times (s) and instantaneous frequency (Hz).
    """
    win = int(round(fs * win_ms / 1000.0))
    hop = int(round(fs * hop_ms / 1000.0))
    window = np.hanning(win)
    freqs = np.fft.rfftfreq(win, 1.0 / fs)
    band = (freqs > nominal_freq * (1 - rel_band)) & (
        freqs < nominal_freq * (1 + rel_band)
    )
    band_idx = np.flatnonzero(band)

    times, values = [], []
    for start in range(0, len(audio) - win, hop):
        spectrum = np.abs(np.fft.rfft(audio[start:start + win] * window))
        local = spectrum[band_idx]
        peak = int(np.argmax(local))
        f = _parabolic_peak(local, peak, freqs[band_idx])
        times.append((start + win / 2.0) / fs)
        values.append(f)
    return np.asarray(times), np.asarray(values)


def _parabolic_peak(mag: np.ndarray, peak: int, freqs: np.ndarray) -> float:
    if 0 < peak < len(mag) - 1:
        a, b, c = mag[peak - 1], mag[peak], mag[peak + 1]
        denom = a - 2 * b + c
        shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
        df = freqs[1] - freqs[0]
        return float(freqs[peak] + shift * df)
    return float(freqs[peak])


def trim_edges(times: np.ndarray, values: np.ndarray, edge: float = 0.4) -> np.ndarray:
    keep = (times > edge) & (times < times[-1] - edge)
    return values[keep]


def vibrato_metrics(freq: np.ndarray, nominal: float) -> dict[str, float]:
    """Summarise vibrato as depth (cents) and dominant modulation rate (Hz)."""
    cents = 1200.0 * np.log2(freq / nominal)
    depth = float(np.ptp(cents))
    centered = cents - cents.mean()
    if len(centered) > 4:
        spectrum = np.abs(np.fft.rfft(centered))
        rate = np.fft.rfftfreq(len(centered), d=0.02)
        rate = rate[1:]
        dominant = float(rate[np.argmax(spectrum[1:])])
    else:
        dominant = 0.0
    return {"vibrato_depth_cents": depth, "vibrato_rate_hz": dominant}
