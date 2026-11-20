"""STFT front-end and per-frame acoustic features.

Input is always a two-channel recording (the two microphones fixed to the same
handheld recorder).  Every feature is computed per analysis frame so that the
classifier and the adaptive gain stage can reuse one common representation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .config import AnalysisConfig


@dataclass
class FrameFeatures:
    times: np.ndarray                  # (T,) frame centre times in seconds
    freqs: np.ndarray                  # (F,) bin centre frequencies
    stft: np.ndarray                   # (2, T, F) complex STFT
    coherence: np.ndarray              # (T, F) magnitude-squared coherence
    band_energy_db: dict               # label -> (T,) per-frame RMS dBFS
    ild_db: np.ndarray                 # (T,) inter-channel level diff (low band)
    crest: np.ndarray                  # (T,) low-band crest factor (dB)
    low_fraction: np.ndarray           # (T,) low-band energy share
    cross_phase: np.ndarray            # (T, F) cross-spectrum phase, radians


def _frame_lengths(cfg: AnalysisConfig, sr: int) -> tuple[int, int]:
    n_fft = int(round(cfg.frame_sec * sr))
    hop = int(round(cfg.hop_sec * sr))
    # Force an even FFT size for the symmetric synthesis window.
    n_fft += n_fft % 2
    return n_fft, max(1, hop)


def stft_channels(audio: np.ndarray, cfg: AnalysisConfig, sr: int):
    """Return STFT tensor (2, T, F), frequencies and frame times."""
    if audio.ndim != 2 or audio.shape[0] != 2:
        raise ValueError("expected shape (2, N): two microphone channels")
    n_fft, hop = _frame_lengths(cfg, sr)
    win = signal.windows.hann(n_fft, sym=False)
    frames = []
    for channel in audio:
        _, _, z = signal.stft(
            channel,
            fs=sr,
            window=win,
            nperseg=n_fft,
            noverlap=n_fft - hop,
            boundary="zeros",
            padded=True,
        )
        frames.append(z.T)  # (T, F)
    z = np.stack(frames, axis=0)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    times = np.arange(z.shape[1]) * hop / sr
    return z, freqs, times, n_fft, hop, win


def istft_channels(
    z: np.ndarray, n_fft: int, hop: int, win: np.ndarray, sr: int, length: int
) -> np.ndarray:
    """Inverse STFT for both channels with overlap-add normalization."""
    out = []
    for channel in range(z.shape[0]):
        _, x = signal.istft(
            z[channel].T,
            fs=sr,
            window=win,
            nperseg=n_fft,
            noverlap=n_fft - hop,
            input_onesided=True,
            boundary=True,
        )
        out.append(x[:length])
    return np.stack(out, axis=0)


def _band_mask(freqs: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    return (freqs >= band[0]) & (freqs <= band[1])


def magnitude_squared_coherence(
    z: np.ndarray, freqs: np.ndarray, cfg: AnalysisConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed cross-channel coherence and cross-spectrum phase.

    |C|^2 = |<S12>|^2 / (<S11> <S22>) with temporal averaging of the
    cross-spectra, which is what makes coherent bass survive while wind
    (decorrelated between capsules) collapses toward zero.
    """
    s11 = z[0].real ** 2 + z[0].imag ** 2
    s22 = z[1].real ** 2 + z[1].imag ** 2
    s12 = z[0] * np.conj(z[1])
    k = max(1, cfg.coherence_smooth_frames)
    kernel = np.ones(k) / k
    s11s = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 0, s11)
    s22s = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 0, s22)
    s12s = np.apply_along_axis(
        lambda v: _smooth_complex(v, kernel), 0, s12
    )
    denom = s11s * s22s
    coh = (s12s.real ** 2 + s12s.imag ** 2) / np.maximum(denom, 1e-20)
    # Silence frames are meaningless: floor their coherence to zero.
    quiet = (s11s + s22s) < (1e-10 * np.max(s11 + s22) + 1e-20)
    coh[quiet] = 0.0
    return np.clip(coh, 0.0, 1.0), np.angle(s12s)


def _smooth_complex(v: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return np.convolve(v.real, kernel, mode="same") + 1j * np.convolve(
        v.imag, kernel, mode="same"
    )


def band_energy_db(z: np.ndarray, freqs: np.ndarray, bands: dict) -> dict:
    """Per-frame RMS energy in dBFS per band, averaged across channels."""
    power = 0.5 * (
        np.abs(z[0]) ** 2 + np.abs(z[1]) ** 2
    )  # (T, F) mean bin power
    result = {}
    total_power = power.sum(axis=1) + 1e-20
    for name, band in bands.items():
        mask = _band_mask(freqs, band)
        result[name] = 10.0 * np.log10(power[:, mask].mean(axis=1) + 1e-20)
    result["_low_fraction"] = (
        power[:, _band_mask(freqs, (20.0, 200.0))].sum(axis=1) / total_power
    )
    return result


def inter_channel_level_db(
    z: np.ndarray, freqs: np.ndarray, band: tuple[float, float]
) -> np.ndarray:
    """Signed low-band level difference ch1 - ch2 in dB."""
    mask = _band_mask(freqs, band)
    p1 = np.abs(z[0][:, mask]).mean(axis=1) ** 2 + 1e-20
    p2 = np.abs(z[1][:, mask]).mean(axis=1) ** 2 + 1e-20
    return 10.0 * np.log10(p1 / p2)


def low_band_crest_db(
    z: np.ndarray, freqs: np.ndarray, band: tuple[float, float]
) -> np.ndarray:
    """Crest factor of the low-band magnitude envelope (dB).

    Sharp drum hits show a high crest; sustained wind shows a low one.
    """
    mask = _band_mask(freqs, band)
    mag = 0.5 * (np.abs(z[0][:, mask]) + np.abs(z[1][:, mask]))
    frame_rms = np.sqrt((mag ** 2).mean(axis=1)) + 1e-20
    frame_peak = mag.max(axis=1) + 1e-20
    return 20.0 * np.log10(frame_peak / frame_rms)


def extract_features(audio: np.ndarray, sr: int, cfg: AnalysisConfig) -> tuple[
    FrameFeatures, int, int, np.ndarray
]:
    z, freqs, times, n_fft, hop, win = stft_channels(audio, cfg, sr)
    coh, phase = magnitude_squared_coherence(z, freqs, cfg)
    energy = band_energy_db(z, freqs, cfg.bands)
    low_fraction = np.clip(energy.pop("_low_fraction"), 0.0, 1.0)
    ild = inter_channel_level_db(z, freqs, cfg.wind_band_hz)
    crest = low_band_crest_db(z, freqs, cfg.wind_band_hz)
    features = FrameFeatures(
        times=times,
        freqs=freqs,
        stft=z,
        coherence=coh,
        band_energy_db=energy,
        ild_db=ild,
        crest=crest,
        low_fraction=low_fraction,
        cross_phase=phase,
    )
    return features, n_fft, hop, win
