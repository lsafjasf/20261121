"""STFT/ISTFT core with end padding and quantifiable bypass error.

The signal is padded by ``n_fft // 2`` samples on both ends (``首尾补齐``)
so every sample is covered by a whole number of windows and the ISTFT
returns exactly ``len(x)`` samples.  With a COLA-compliant Hann window and
hop = n_fft // 4 the bypass path (identity mask) reconstructs the input
PCM to within floating-point round-off; :func:`reconstruction_report`
quantifies that error.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StftConfig:
    sample_rate: int = 44100
    n_fft: int = 4096
    hop: int = 1024
    gain_floor: float = 0.0  # minimum per-bin gain when fully masked

    @property
    def n_bins(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def bin_hz(self) -> float:
        return self.sample_rate / self.n_fft

    @property
    def frame_sec(self) -> float:
        return self.hop / self.sample_rate


@dataclass
class StftProcessor:
    config: StftConfig = field(default_factory=StftConfig)

    def __post_init__(self) -> None:
        cfg = self.config
        self._window = np.hanning(cfg.n_fft + 2)[1:-1].astype(np.float64)
        self._pad = cfg.n_fft // 2

    # ------------------------------------------------------------------ stft
    def stft(self, x: np.ndarray) -> np.ndarray:
        """Return complex spectrum, shape (n_frames, n_bins)."""
        cfg = self.config
        x = np.asarray(x, dtype=np.float64)
        padded = np.pad(x, self._pad, mode="constant")
        n_frames = 1 + (len(padded) - cfg.n_fft) // cfg.hop
        frames = np.lib.stride_tricks.as_strided(
            padded,
            shape=(n_frames, cfg.n_fft),
            strides=(padded.strides[0] * cfg.hop, padded.strides[0]),
        )
        return np.fft.rfft(frames * self._window, axis=1)

    # ----------------------------------------------------------------- istft
    def istft(self, spec: np.ndarray, out_length: int) -> np.ndarray:
        """Inverse STFT with window-envelope normalisation."""
        cfg = self.config
        frames = np.fft.irfft(spec, n=cfg.n_fft, axis=1) * self._window
        total = (len(frames) - 1) * cfg.hop + cfg.n_fft
        out = np.zeros(total, dtype=np.float64)
        envelope = np.zeros(total, dtype=np.float64)
        win_sq = self._window ** 2
        for i, frame in enumerate(frames):
            start = i * cfg.hop
            out[start:start + cfg.n_fft] += frame
            envelope[start:start + cfg.n_fft] += win_sq
        nz = envelope > 1e-8
        out[nz] /= envelope[nz]
        # strip the end padding and return exactly out_length samples
        return out[self._pad:self._pad + out_length]

    # -------------------------------------------------------------- process
    def process(self, x: np.ndarray, mask: np.ndarray | None,
                depth: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Apply a [0, 1] mask as attenuation; returns (y, gain_used).

        ``mask=None`` (or all-zero) is the bypass path.
        """
        spec = self.stft(x)
        if mask is None:
            gain = np.ones(spec.shape, dtype=np.float64)
        else:
            mask = np.asarray(mask, dtype=np.float64)
            if mask.shape != spec.shape:
                raise ValueError(
                    f"mask shape {mask.shape} != spectrum {spec.shape}")
            gain = 1.0 - depth * np.clip(mask, 0.0, 1.0)
            gain = np.maximum(gain, self.config.gain_floor)
        y = self.istft(spec * gain, len(x))
        return y, gain


def reconstruction_report(x: np.ndarray, y: np.ndarray) -> dict:
    """Quantify PCM error between reference ``x`` and reconstruction ``y``."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = min(len(x), len(y))
    diff = y[:n] - x[:n]
    ref_power = float(np.mean(x[:n] ** 2)) or 1e-30
    err_power = float(np.mean(diff ** 2))
    return {
        "length_match": len(x) == len(y),
        "max_abs_err": float(np.max(np.abs(diff))) if n else float("nan"),
        "rmse": float(np.sqrt(err_power)),
        "snr_db": float(10.0 * np.log10(ref_power / max(err_power, 1e-300))),
    }


def band_energy(spec: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Total energy of a spectrum, optionally weighted by a mask."""
    power = np.abs(spec) ** 2
    if mask is not None:
        power = power * mask
    return float(np.sum(power))
