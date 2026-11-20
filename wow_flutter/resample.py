"""Banded overlap-add resampling for time-base correction.

The program audio is corrected band by band (short overlapping time bands).
Within each band the correction map is approximated by a local constant
resampling ratio plus a fractional offset, applied with band-limited
(windowed sinc) interpolation.  A Hann window and 50% overlap-add join the
bands without discontinuities even when the local speed varies.
"""

from __future__ import annotations

import numpy as np


def _sinc_table(taps: int, over: int) -> np.ndarray:
    """Precompute a windowed-sinc fractional-delay lookup table."""
    base = np.arange(-taps // 2 + 1, taps // 2 + 1)
    frac = np.linspace(0.0, 1.0, over, endpoint=False)
    table = np.empty((over, taps))
    for j, f in enumerate(frac):
        x = base - f
        kern = np.sinc(x) * np.hanning(taps)
        table[j] = kern / kern.sum()
    return table


def _interpolate(x: np.ndarray, positions: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Band-limited interpolation at floating sample positions."""
    over, taps = table.shape
    half = taps // 2
    pad = half + 2
    padded = np.pad(x, (pad, pad), mode="constant")
    # Snap near-integer positions to exact integers.  Without this, a fitted
    # local ratio such as 0.9999999999999997 accumulates to positions like
    # 545.9999999999999, which the fractional table would round the wrong way
    # and introduce a whole-sample discontinuity.
    nearest = np.round(positions)
    positions = np.where(np.abs(positions - nearest) < 1e-6, nearest, positions)
    base = np.floor(positions).astype(np.int64)
    frac_idx = np.round((positions - base) * over).astype(np.int64) % over
    out = np.zeros_like(positions)
    for k in range(taps):
        src = base + (k - half + 1) + pad
        valid = (src >= 0) & (src < len(padded))
        contrib = np.zeros_like(positions)
        contrib[valid] = padded[src[valid]]
        out += table[frac_idx, k] * contrib
    return out


def banded_resample(
    audio: np.ndarray,
    input_map: np.ndarray,
    *,
    band_ms: float = 46.0,
    sample_rate: float | None = None,
    sinc_taps: int = 16,
    table_oversample: int = 64,
) -> np.ndarray:
    """Resample ``audio`` onto a correct time base.

    Parameters
    ----------
    audio:
        Mono program audio on the warped (tape) time base.
    input_map:
        ``input_map[n]`` gives the input-sample position used to synthesise
        output sample ``n``; produced by
        :func:`wow_flutter.warp.estimate_warp`.
    band_ms:
        Band length in milliseconds.  Short enough that the local warp is
        nearly affine, long enough for clean overlap-add.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1:
        raise ValueError("audio must be mono")
    n_out = len(audio)
    if len(input_map) < n_out:
        pad = np.full(n_out - len(input_map), input_map[-1])
        map_ = np.concatenate((input_map, pad))
    else:
        map_ = input_map[:n_out]
    if sample_rate is None:
        raise ValueError("sample_rate is required")

    band = max(sinc_taps * 2, int(round(sample_rate * band_ms / 1000.0)))
    hop = band // 2
    table = _sinc_table(sinc_taps, table_oversample)
    window = np.hanning(band)
    acc = np.zeros(n_out)
    weight = np.zeros(n_out)

    for start in range(0, n_out, hop):
        end = min(start + band, n_out)
        length = end - start
        if length < sinc_taps:
            break
        local_map = map_[start:end]
        # Fit an affine local map: input position = ratio*n + offset.
        n_idx = np.arange(length)
        ratio, offset = np.polyfit(n_idx, local_map, 1)
        positions = ratio * n_idx + offset
        band_out = _interpolate(audio, positions, table)
        w = window[:length] if length == band else np.hanning(length)
        acc[start:end] += band_out * w
        weight[start:end] += w

    nonzero = weight > 1e-8
    acc[nonzero] /= weight[nonzero]
    return acc
