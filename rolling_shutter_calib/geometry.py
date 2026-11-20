"""Stripe geometry estimation and row-profile demodulation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares


@dataclass
class GeometryResult:
    """Orientation of the flicker stripes in an image."""

    angle_deg: float          # stripe axis angle, wrapped to (-90, 90]
    coherence: float          # structure-tensor anisotropy in [0, 1]


def estimate_stripe_angle(frame: np.ndarray, sigma: float = 5.0) -> GeometryResult:
    """Estimate the angle of nearly-horizontal periodic stripes.

    The structure tensor of the blurred gradient field gives the
    dominant orientation.  A stripe axis parallel to image rows
    (horizontal bands) yields an angle close to 0 degrees.
    """
    gray = frame.astype(np.float64)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    ksize = int(max(3, round(sigma * 2)) | 1)
    jxx = cv2.GaussianBlur(grad_x * grad_x, (ksize, ksize), sigma)
    jyy = cv2.GaussianBlur(grad_y * grad_y, (ksize, ksize), sigma)
    jxy = cv2.GaussianBlur(grad_x * grad_y, (ksize, ksize), sigma)

    jxx = float(np.mean(jxx))
    jyy = float(np.mean(jyy))
    jxy = float(np.mean(jxy))
    trace = jxx + jyy
    if trace <= 1e-12:
        return GeometryResult(0.0, 0.0)
    diff = jxx - jyy
    # Orientation of the gradient (normal to stripes), axis angle is
    # shifted by 90 degrees.
    grad_angle = 0.5 * np.arctan2(2.0 * jxy, diff)
    axis_angle = grad_angle - np.pi / 2.0
    angle_deg = float(np.rad2deg(_wrap_pi(axis_angle)))
    denom = float(np.hypot(diff, 2.0 * jxy))
    coherence = float(np.clip(denom / trace, 0.0, 1.0))
    return GeometryResult(angle_deg, coherence)


def rectify_frame(frame: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate the image so stripes become horizontal (axis angle 0)."""
    if angle_deg == 0.0:
        return frame
    height, width = frame.shape[:2]
    matrix = cv2.getRotationMatrix2D((width * 0.5, height * 0.5), angle_deg, 1.0)
    return cv2.warpAffine(
        frame, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )


def row_profile(frame: np.ndarray, trim_ratio: float = 0.1) -> np.ndarray:
    """Robust mean across columns of a rectified frame, per row."""
    gray = frame.astype(np.float64)
    n_trim = int(gray.shape[1] * trim_ratio)
    if n_trim * 2 >= gray.shape[1]:
        n_trim = 0
    if n_trim:
        gray = gray[:, n_trim:-n_trim]
    return np.mean(gray, axis=1)


def harmonic_snr(profile: np.ndarray, fundamental_cpr: float, orders=(1, 3, 5, 7)) -> dict[int, float]:
    """Per-harmonic SNR from the FFT magnitude over a local noise floor."""
    signal = detrend(profile)
    n = signal.size
    spectrum = np.abs(np.fft.fft(signal))
    rows = np.arange(n, dtype=np.float64)
    result: dict[int, float] = {}
    for k in orders:
        target = abs(k * fundamental_cpr)
        if target <= 0.0 or target >= 0.5:
            result[k] = 0.0
            continue
        bin_idx = int(round(target * n))
        amplitude = float(
            2.0
            * abs(
                np.mean(
                    signal
                    * np.exp(-1j * 2.0 * np.pi * target * rows)
                )
            )
        )
        neighborhood = np.concatenate(
            [
                spectrum[max(1, bin_idx - 4) : bin_idx - 1],
                spectrum[bin_idx + 2 : bin_idx + 5],
            ]
        )
        floor = max(float(2.0 * np.mean(neighborhood) / n), 1e-9)
        result[k] = amplitude / floor
    return result


def detrend(profile: np.ndarray, poly_order: int = 3) -> np.ndarray:
    """Remove slow illumination shading while keeping stripe harmonics."""
    rows = np.arange(profile.size, dtype=np.float64)
    coeff = np.polyfit(rows, profile, poly_order)
    return profile - np.polyval(coeff, rows)


def background_trend(profile: np.ndarray, poly_order: int = 3) -> np.ndarray:
    """Return the smooth shading component removed by :func:`detrend`."""
    rows = np.arange(profile.size, dtype=np.float64)
    coeff = np.polyfit(rows, profile, poly_order)
    return np.polyval(coeff, rows)


def signed_spatial_frequency(profile: np.ndarray) -> tuple[float, float, dict]:
    """Estimate signed fundamental cycles/row.

    Returns
    -------
    frequency_cycles_per_row : signed value; positive means phase
        increases toward the bottom row.
    snr : fundamental amplitude over the median spectral floor.
    info : debug dictionary with the FFT and fitted complex amplitude.

    A real signal has conjugate-symmetric FFT magnitudes, so the sign
    cannot come from peak magnitudes: the FFT locates the unsigned
    frequency, then a positive-frequency complex sinusoid is fitted by
    least squares.  Its phase slope over the rows carries the sign,
    because a negative-frequency component rotates the other way.
    """
    signal = detrend(profile)
    n = signal.size
    spectrum = np.fft.fft(signal)
    freqs = np.fft.fftfreq(n)
    magnitudes = np.abs(spectrum)
    unsigned_freq, peak_amp = _fundamental_frequency(signal, magnitudes, freqs)

    rows = np.arange(n, dtype=np.float64)
    unsigned_freq = _refine_frequency(signal, rows, unsigned_freq)
    peak_amp = float(
        2.0
        * abs(np.mean(signal * np.exp(-1j * 2.0 * np.pi * unsigned_freq * rows)))
    )
    # DFT coefficient of a positive-frequency complex sinusoid.  For a
    # real row profile its phase sign flips with the scan direction.
    coefficient = np.mean(signal * np.exp(-1j * 2.0 * np.pi * unsigned_freq * rows))
    # Under a known phase reference (genlock) the DFT phase sign gives
    # the scan direction; the caller treats this as ambiguous otherwise.
    phase = float(np.angle(coefficient))
    signed_freq = float(np.sign(phase) * unsigned_freq)
    # Mask the harmonic comb before estimating the noise floor, so the
    # higher harmonics of a square wave do not count as noise.
    harmonic_bins = np.round(
        unsigned_freq * n * np.arange(1, 9)
    ).astype(int)
    mask = np.ones(n, dtype=bool)
    mask[0] = False
    for bin_idx in harmonic_bins:
        if 0 < bin_idx < n:
            mask[bin_idx] = False
            mask[n - bin_idx] = False
    noise_bins = magnitudes[mask]
    # DFT bins of unit-amplitude sinusoids have magnitude n/2; convert
    # the floor to the same one-sided amplitude scale as peak_amp.
    spectral_floor = max(float(2.0 * np.sqrt(np.mean(noise_bins**2)) / n), 1e-9)
    snr = float(peak_amp / spectral_floor)
    return signed_freq, snr, {
        "spectrum": spectrum,
        "coefficient": coefficient,
        "phase": phase,
        "spectral_floor_ratio": float(peak_amp / spectral_floor),
    }


def _fundamental_frequency(
    signal: np.ndarray, magnitudes: np.ndarray, freqs: np.ndarray
):
    """Locate the fundamental of a harmonic comb, not the loudest bin.

    A coarse FFT-bin search handles alias placement; a fine orthogonal
    sine/cosine grid then refines the frequency well below one FFT bin,
    which matters because the same frequency error is multiplied by every
    harmonic order during exposure fitting.
    """
    n = magnitudes.size
    positive = np.arange(1, n // 2)
    best_idx = positive[0]
    best_score = -np.inf
    for idx in positive:
        # Comb score: mean energy at integer multiples (normalization
        # avoids favoring the lowest bin, which collects shading leaks).
        multiples = np.arange(idx, n // 2, idx)
        score = float(np.mean(magnitudes[multiples]))
        if score > best_score:
            best_score = score
            best_idx = int(idx)
    coarse, amplitude = _parabolic_peak(magnitudes, freqs, best_idx)
    coarse = abs(coarse)
    bin_width = abs(freqs[1] - freqs[0])
    rows = np.arange(n, dtype=np.float64)
    grid = np.linspace(coarse - bin_width, coarse + bin_width, 81)
    powers = np.empty_like(grid)
    for j, candidate in enumerate(grid):
        angle = 2.0 * np.pi * candidate * rows
        design = np.column_stack([np.cos(angle), np.sin(angle)])
        coeff, *_ = np.linalg.lstsq(design, signal, rcond=None)
        powers[j] = np.sum((design @ coeff) ** 2)
    best = int(np.argmax(powers))
    refined_freq = float(grid[best])
    refined_amp = float(np.sqrt(max(powers[best], 0.0)) * np.sqrt(2.0 / n))
    return refined_freq, refined_amp


def _parabolic_peak(magnitudes: np.ndarray, freqs: np.ndarray, idx: int):
    """Sub-bin peak refinement by quadratic interpolation on magnitudes."""
    n = magnitudes.size
    left = magnitudes[(idx - 1) % n]
    center = magnitudes[idx]
    right = magnitudes[(idx + 1) % n]
    denom = left - 2.0 * center + right
    offset = 0.0 if denom == 0.0 else 0.5 * (left - right) / denom
    bin_freq = freqs[1] - freqs[0]
    refined = freqs[idx] + offset * bin_freq
    amplitude = center - 0.25 * (left - right) * offset
    return float(refined), float(amplitude)


def _refine_frequency(signal: np.ndarray, rows: np.ndarray, f0: float) -> float:
    """Joint nonlinear fit of fundamental frequency plus odd harmonics."""
    orders = (1, 3, 5, 7)

    def unpack(theta: np.ndarray):
        freq = theta[0]
        params = theta[1:].reshape(-1, 2)
        return freq, params

    def residuals(theta: np.ndarray) -> np.ndarray:
        freq, params = unpack(theta)
        model = np.zeros_like(signal)
        for (a, b), k in zip(params, orders):
            angle = 2.0 * np.pi * k * freq * rows
            model += a * np.cos(angle) + b * np.sin(angle)
        return signal - model

    theta0 = [f0]
    for k in orders:
        angle = 2.0 * np.pi * k * f0 * rows
        design = np.column_stack([np.cos(angle), np.sin(angle)])
        coeff, *_ = np.linalg.lstsq(design, signal, rcond=None)
        theta0.extend(coeff.tolist())
    result = least_squares(
        residuals,
        np.asarray(theta0, dtype=np.float64),
        method="lm",
        max_nfev=2000,
    )
    return float(abs(result.x[0]))


def _wrap_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi
