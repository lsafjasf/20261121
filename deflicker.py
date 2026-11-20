"""Deflicker processor for footage shot under flickering lights.

Estimates a per-frame, per-row brightness modulation from a stable
reference region of the frame, then applies the inverse correction at
the original resolution. Moving foreground that enters the reference
region is down-weighted so it does not corrupt the estimate. Slow,
genuine exposure changes (e.g. a global dimming ramp) are separated
from fast flicker via temporal trend decomposition and are preserved.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DeflickerConfig:
    # Reference region as (x, y, w, h) in pixels, or None for auto-select.
    ref_roi: tuple | None = None
    # Grid size used when auto-selecting the reference region.
    auto_grid: int = 4
    # Half-window (frames) of the temporal trend filter. Windows longer
    # than this are treated as genuine exposure change and preserved.
    trend_radius: int = 8
    # Half-window (frames) of the running temporal median used to build
    # the background model for motion down-weighting.
    bg_radius: int = 5
    # Pixels deviating from the background by more than this many
    # sigmas (robust) are treated as moving foreground.
    motion_sigma: float = 3.0
    # Minimum weight kept for foreground pixels (never fully zero so a
    # row that is briefly fully covered still yields an estimate).
    min_weight: float = 0.05
    # Spatial smoothing (rows) applied to the row-modulation profile.
    row_smooth: int = 5
    # Clamp for the final correction gain, for safety.
    gain_clip: tuple = (0.5, 2.0)


def _to_gray_float(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    return gray.astype(np.float32) / 255.0


def _median_along_time(stack: np.ndarray, radius: int) -> np.ndarray:
    """Sliding median along axis 0 of a (T, ...) stack."""
    n = stack.shape[0]
    out = np.empty_like(stack)
    for t in range(n):
        lo = max(0, t - radius)
        hi = min(n, t + radius + 1)
        out[t] = np.median(stack[lo:hi], axis=0)
    return out


def _gauss_along_time(stack: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian low-pass along axis 0 of a (T, H) stack."""
    k = int(np.ceil(3.0 * sigma)) * 2 + 1
    return cv2.GaussianBlur(stack, (1, k), sigma,
                            borderType=cv2.BORDER_REFLECT)


class DeflickerProcessor:
    """Offline (whole-clip) deflicker built around a reference region."""

    def __init__(self, config: DeflickerConfig | None = None):
        self.cfg = config or DeflickerConfig()
        self.diagnostics: dict = {}

    def _auto_ref_roi(self, grays: np.ndarray) -> tuple:
        """Pick the most temporally stable full-height vertical strip.

        Row-varying flicker is estimated per row, so the reference
        region must span every row; a vertical strip gives that while
        still avoiding busy horizontal bands.
        """
        h, w = grays.shape[1:]
        g = self.cfg.auto_grid
        best, best_score = None, np.inf
        for gx in range(g):
            x0, x1 = gx * w // g, (gx + 1) * w // g
            strip = grays[:, :, x0:x1]
            series = strip.mean(axis=(1, 2))
            score = series.std() / (series.mean() + 1e-6)
            if score < best_score:
                best_score = score
                best = (x0, 0, x1 - x0, h)
        return best

    def _motion_weights(self, ref: np.ndarray) -> np.ndarray:
        """Per-pixel weights: 1 for static background, low for movers."""
        background = _median_along_time(ref, self.cfg.bg_radius)
        residual = np.abs(ref - background)
        mad = np.median(residual, axis=0) + 1e-4
        z = residual / (1.4826 * mad[None])
        w = 1.0 / (1.0 + (z / self.cfg.motion_sigma) ** 2)
        return np.clip(w, self.cfg.min_weight, 1.0).astype(np.float32)

    @staticmethod
    def _row_profile(ref: np.ndarray, weights: np.ndarray) -> np.ndarray:
        wsum = weights.sum(axis=2) + 1e-6
        return (ref * weights).sum(axis=2) / wsum              # (T, H_ref)

    def _flicker_gain(self, row_prof: np.ndarray):
        """Split the row profile into fast flicker and slow trend.

        Works in the log domain: the slow Gaussian trend holds genuine
        exposure changes (e.g. a dimming ramp) and the static row
        profile of the scene; everything faster is flicker.
        """
        log_prof = np.log(np.maximum(row_prof, 1e-3))
        trend = _gauss_along_time(log_prof, self.cfg.trend_radius / 2.0)
        flicker = log_prof - trend
        if self.cfg.row_smooth > 1:
            k = self.cfg.row_smooth | 1
            flicker = cv2.GaussianBlur(
                flicker, (k, 1), 0, borderType=cv2.BORDER_REPLICATE)
        return np.exp(flicker).astype(np.float32), np.exp(trend[:, 0])

    def _estimate(self, grays: np.ndarray, roi: tuple):
        """Return total flicker gain (T, H_full) plus diagnostics."""
        x, y, w, h = roi
        ref = grays[:, y:y + h, x:x + w]

        # Pass 1: uniform weights give a rough gain so the ref region
        # can be (de)flickered before motion detection -- otherwise the
        # flicker itself would look like motion and poison the weights.
        gain1, _ = self._flicker_gain(
            self._row_profile(ref, np.ones_like(ref)))
        ref_corr = ref / gain1[:, :, None]

        # Pass 2: motion-aware weights on the deflickered reference.
        weights = self._motion_weights(ref_corr)
        gain, trend = self._flicker_gain(self._row_profile(ref, weights))

        # Interpolate the reference-row profile to full-frame rows so the
        # correction can be applied at the original resolution.
        full_h = grays.shape[1]
        src_y = (np.arange(h) + 0.5) / h * full_h
        dst_y = np.arange(full_h, dtype=np.float32)
        row_full = np.empty((gain.shape[0], full_h), np.float32)
        for t in range(gain.shape[0]):
            row_full[t] = np.interp(dst_y, src_y, gain[t],
                                    left=gain[t, 0], right=gain[t, -1])
        frame_flicker = row_full.mean(axis=1)
        return frame_flicker, row_full, trend

    def process(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        grays = np.stack([_to_gray_float(f) for f in frames])
        roi = self.cfg.ref_roi or self._auto_ref_roi(grays)
        frame_flicker, row_flicker, trend = self._estimate(grays, roi)

        lo, hi = self.cfg.gain_clip
        gain = np.clip(row_flicker, lo, hi)                    # (T, H)

        out = []
        for t, frame in enumerate(frames):
            f = frame.astype(np.float32) / 255.0
            g = gain[t][:, None]
            if f.ndim == 3:
                g = g[..., None]
            corrected = f / g
            out.append(np.clip(corrected * 255.0, 0, 255).astype(np.uint8))

        self.diagnostics = {
            "ref_roi": roi,
            "frame_flicker": frame_flicker,
            "exposure_trend": trend,
            "row_flicker": row_flicker,
        }
        return out
