"""Estimate rolling-shutter parameters from rectified row profiles.

The flicker stripes form a line-frequency comb.  Their fundamental
spatial frequency gives the per-row readout time while the decay of
harmonic amplitudes gives the per-row exposure time.  Every failure mode
that makes the inverse problem non-identifiable is reported explicitly
instead of silently returning a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import detrend
from .model import exposure_transfer, led_harmonic, normalized_sinc


@dataclass
class Solvability:
    """Human- and machine-readable solvability diagnostics."""

    solvable: bool = True
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def fail(self, reason: str) -> None:
        self.solvable = False
        self.reasons.append(reason)

    def warn(self, text: str) -> None:
        self.warnings.append(text)


@dataclass
class FrameEstimate:
    frame_index: int
    angle_deg: float
    coherence: float
    freq_signed_cpr: float
    snr: float
    t_line_s: float | None
    t_line_candidates_s: list[float]
    t_exp_s: float | None
    scan_direction: int | None          # +1 top-down, -1 bottom-up, None unknown
    direction_ambiguous: bool
    harmonic_snr: dict[int, float]
    solvability: Solvability
    profile: np.ndarray
    detrended: np.ndarray
    model_profile: np.ndarray | None


def fit_harmonics(
    profile: np.ndarray, fundamental_cpr: float, orders: tuple[int, ...] = (1, 3, 5, 7)
) -> dict[int, complex]:
    """Complex least-squares amplitudes of harmonics on a real profile.

    Fitting sine/cosine pairs (rather than FFT bins) works for any
    non-integer spatial frequency and returns sign-bearing phases.
    """
    rows = np.arange(profile.size, dtype=np.float64)
    signal = detrend(profile)
    window = np.hanning(profile.size)
    design = [np.ones_like(rows)]
    for k in orders:
        angle = 2.0 * np.pi * k * fundamental_cpr * rows
        design.append(window * np.cos(angle))
        design.append(window * np.sin(angle))
    design = np.vstack(design).T
    coeff, *_ = np.linalg.lstsq(design, window * signal, rcond=None)
    harmonics: dict[int, complex] = {}
    offset = 1
    for k in orders:
        a = coeff[offset]
        b = coeff[offset + 1]
        harmonics[k] = complex(a, -b)
        offset += 2
    return harmonics


def line_time_candidates(
    freq_cpr_abs: float, f_led_hz: float, prior_t_line_s: tuple[float, float]
) -> list[float]:
    """Enumerate t_line values compatible with an aliased spatial frequency.

    A true signed frequency ``f0`` sampled at one sample per row appears
    as ``|f0 mod 1|`` folded into [0, 0.5].  Every candidate
    ``t_line = m / (f_led * 2k)`` ... is enumerated against the prior.
    """
    if freq_cpr_abs <= 0.0:
        return []
    folded = min(freq_cpr_abs, 1.0 - freq_cpr_abs)
    if folded <= 0.0:
        return []
    lo, hi = prior_t_line_s
    candidates: list[float] = []
    # Unaliased case: the observed frequency is already below Nyquist.
    if lo <= folded / f_led_hz <= hi:
        candidates.append(folded / f_led_hz)
    # |f0| = n +/- folded for integer n >= 1.
    for n in range(1, 10_000):
        for signed_freq in (n - folded, n + folded):
            if signed_freq <= 0.0:
                continue
            t_line = signed_freq / f_led_hz
            if lo <= t_line <= hi:
                candidates.append(t_line)
        if n / f_led_hz > hi:
            break
    return sorted(set(round(c, 15) for c in candidates))


def estimate_exposure(
    harmonics: dict[int, complex],
    f_led_hz: float,
    duty: float,
    t_line_s: float,
    search_s: tuple[float, float] = (1e-6, 5e-3),
    harmonic_snr: dict[int, float] | None = None,
    min_harmonic_snr: float = 3.0,
    frame_height: int = 1080,
    mean_level: float | None = None,
    black_level: float = 0.0,
    brightness_span: float | None = None,
) -> tuple[float | None, float, dict[int, float], float, list[dict]]:
    """Recover t_exp from harmonic amplitude ratios.

    Returns ``(t_exp, best_rmse, observed_ratios)``.  Ratios normalize out
    the unknown brightness/contrast scale.  A ``None`` result means every
    harmonic sits at an exposure null so exposure is unobservable.
    """
    period = 1.0 / f_led_hz
    base = max(abs(harmonics.get(1, 0.0)), 1e-12)
    usable = [
        k
        for k in harmonics
        if k >= 3
        and abs(harmonics[k]) > 1e-9
        and (
            harmonic_snr is None
            or harmonic_snr.get(k, 0.0) >= min_harmonic_snr
        )
    ]
    observed = {
        k: abs(a) / base
        for k, a in harmonics.items()
        if k in usable
    }
    if not observed:
        return None, np.inf, {}

    upper = min(search_s[1], frame_height * t_line_s)
    lower = max(search_s[0], 1e-7)
    if upper <= lower:
        return None, np.inf, {}
    grid = np.geomspace(lower, upper, 6000)
    errors = np.empty_like(grid)
    for idx, t_exp in enumerate(grid):
        err = 0.0
        for k, obs_ratio in observed.items():
            pred = (
                abs(led_harmonic(k, duty))
                * abs(exposure_transfer(k, t_exp, period))
                / max(
                    abs(led_harmonic(1, duty))
                    * abs(exposure_transfer(1, t_exp, period)),
                    1e-12,
                )
            )
            err += (np.log(max(pred, 1e-6)) - np.log(max(obs_ratio, 1e-6))) ** 2
        errors[idx] = err / len(observed)
    rmse = float(np.exp(np.sqrt(np.min(errors))))
    t_exp, lobe_margin, lobe_table = _select_with_fundamental(
        grid,
        errors,
        harmonics,
        f_led_hz,
        duty,
        mean_level=mean_level,
        black_level=black_level,
        brightness_span=brightness_span,
    )
    return t_exp, rmse, observed, lobe_margin, lobe_table


def _select_with_fundamental(
    grid: np.ndarray,
    ratio_errors: np.ndarray,
    harmonics: dict[int, complex],
    f_led_hz: float,
    duty: float,
    lobe_tolerance: float = 0.15,
    mean_level: float | None = None,
    black_level: float = 0.0,
    brightness_span: float | None = None,
) -> tuple[float, float, list[dict]]:
    """Resolve sinc-lobe degeneracy using the fundamental amplitude.

    Every lobe gives the same set of harmonic *ratios* and, with a free
    brightness scale, identical harmonic magnitudes.  The DC component
    breaks the symmetry: it equals ``black + G * duty`` regardless of
    exposure, while the AC scale grows with exposure, so the AC/DC
    ratio is unique per lobe.
    """
    period = 1.0 / f_led_hz
    scored: list[tuple[float, float]] = []
    lobe_table: list[dict] = []
    best_t = float(grid[int(np.argmin(ratio_errors))])
    minima = _strict_local_minima(ratio_errors, lobe_tolerance)
    if minima.size == 0:
        minima = np.array([int(np.argmin(ratio_errors))])

    orders = sorted(harmonics)
    measured = np.array([abs(harmonics[k]) for k in orders])
    led_mag = np.array([abs(led_harmonic(k, duty)) for k in orders])
    c1 = abs(harmonics[1])
    led1 = max(abs(led_harmonic(1, duty)), 1e-12)
    for idx in minima:
        t_exp = float(grid[idx])
        transfer = np.array(
            [abs(normalized_sinc(k * t_exp / period)) for k in orders]
        )
        template = led_mag * transfer
        scale = float(np.dot(template, measured) / max(np.dot(template, template), 1e-12))
        # Relative harmonic shape error (identical across true lobes).
        shape_error = float(
            np.mean(((scale * template - measured) / np.maximum(measured, 1e-9)) ** 2)
        )
        lobe_error = shape_error
        if brightness_span is not None:
            # The fundamental transfer is unique per lobe.  Given the
            # observed on-off grey span G_obs, the expected fundamental
            # amplitude (real-signal convention) is
            # 2 G_obs |c_led(1)| |sinc(t_exp/T)|; compare it directly
            # with the measured fundamental.
            expected_c1 = (
                2.0 * brightness_span * led1 * transfer[orders.index(1)]
            )
            lobe_error += ((expected_c1 - c1) / max(c1, 1e-9)) ** 2
        scored.append((float(lobe_error), t_exp))
        lobe_table.append(
            {
                "t_exp_s": t_exp,
                "shape_error": shape_error,
                "lobe_error": float(lobe_error),
                "brightness_scale_g": float(scale),
            }
        )
    scored.sort(key=lambda item: item[0])
    best_error, best_t = scored[0]
    if len(scored) > 1:
        # Relative gap to the runner-up lobe; a small gap means the
        # brightness anchor cannot tell two sinc lobes apart.
        lobe_margin = float(
            (scored[1][0] - best_error) / max(abs(best_error), 1e-12)
        )
    else:
        lobe_margin = float("inf")
    return best_t, lobe_margin, lobe_table


def _strict_local_minima(values: np.ndarray, tolerance: float) -> np.ndarray:
    """Indices of isolated local minima within ``tolerance`` of the best."""
    interior = values[1:-1]
    is_min = (interior < values[:-2]) & (interior <= values[2:])
    idx = np.flatnonzero(is_min) + 1
    floor = values.min()
    return idx[values[idx] <= floor + tolerance]


def exposure_null_distance(t_exp_s: float, f_led_hz: float, orders=(1, 3, 5)) -> float:
    """Smallest |sinc(k t_exp / T)| over measured orders.

    Near an integer-cycle null the carrying harmonics vanish into the
    noise and the exposure estimate becomes ill-conditioned.
    """
    period = 1.0 / f_led_hz
    return float(min(abs(normalized_sinc(k * t_exp_s / period)) for k in orders))


def model_row_profile(
    rows: np.ndarray,
    fundamental_cpr: float,
    harmonics: dict[int, complex],
    mean_level: float,
) -> np.ndarray:
    """Reconstruct the fitted periodic profile for residual maps.

    ``harmonics[k]`` holds the full-amplitude complex coefficient from
    :func:`fit_harmonics`, where the real profile is
    ``a cos(x) + b sin(x)`` with ``coefficient = a - j b``.
    """
    out = np.full(rows.shape, mean_level, dtype=np.float64)
    for k, coefficient in harmonics.items():
        angle = 2.0 * np.pi * k * fundamental_cpr * rows
        out += coefficient.real * np.cos(angle) - coefficient.imag * np.sin(angle)
    return out
