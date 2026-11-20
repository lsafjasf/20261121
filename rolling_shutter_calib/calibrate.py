"""End-to-end rolling-shutter calibration pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import cv2
import numpy as np

from .estimator import (
    FrameEstimate,
    Solvability,
    estimate_exposure,
    exposure_null_distance,
    fit_harmonics,
    line_time_candidates,
    model_row_profile,
)
from .geometry import (
    detrend,
    estimate_stripe_angle,
    harmonic_snr as spectral_harmonic_snr,
    rectify_frame,
    row_profile,
    signed_spatial_frequency,
)


@dataclass
class CalibrationConfig:
    f_led_hz: float
    duty: float = 0.5
    min_snr: float = 5.0
    min_coherence: float = 0.25
    min_cycles_per_frame: float = 1.5
    min_null_margin: float = 0.08
    max_ratio_error: float = 1.6
    t_line_prior_s: tuple[float, float] = (1e-6, 200e-6)
    t_exp_search_s: tuple[float, float] = (1e-6, 5e-3)
    genlock: bool = False
    brightness_span: float | None = None  # known LED on-off grey scale G


@dataclass
class ClipResult:
    config: dict
    frames_processed: int
    frames_solvable: int
    t_line_s: float | None
    t_exp_s: float | None
    scan_direction: int | None
    direction_ambiguous: bool
    rotation_deg: float | None
    solvable: bool
    reasons: list[str]
    warnings: list[str]
    frame_results: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_frame(
    frame: np.ndarray, index: int, cfg: CalibrationConfig
) -> FrameEstimate:
    """Run geometry, demodulation and solvability checks on one frame."""
    solv = Solvability()
    geometry = estimate_stripe_angle(frame)
    rectified = rectify_frame(frame, geometry.angle_deg)
    profile = row_profile(rectified)
    freq_signed, snr, _ = signed_spatial_frequency(profile)
    freq_abs = abs(freq_signed)

    if geometry.coherence < cfg.min_coherence:
        solv.fail(f"stripe coherence {geometry.coherence:.2f} below threshold")
    if snr < cfg.min_snr:
        solv.fail(f"fundamental SNR {snr:.2f} below threshold {cfg.min_snr}")

    rows = float(profile.size)
    cycles = freq_abs * rows
    if cycles < cfg.min_cycles_per_frame:
        solv.fail(
            f"only {cycles:.2f} stripe cycles in frame; need >= "
            f"{cfg.min_cycles_per_frame}"
        )

    candidates = line_time_candidates(
        freq_abs, cfg.f_led_hz, cfg.t_line_prior_s
    )
    if not candidates:
        solv.fail("no line time inside prior for observed stripe frequency")

    harmonics = fit_harmonics(profile, freq_signed)
    harmonic_snr = spectral_harmonic_snr(profile, freq_signed)
    measured_span = float(
        np.percentile(profile, 98) - np.percentile(profile, 2)
    )
    brightness_span = (
        cfg.brightness_span if cfg.brightness_span is not None else measured_span
    )
    t_line = candidates[0] if candidates else None
    t_exp = None
    model_profile = None
    if len(candidates) > 1:
        solv.warn(
            f"{len(candidates)} aliased line-time candidates in prior; "
            "tighten t_line_prior_s"
        )
    if t_line is not None and solv.solvable:
        t_exp, ratio_error, _, lobe_separation, lobe_table = estimate_exposure(
            harmonics,
            cfg.f_led_hz,
            cfg.duty,
            t_line,
            cfg.t_exp_search_s,
            harmonic_snr=harmonic_snr,
            frame_height=profile.size,
            brightness_span=max(brightness_span, 1.0),
        )
        if t_exp is None:
            solv.fail("no higher harmonics above noise; exposure unobservable")
        else:
            if ratio_error > cfg.max_ratio_error:
                solv.fail(
                    f"harmonic ratio fit error {ratio_error:.2f} too large"
                )
            true_lobe = min(lobe_table, key=lambda row: row["lobe_error"])
            strong_orders = tuple(
                k for k, snr_k in harmonic_snr.items() if snr_k >= 5.0
            ) or tuple(sorted(harmonics))
            true_margin = _weighted_null_margin(
                true_lobe["t_exp_s"],
                cfg.f_led_hz,
                strong_orders,
                harmonic_snr,
            )
            if true_margin < cfg.min_null_margin:
                solv.fail(
                    f"best sinc lobe near null (margin {true_margin:.3f}); "
                    "exposure time unobservable"
                )
            usable_orders = tuple(
                k for k, snr_k in harmonic_snr.items() if k > 1 and snr_k >= 3.0
            )
            if usable_orders:
                margin = _weighted_null_margin(
                    t_exp, cfg.f_led_hz, usable_orders, harmonic_snr
                )
            else:
                margin = 1.0
            if margin < cfg.min_null_margin:
                solv.fail(
                    f"exposure within sinc null (margin {margin:.3f}); "
                    "exposure time unobservable"
                )
            if t_exp >= 1.0 / cfg.f_led_hz and margin < cfg.min_null_margin * 2:
                solv.warn("multi-period exposure close to a harmonic null")
            rows_idx = np.arange(profile.size, dtype=np.float64)
            model_profile = model_row_profile(
                rows_idx, freq_signed, harmonics, float(np.mean(profile))
            )

    direction, direction_ambiguous = _resolve_direction(
        cfg, freq_signed, harmonics, profile.size
    )
    if direction_ambiguous:
        solv.warn("scan direction ambiguous without genlock or asymmetric phase")

    return FrameEstimate(
        frame_index=index,
        angle_deg=geometry.angle_deg,
        coherence=geometry.coherence,
        freq_signed_cpr=freq_signed,
        snr=snr,
        t_line_s=t_line,
        t_line_candidates_s=candidates,
        t_exp_s=t_exp,
        scan_direction=direction,
        direction_ambiguous=direction_ambiguous,
        harmonic_snr=harmonic_snr,
        solvability=solv,
        profile=profile,
        detrended=profile - model_profile if model_profile is not None else profile * np.nan,
        model_profile=model_profile,
    )


def _residual(profile: np.ndarray, fundamental: float, harmonics: dict[int, complex]):
    rows = np.arange(profile.size, dtype=np.float64)
    signal = detrend(profile)
    model = model_row_profile(rows, fundamental, harmonics, 0.0)
    return signal - model


def _weighted_null_margin(t_exp: float, f_led_hz: float, orders, snr_map) -> float:
    """SNR-weighted distance to a sinc null across usable harmonics."""
    weights = np.array([max(snr_map.get(k, 0.0), 1e-9) for k in orders])
    margins = np.array([
        exposure_null_distance(t_exp, f_led_hz, orders=(k,)) for k in orders
    ])
    return float(np.sum(weights * margins) / np.sum(weights))


def _resolve_direction(
    cfg: CalibrationConfig,
    freq_signed: float,
    harmonics: dict[int, complex],
    n_rows: int,
) -> tuple[int | None, bool]:
    """Determine top-down (+1) vs bottom-up (-1) readout.

    Under genlock the row-zero LED phase is known (zero), so the sign of
    the analytic spatial frequency maps directly onto readout direction.
    Free-running captures only determine the sign up to a global phase
    flip and are reported as ambiguous.
    """
    if not cfg.genlock:
        return None, True
    coefficient = harmonics.get(1)
    if coefficient is None or abs(coefficient) < 1e-9:
        return None, True
    # genlock: start time t0(r) = -s r t_line and the LED phase is zero
    # at the frame trigger, so the spatial frequency sign equals s.
    return (1 if freq_signed > 0.0 else -1), False


def calibrate_video(
    video_path: str | Path, cfg: CalibrationConfig
) -> ClipResult:
    """Read a clip and aggregate per-frame estimates into a clip result."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    frame_estimates: list[FrameEstimate] = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_estimates.append(estimate_frame(frame, index, cfg))
        index += 1
    cap.release()
    return aggregate(frame_estimates, cfg)


def aggregate(estimates: list[FrameEstimate], cfg: CalibrationConfig) -> ClipResult:
    good = [e for e in estimates if e.solvability.solvable]
    reasons = sorted({r for e in estimates for r in e.solvability.reasons})
    warnings = sorted({w for e in estimates for w in e.solvability.warnings})

    t_line = _median([e.t_line_s for e in good])
    t_exp = _median([e.t_exp_s for e in good])
    rotation = float(np.median([e.angle_deg for e in good])) if good else None

    directions = [e.scan_direction for e in good if e.scan_direction is not None]
    direction_ambiguous = not directions or any(e.direction_ambiguous for e in good)
    if directions:
        direction = int(np.sign(np.median(directions)))
    else:
        direction = None

    result = ClipResult(
        config=asdict(cfg),
        frames_processed=len(estimates),
        frames_solvable=len(good),
        t_line_s=t_line,
        t_exp_s=t_exp,
        scan_direction=direction,
        direction_ambiguous=direction_ambiguous,
        rotation_deg=rotation,
        solvable=bool(good) and t_line is not None and t_exp is not None,
        reasons=reasons,
        warnings=warnings,
    )
    result.frame_results = [_frame_dict(e) for e in estimates]
    return result


def _frame_dict(estimate: FrameEstimate) -> dict:
    return {
        "frame_index": estimate.frame_index,
        "angle_deg": estimate.angle_deg,
        "coherence": estimate.coherence,
        "freq_signed_cpr": estimate.freq_signed_cpr,
        "snr": estimate.snr,
        "t_line_s": estimate.t_line_s,
        "t_line_candidates_s": estimate.t_line_candidates_s,
        "t_exp_s": estimate.t_exp_s,
        "scan_direction": estimate.scan_direction,
        "direction_ambiguous": estimate.direction_ambiguous,
        "harmonic_snr": estimate.harmonic_snr,
        "solvable": estimate.solvability.solvable,
        "reasons": estimate.solvability.reasons,
        "warnings": estimate.solvability.warnings,
    }


def _median(values: list):
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def save_result(result: ClipResult, path: str | Path) -> None:
    Path(path).write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True))
