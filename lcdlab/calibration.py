from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import LabConfig
from .model import ResponseParams
from .synth import CalibrationClip, Segment


@dataclass(slots=True)
class FitResult:
    scope: str
    direction: str
    channel: str
    transition: str
    alpha: float
    tau_frames: float
    samples: int
    fit_rmse: float
    profile_rmse: float

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


@dataclass(slots=True)
class CalibrationResult:
    params: ResponseParams
    fits: list[FitResult]
    curves: dict[str, dict[str, np.ndarray]]
    mean_step_curves: dict[str, dict[str, np.ndarray]]

    def global_taus(self) -> dict[str, list[float]]:
        return {
            "rise": self.params.tau_rise.tolist(),
            "fall": self.params.tau_fall.tolist(),
        }


@dataclass(slots=True)
class _TransitionSamples:
    rise_dx: np.ndarray
    rise_dy: np.ndarray
    fall_dx: np.ndarray
    fall_dy: np.ndarray


def _collect_samples(
    source: np.ndarray,
    captured: np.ndarray,
    frames: Iterable[int],
) -> _TransitionSamples:
    rise_dx: list[np.ndarray] = []
    rise_dy: list[np.ndarray] = []
    fall_dx: list[np.ndarray] = []
    fall_dy: list[np.ndarray] = []
    for frame_index in frames:
        if frame_index < 1:
            continue
        previous = captured[frame_index - 1]
        target = source[frame_index]
        observed = captured[frame_index]
        dx = target - previous
        dy = observed - previous
        usable = (np.abs(dx) > 0.012) & (observed > 0.003) & (observed < 0.997)
        rising = usable & (dx > 0)
        falling = usable & (dx < 0)
        if np.any(rising):
            rise_dx.append(dx[rising])
            rise_dy.append(dy[rising])
        if np.any(falling):
            fall_dx.append(dx[falling])
            fall_dy.append(dy[falling])
    if not rise_dx:
        rise_dx = [np.zeros(1)]
        rise_dy = [np.zeros(1)]
    if not fall_dx:
        fall_dx = [np.ones(1)]
        fall_dy = [np.ones(1)]
    return _TransitionSamples(
        rise_dx=np.concatenate(rise_dx).astype(np.float64),
        rise_dy=np.concatenate(rise_dy).astype(np.float64),
        fall_dx=np.concatenate(fall_dx).astype(np.float64),
        fall_dy=np.concatenate(fall_dy).astype(np.float64),
    )


def _fit_alpha(dx: np.ndarray, dy: np.ndarray) -> tuple[float, float]:
    energy = float(np.dot(dx, dx))
    if energy <= 1e-9:
        return 0.25, 0.0
    alpha = float(np.dot(dx, dy) / energy)
    alpha = float(np.clip(alpha, 0.015, 0.985))
    residual = dy - alpha * dx
    rmse = float(np.sqrt(np.mean(residual * residual)))
    return alpha, rmse


def _profile_rmse(
    dx: np.ndarray,
    dy: np.ndarray,
    alpha: float,
    bins: int = 16,
) -> float:
    if len(dx) == 0:
        return 0.0
    edges = np.quantile(dx, np.linspace(0.0, 1.0, bins + 1))
    residual: list[float] = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (dx >= lower) & (dx <= upper)
        if np.count_nonzero(selected) < 8:
            continue
        local_alpha = float(np.dot(dx[selected], dy[selected]) / np.dot(
            dx[selected], dx[selected]
        ))
        residual.append((local_alpha - alpha) ** 2)
    return float(np.sqrt(np.mean(residual))) if residual else 0.0


def _channel_fit(
    source: np.ndarray,
    captured: np.ndarray,
    frames: Iterable[int],
    channel_index: int,
    channel: str,
    scope: str,
    direction: str,
) -> list[FitResult]:
    samples = _collect_samples(
        source[..., channel_index], captured[..., channel_index], frames
    )
    output: list[FitResult] = []
    for transition, dx, dy in (
        ("rise", samples.rise_dx, samples.rise_dy),
        ("fall", samples.fall_dx, samples.fall_dy),
    ):
        alpha, rmse = _fit_alpha(dx, dy)
        tau = float(-1.0 / np.log1p(-alpha))
        output.append(
            FitResult(
                scope=scope,
                direction=direction,
                channel=channel,
                transition=transition,
                alpha=alpha,
                tau_frames=tau,
                samples=int(dx.size),
                fit_rmse=rmse,
                profile_rmse=_profile_rmse(dx, dy, alpha),
            )
        )
    return output


def _segment_frames(segment: Segment) -> list[int]:
    return list(range(segment.start, segment.end))


def _extract_step_curves(
    clip: CalibrationClip,
    params: ResponseParams,
    config: LabConfig,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, np.ndarray]],
]:
    curves: dict[str, dict[str, np.ndarray]] = {"rise": {}, "fall": {}}
    measured: dict[str, dict[str, np.ndarray]] = {"rise": {}, "fall": {}}
    for channel_index, channel in enumerate(config.channels):
        for transition in ("rise", "fall"):
            segments = [
                segment
                for segment in clip.segments
                if segment.kind == "flash" and transition in segment.name
            ]
            segment = segments[0]
            height, width = clip.captured.shape[1:3]
            x_margin = max(1, width // 6)
            y_margin = max(1, height // 6)
            trace_start = max(0, segment.start - 1)
            crop = clip.captured[
                trace_start : segment.end,
                y_margin : height - y_margin,
                x_margin : width - x_margin,
                channel_index,
            ]
            trace = np.mean(crop, axis=(1, 2)).astype(np.float64)
            initial = float(trace[0])
            target = 1.0 if transition == "rise" else 0.0
            span = target - initial
            normalized_measured = (trace[1:] - initial) / span
            alpha = (
                params.alpha_rise[channel_index]
                if transition == "rise"
                else params.alpha_fall[channel_index]
            )
            length = segment.end - segment.start
            normalized_model = np.empty(length, dtype=np.float64)
            normalized_model[0] = 1.0 - np.power(1.0 - alpha, 1)
            if length > 1:
                normalized_model[1:] = 1.0 - np.power(
                    1.0 - alpha, np.arange(2, length + 1)
                )
            curves[transition][channel] = normalized_model
            measured[transition][channel] = normalized_measured
    return curves, measured


def calibrate(clip: CalibrationClip, config: LabConfig) -> CalibrationResult:
    frames = list(range(1, clip.source.shape[0]))
    fits: list[FitResult] = []
    for channel_index, channel in enumerate(config.channels):
        fits.extend(
            _channel_fit(
                clip.source,
                clip.captured,
                frames,
                channel_index,
                channel,
                "global",
                "all",
            )
        )
        for segment in clip.segments:
            if segment.kind != "sweep" or segment.direction is None:
                continue
            fits.extend(
                _channel_fit(
                    clip.source,
                    clip.captured,
                    _segment_frames(segment),
                    channel_index,
                    channel,
                    "directional",
                    segment.direction,
                )
            )

    alpha_rise = np.empty(3, dtype=np.float64)
    alpha_fall = np.empty(3, dtype=np.float64)
    for channel_index, channel in enumerate(config.channels):
        rise = next(
            fit
            for fit in fits
            if fit.scope == "global"
            and fit.channel == channel
            and fit.transition == "rise"
        )
        fall = next(
            fit
            for fit in fits
            if fit.scope == "global"
            and fit.channel == channel
            and fit.transition == "fall"
        )
        alpha_rise[channel_index] = rise.alpha
        alpha_fall[channel_index] = fall.alpha
    params = ResponseParams(alpha_rise=alpha_rise, alpha_fall=alpha_fall)
    curves, measured = _extract_step_curves(clip, params, config)
    return CalibrationResult(
        params=params,
        fits=fits,
        curves=curves,
        mean_step_curves=measured,
    )


def save_calibration(result: CalibrationResult, config: LabConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scope",
        "direction",
        "channel",
        "transition",
        "alpha",
        "tau_frames",
        "samples",
        "fit_rmse",
        "profile_rmse",
    ]
    with (out_dir / "calibration_fits.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for fit in result.fits:
            writer.writerow(fit.to_dict())

    grouped: dict[str, dict[str, list[float]]] = {}
    for channel, tau_rise, tau_fall in zip(
        config.channels, result.params.tau_rise, result.params.tau_fall
    ):
        grouped[channel] = {
            "alpha_rise": float(
                result.params.alpha_rise[config.channels.index(channel)]
            ),
            "alpha_fall": float(
                result.params.alpha_fall[config.channels.index(channel)]
            ),
            "tau_rise_frames": float(tau_rise),
            "tau_fall_frames": float(tau_fall),
            "tau_rise_ms": float(tau_rise / config.fps * 1000.0),
            "tau_fall_ms": float(tau_fall / config.fps * 1000.0),
        }

    payload = {
        "model": "asymmetric switched first-order response",
        "fps": config.fps,
        "instability_gain_threshold": config.instability_gain,
        "channels": grouped,
        "global_fits": [fit.to_dict() for fit in result.fits if fit.scope == "global"],
        "directional_fits": [
            fit.to_dict() for fit in result.fits if fit.scope == "directional"
        ],
    }
    (out_dir / "calibration.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    np.savez_compressed(
        out_dir / "calibration_curves.npz",
        **{
            f"{kind}_{channel}": values
            for kind, channels in result.curves.items()
            for channel, values in channels.items()
        },
        **{
            f"measured_{kind}_{channel}": values
            for kind, channels in result.mean_step_curves.items()
            for channel, values in channels.items()
        },
    )
