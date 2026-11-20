"""Command-line interface for synthesis, calibration and reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .calibrate import CalibrationConfig, calibrate_video
from .estimator import fit_harmonics, model_row_profile
from .geometry import (
    background_trend,
    estimate_stripe_angle,
    rectify_frame,
    row_profile,
    signed_spatial_frequency,
)
from .synth import SynthConfig, write_video
from .visualize import profile_comparison, residual_map, truth_comparison
import cv2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rs-calib",
        description="Rolling-shutter calibration from LED flicker video.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    synth = sub.add_parser("synthesize", help="render a synthetic clip")
    _add_synth_args(synth)

    calib = sub.add_parser("calibrate", help="estimate shutter parameters")
    calib.add_argument("video", type=Path)
    calib.add_argument("--f-led", type=float, required=True, help="LED frequency Hz")
    calib.add_argument("--duty", type=float, default=0.5)
    calib.add_argument("--genlock", action="store_true")
    calib.add_argument("--truth", type=Path, default=None)
    calib.add_argument("--out-dir", type=Path, default=Path("artifacts/out"))
    calib.add_argument("--min-snr", type=float, default=5.0)
    calib.add_argument("--t-line-min-us", type=float, default=1.0)
    calib.add_argument("--t-line-max-us", type=float, default=200.0)
    calib.add_argument(
        "--brightness-span",
        type=float,
        default=None,
        help="calibrated LED on-off grey-level span G; resolves long "
        "exposure sinc-lobe ambiguity",
    )
    return parser


def _add_synth_args(parser: argparse.ArgumentParser) -> None:
    defaults = SynthConfig()
    parser.add_argument("--out", type=Path, default=Path("artifacts/synthetic.avi"))
    for name, value in defaults.to_dict().items():
        if isinstance(value, bool):
            parser.add_argument(f"--{name.replace('_', '-')}", action="store_true")
        else:
            parser.add_argument(f"--{name.replace('_', '-')}", type=type(value), default=value)


def _synth_config_from_args(args: argparse.Namespace) -> SynthConfig:
    allowed = SynthConfig.__dataclass_fields__
    values = {key: getattr(args, key) for key in allowed}
    return SynthConfig(**values)


def run_synthesize(args: argparse.Namespace) -> None:
    cfg = _synth_config_from_args(args)
    video, truth = write_video(cfg, args.out)
    print(f"wrote {video} and {truth}")


def run_calibrate(args: argparse.Namespace) -> None:
    cfg = CalibrationConfig(
        f_led_hz=args.f_led,
        duty=args.duty,
        min_snr=args.min_snr,
        genlock=args.genlock,
        brightness_span=args.brightness_span,
        t_line_prior_s=(args.t_line_min_us * 1e-6, args.t_line_max_us * 1e-6),
    )
    result = calibrate_video(args.video, cfg)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.out_dir / "estimate.json"
    result_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True))

    _write_residual_artifacts(args.video, cfg, args.out_dir)
    if args.truth is not None:
        _write_truth_comparison(args.truth, result, args.out_dir)

    print(json.dumps(_summary(result), indent=2))
    print(f"artifacts written to {args.out_dir}")


def _write_residual_artifacts(video: Path, cfg: CalibrationConfig, out_dir: Path) -> None:
    cap = cv2.VideoCapture(str(video))
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        geometry = estimate_stripe_angle(frame)
        rectified = rectify_frame(frame, geometry.angle_deg)
        profile = row_profile(rectified)
        freq, _, _ = signed_spatial_frequency(profile)
        harmonics = fit_harmonics(profile, freq)
        rows = np.arange(profile.size, dtype=np.float64)
        model = (
            background_trend(profile)
            + model_row_profile(rows, freq, harmonics, 0.0)
        )
        border = max(profile.size // 20, 4)
        profile_plot = profile[border:-border]
        model_plot = model[border:-border]
        observed_2d = np.repeat(profile[:, None], 120, axis=1)
        model_2d = np.repeat(model[:, None], 120, axis=1)
        residual_map(
            observed_2d,
            model_2d,
            out_dir / f"residual_frame_{index:02d}.png",
            title=f"frame {index}",
        )
        profile_comparison(
            profile_plot, model_plot, out_dir / f"profile_frame_{index:02d}.png"
        )
        index += 1
    cap.release()


def _write_truth_comparison(truth_path: Path, result, out_dir: Path) -> None:
    truth = json.loads(truth_path.read_text())
    estimate = {
        "t_line_us": (result.t_line_s or float("nan")) * 1e6,
        "t_exp_us": (result.t_exp_s or float("nan")) * 1e6,
        "scan_direction": result.scan_direction,
    }
    truth_comparison(truth, estimate, out_dir / "truth_comparison.png")
    report = {
        "line_time_error_percent": _percent_error(
            truth["t_line_us"], estimate["t_line_us"]
        ),
        "exposure_error_percent": _percent_error(
            truth["t_exp_us"], estimate["t_exp_us"]
        ),
        "direction_correct": truth["scan_direction"] == estimate["scan_direction"],
        "rotation_error_deg": abs(truth["rotation_deg"] + (result.rotation_deg or 0.0)),
        "solvable": result.solvable,
        "reasons": result.reasons,
    }
    (out_dir / "truth_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))


def _percent_error(truth_v: float, est_v: float) -> float:
    if not np.isfinite(est_v) or truth_v == 0:
        return float("nan")
    return 100.0 * (est_v - truth_v) / truth_v


def _summary(result) -> dict:
    return {
        "solvable": result.solvable,
        "frames_solvable": result.frames_solvable,
        "t_line_us": None if result.t_line_s is None else result.t_line_s * 1e6,
        "t_exp_us": None if result.t_exp_s is None else result.t_exp_s * 1e6,
        "scan_direction": result.scan_direction,
        "direction_ambiguous": result.direction_ambiguous,
        "rotation_deg": result.rotation_deg,
        "reasons": result.reasons,
        "warnings": result.warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "synthesize":
        run_synthesize(args)
    elif args.command == "calibrate":
        run_calibrate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
