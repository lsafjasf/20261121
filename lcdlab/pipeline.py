from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .calibration import calibrate, save_calibration
from .config import LabConfig
from .metrics import compute_metrics, save_metrics
from .report import write_report
from .rendering import (
    render_calibration_plots,
    render_comparison,
    render_contact_sheet,
    render_diagnostics,
    render_metric_chart,
    save_metadata,
    save_video,
)
from .restoration import restore_sequence
from .synth import build_calibration_clip, build_demo_clip


def run_experiment(config: LabConfig, make_comparison_video: bool = True) -> dict:
    started = time.perf_counter()
    out_dir = config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/7] Building calibration and demo recordings")
    calibration_clip = build_calibration_clip(config)
    demo_clip = build_demo_clip(config)

    print("[2/7] Estimating per-channel rise/fall response")
    calibration = calibrate(calibration_clip, config)
    save_calibration(calibration, config, out_dir)
    for channel, tau_rise, tau_fall in zip(
        config.channels, calibration.params.tau_rise, calibration.params.tau_fall
    ):
        print(
            f"      {channel}: rise {tau_rise:5.2f} frames / "
            f"fall {tau_fall:5.2f} frames"
        )

    print("[3/7] Inverting the demo recording")
    restoration = restore_sequence(demo_clip.captured, calibration.params, config)
    metrics = compute_metrics(demo_clip, restoration, config)
    save_metrics(metrics, out_dir)

    print("[4/7] Rendering videos")
    save_video(out_dir / "calibration_source.mp4", calibration_clip.source, config.fps)
    save_video(
        out_dir / "calibration_capture.mp4", calibration_clip.captured, config.fps
    )
    save_video(out_dir / "captured.mp4", demo_clip.captured, config.fps, crf=15)
    save_video(out_dir / "restored.mp4", restoration.frames, config.fps, crf=15)
    save_video(out_dir / "ground_truth.mp4", demo_clip.source, config.fps, crf=15)

    print("[5/7] Building synchronized comparison")
    if make_comparison_video:
        render_comparison(
            demo_clip,
            restoration,
            metrics,
            config,
            out_dir / "comparison.mp4",
        )
    save_metadata(demo_clip, calibration_clip, config, out_dir)

    print("[6/7] Drawing calibration and diagnostic charts")
    render_calibration_plots(calibration, config, out_dir)
    render_diagnostics(demo_clip, restoration, metrics, config, out_dir)
    render_metric_chart(metrics, out_dir)
    render_contact_sheet(demo_clip, restoration, config, out_dir)

    print("[7/7] Writing interactive report")
    write_report(calibration, metrics, config, out_dir)
    elapsed = time.perf_counter() - started
    summary = {
        "output_dir": str(out_dir.resolve()),
        "elapsed_seconds": elapsed,
        "ghost_rms_original": metrics["ghost"]["captured"]["rms_8bit"],
        "ghost_rms_restored": metrics["ghost"]["restored"]["rms_8bit"],
        "overshoot_rms_original": metrics["overshoot"]["captured"][
            "rms_excursion_8bit"
        ],
        "overshoot_rms_restored": metrics["overshoot"]["restored"][
            "rms_excursion_8bit"
        ],
        "report": str((out_dir / "report.html").resolve()),
        "metrics": str((out_dir / "metrics.json").resolve()),
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
