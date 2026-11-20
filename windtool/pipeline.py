"""End-to-end analysis pipeline shared by the CLI and tests."""

from __future__ import annotations

import os

import numpy as np

from .config import AnalysisConfig
from .drum_phase import check_drum_segments
from .dsp import extract_features
from .io import load_stereo, save_stereo, write_json
from .reduction import (
    apply_reduction,
    build_gain_envelope,
    measure_wind_reduction,
)
from .report import file_band_energy, plot_overview, write_html
from .segmentation import LABEL_NAMES, build_segments, classify_segments


def analyze_file(
    input_wav: str,
    out_dir: str,
    cfg: AnalysisConfig | None = None,
    report_title: str | None = None,
) -> dict:
    cfg = cfg or AnalysisConfig()
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_wav))[0]

    audio, sr = load_stereo(input_wav)
    features, n_fft, hop, win = extract_features(audio, sr, cfg)

    segments = build_segments(features, cfg)
    segments = classify_segments(segments, features, cfg)

    gain = build_gain_envelope(features, segments, cfg)
    processed = apply_reduction(features, gain, n_fft, hop, win, sr, audio.shape[1])

    processed_wav = os.path.join(out_dir, f"{base}_processed.wav")
    save_stereo(processed_wav, processed, sr)

    wind_rows = measure_wind_reduction(audio, processed, sr, segments, cfg)
    drum_rows = check_drum_segments(segments, features, cfg, sr)
    band_energy = file_band_energy(audio, sr, cfg.bands)

    figure_path = os.path.join(out_dir, f"{base}_overview.png")
    plot_overview(features, segments, gain, figure_path)

    payload = {
        "input": os.path.abspath(input_wav),
        "sample_rate": sr,
        "duration_sec": round(audio.shape[1] / sr, 3),
        "file_band_energy_dbfs": band_energy,
        "segments": [
            {
                "index": s.index,
                "start_sec": round(s.start_sec, 3),
                "end_sec": round(s.end_sec, 3),
                "duration_sec": round(s.duration_sec, 3),
                "label": s.label,
                "label_name": LABEL_NAMES[s.label],
                "target_attenuation_db": s.attenuation_db,
                "metrics": s.metrics,
                "reasons": s.reasons,
            }
            for s in segments
        ],
        "drum_phase_check": drum_rows,
        "wind_reduction_check": wind_rows,
        "outputs": {
            "processed_wav": os.path.abspath(processed_wav),
            "overview_png": os.path.abspath(figure_path),
        },
    }
    json_path = os.path.join(out_dir, f"{base}_report.json")
    write_json(json_path, payload)

    html_path = os.path.join(out_dir, f"{base}_report.html")
    write_html(
        html_path,
        report_title or f"Wind-noise annotation — {base}",
        segments,
        figure_path,
        input_wav,
        processed_wav,
        wind_rows,
        drum_rows,
        band_energy,
    )
    payload["outputs"]["report_json"] = os.path.abspath(json_path)
    payload["outputs"]["report_html"] = os.path.abspath(html_path)
    return payload
