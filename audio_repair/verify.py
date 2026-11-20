"""Headless verification of the repair pipeline.

Scenarios:
  1. bypass      -- identity mask; reconstruction error vs original PCM.
  2. chirps      -- crossed chirps + transients; circle the up-chirp and
                    measure its attenuation plus energy change of the
                    non-selected down-chirp and clicks.
  3. hum         -- tonal hum removal; music bed must be untouched.
  4. overlap     -- overlapping strokes saturate at weight 1.

Run:  python3 -m audio_repair.verify [--out artifacts]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .dsp import StftConfig, StftProcessor, band_energy, reconstruction_report
from .io_wav import save_wav
from .strokes import Stroke, combine_strokes
from .synth import crossed_chirps, hum_noise


def _db(ratio: float) -> float:
    return 10.0 * np.log10(max(ratio, 1e-300))


def scenario_bypass(proc: StftProcessor) -> dict:
    rng = np.random.default_rng(42)
    sr = proc.config.sample_rate
    t = np.arange(sr * 2) / sr
    x = 0.4 * np.sin(2 * np.pi * 440 * t) + 0.2 * rng.standard_normal(len(t))
    y, _ = proc.process(x, None)
    rep = reconstruction_report(x, y)
    edge = int(0.02 * sr)
    rep["edge_max_err"] = float(max(
        np.max(np.abs(y[:edge] - x[:edge])),
        np.max(np.abs(y[-edge:] - x[-edge:]))))
    rep["pass"] = (rep["length_match"] and rep["max_abs_err"] < 1e-10
                   and rep["edge_max_err"] < 1e-10)
    return rep


def scenario_chirps(proc: StftProcessor, out: Path | None) -> dict:
    fix = crossed_chirps(sr=proc.config.sample_rate)
    cfg = proc.config
    spec = proc.stft(fix["mix"])
    n_frames, n_bins = spec.shape
    up_stroke = Stroke.from_points(fix["trajectories"]["up"], radius_hz=150.0)
    mask_up = combine_strokes([up_stroke], n_frames, n_bins,
                              cfg.frame_sec, cfg.bin_hz)
    y, _ = proc.process(fix["mix"], mask_up)
    spec_after = proc.stft(y)

    down_stroke = Stroke.from_points(fix["trajectories"]["down"],
                                     radius_hz=150.0)
    mask_down = combine_strokes([down_stroke], n_frames, n_bins,
                                cfg.frame_sec, cfg.bin_hz)

    # clicks: broadband, so measure in time windows across all bins,
    # excluding bins claimed by the selected mask (fair "unselected" test)
    mask_clicks = np.zeros_like(mask_up)
    for tc in fix["click_times"]:
        f0 = int((tc - 0.03) / cfg.frame_sec)
        f1 = int((tc + 0.03) / cfg.frame_sec)
        mask_clicks[max(f0, 0):f1 + 1, :] = 1.0
    mask_clicks *= (mask_up < 0.05)

    def change(mask: np.ndarray) -> float:
        before = band_energy(spec, mask)
        after = band_energy(spec_after, mask)
        return _db(after / max(before, 1e-300))

    rep = {
        "selected_up_chirp_db": change(mask_up),
        "unselected_down_chirp_db": change(mask_down),
        "unselected_clicks_db": change(mask_clicks),
        "mask_coverage": float(np.mean(mask_up > 0.5)),
    }
    rep["pass"] = (rep["selected_up_chirp_db"] < -15.0
                   and abs(rep["unselected_down_chirp_db"]) < 1.5
                   and abs(rep["unselected_clicks_db"]) < 1.0)
    if out:
        save_wav(out / "chirps_mix.wav", fix["mix"], cfg.sample_rate)
        save_wav(out / "chirps_repaired.wav", y, cfg.sample_rate)
    return rep


def scenario_hum(proc: StftProcessor, out: Path | None) -> dict:
    fix = hum_noise(sr=proc.config.sample_rate)
    cfg = proc.config
    spec = proc.stft(fix["mix"])
    n_frames, n_bins = spec.shape
    dur = fix["dur"]
    strokes = [Stroke.from_points([(0.0, f), (dur, f)], radius_hz=35.0)
               for f in fix["hum_freqs"]]
    mask = combine_strokes(strokes, n_frames, n_bins,
                           cfg.frame_sec, cfg.bin_hz)
    y, _ = proc.process(fix["mix"], mask)
    spec_after = proc.stft(y)

    freqs = np.arange(n_bins) * cfg.bin_hz
    hum_band = np.zeros_like(mask)
    for f in fix["hum_freqs"]:
        hum_band[:, np.abs(freqs - f) <= 30.0] = 1.0
    music_band = np.zeros_like(mask)
    for f in fix["music_freqs"]:
        music_band[:, np.abs(freqs - f) <= 15.0] = 1.0

    rep = {
        "hum_band_db": _db(band_energy(spec_after, hum_band)
                           / band_energy(spec, hum_band)),
        "music_band_db": _db(band_energy(spec_after, music_band)
                             / band_energy(spec, music_band)),
    }
    rep["pass"] = (rep["hum_band_db"] < -20.0
                   and abs(rep["music_band_db"]) < 1.0)
    if out:
        save_wav(out / "hum_mix.wav", fix["mix"], cfg.sample_rate)
        save_wav(out / "hum_repaired.wav", y, cfg.sample_rate)
    return rep


def scenario_overlap(proc: StftProcessor) -> dict:
    cfg = proc.config
    n_frames, n_bins = 40, cfg.n_bins
    pts = [(0.0, 1000.0), (n_frames * cfg.frame_sec, 1000.0)]
    s1 = Stroke.from_points(pts, radius_hz=100.0, depth=1.0)
    s2 = Stroke.from_points(pts, radius_hz=100.0, depth=1.0)
    single = combine_strokes([s1], n_frames, n_bins,
                             cfg.frame_sec, cfg.bin_hz)
    double = combine_strokes([s1, s2], n_frames, n_bins,
                             cfg.frame_sec, cfg.bin_hz)
    half = Stroke.from_points(pts, radius_hz=100.0, depth=0.5)
    half2 = combine_strokes([half, half], n_frames, n_bins,
                            cfg.frame_sec, cfg.bin_hz)
    rep = {
        "single_max": float(single.max()),
        "double_max": float(double.max()),
        "half_depth_union_max": float(half2.max()),
    }
    expected_half = 1.0 - (1.0 - rep["single_max"] * 0.5) ** 2
    rep["pass"] = (rep["single_max"] <= 1.0
                   and rep["double_max"] <= 1.0
                   and rep["double_max"] >= rep["single_max"]
                   and abs(rep["half_depth_union_max"] - expected_half) < 1e-9)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None,
                    help="directory for before/after wav files")
    args = ap.parse_args()
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)

    proc = StftProcessor(StftConfig(sample_rate=44100))
    results = {
        "bypass": scenario_bypass(proc),
        "chirps": scenario_chirps(proc, args.out),
        "hum": scenario_hum(proc, args.out),
        "overlap": scenario_overlap(proc),
    }
    ok = True
    for name, rep in results.items():
        status = "PASS" if rep.pop("pass") else "FAIL"
        ok &= status == "PASS"
        body = "  ".join(f"{k}={v:.4g}" if isinstance(v, float)
                         else f"{k}={v}" for k, v in rep.items())
        print(f"[{status}] {name:8s} {body}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
