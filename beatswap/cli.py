"""Command-line entry point for the beatswap splice finder."""

from __future__ import annotations

import argparse
import os

import numpy as np

from . import align, audio, candidates, features, render, report


def run(path_a, path_b, outdir, bars=8, beats_per_bar=4, top_k=5,
        fade_ms=15.0):
    os.makedirs(outdir, exist_ok=True)
    sr = audio.TARGET_SR
    xa, sr = audio.load_wav(path_a, sr)
    xb, sr = audio.load_wav(path_b, sr)
    xa, gain_a = audio.normalize_loudness(xa)
    xb, gain_b = audio.normalize_loudness(xb)

    env_a, fps = features.onset_envelope(xa, sr)
    env_b, _ = features.onset_envelope(xb, sr)
    bpm_a = features.estimate_tempo(env_a, fps)
    bpm_b = features.estimate_tempo(env_b, fps)
    beats_a = features.track_beats(env_a, fps, bpm_a)
    beats_b = features.track_beats(env_b, fps, bpm_b)

    dec = 2  # decimate envelopes for DTW to bound memory/time
    pa, pb, dtw_cost = align.dtw_path(env_a[::dec], env_b[::dec])
    mapper = align.make_frame_mapper(pa * dec, pb * dec)

    chosen, scored, diag = candidates.find_candidates(
        env_a, beats_a, env_b, beats_b, mapper, fps,
        bars=bars, beats_per_bar=beats_per_bar, top_k=top_k)

    previews = []
    for rank, c in enumerate(chosen, 1):
        fa0, fa1 = c["a_entry_frame"], c["a_exit_frame"]
        fb0, fb1 = c["b_entry_frame"], c["b_exit_frame"]
        c["a_entry_s"] = round(fa0 / fps, 3)
        c["a_exit_s"] = round(fa1 / fps, 3)
        c["b_entry_s"] = round(fb0 / fps, 3)
        c["b_exit_s"] = round(fb1 / fps, 3)
        out_ab = os.path.join(outdir, f"preview{rank}_A_with_B_middle.wav")
        out_ba = os.path.join(outdir, f"preview{rank}_B_with_A_middle.wav")
        joints = []
        for out_path, head, hc, mid, mi, mo, tail, ti in (
                (out_ab, xa, fa0 / fps, xb, fb0 / fps, fb1 / fps,
                 xa, fa1 / fps),
                (out_ba, xb, fb0 / fps, xa, fa0 / fps, fa1 / fps,
                 xb, fb1 / fps)):
            joined, (j1, j2) = render.render_swap(
                head, hc, mid, mi, mo, tail, ti, sr, fade_ms)
            audio.save_wav(out_path, joined, sr)
            for tag, jc in (("entry", j1), ("exit", j2)):
                clip_path = out_path.replace(
                    ".wav", f"_joint_{tag}.wav")
                audio.save_wav(clip_path,
                               render.joint_clip(joined, jc, sr), sr)
                joints.append(clip_path)
        previews.append(dict(rank=rank, a_with_b_middle=out_ab,
                             b_with_a_middle=out_ba,
                             joint_clips=joints))

    t_a = np.arange(len(env_a)) / fps
    t_b = np.arange(len(env_b)) / fps
    windows = [(c["a_entry_s"], c["a_exit_s"]) for c in chosen]
    report.plot_envelopes(os.path.join(outdir, "evidence_envelopes.svg"),
                          t_a, env_a, t_b, env_b, windows,
                          "Onset envelopes (green = candidate swap windows)")
    report.plot_dtw(os.path.join(outdir, "evidence_dtw.svg"),
                    pa * dec, pb * dec, fps,
                    "DTW warping path (grey = identical clocks)")
    report.plot_scores(os.path.join(outdir, "evidence_scores.svg"),
                       [c["entry_beat"] for c in scored],
                       [c["score"] for c in scored],
                       [c["entry_beat"] for c in chosen],
                       "Swap-window score vs entry beat")

    payload = dict(
        inputs=dict(a=path_a, b=path_b),
        loudness_gain_db=dict(a=round(gain_a, 2), b=round(gain_b, 2)),
        tempo_bpm=dict(a=round(bpm_a, 2), b=round(bpm_b, 2)),
        beats_detected=dict(a=len(beats_a), b=len(beats_b)),
        dtw_mean_step_cost=round(dtw_cost, 4),
        diagnostics=diag,
        candidates=chosen,
        previews=previews,
        evidence_files=[
            "evidence_envelopes.svg", "evidence_dtw.svg",
            "evidence_scores.svg", "report.json"])
    report.write_json(os.path.join(outdir, "report.json"), payload)

    print(f"A: {path_a}  ({bpm_a:.1f} BPM, {len(beats_a)} beats, "
          f"gain {gain_a:+.1f} dB)")
    print(f"B: {path_b}  ({bpm_b:.1f} BPM, {len(beats_b)} beats, "
          f"gain {gain_b:+.1f} dB)")
    print(f"tempo ratio A/B: {diag['tempo_ratio_a_over_b']:.4f}  "
          f"DTW cost: {dtw_cost:.3f}")
    print(f"\nTop {len(chosen)} splice candidates ({bars} bars = "
          f"{bars * beats_per_bar} beats):")
    for rank, c in enumerate(chosen, 1):
        flag = "  <-- WARNINGS: " + "; ".join(c["warnings"]) \
            if c["warnings"] else ""
        print(f"  #{rank} beats {c['entry_beat']:>3}-{c['exit_beat']:>3} "
              f"A[{c['a_entry_s']:7.2f}s .. {c['a_exit_s']:7.2f}s] <-> "
              f"B[{c['b_entry_s']:7.2f}s .. {c['b_exit_s']:7.2f}s] "
              f"score={c['score']:.3f} "
              f"(entry {c['entry_sim']:.2f} exit {c['exit_sim']:.2f} "
              f"interior {c['interior_sim']:.2f}){flag}")
    print(f"\nPreviews and evidence written to {outdir}/")
    return payload


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="beatswap",
        description="Find beat-aligned splice points to swap a middle "
                    "section between two recordings of the same "
                    "accompaniment (robust to loudness and small tempo "
                    "differences).")
    p.add_argument("audio_a")
    p.add_argument("audio_b")
    p.add_argument("--out", default="beatswap_out")
    p.add_argument("--bars", type=int, default=8)
    p.add_argument("--beats-per-bar", type=int, default=4)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--fade-ms", type=float, default=15.0)
    args = p.parse_args(argv)
    run(args.audio_a, args.audio_b, args.out, bars=args.bars,
        beats_per_bar=args.beats_per_bar, top_k=args.top_k,
        fade_ms=args.fade_ms)


if __name__ == "__main__":
    main()
