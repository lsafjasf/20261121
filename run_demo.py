"""End-to-end validation of pilot-driven tape wow/flutter restoration.

Runs the full pipeline on synthetic material containing a stable note and a
deliberately vibratoed note, with slow drift, wow, flutter and pilot
dropouts.  Prints quantitative evidence that:

1. the stable note's frequency variance collapses (machine wow removed),
2. the performer's own vibrato depth/rate is preserved,
3. dropouts are bridged by bounded extrapolation.
"""

from __future__ import annotations

import os
import wave

import numpy as np

from wow_flutter.synthesis import generate_tape
from wow_flutter.pilot import track_pilot
from wow_flutter.warp import estimate_warp
from wow_flutter.resample import banded_resample
from wow_flutter.measure import note_frequency_track, trim_edges, vibrato_metrics


def main() -> None:
    out_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(out_dir, exist_ok=True)

    tape = generate_tape()
    fs = tape.sample_rate

    track = track_pilot(tape.pilot, fs, 1000.0)
    warp = estimate_warp(track, fs)
    restored = banded_resample(tape.program, warp.input_map, sample_rate=fs)

    stable = 392.0
    vib_note = 523.25
    results = {}
    for label, sig in (("warped", tape.program), ("restored", restored)):
        st, sf = note_frequency_track(sig, fs, stable)
        vt, vf = note_frequency_track(sig, fs, vib_note)
        sf = trim_edges(st, sf)
        vf_trim = trim_edges(vt, vf)
        results[label] = {
            "stable_std_hz": float(np.std(sf)),
            "stable_var_hz2": float(np.var(sf)),
            "vibrato": vibrato_metrics(vf_trim, vib_note),
        }

    # Clean reference for the vibrato ground truth.
    ct, clean_vf = note_frequency_track(tape.clean_program, fs, vib_note)
    clean_vf = trim_edges(ct, clean_vf)
    clean_vib = vibrato_metrics(clean_vf, vib_note)

    print("=" * 68)
    print("PILOT TRACKING")
    print(f"  frames={len(track.freq)} reliable={track.reliable.mean()*100:.1f}%")
    print(f"  detected dropout frames={np.mean(~track.reliable)*100:.1f}%")
    print(f"  extrapolated warp frames={warp.extrapolated.mean()*100:.1f}%")
    print("-" * 68)
    print("STABLE NOTE (392 Hz): machine-induced frequency wander")
    wv, rv = results["warped"], results["restored"]
    print(f"  warped    std={wv['stable_std_hz']:.4f} Hz  var={wv['stable_var_hz2']:.4f}")
    print(f"  restored  std={rv['stable_std_hz']:.4f} Hz  var={rv['stable_var_hz2']:.4f}")
    reduction = 1 - rv["stable_var_hz2"] / wv["stable_var_hz2"]
    print(f"  variance reduction = {reduction*100:.1f}%")
    print("-" * 68)
    print("VIBRATO NOTE (523.25 Hz): performer expression must survive")
    print(f"  clean ground truth depth={clean_vib['vibrato_depth_cents']:.1f} cents"
          f"  rate={clean_vib['vibrato_rate_hz']:.2f} Hz")
    print(f"  warped   depth={wv['vibrato']['vibrato_depth_cents']:.1f} cents"
          f"  rate={wv['vibrato']['vibrato_rate_hz']:.2f} Hz")
    print(f"  restored depth={rv['vibrato']['vibrato_depth_cents']:.1f} cents"
          f"  rate={rv['vibrato']['vibrato_rate_hz']:.2f} Hz")
    print("=" * 68)

    _plot(tape, track, warp, restored, results, out_dir, fs)
    _save_wav(tape, restored, out_dir, fs)


def _save_wav(tape, restored, out_dir, fs) -> None:
    def write(name, signal_):
        peak = max(1e-9, float(np.max(np.abs(signal_))))
        pcm = np.int16(signal_ / peak * 0.9 * 32767)
        path = os.path.join(out_dir, name)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(fs))
            wf.writeframes(pcm.tobytes())

    write("program_warped.wav", tape.program)
    write("program_restored.wav", restored)
    write("pilot_with_dropouts.wav", tape.pilot)
    print(f"audio saved to {out_dir}/*.wav")


def _plot(tape, track, warp, restored, results, out_dir, fs) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(tape.program)) / fs
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=False)

    ax = axes[0]
    ax.plot(track.frame_times, (track.freq - 1000.0), lw=1, label="tracked pilot IF error")
    ax.plot(track.frame_times, 1000.0 * warp.frame_speed_error, lw=1, label="smoothed estimate")
    ax.fill_between(track.frame_times, ax.get_ylim()[0], ax.get_ylim()[1],
                    where=~track.reliable, color="red", alpha=0.15, label="dropout")
    ax.set_ylabel("pilot freq error (Hz)"); ax.legend(fontsize=8); ax.set_title("Pilot tracking")

    ax = axes[1]
    ax.plot(t, tape.speed_error * 100, lw=1, label="true speed error")
    ax.plot(warp.frame_times, warp.frame_speed_error * 100, lw=1, label="estimated")
    ax.set_ylabel("speed error (%)"); ax.legend(fontsize=8); ax.set_title("Warp estimate")

    for sig, style, name in ((tape.program, ":", "warped"), (restored, "-", "restored")):
        tt, ff = note_frequency_track(sig, fs, 392.0)
        axes[2].plot(tt, ff - 392.0, style, lw=1.3, label=name)
    axes[2].set_ylabel("392 Hz note error (Hz)"); axes[2].legend(fontsize=8)
    axes[2].set_title("Stable note: variance should shrink")

    for sig, style, name in ((tape.program, ":", "warped"), (restored, "-", "restored")):
        tt, ff = note_frequency_track(sig, fs, 523.25)
        axes[3].plot(tt, 1200 * np.log2(ff / 523.25), style, lw=1.3, label=name)
    axes[3].set_ylabel("vibrato (cents)"); axes[3].set_xlabel("time (s)")
    axes[3].legend(fontsize=8); axes[3].set_title("Vibrato note: expression preserved")

    fig.tight_layout()
    path = os.path.join(out_dir, "restoration_report.png")
    fig.savefig(path, dpi=110)
    print(f"plot saved to {path}")


if __name__ == "__main__":
    main()
