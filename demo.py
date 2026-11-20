"""Demo: synthesize a degraded stereo transfer of a mono record, restore it
with the streaming chain in restore.py, and report / export the results.

Outputs (written to ./out):
  sample_clean.wav      reference mono mix (as identical stereo pair)
  sample_degraded.wav   cartridge transfer: gain/delay mismatch + scratches
  sample_restored.wav   streaming-restored stereo
  patch_XX_<ch>_before.wav / patch_XX_<ch>_after.wav   A/B snippet per repair

Console report lists every patched sample range, the transient
preservation ratio for the drum hits, and scratch removal in dB.
"""

import os

import numpy as np
from scipy import io, signal

from restore import StereoClickRestorer, calibrate

FS = 44100
DUR = 6.0
BLOCK = 1024
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def synth_clean(fs=FS, dur=DUR, seed=7):
    """Mono 'record': harmonic bed + alternating kick/snare hits."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * dur)) / fs
    bed = np.zeros_like(t)
    for f, a in [(110.0, 0.30), (165.0, 0.18), (220.0, 0.12), (330.0, 0.07)]:
        bed += a * np.sin(2 * np.pi * f * t + 0.3 * np.sin(2 * np.pi * 0.5 * t))
    bed *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.25 * t) ** 2
    bed += 0.004 * rng.standard_normal(len(t))          # faint groove noise

    hits = []                                            # (t0, kind) ground truth
    hit_times = [0.50, 1.25, 2.00, 2.75, 3.50, 4.25, 5.00]
    for i, t0 in enumerate(hit_times):
        kind = "kick" if i % 2 == 0 else "snare"
        n = int(0.30 * fs)
        tt = np.arange(n) / fs
        i0 = int(t0 * fs)
        if kind == "kick":
            f_inst = 45 + 75 * np.exp(-tt / 0.02)
            ph = 2 * np.pi * np.cumsum(f_inst) / fs
            hit = 0.9 * np.sin(ph) * np.exp(-tt / 0.07)
        else:
            noise = rng.standard_normal(n)
            b, a = signal.butter(3, [1500, 5000], btype="band", fs=fs)
            hit = 0.55 * signal.lfilter(b, a, noise) * np.exp(-tt / 0.05)
            hit += 0.4 * np.sin(2 * np.pi * 190 * tt) * np.exp(-tt / 0.09)
        sl = slice(i0, i0 + n)
        bed[sl] += hit[: len(bed[sl])]
        hits.append((t0, kind))
    return bed, hits


def add_scratch(x, fs, t0, amp=1.4, dur_ms=1.8, seed=0):
    """Sharp single-sided scratch: pop + short decaying HF burst."""
    rng = np.random.default_rng(seed)
    n = int(dur_ms * fs / 1000)
    i0 = int(t0 * fs)
    tt = np.arange(n) / fs
    burst = rng.standard_normal(n) * np.exp(-tt / (dur_ms / 4000))
    b, a = signal.butter(2, 2000, btype="high", fs=fs)
    burst = signal.lfilter(b, a, burst)
    burst /= np.max(np.abs(burst)) + 1e-12
    x[i0:i0 + n] += amp * burst
    x[i0] += 0.7 * amp * np.sign(rng.standard_normal())  # the initial pop
    return i0, i0 + n


def make_degraded(clean, fs=FS):
    """Stereo cartridge transfer: R quieter + delayed, different scratches."""
    rng = np.random.default_rng(123)
    n = len(clean)
    delay = 7
    gain_r = 0.82
    left = clean + 0.0006 * rng.standard_normal(n)       # preamp noise (uncorrelated)
    right = gain_r * np.concatenate([np.zeros(delay), clean[:-delay]])
    right += 0.0006 * rng.standard_normal(n)

    scratches = []                                       # (t0, ch, (i0, i1))
    for t0, ch, seed in [(1.02, "L", 11), (2.005, "L", 12),
                         (2.33, "R", 13), (3.90, "L", 14), (4.60, "R", 15)]:
        tgt = left if ch == "L" else right
        i0, i1 = add_scratch(tgt, fs, t0, seed=seed)
        scratches.append((t0, ch, (i0, i1)))
    return left, right, dict(delay=delay, gain=gain_r), scratches


# ---------------------------------------------------------------------------
# Streaming restoration
# ---------------------------------------------------------------------------

def run_streaming(left, right, fs=FS, block=BLOCK, calib_len=None):
    """Calibrate on the stream head, then restore block by block."""
    calib_len = calib_len or int(0.45 * fs)              # head is hit-free
    lag, gain = calibrate(left[:calib_len], right[:calib_len])
    rest = StereoClickRestorer(fs, lag=lag, gain=gain)
    out_l, out_r = [], []
    for i in range(0, len(left), block):
        lo, ro = rest.process(left[i:i + block], right[i:i + block])
        out_l.append(lo)
        out_r.append(ro)
    lo, ro = rest.flush()
    out_l.append(lo)
    out_r.append(ro)
    rec_l = np.concatenate(out_l)
    rec_r = np.concatenate(out_r)
    n = min(len(rec_l), len(left))
    return rec_l[:n], rec_r[:n], rest.patches, (lag, gain)


# ---------------------------------------------------------------------------
# Metrics + export
# ---------------------------------------------------------------------------

def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def transient_preservation(clean, rec_l, rec_r, hits, patches, fs=FS,
                           win_ms=150.0, corr_min=0.90, untouched_min=0.95):
    """Per drum hit: waveform correlation + fraction of un-patched samples."""
    n_win = int(win_ms * fs / 1000)
    rows = []
    for t0, kind in hits:
        s = int(t0 * fs)
        e = min(s + n_win, len(clean))
        mid_rec = 0.5 * (rec_l[s:e] + rec_r[s:e])
        c = _corr(clean[s:e], mid_rec)
        patched = np.zeros(e - s, dtype=bool)
        for p in patches:                       # a patch touches one channel;
            ps = max(s, p["start"])             # count it half-weight on the mid
            pe = min(e, p["end"])
            if pe > ps:
                patched[ps - s:pe - s] = True
        untouched = 1.0 - patched.mean()
        rows.append(dict(t0=t0, kind=kind, corr=c, untouched=untouched,
                         ok=(c >= corr_min and untouched >= untouched_min)))
    ratio = sum(r["ok"] for r in rows) / len(rows)
    return rows, ratio


def scratch_reduction(ref, before, after, scratches, fs=FS, guard_ms=8.0):
    """Error-energy reduction vs reference inside each scratch window.

    `before`/`after` are per-channel dicts {'L': ..., 'R': ...} already
    aligned to `ref`; scratch windows are given in the same timeline.
    """
    g = int(guard_ms * fs / 1000)
    out = []
    for t0, ch, (i0, i1) in scratches:
        s, e = max(0, i0 - g), min(len(ref), i1 + g)
        err_b = np.sum((before[ch][s:e] - ref[s:e]) ** 2)
        err_a = np.sum((after[ch][s:e] - ref[s:e]) ** 2)
        red_db = 10 * np.log10((err_b + 1e-12) / (err_a + 1e-12))
        out.append((t0, ch, red_db))
    return out


def write_wav(path, left, right=None, fs=FS):
    right = left if right is None else right
    data = np.stack([left, right], axis=1)
    peak = np.abs(data).max()
    if peak > 0.98:                          # avoid clipping the int16 export
        data = data * (0.98 / peak)
    io.wavfile.write(path, fs, (data * 32767).astype(np.int16))


def export_snippets(deg_l, deg_r, rec_l, rec_r, patches, fs=FS, half_ms=60.0):
    h = int(half_ms * fs / 1000)
    for i, p in enumerate(patches, 1):
        c = (p["start"] + p["end"]) // 2
        s, e = max(0, c - h), c + h
        tag = f"patch_{i:02d}_{p['channel']}"
        write_wav(os.path.join(OUT, f"{tag}_before.wav"), deg_l[s:e], deg_r[s:e])
        write_wav(os.path.join(OUT, f"{tag}_after.wav"), rec_l[s:e], rec_r[s:e])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT, exist_ok=True)
    clean, hits = synth_clean()
    deg_l, deg_r, truth, scratches = make_degraded(clean)
    print(f"[synth] {DUR:.1f}s @ {FS} Hz | cartridge: R gain={truth['gain']}, "
          f"R delay={truth['delay']} samples | scratches: "
          + ", ".join(f"{t0:.3f}s/{ch}" for t0, ch, _ in scratches))

    rec_l, rec_r, patches, (lag, gain) = run_streaming(deg_l, deg_r)
    print(f"[calib] estimated lag={lag} (true {truth['delay']}), "
          f"gain={gain:.4f} (true {1 / truth['gain']:.4f})")

    print(f"\n[patches] {len(patches)} repairs (block-streaming, "
          f"block={BLOCK}, latency=look+align):")
    print(f"  {'#':>3} {'channel':>7} {'start':>8} {'end':>8} "
          f"{'len(smp)':>8} {'len(ms)':>7} {'confidence':>10}")
    for i, p in enumerate(patches, 1):
        ln = p["end"] - p["start"]
        print(f"  {i:>3} {p['channel']:>7} {p['start']:>8} {p['end']:>8} "
              f"{ln:>8} {1000 * ln / FS:>7.2f} {p['confidence']:>10.1f}")

    # Align everything to the clean-reference timeline.  The restorer delays
    # the leading channel by |lag| samples, so all outputs sit |lag| samples
    # late; the degraded R is additionally gain-compensated for comparison.
    off = abs(lag)
    n = len(clean) - off
    ref = clean[:n]
    rec_al = {"L": rec_l[off:off + n], "R": rec_r[off:off + n]}
    deg_al = {"L": deg_l[off:off + n], "R": gain * deg_r[off:off + n]}
    patches_al = [dict(p, start=p["start"] - off, end=p["end"] - off)
                  for p in patches]
    scratches_al = [(t0, ch, (i0 - off, i1 - off)) for t0, ch, (i0, i1)
                    in scratches]

    rows, ratio = transient_preservation(ref, rec_al["L"], rec_al["R"],
                                         hits, patches_al)
    print("\n[transients] drum-hit preservation:")
    print(f"  {'t(s)':>6} {'kind':>6} {'corr':>7} {'untouched':>10} {'kept':>5}")
    for r in rows:
        print(f"  {r['t0']:>6.2f} {r['kind']:>6} {r['corr']:>7.3f} "
              f"{r['untouched']:>10.3f} {'yes' if r['ok'] else 'NO':>5}")
    print(f"  preservation ratio: "
          f"{sum(r['ok'] for r in rows)}/{len(rows)} = {ratio:.2%}")

    print("\n[scratches] error-energy reduction vs clean reference:")
    for t0, ch, red in scratch_reduction(ref, deg_al, rec_al, scratches_al):
        print(f"  {t0:>5.3f}s ch={ch}: {red:6.1f} dB")

    write_wav(os.path.join(OUT, "sample_clean.wav"), clean)
    write_wav(os.path.join(OUT, "sample_degraded.wav"), deg_l, deg_r)
    write_wav(os.path.join(OUT, "sample_restored.wav"), rec_l, rec_r)
    export_snippets(deg_l, deg_r, rec_l, rec_r, patches)
    print(f"\n[audio] WAVs written to {OUT}/ "
          f"(full before/after + per-patch A/B snippets)")


if __name__ == "__main__":
    main()
