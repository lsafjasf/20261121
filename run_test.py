"""Run the dual-line synthesizer on the generated test recording and
report per-interval source attribution and output continuity."""
import wave
import numpy as np

import mixer
from gen_test_audio import SR, OUT_DIR


def read_wav(path):
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        pcm = np.frombuffer(w.readframes(n), dtype=np.int16)
    return pcm.astype(np.float64) / 32768.0


def write_wav(path, x):
    pcm = (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def intervals(labels):
    """Collapse per-frame labels into (start_s, end_s, label) runs."""
    runs = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append((start * mixer.FRAME / SR, i * mixer.FRAME / SR, labels[start]))
            start = i
    return runs


def continuity(out, runs):
    """Max |x[t]-x[t-1]| inside each run and at each switch boundary."""
    step = np.abs(np.diff(out))
    rows = []
    for (t0, t1, label) in runs:
        a, b = int(t0 * SR), int(t1 * SR)
        inner = step[a + 1:b - 1].max() if b - a > 2 else 0.0
        rows.append((t0, t1, label, inner))
    bounds = []
    for k in range(1, len(runs)):
        b = int(runs[k][0] * SR)
        lo, hi = max(0, b - 4), min(len(step), b + 4)
        bounds.append((runs[k][0], runs[k - 1][2], runs[k][2],
                       step[lo:hi].max()))
    return rows, bounds


def main():
    main_x = read_wav(f"{OUT_DIR}/main.wav")
    back_x = read_wav(f"{OUT_DIR}/backup.wav")
    synth = mixer.Synthesizer(main_x, back_x, sr=SR)
    out, labels = synth.run()
    write_wav(f"{OUT_DIR}/output.wav", out)

    print(f"estimated backup delay: {synth.delay} samples "
          f"({synth.delay / SR * 1000:.1f} ms)")
    runs = intervals(labels)
    rows, bounds = continuity(out, runs)

    print("\nper-interval source attribution and continuity:")
    print(f"{'interval(s)':>14}  {'source':<7}  {'max inner step':>14}")
    for t0, t1, label, inner in rows:
        print(f"{t0:6.2f}-{t1:<6.2f}  {label:<7}  {inner:14.4f}")

    print("\nswitch-boundary continuity (max step within +/-4 samples):")
    print(f"{'at(s)':>7}  {'transition':<15}  {'max step':>9}")
    for t, a, b, s in bounds:
        print(f"{t:7.2f}  {a + '->' + b:<15}  {s:9.4f}")

    # reference: max natural step of the 1 kHz pilot at amplitude 0.5
    ref = 0.5 * 2 * np.pi * 1000 / SR
    print(f"\nreference: natural max step of 1 kHz pilot ~= {ref:.4f}")
    print(f"wrote {OUT_DIR}/output.wav")


if __name__ == "__main__":
    main()
