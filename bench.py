"""PLC test bench: builds test signals, drops packets, runs the PLC,
and writes listenable 16-bit WAV files plus a loss-pattern report.

Usage:
    python3 bench.py            # generate out/*.wav
    python3 bench.py --rt       # also report real-time processing factor
"""

import math
import os
import random
import sys
import time
import wave

from plc import StreamPLC

FS = 16000
FRAME_MS = 20.0
FRAME = int(FS * FRAME_MS / 1000)  # 320 samples
OUT_DIR = "out"


# ----------------------------------------------------------------------
# test signals
# ----------------------------------------------------------------------
def gen_voiced(dur_s, f0=120.0, amp=0.5):
    """Harmonic 'voiced-like' signal with slight vibrato."""
    n = int(dur_s * FS)
    out = []
    phase = 0.0
    for i in range(n):
        t = i / FS
        f = f0 * (1.0 + 0.02 * math.sin(2.0 * math.pi * 3.0 * t))
        phase += 2.0 * math.pi * f / FS
        s = sum(math.sin(k * phase) / k for k in range(1, 6))
        out.append(amp * 0.5 * s)
    return out


def gen_silence(dur_s, floor=1e-4):
    """Digital silence with a tiny noise floor."""
    rng = random.Random(1234)
    return [floor * (rng.random() * 2.0 - 1.0) for _ in range(int(dur_s * FS))]


def gen_abrupt_stop(dur_s=1.6, stop_at=0.8):
    """Voiced signal that cuts to zero mid-stream (talker hangs up)."""
    sig = gen_voiced(dur_s)
    cut = int(stop_at * FS)
    for i in range(cut, len(sig)):
        sig[i] = 0.0
    return sig


# ----------------------------------------------------------------------
# loss patterns: set of lost frame indices
# ----------------------------------------------------------------------
def frames_of(samples):
    return [samples[i:i + FRAME] for i in range(0, len(samples) - FRAME + 1, FRAME)]


def run_case(name, samples, lost_frames, rt=False):
    plc = StreamPLC(sample_rate=FS, frame_ms=FRAME_MS)
    frames = frames_of(samples)
    out = []
    worst_us = 0.0
    total_us = 0.0
    for idx, fr in enumerate(frames):
        lost = idx in lost_frames
        t0 = time.perf_counter()
        res = plc.process(None if lost else fr, lost)
        dt = (time.perf_counter() - t0) * 1e6
        worst_us = max(worst_us, dt)
        total_us += dt
        out.extend(res)
    if rt:
        n = len(frames)
        print("[%s] avg %.1f us/frame, worst %.1f us/frame (budget %.0f us)"
              % (name, total_us / n, worst_us, FRAME_MS * 1000))
    return out


def write_wav(path, samples):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(FS)
        data = bytearray()
        for x in samples:
            v = int(max(-1.0, min(1.0, x)) * 32767.0)
            data += v.to_bytes(2, "little", signed=True)
        w.writeframes(bytes(data))


def sec_to_frame(t):
    return int(t * FS / FRAME)


CASES = [
    # (name, signal, lost frame indices)
    ("voiced", gen_voiced(2.0),
     {sec_to_frame(0.5)}                                  # single packet
     | set(range(sec_to_frame(1.0), sec_to_frame(1.0) + 3))    # burst x3
     | set(range(sec_to_frame(1.5), sec_to_frame(1.5) + 12))), # long burst
    ("silence", gen_silence(1.0),
     set(range(sec_to_frame(0.4), sec_to_frame(0.4) + 5))),
    ("abrupt_stop", gen_abrupt_stop(),
     set(range(sec_to_frame(0.78), sec_to_frame(0.78) + 4))   # spans the cut
     | set(range(sec_to_frame(1.2), sec_to_frame(1.2) + 25))), # long, post-stop
]


def main():
    rt = "--rt" in sys.argv
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, samples, lost in CASES:
        concealed = run_case(name, samples, lost, rt=rt)
        lossy = list(samples)
        for f in lost:
            for i in range(f * FRAME, (f + 1) * FRAME):
                if i < len(lossy):
                    lossy[i] = 0.0
        write_wav(os.path.join(OUT_DIR, name + "_clean.wav"), samples)
        write_wav(os.path.join(OUT_DIR, name + "_lossy.wav"), lossy)
        write_wav(os.path.join(OUT_DIR, name + "_plc.wav"), concealed)
        print("[%s] lost frames: %s -> out/%s_{clean,lossy,plc}.wav"
              % (name, sorted(lost), name))


if __name__ == "__main__":
    main()
