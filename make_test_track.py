"""Generate testdata/A.wav and testdata/B.wav for beatswap.
Both versions share the same 8-bar accompaniment loop (syncopated kick,
snare ghosts, off-beat bass) repeated 4 times. Version B is 2% faster,
8 dB quieter, and has one EXTRA BEAT inserted in bar 20 (0-based) to
trigger the mis-match warning. The repeated loops produce multiple
equivalent splice candidates.
"""

import numpy as np
from scipy import io, signal

SR = 22050
BPM_A = 120.0
BPM_B = 122.4               # +2% tempo
GAIN_B = 10 ** (-8 / 20)    # -8 dB
N_LOOPS = 4
BARS_PER_LOOP = 8
EXTRA_BEAT_BAR = 20         # 0-based bar in B that gets a 5th beat

rng = np.random.default_rng(7)


def kick(t):
    f = 120 * np.exp(-t * 30) + 45
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * np.exp(-t * 22)


def snare(t):
    n = rng.standard_normal(len(t))
    b, a = signal.butter(2, [1500 / (SR / 2), 6000 / (SR / 2)], btype="band")
    return signal.lfilter(b, a, n) * np.exp(-t * 30) * 0.8


def hat(t, accent=1.0):
    n = rng.standard_normal(len(t))
    b, a = signal.butter(2, 7000 / (SR / 2), btype="high")
    return signal.lfilter(b, a, n) * np.exp(-t * 90) * 0.35 * accent


def bass(t, freq):
    x = np.sign(np.sin(2 * np.pi * freq * t))
    b, a = signal.butter(2, 350 / (SR / 2), btype="low")
    return signal.lfilter(b, a, x) * np.exp(-t * 4) * 0.5


def pad(t, freqs, dur):
    env = np.minimum(t / 0.4, 1.0) * np.minimum((dur - t) / 0.4, 1.0)
    env = np.clip(env, 0, 1)
    out = np.zeros(len(t))
    for f in freqs:
        out += np.sin(2 * np.pi * f * t + rng.uniform(0, 6))
    return out / max(len(freqs), 1) * env * 0.12


CHORDS = {
    0: [220.0, 261.63, 329.63], 1: [220.0, 261.63, 329.63],
    2: [174.61, 220.0, 261.63], 3: [174.61, 220.0, 261.63],
    4: [130.81, 164.81, 196.0], 5: [130.81, 164.81, 196.0],
    6: [196.0, 246.94, 293.66], 7: [196.0, 246.94, 293.66],
}
BASS_ROOT = [55.0, 55.0, 43.65, 43.65, 32.7, 32.7, 49.0, 49.0]


def add_event(buf, t_sec, sig):
    i = int(t_sec * SR)
    j = min(len(buf), i + len(sig))
    if i < j:
        buf[i:j] += sig[:j - i]


def bar_events(buf, bar_start_sec, beat_sec, chord_idx, vel=1.0):
    """One bar of the syncopated pattern, 4 beats starting at bar_start_sec."""
    s = bar_start_sec
    for beat in range(4):
        add_event(buf, s + beat * beat_sec,
                  kick(np.arange(int(0.15 * SR)) / SR) * vel)
    # syncopated extra kick on the "and" of 4
    add_event(buf, s + 3.5 * beat_sec,
              kick(np.arange(int(0.15 * SR)) / SR) * 0.8 * vel)
    # snare on 2 and 4, syncopated ghost on the "and" of 3
    t = np.arange(int(0.12 * SR)) / SR
    add_event(buf, s + 1.0 * beat_sec, snare(t) * vel)
    add_event(buf, s + 3.0 * beat_sec, snare(t) * vel)
    add_event(buf, s + 2.5 * beat_sec, snare(t) * 0.35 * vel)
    # eighth-note hats, accents pushed off the beat (syncopation)
    th = np.arange(int(0.05 * SR)) / SR
    for e in range(8):
        accent = 1.0 if e % 2 == 1 else 0.55
        add_event(buf, s + e * 0.5 * beat_sec, hat(th, accent) * vel)
    # syncopated bass: 1, and-of-2, 3, and-of-4
    tb = np.arange(int(0.45 * beat_sec * SR)) / SR
    root = BASS_ROOT[chord_idx]
    for off in (0.0, 1.5, 2.0, 3.5):
        add_event(buf, s + off * beat_sec, bass(tb, root) * vel)
    # pad for the whole bar
    dur = 4 * beat_sec
    tp = np.arange(int(dur * SR)) / SR
    add_event(buf, s, pad(tp, CHORDS[chord_idx], dur) * vel)


def render(bpm, extra_beat_bar=None):
    beat_sec = 60.0 / bpm
    total_beats = N_LOOPS * BARS_PER_LOOP * 4 + (1 if extra_beat_bar is not None else 0)
    buf = np.zeros(int((total_beats * beat_sec + 1.0) * SR))
    t = 0.0
    bar = 0
    for loop in range(N_LOOPS):
        for bar_in_loop in range(BARS_PER_LOOP):
            vel = 1.0 + rng.uniform(-0.03, 0.03)
            bar_events(buf, t, beat_sec, bar_in_loop, vel)
            t += 4 * beat_sec
            if extra_beat_bar is not None and bar == extra_beat_bar:
                # deliberately inserted extra beat: repeat beat 1 material
                add_event(buf, t, kick(np.arange(int(0.15 * SR)) / SR))
                th = np.arange(int(0.05 * SR)) / SR
                add_event(buf, t + 0.5 * beat_sec, hat(th, 1.0))
                t += beat_sec
            bar += 1
    return buf


def main():
    a = render(BPM_A)
    b = render(BPM_B, extra_beat_bar=EXTRA_BEAT_BAR)
    a = a / (np.max(np.abs(a)) + 1e-12) * 0.9
    b = b / (np.max(np.abs(b)) + 1e-12) * 0.9 * GAIN_B  # quieter recording
    for name, x in (("A", a), ("B", b)):
        io.wavfile.write(f"testdata/{name}.wav", SR, (x * 32767).astype(np.int16))
        print(f"testdata/{name}.wav  {len(x) / SR:.1f}s")


if __name__ == "__main__":
    main()
