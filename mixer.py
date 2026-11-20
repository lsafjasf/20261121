"""Dual-input recording synthesizer.

Uses the main line while it is healthy. When the main line freezes or
drops frames, switches to a backup line carrying the same content with
a small delay -- but only after verifying content correlation. Handles
both-lines-bad (mute) and switch-back after the main line recovers.
"""
import numpy as np

FRAME = 160            # 10 ms @ 16 kHz
FREEZE_FRAMES = 3      # consecutive identical frames => frozen
FREEZE_LOOKBACK = 4    # a frame repeating any of the last K frames is a repeat
BAD_HOLD_FRAMES = 2    # consecutive bad frames before leaving a source
RECOVER_FRAMES = 5     # consecutive good frames before trusting main again
CORR_THRESHOLD = 0.6   # min normalized correlation to allow a switch
SILENCE_RMS = 0.01     # below this a frame counts as silence
CLIP_LEVEL = 0.98
CLIP_RATIO = 0.01
FADE = 64              # crossfade length in samples (4 ms)


def frame_rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if len(x) else 0.0


class LineHealth:
    """Per-frame health of one input line."""

    def __init__(self, x):
        self.x = x
        self.n_frames = len(x) // FRAME
        self.rms = np.zeros(self.n_frames)
        self.frozen = np.zeros(self.n_frames, dtype=bool)
        self.zero = np.zeros(self.n_frames, dtype=bool)
        self.clipped = np.zeros(self.n_frames, dtype=bool)
        self._run()

    def _run(self):
        history = []
        run = 0
        for i in range(self.n_frames):
            f = self.x[i * FRAME:(i + 1) * FRAME]
            self.rms[i] = frame_rms(f)
            self.zero[i] = not np.any(f)
            self.clipped[i] = np.mean(np.abs(f) >= CLIP_LEVEL) > CLIP_RATIO
            # exact digital repeat of any recent frame => frozen buffer;
            # real ADC noise makes exact repeats of live audio impossible
            if any(np.array_equal(f, h) for h in history):
                run += 1
            else:
                run = 0
            self.frozen[i] = run >= FREEZE_FRAMES - 1
            history = ([f] + history)[:FREEZE_LOOKBACK]

    def bad(self, i):
        return bool(self.frozen[i] or self.zero[i] or self.clipped[i])

    def silent(self, i):
        return bool(self.rms[i] < SILENCE_RMS)


def norm_corr(a, b):
    """Normalized correlation between equal-length segments."""
    a = a.astype(np.float64) - np.mean(a)
    b = b.astype(np.float64) - np.mean(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def estimate_delay(main, backup, max_lag=int(0.25 * 16000)):
    """Lag of backup vs main, in samples (backup lags by this amount).

    Estimated once from an early mutually healthy window via FFT
    cross-correlation. For periodic content (pilot tone) several lags
    correlate equally well, so the peak nearest zero lag is chosen.
    """
    n = min(len(main), len(backup), 16000)
    a = main[:n].astype(np.float64)
    b = backup[:n].astype(np.float64)
    size = 1 << (2 * n - 1).bit_length()
    fa = np.fft.rfft(a, size)
    fb = np.fft.rfft(b, size)
    corr = np.fft.irfft(fb * np.conj(fa), size)
    lags = np.concatenate([np.arange(0, max_lag + 1),
                           np.arange(size - max_lag, size)])
    vals = corr[lags]
    peak = np.max(vals)
    near = np.where(vals >= 0.99 * peak)[0]
    lag = lags[near[np.argmin(np.abs(lags[near]))]]
    if lag > size // 2:
        lag -= size
    return int(lag)


MAIN, BACKUP, MUTE = "main", "backup", "mute"


class Synthesizer:
    """Frame-level switch state machine plus gapless renderer."""

    def __init__(self, main, backup, sr=16000):
        self.sr = sr
        self.main = main.astype(np.float64)
        self.backup = backup.astype(np.float64)
        self.n = min(len(self.main), len(self.backup))
        self.main_h = LineHealth(self.main)
        self.back_h = LineHealth(self.backup)
        self.delay = estimate_delay(self.main, self.backup)
        # backup aligned onto the main timeline: back_al[t] ~ main[t]
        d = self.delay
        if d >= 0:
            self.back_al = np.concatenate([self.backup[d:], np.zeros(d)])
        else:
            self.back_al = np.concatenate([np.zeros(-d), self.backup[:d]])
        self.back_al = self.back_al[:self.n]

    def content_matches(self, t, w=480):
        """Check main vs aligned backup over a window ending at t.

        Silence on both sides counts as compatible (correlation is
        undefined for silence); otherwise require correlation.
        """
        if t - w < 0 or t > self.n:
            return True, 1.0
        ma = self.main[t - w:t]
        ba = self.back_al[t - w:t]
        if frame_rms(ma) < SILENCE_RMS and frame_rms(ba) < SILENCE_RMS:
            return True, 1.0
        c = norm_corr(ma, ba)
        return c >= CORR_THRESHOLD, c

    def decide(self):
        """Return per-frame source labels."""
        n_f = self.main_h.n_frames
        labels = []
        state = MAIN
        main_bad = back_bad = main_good = back_good = 0
        for i in range(n_f):
            mb = self.main_h.bad(i)
            bb = self.back_h.bad(i)
            main_bad = main_bad + 1 if mb else 0
            back_bad = back_bad + 1 if bb else 0
            main_good = 0 if mb else main_good + 1
            back_good = 0 if bb else back_good + 1
            t = i * FRAME
            if state == MAIN:
                if main_bad >= BAD_HOLD_FRAMES:
                    ok, _ = self.content_matches(t)
                    if not bb and ok:
                        state = BACKUP
                    else:
                        state = MUTE
            elif state == BACKUP:
                if back_bad >= BAD_HOLD_FRAMES:
                    if not mb and main_good >= RECOVER_FRAMES:
                        ok, _ = self.content_matches(t)
                        state = MAIN if ok else MUTE
                    else:
                        state = MUTE
                elif main_good >= RECOVER_FRAMES:
                    ok, _ = self.content_matches(t)
                    if ok:
                        state = MAIN
            else:  # MUTE
                if not mb and main_good >= RECOVER_FRAMES:
                    state = MAIN
                elif not bb and back_good >= RECOVER_FRAMES:
                    ok, _ = self.content_matches(t)
                    if ok:
                        state = BACKUP
            labels.append(state)
        return labels

    def render(self, labels):
        """Render output with raised-cosine crossfades at transitions."""
        n_f = len(labels)
        src_sig = {MAIN: self.main, BACKUP: self.back_al, MUTE: np.zeros(self.n)}
        out = np.zeros(self.n)
        for i, s in enumerate(labels):
            out[i * FRAME:(i + 1) * FRAME] = src_sig[s][i * FRAME:(i + 1) * FRAME]
        half = FADE // 2
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, FADE))
        for i in range(1, n_f):
            if labels[i] != labels[i - 1]:
                b = i * FRAME
                lo, hi = max(0, b - half), min(self.n, b + half)
                seg = ramp[lo - (b - half):hi - (b - half)]
                prev = src_sig[labels[i - 1]][lo:hi]
                nxt = src_sig[labels[i]][lo:hi]
                out[lo:hi] = (1 - seg) * prev + seg * nxt
        return out

    def run(self):
        labels = self.decide()
        return self.render(labels), labels
