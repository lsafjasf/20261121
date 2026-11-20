"""Streaming restoration chain for mono records digitized with a stereo cartridge.

Pipeline (all steps causal, block-based, constant latency):
  1. Cartridge calibration: estimate inter-channel delay (cross-correlation)
     and gain (least squares) from a short head of the stream.
  2. Alignment: delay the leading channel and scale the quieter channel so
     both channels carry the same mono signal.
  3. Detection: the difference signal L - R cancels common-mode music
     (including short shared transients such as drum hits); single-sided
     impulse contamination (scratches) sticks out.  Robust Median/MAD
     thresholds with hysteresis delimit candidate click regions.
  4. Verification: a region is only patched when exactly one channel
     deviates from the mid signal with enough confidence, so shared
     transients and alignment residue are left untouched.
  5. Repair: the corrupted channel is crossfaded to the clean channel
     over the detected region only.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 1. Cartridge calibration: delay + gain
# ---------------------------------------------------------------------------

def estimate_delay(ref: np.ndarray, targ: np.ndarray, max_lag: int = 512) -> int:
    """Return how many samples `targ` lags behind `ref` (negative = leads)."""
    n = min(len(ref), len(targ))
    ref = np.asarray(ref[:n], dtype=float) - np.mean(ref[:n])
    targ = np.asarray(targ[:n], dtype=float) - np.mean(targ[:n])
    nfft = 1 << (2 * n - 1).bit_length()
    xc = np.fft.irfft(np.fft.rfft(targ, nfft) * np.conj(np.fft.rfft(ref, nfft)), nfft)
    lags = np.concatenate([np.arange(0, max_lag + 1), np.arange(nfft - max_lag, nfft)])
    best = lags[np.argmax(np.abs(xc[lags]))]
    return int(best) if best <= max_lag else int(best - nfft)


def estimate_gain(ref: np.ndarray, targ: np.ndarray, lag: int) -> float:
    """Least-squares gain g so that g * targ(aligned) ~= ref."""
    if lag > 0:                      # targ lags ref
        a = ref[:-lag]
        b = targ[lag:]
    elif lag < 0:                    # targ leads ref
        a = ref[-lag:]
        b = targ[:lag]
    else:
        a, b = ref, targ
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    denom = float(np.dot(b, b))
    return float(np.dot(a, b) / denom) if denom > 0 else 1.0


def calibrate(left: np.ndarray, right: np.ndarray, max_lag: int = 512):
    """Estimate (lag_of_right_vs_left, gain_for_right) from a calibration head."""
    lag = estimate_delay(left, right, max_lag)
    gain = estimate_gain(left, right, lag)
    return lag, gain


# ---------------------------------------------------------------------------
# 2./3./4./5. Streaming detector + patcher
# ---------------------------------------------------------------------------

def _robust_stats(x: np.ndarray):
    med = float(np.median(x))
    mad = 1.4826 * float(np.median(np.abs(x - med)))
    return med, max(mad, 1e-12)


def _envelope(x: np.ndarray, win: int = 9) -> np.ndarray:
    kernel = np.ones(win) / win
    return np.convolve(np.abs(x), kernel, mode="same")


def _regions_above(mask: np.ndarray):
    """Return list of [start, end) runs where mask is True."""
    runs = []
    in_run = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_run:
            in_run, start = True, i
        elif not v and in_run:
            runs.append([start, i])
            in_run = False
    if in_run:
        runs.append([start, len(mask)])
    return runs


class StereoClickRestorer:
    """Causal, block-based stereo click remover for aligned mono sources.

    Usage:
        r = StereoClickRestorer(fs, lag, gain)
        for lb, rb in blocks:
            lo, ro = r.process(lb, rb)   # may be shorter while buffering
        lo, ro = r.flush()

    `r.patches` collects one dict per repair: absolute sample range
    [start, end), corrupted channel, and confidence ratio.
    """

    def __init__(self, fs: int, lag: int = 0, gain: float = 1.0,
                 look: int = 256, fade: int = 16,
                 k_hi: float = 8.0, k_lo: float = 4.0,
                 merge_ms: float = 2.0, max_ms: float = 6.0,
                 min_conf: float = 4.0):
        self.fs = fs
        self.lag = int(lag)
        self.gain = float(gain)
        self.look = int(look)
        self.fade = int(fade)
        self.k_hi = k_hi
        self.k_lo = k_lo
        self.merge_gap = int(merge_ms * fs / 1000)
        self.max_len = int(max_ms * fs / 1000)
        self.min_conf = min_conf

        n = abs(self.lag)
        # lag > 0: right lags left -> delay left; lag < 0: delay right.
        self._fifo_l = np.zeros(n if self.lag > 0 else 0)
        self._fifo_r = np.zeros(n if self.lag < 0 else 0)
        self._buf_l = np.zeros(0)
        self._buf_r = np.zeros(0)
        self.emitted = 0
        self.patches: list[dict] = []

    # -- alignment ---------------------------------------------------------
    def _align(self, left: np.ndarray, right: np.ndarray):
        right = self.gain * right
        left = self._delay(left, "_fifo_l")
        right = self._delay(right, "_fifo_r")
        return left, right

    def _delay(self, x: np.ndarray, fifo_name: str):
        fifo = getattr(self, fifo_name)
        if fifo.size == 0:
            return x
        joined = np.concatenate([fifo, x])
        setattr(self, fifo_name, joined[-fifo.size:].copy())
        return joined[:len(x)]

    # -- main entry ---------------------------------------------------------
    def process(self, left: np.ndarray, right: np.ndarray):
        left, right = self._align(np.asarray(left, dtype=float),
                                  np.asarray(right, dtype=float))
        self._buf_l = np.concatenate([self._buf_l, left])
        self._buf_r = np.concatenate([self._buf_r, right])
        self._detect_and_patch()
        n_emit = max(0, len(self._buf_l) - self.look)
        return self._emit(n_emit)

    def flush(self):
        pad = np.zeros(self.look)
        self._buf_l = np.concatenate([self._buf_l, pad])
        self._buf_r = np.concatenate([self._buf_r, pad])
        self._detect_and_patch()
        return self._emit(len(self._buf_l) - self.look)

    def _emit(self, n: int):
        out_l, out_r = self._buf_l[:n], self._buf_r[:n]
        self._buf_l, self._buf_r = self._buf_l[n:], self._buf_r[n:]
        self.emitted += n
        return out_l.copy(), out_r.copy()

    # -- detection ----------------------------------------------------------
    def _detect_and_patch(self):
        diff = self._buf_l - self._buf_r          # common-mode music cancels
        env = _envelope(diff)
        med, sig = _robust_stats(env)
        hi = med + self.k_hi * sig
        lo = med + self.k_lo * sig

        runs = _regions_above(env > lo)
        # keep only runs that cross the high threshold (seed) somewhere
        runs = [[s, e] for s, e in runs if env[s:e].max() > hi]
        # merge runs separated by tiny gaps (one click, multiple peaks)
        merged = []
        for s, e in runs:
            if merged and s - merged[-1][1] < self.merge_gap:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        # an open run touching the buffer tail may still grow: defer it
        if merged and merged[-1][1] == len(env) and env[-1] > lo:
            merged.pop()

        for s, e in merged:
            if e - s > self.max_len:
                continue                      # too long for a click: music
            self._patch(s, e)

    # -- repair ---------------------------------------------------------------
    def _patch(self, s: int, e: int):
        # Which channel is corrupted?  Deviation-from-mid is symmetric by
        # construction, so instead measure how inconsistent each channel is
        # with its OWN context: bridge the gap by linear interpolation
        # between the samples just outside the region and integrate the
        # prediction error.  The clean channel follows its neighbourhood,
        # the contaminated one does not.
        err_l = self._interp_error(self._buf_l, s, e)
        err_r = self._interp_error(self._buf_r, s, e)
        bad_is_l = err_l >= err_r
        conf = max(err_l, err_r) / (min(err_l, err_r) + 1e-12)
        if conf < self.min_conf:
            return                            # both channels agree: leave it

        a = max(0, s - self.fade)
        b = min(len(self._buf_l), e + self.fade)
        n = b - a
        weight = np.ones(n)
        f = min(self.fade, n // 2)
        if f > 0:
            ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, f))
            weight[:f] = ramp
            weight[-f:] = ramp[::-1]

        dst = self._buf_l if bad_is_l else self._buf_r
        src = self._buf_r if bad_is_l else self._buf_l
        dst[a:b] = (1.0 - weight) * dst[a:b] + weight * src[a:b]

        self.patches.append({
            "start": self.emitted + s,
            "end": self.emitted + e,
            "channel": "L" if bad_is_l else "R",
            "confidence": conf,
        })

    @staticmethod
    def _interp_error(x: np.ndarray, s: int, e: int) -> float:
        n = e - s
        a = x[s - 1] if s > 0 else x[min(e, len(x) - 1)]
        b = x[e] if e < len(x) else x[max(s - 1, 0)]
        interp = a + (b - a) * (np.arange(1, n + 1) / (n + 1))
        return float(np.sum((x[s:e] - interp) ** 2))
