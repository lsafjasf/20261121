"""Streaming mono packet-loss concealment (PLC) engine.

Frame-based, zero extra algorithmic latency: feed exactly one frame per
`process()` call (PCM samples, or None when the packet is lost) and get
exactly one frame of playable output back.

Strategy
--------
- Good frames pass through bit-exact and refresh the history buffer.
- On loss, the pitch period of the recent signal is estimated with a
  normalized autocorrelation search.
- Short gaps (<= short_gap_ms): periodic continuation, y[n] = y[n - T],
  which is click-free because the read pointer always stays one period
  behind the write pointer.
- Long gaps: the continuation keeps running but its gain decays
  exponentially (time constant decay_tau_ms) so energy fades smoothly
  to silence instead of droning forever.
- Packet recovery: the concealment tail is crossfaded into the real
  frame with a raised-cosine ramp (fade_ms) to avoid a hard cut.

Pure standard library; works on float samples in [-1, 1].
"""

import math


class StreamPLC:
    def __init__(self, sample_rate=16000, frame_ms=20.0,
                 min_pitch_ms=2.5, max_pitch_ms=15.0,
                 short_gap_ms=60.0, decay_tau_ms=40.0, fade_ms=10.0):
        self.fs = sample_rate
        self.frame_size = int(round(sample_rate * frame_ms / 1000.0))
        self.min_pitch = max(1, int(round(sample_rate * min_pitch_ms / 1000.0)))
        self.max_pitch = max(self.min_pitch + 1,
                             int(round(sample_rate * max_pitch_ms / 1000.0)))
        self.short_gap_frames = max(1, int(round(short_gap_ms / frame_ms)))
        self.fade_len = max(1, int(round(sample_rate * fade_ms / 1000.0)))
        self.decay_step = math.exp(-1000.0 / (decay_tau_ms * sample_rate))

        # History must cover the pitch search window plus one frame.
        self.hist_len = 2 * self.max_pitch + self.frame_size + self.fade_len
        self.hist = [0.0] * self.hist_len
        self.loss_run = 0            # consecutive lost frames so far
        self.pitch = self.min_pitch  # last pitch estimate, in samples
        self.gain = 1.0              # current concealment gain

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def process(self, frame, lost):
        """Consume one frame; return the output frame (list of floats).

        `frame` is a sequence of self.frame_size floats, or None when the
        packet was lost (`lost` must then be True).
        """
        if lost:
            return self._lost_frame()
        if frame is None or len(frame) != self.frame_size:
            raise ValueError("good frame must have %d samples" % self.frame_size)
        if self.loss_run > 0:
            return self._recover(frame)
        self._push_history(frame)
        return list(frame)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _push_history(self, samples):
        self.hist = (self.hist + list(samples))[-self.hist_len:]

    def _estimate_pitch(self):
        """Normalized autocorrelation over the recent history."""
        win = 2 * self.max_pitch
        h = self.hist[-win:]
        n = len(h)
        e0 = sum(x * x for x in h)
        if e0 < 1e-10:
            return self.pitch
        best_lag, best_corr = self.pitch, -1.0
        for lag in range(self.min_pitch, self.max_pitch + 1):
            num = 0.0
            den = 0.0
            for i in range(lag, n):
                b = h[i - lag]
                num += h[i] * b
                den += b * b
            if den <= 0.0:
                continue
            corr = num / math.sqrt(den * e0)
            if corr > best_corr:
                best_corr = corr
                best_lag = lag
        return best_lag

    def _synthesize(self, count, decaying):
        """Generate `count` concealment samples from current history.

        Does not mutate state; returns (samples, end_gain).
        """
        buf = self.hist
        base = len(buf)
        out = []
        g = self.gain
        for i in range(count):
            if decaying:
                g *= self.decay_step
                if g < 1e-3:
                    g = 0.0
            idx = base + i - self.pitch
            ref = buf[idx] if idx < base else out[idx - base]
            out.append(g * ref)
        return out, g

    def _lost_frame(self):
        if self.loss_run == 0:
            self.pitch = self._estimate_pitch()
            self.gain = 1.0
        self.loss_run += 1
        decaying = self.loss_run > self.short_gap_frames
        out, self.gain = self._synthesize(self.frame_size, decaying)
        self._push_history(out)
        return out

    def _recover(self, frame):
        fade = min(self.fade_len, self.frame_size)
        decaying = self.loss_run > self.short_gap_frames
        virt, _ = self._synthesize(fade, decaying)
        out = list(frame)
        for i in range(fade):
            w = 0.5 - 0.5 * math.cos(math.pi * (i + 1) / (fade + 1))
            out[i] = (1.0 - w) * virt[i] + w * frame[i]
        self.loss_run = 0
        self.gain = 1.0
        self._push_history(out)
        return out
