"""Streaming asynchronous resampler with a continuously variable ratio.

The ratio (slave samples consumed per master sample) can change on every
output block.  Fractional input positions are tracked globally, so ratio
changes are sample-continuous: there is never a discontinuity, duplicated
sample, or gap in the reconstructed waveform.  Missing input is zero-filled
and counted as an underrun instead of being silently interpolated away.

Internals use a fixed ring buffer; all work per block is O(block size).
"""

from __future__ import annotations

import math

import numpy as np


class AsyncResampler:
    def __init__(self, kind: str = "linear", capacity: int = 1 << 16) -> None:
        if kind not in ("linear", "cubic"):
            raise ValueError("kind must be 'linear' or 'cubic'")
        self.kind = kind
        self.capacity = int(capacity)
        self._ring = np.zeros(self.capacity, dtype=np.float64)
        self._head = 0  # ring index of global sample 0
        self._count = 0  # samples currently queued
        self._base = 0  # global index of the oldest queued sample
        self._pos = 0.0  # global fractional position of next output
        self.samples_in = 0
        self.samples_out = 0
        self.underrun_samples = 0
        self._last_positions = np.zeros(0, dtype=np.float64)

    def reset(self) -> None:
        self._ring.fill(0.0)
        self._head = 0
        self._count = 0
        self._base = 0
        self._pos = 0.0
        self.samples_in = 0
        self.samples_out = 0
        self.underrun_samples = 0
        self._last_positions = np.zeros(0, dtype=np.float64)

    @property
    def position(self) -> float:
        return self._pos

    @property
    def queued(self) -> int:
        return self._count

    @property
    def last_positions(self) -> np.ndarray:
        return self._last_positions

    @property
    def left_margin(self) -> int:
        """Extra queued samples retained to the left of the read position."""
        return 1 if self.kind == "cubic" else 0

    def push(self, samples: np.ndarray) -> None:
        samples = np.ascontiguousarray(samples, dtype=np.float64)
        take = min(samples.shape[0], self.capacity - self._count)
        if take == 0:
            return
        tail = (self._head + self._count) % self.capacity
        first = min(take, self.capacity - tail)
        self._ring[tail:tail + first] = samples[:first]
        if take > first:
            self._ring[:take - first] = samples[first:take]
        self._count += take
        self.samples_in += take

    def _gather(self, rel: np.ndarray, offset: int) -> np.ndarray:
        # Clamp-to-edge only for taps that fall outside the queued window;
        # interior taps index real samples and are never altered.
        safe = np.clip(rel + offset, 0, max(self._count - 1, 0))
        idx = (self._head + safe) % self.capacity
        return self._ring[idx]

    def read(self, count: int, ratio: float) -> np.ndarray:
        count = int(count)
        out = np.zeros(count, dtype=np.float64)
        if count == 0:
            return out
        positions = self._pos + np.arange(count, dtype=np.float64) * ratio
        self._last_positions = positions.copy()
        floors = np.floor(positions).astype(np.int64)
        frac = positions - floors
        rel = floors - self._base
        # A frame is only valid when its central sample is queued.  Taps
        # just outside the edges (startup/held lookahead) are clamped.
        valid = rel < self._count
        missing = count - int(np.count_nonzero(valid))

        if self.kind == "linear":
            r = rel
            y0 = self._gather(r, 0)
            y1 = self._gather(r, 1)
            interp = y0 + frac * (y1 - y0)
        else:
            r = rel
            t = frac
            t2 = t * t
            ym1 = self._gather(r, -1)
            y0 = self._gather(r, 0)
            y1 = self._gather(r, 1)
            y2 = self._gather(r, 2)
            interp = 0.5 * (
                2.0 * y0
                + (-ym1 + y2) * t
                + (2.0 * ym1 - 5.0 * y0 + 4.0 * y1 - y2) * t2
                + (-ym1 + 3.0 * y0 - 3.0 * y1 + y2) * t2 * t
            )

        out[valid] = interp[valid]  # invalid frames stay exactly zero
        self.underrun_samples += missing
        self.samples_out += count
        self._pos = positions[-1] + ratio

        # Keep every cell the next read can reference.  Linear needs the
        # cell at floor(next_position) as its left anchor; cubic needs one
        # extra sample on the left.  The accounting identity stays exact
        # because the retained margin is counted inside ``queued``.
        left_margin = 1 if self.kind == "cubic" else 0
        keep_from_global = int(math.floor(self._pos)) - left_margin
        drop = keep_from_global - self._base
        if drop > 0:
            self._head = (self._head + drop) % self.capacity
            self._count -= drop
            self._base += drop
        return out
