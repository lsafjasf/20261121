"""Bounded FIFO that never silently discards or fabricates samples.

Every capacity violation is counted explicitly so tests can prove that the
adaptive mixer never hits one.  A real-time process should treat a non-zero
counter as an engineering fault (oversized drift / undersized buffer), not
as normal operation.
"""

from __future__ import annotations

import numpy as np


class DriftBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._buf = np.zeros(self.capacity, dtype=np.float64)
        self._head = 0  # index of oldest sample
        self._size = 0
        self.overruns = 0  # pushes rejected because the buffer was full
        self.underruns = 0  # pops served with fewer samples than requested

    def __len__(self) -> int:
        return self._size

    @property
    def fill(self) -> int:
        return self._size

    def clear_counters(self) -> None:
        self.overruns = 0
        self.underruns = 0

    def push(self, samples: np.ndarray) -> int:
        """Append samples; returns the number actually stored."""
        samples = np.ascontiguousarray(samples, dtype=np.float64)
        free = self.capacity - self._size
        take = min(free, samples.shape[0])
        if take < samples.shape[0]:
            self.overruns += int(samples.shape[0] - take)
        if take == 0:
            return 0
        tail = (self._head + self._size) % self.capacity
        first = min(take, self.capacity - tail)
        self._buf[tail:tail + first] = samples[:first]
        if take > first:
            self._buf[:take - first] = samples[first:take]
        self._size += take
        return take

    def pop(self, count: int) -> np.ndarray:
        """Remove up to ``count`` samples; short reads bump ``underruns``."""
        count = int(count)
        out = np.zeros(count, dtype=np.float64)
        take = min(count, self._size)
        if take < count:
            self.underruns += int(count - take)
        if take:
            first = min(take, self.capacity - self._head)
            out[:first] = self._buf[self._head:self._head + first]
            if take > first:
                out[first:take] = self._buf[:take - first]
            self._head = (self._head + take) % self.capacity
            self._size -= take
        return out

    def peek(self, count: int) -> np.ndarray:
        """Read without consuming; zero-padded on underrun."""
        count = int(count)
        out = np.zeros(count, dtype=np.float64)
        take = min(count, self._size)
        first = min(take, self.capacity - self._head)
        out[:first] = self._buf[self._head:self._head + first]
        if take > first:
            out[first:take] = self._buf[:take - first]
        return out
