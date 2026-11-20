"""Mixing engines.

``MonitorEngine`` keeps the master device as the timing reference: master
frames pass through untouched and only the slave stream is re-clocked.
``NaiveEngine`` is the deliberately naive baseline (fixed 1:1 ratio) used to
show what happens when independent clocks are assumed to run at the same rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .controller import FillController
from .resampler import AsyncResampler
from .ringbuffer import DriftBuffer


@dataclass
class EngineMetrics:
    fill_history: list[int] = field(default_factory=list)
    ratio_history: list[float] = field(default_factory=list)
    position_history: list = field(default_factory=list)
    master_frames_in: int = 0
    slave_frames_in: int = 0
    frames_out: int = 0


class MonitorEngine:
    """Slave drift is absorbed by buffer occupancy + async resampling."""

    def __init__(
        self,
        buffer_capacity: int,
        target_fill: int,
        kp: float,
        ki: float,
        resampler_kind: str = "linear",
        master_gain: float = 0.5,
        slave_gain: float = 0.5,
    ) -> None:
        self.buffer = DriftBuffer(buffer_capacity)
        self.controller = FillController(target_fill, kp, ki)
        self.resampler = AsyncResampler(resampler_kind)
        self.master_gain = master_gain
        self.slave_gain = slave_gain
        self.metrics = EngineMetrics()

    @property
    def faults(self) -> int:
        return self.buffer.overruns + self.buffer.underruns + (
            self.resampler.underrun_samples
        )

    def process_block(
        self, master_block: np.ndarray, slave_block: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        master_block = np.asarray(master_block, dtype=np.float64)
        n = master_block.shape[0]

        stored = self.buffer.push(slave_block)
        ratio = self.controller.update(self.buffer.fill)
        # Keep a small lookahead inside the resampler so its fractional
        # read position never runs ahead of queued data.
        target_queued = int(np.ceil(n * ratio)) + 8
        take = max(0, target_queued - self.resampler.queued)
        self.resampler.push(self.buffer.pop(min(take, self.buffer.fill)))
        slave_out = self.resampler.read(n, ratio)
        mixed = (
            self.master_gain * master_block
            + self.slave_gain * slave_out
        )

        self.metrics.master_frames_in += n
        self.metrics.slave_frames_in += stored
        self.metrics.frames_out += n
        self.metrics.fill_history.append(self.buffer.fill)
        self.metrics.ratio_history.append(ratio)
        self.metrics.position_history.append(
            self.resampler.last_positions.copy()
        )
        return mixed, slave_out


class NaiveEngine:
    """Baseline that assumes both clocks are exactly 48 kHz."""

    def __init__(
        self,
        buffer_capacity: int,
        master_gain: float = 0.5,
        slave_gain: float = 0.5,
    ) -> None:
        self.buffer = DriftBuffer(buffer_capacity)
        self.resampler = AsyncResampler("linear")
        self.master_gain = master_gain
        self.slave_gain = slave_gain
        self.metrics = EngineMetrics()

    @property
    def faults(self) -> int:
        return self.buffer.overruns + self.resampler.underrun_samples

    def process_block(
        self, master_block: np.ndarray, slave_block: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        master_block = np.asarray(master_block, dtype=np.float64)
        n = master_block.shape[0]

        stored = self.buffer.push(slave_block)
        self.resampler.push(self.buffer.pop(min(n, self.buffer.fill)))
        slave_out = self.resampler.read(n, 1.0)
        mixed = (
            self.master_gain * master_block
            + self.slave_gain * slave_out
        )

        self.metrics.master_frames_in += n
        self.metrics.slave_frames_in += stored
        self.metrics.frames_out += n
        self.metrics.fill_history.append(self.buffer.fill)
        self.metrics.ratio_history.append(1.0)
        self.metrics.position_history.append(
            self.resampler.last_positions.copy()
        )
        return mixed, slave_out
