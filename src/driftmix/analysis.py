"""Evidence collection: frame accounting, silence search, fidelity checks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .scenario import SAMPLE_RATE, SLAVE_TONE_HZ, RunResult


@dataclass
class Report:
    name: str
    total_blocks: int
    block_size_ok: bool
    unique_block_sizes: tuple
    buffer_overruns: int
    resampler_underruns: int
    silent_blocks: int
    lowest_block_rms: float
    accounting_residual: int
    fractional_slack: float
    slave_rms_error: float
    switch_fill: int
    max_fill_deviation: int

    @property
    def fault_free(self) -> bool:
        return (
            self.block_size_ok
            and self.buffer_overruns == 0
            and self.resampler_underruns == 0
            and self.silent_blocks == 0
            and self.accounting_residual == 0
        )


def block_rms(signal: np.ndarray, block: int) -> np.ndarray:
    n = signal.shape[0] // block
    trimmed = signal[: n * block].reshape(n, block)
    return np.sqrt(np.mean(trimmed * trimmed, axis=1))


def zero_blocks(signal: np.ndarray, block: int) -> int:
    """Count blocks that are literally all-zero (a dropped/silent block)."""
    n = signal.shape[0] // block
    trimmed = signal[: n * block].reshape(n, block)
    return int(np.count_nonzero(np.all(trimmed == 0.0, axis=1)))


def ideal_slave_on_master_grid(
    result: RunResult, sample_positions: np.ndarray
) -> np.ndarray:
    """Ideal band-limited samples of the slave tone at the read positions.

    ``sample_positions`` are the resampler's global fractional slave
    indices for every output frame, so this isolates interpolation error
    from the (intentionally adaptive) clock-ratio tracking.
    """
    phase = 2.0 * np.pi * SLAVE_TONE_HZ * sample_positions / SAMPLE_RATE
    return np.sin(phase)


def fractional_slack(result: RunResult) -> float:
    """Sub-sample part of the resampler read position (must be in [0, 1))."""
    return result.resampler_position_end - math.floor(
        result.resampler_position_end
    )


def accounting_residual(result: RunResult) -> int:
    """Integer identity over every slave sample the device produced.

    produced == floor(consumed_position) + effective_queue
                + buffer.fill + overruns,
    where effective_queue excludes the resampler's left interpolation
    margin.  A non-zero result proves a sample was lost or invented.
    """
    consumed = math.floor(result.resampler_position_end)
    effective_queue = (
        result.resampler_queued_end - result.resampler_left_margin
    )
    accounted = (
        consumed
        + effective_queue
        + result.buffer_fill_end
        + result.buffer_overruns
    )
    return int(result.slave_produced - accounted)


def analyze(result: RunResult) -> Report:
    cfg = result.cfg
    block = cfg.block
    sizes = np.asarray(result.block_frames)
    rms = block_rms(result.slave_out, block)

    ideal = ideal_slave_on_master_grid(result, result.slave_positions)
    rms_err = float(np.sqrt(np.mean((result.slave_out - ideal) ** 2)))

    target = int(cfg.target_seconds * SAMPLE_RATE)
    switch_block = int(cfg.fast_seconds * SAMPLE_RATE / block)

    return Report(
        name=result.engine_name,
        total_blocks=int(sizes.shape[0]),
        block_size_ok=bool(np.all(sizes == block)),
        unique_block_sizes=tuple(int(s) for s in np.unique(sizes)),
        buffer_overruns=result.buffer_overruns,
        resampler_underruns=result.resampler_underruns,
        silent_blocks=zero_blocks(result.slave_out, block),
        lowest_block_rms=float(np.min(rms)),
        accounting_residual=accounting_residual(result),
        fractional_slack=fractional_slack(result),
        slave_rms_error=rms_err,
        switch_fill=int(result.fill[switch_block - 1]),
        max_fill_deviation=int(np.max(np.abs(result.fill - target))),
    )
