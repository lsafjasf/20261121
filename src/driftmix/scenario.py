"""Deterministic simulation of two clock-independent capture devices.

The master device defines block timing (exactly ``block`` master frames per
callback, no exceptions).  The slave device advances on its own fractional
clock, whose rate relative to the master follows a piecewise drift profile:
fast first, then slow.  Both devices render sine waves indexed by their own
sample counters, exactly as independent ADCs would do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .engine import MonitorEngine, NaiveEngine

SAMPLE_RATE = 48_000
BLOCK = 480  # 10 ms master callback
MASTER_TONE_HZ = 300.0
SLAVE_TONE_HZ = 700.0


@dataclass(frozen=True)
class ScenarioConfig:
    seconds: float = 60.0
    block: int = BLOCK
    drift_ppm: float = 5_000.0  # +/- 0.5 %
    fast_seconds: float = 20.0
    buffer_seconds: float = 0.1
    target_seconds: float = 0.05
    kp: float = 1.0e-5
    ki: float = 1.0e-6
    resampler_kind: str = "linear"

    @property
    def n_blocks(self) -> int:
        return int(round(self.seconds * SAMPLE_RATE / self.block))


def slave_drift(t_master: float, cfg: ScenarioConfig) -> float:
    """Fractional rate offset of the slave clock at master time t."""
    d = cfg.drift_ppm * 1e-6
    return d if t_master < cfg.fast_seconds else -d


def master_signal(n: int) -> np.ndarray:
    idx = np.arange(n, dtype=np.float64)
    return np.sin(2.0 * np.pi * MASTER_TONE_HZ * idx / SAMPLE_RATE)


def slave_signal(first: int, count: int) -> np.ndarray:
    idx = np.arange(first, first + count, dtype=np.float64)
    return np.sin(2.0 * np.pi * SLAVE_TONE_HZ * idx / SAMPLE_RATE)


@dataclass
class RunResult:
    cfg: ScenarioConfig
    engine_name: str
    master_out: np.ndarray
    slave_out: np.ndarray
    mixed: np.ndarray
    slave_produced: int
    block_frames: list
    fill: np.ndarray
    ratios: np.ndarray
    slave_positions: np.ndarray
    buffer_overruns: int
    resampler_underruns: int
    buffer_fill_end: int = 0
    resampler_queued_end: int = 0
    resampler_position_end: float = 0.0
    resampler_left_margin: int = 0


def slave_schedule(cfg: ScenarioConfig) -> np.ndarray:
    """Slave samples delivered per master block (independent clock).

    A floating phase accumulator integrates the piecewise-constant slave
    rate; the integer part is emitted each block and the fraction carries
    over, so the schedule is unbiased over the whole run.
    """
    t_mid = (np.arange(cfg.n_blocks) + 0.5) * cfg.block / SAMPLE_RATE
    drift = np.where(t_mid < cfg.fast_seconds, 1.0, -1.0)
    rate = 1.0 + drift * cfg.drift_ppm * 1e-6
    phase = np.cumsum(cfg.block * rate)
    emitted = np.floor(phase).astype(np.int64)
    schedule = np.empty_like(emitted)
    schedule[0] = emitted[0]
    schedule[1:] = np.diff(emitted)
    return schedule


def _run(engine, cfg: ScenarioConfig, name: str) -> RunResult:
    target = int(cfg.target_seconds * SAMPLE_RATE)
    schedule = slave_schedule(cfg)
    total_slave = target + int(schedule.sum())

    # Render every slave ADC sample up front, indexed by its own counter.
    slave_all = slave_signal(0, total_slave)
    master_all = master_signal(cfg.n_blocks * cfg.block)

    # Prefill to the nominal target, as a real monitor does at startup.
    engine.buffer.push(slave_all[:target])
    slave_next = target

    mixed_parts: list[np.ndarray] = []
    slave_parts: list[np.ndarray] = []
    block_frames: list[int] = []

    for b in range(cfg.n_blocks):
        n_slave = int(schedule[b])
        slave_block = slave_all[slave_next:slave_next + n_slave]
        slave_next += n_slave
        master_block = master_all[b * cfg.block:(b + 1) * cfg.block]

        mixed, slave_out = engine.process_block(master_block, slave_block)
        mixed_parts.append(mixed)
        slave_parts.append(slave_out)
        block_frames.append(master_block.shape[0])

    return RunResult(
        cfg=cfg,
        engine_name=name,
        master_out=master_all,
        slave_out=np.concatenate(slave_parts),
        mixed=np.concatenate(mixed_parts),
        slave_produced=total_slave,
        block_frames=block_frames,
        fill=np.asarray(engine.metrics.fill_history),
        ratios=np.asarray(engine.metrics.ratio_history),
        slave_positions=np.concatenate(engine.metrics.position_history),
        buffer_overruns=engine.buffer.overruns,
        resampler_underruns=engine.resampler.underrun_samples,
        buffer_fill_end=engine.buffer.fill,
        resampler_queued_end=engine.resampler.queued,
        resampler_position_end=engine.resampler.position,
        resampler_left_margin=engine.resampler.left_margin,
    )


def run_adaptive(cfg: ScenarioConfig | None = None) -> RunResult:
    cfg = cfg or ScenarioConfig()
    capacity = int(cfg.buffer_seconds * SAMPLE_RATE)
    target = int(cfg.target_seconds * SAMPLE_RATE)
    engine = MonitorEngine(
        capacity, target, cfg.kp, cfg.ki, cfg.resampler_kind
    )
    return _run(engine, cfg, "adaptive")


def run_naive(cfg: ScenarioConfig | None = None) -> RunResult:
    cfg = cfg or ScenarioConfig()
    capacity = int(cfg.buffer_seconds * SAMPLE_RATE)
    engine = NaiveEngine(capacity)
    return _run(engine, cfg, "naive")
