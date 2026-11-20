import numpy as np

from driftmix.analysis import (
    accounting_residual,
    analyze,
    block_rms,
    zero_blocks,
)
from driftmix.scenario import (
    SAMPLE_RATE,
    ScenarioConfig,
    run_adaptive,
    run_naive,
    slave_schedule,
)


def _cfg(**over):
    base = dict(seconds=60.0, fast_seconds=20.0, drift_ppm=5000.0)
    base.update(over)
    return ScenarioConfig(**base)


def test_master_block_size_never_changes():
    result = run_adaptive(_cfg())
    assert set(result.block_frames) == {480}
    assert len(result.block_frames) == result.cfg.n_blocks


def test_adaptive_run_is_completely_fault_free():
    result = run_adaptive(_cfg())
    report = analyze(result)
    assert report.fault_free
    assert report.buffer_overruns == 0
    assert report.resampler_underruns == 0
    assert report.silent_blocks == 0
    assert report.accounting_residual == 0


def test_no_periodic_silence_in_adaptive_output():
    result = run_adaptive(_cfg())
    rms = block_rms(result.slave_out, 480)
    assert zero_blocks(result.slave_out, 480) == 0
    # Every 10 ms block carries healthy signal energy.
    assert np.min(rms) > 0.5


def test_sample_accounting_closes_for_both_engines():
    for runner in (run_adaptive, run_naive):
        result = runner(_cfg())
        assert accounting_residual(result) == 0


def test_adaptive_buffer_stays_inside_capacity():
    result = run_adaptive(_cfg())
    capacity = int(result.cfg.buffer_seconds * SAMPLE_RATE)
    assert result.fill.min() > 0
    assert result.fill.max() < capacity


def test_naive_baseline_demonstrates_failure_modes():
    result = run_naive(_cfg())
    report = analyze(result)
    assert not report.fault_free
    assert report.buffer_overruns > 0
    assert report.silent_blocks > 0


def test_drift_profile_runs_fast_then_slow():
    cfg = _cfg()
    schedule = slave_schedule(cfg)
    fast = schedule[: cfg.n_blocks // 3]
    slow = schedule[-cfg.n_blocks // 3:]
    assert fast.mean() > 480
    assert slow.mean() < 480


def test_resampled_slave_matches_ideal_read_positions():
    from driftmix.analysis import ideal_slave_on_master_grid

    result = run_adaptive(_cfg())
    ideal = ideal_slave_on_master_grid(result, result.slave_positions)
    rms_error = np.sqrt(np.mean((result.slave_out - ideal) ** 2))
    assert rms_error < 1e-3


def test_master_path_passes_through_unshifted():
    result = run_adaptive(_cfg())
    direct = 0.5 * result.master_out + 0.5 * result.slave_out
    assert np.allclose(direct, result.mixed, atol=1e-12)
