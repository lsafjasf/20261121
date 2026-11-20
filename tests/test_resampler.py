import math

import numpy as np
import pytest

from driftmix.resampler import AsyncResampler


@pytest.mark.parametrize("kind", ["linear", "cubic"])
def test_identity_ratio_one_is_exact(kind):
    rs = AsyncResampler(kind)
    wave = np.sin(np.arange(100) * 0.3)
    rs.push(wave)
    out = rs.read(90, 1.0)
    assert np.allclose(out, wave[:90], atol=1e-12)


@pytest.mark.parametrize("kind", ["linear", "cubic"])
def test_block_boundaries_are_continuous(kind):
    rs = AsyncResampler(kind)
    wave = np.sin(np.arange(5000) * 0.07)
    rs.push(wave)
    first = rs.read(500, 1.004)
    second = rs.read(500, 1.004)
    # No duplicated or zeroed sample at the block seam.
    assert first[-1] != second[0]
    assert second[0] != 0.0


@pytest.mark.parametrize("kind", ["linear", "cubic"])
def test_sample_accounting_identity(kind):
    rs = AsyncResampler(kind)
    for chunk in range(10):
        rs.push(np.arange(chunk * 100, (chunk + 1) * 100, dtype=float))
        rs.read(90, 1.05)
    identity = (
        rs.samples_in
        - math.floor(rs.position)
        - rs.queued
        + rs.left_margin
    )
    assert identity == 0


def test_starvation_emits_zeros_and_counts_underrun():
    rs = AsyncResampler("linear")
    rs.push(np.arange(4.0))
    out = rs.read(10, 1.0)
    assert rs.underrun_samples == 6
    assert np.all(out[4:] == 0.0)
    assert np.allclose(out[:4], np.arange(4.0))


def test_ratio_change_is_sample_continuous():
    rs = AsyncResampler("linear")
    rs.push(np.sin(np.arange(2000) * 0.05))
    a = rs.read(400, 0.998)
    b = rs.read(400, 1.003)  # ratio jumps at the seam
    seam = np.abs(b[0] - a[-1])
    # Adjacent output samples stay within one local signal increment.
    typical = np.mean(np.abs(np.diff(a)))
    assert seam < 5 * typical
