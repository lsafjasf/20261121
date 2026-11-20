import numpy as np

from driftmix.ringbuffer import DriftBuffer


def test_push_pop_fifo_order():
    buf = DriftBuffer(8)
    buf.push(np.arange(5.0))
    assert np.array_equal(buf.pop(5), np.arange(5.0))
    assert buf.fill == 0


def test_wraparound_preserves_order():
    buf = DriftBuffer(8)
    for start in range(0, 40, 4):
        buf.push(np.arange(start, start + 4.0))
        out = buf.pop(4)
        assert np.array_equal(out, np.arange(start, start + 4.0))


def test_overrun_is_counted_not_silent():
    buf = DriftBuffer(4)
    stored = buf.push(np.arange(10.0))
    assert stored == 4
    assert buf.overruns == 6
    assert np.array_equal(buf.pop(4), np.arange(4.0))


def test_underrun_is_counted_and_zero_padded():
    buf = DriftBuffer(8)
    buf.push(np.array([1.0, 2.0]))
    out = buf.pop(5)
    assert buf.underruns == 3
    assert np.array_equal(out, [1.0, 2.0, 0.0, 0.0, 0.0])
