import os
import threading

import numpy as np

from driftmix.engine import MonitorEngine
from driftmix.live import RawFifoReader


def _writer(path, frames, ready, closed):
    ready.set()  # reader may now open the FIFO; open() blocks until it does
    with open(path, "wb") as fh:
        fh.write(frames.astype("<f4").tobytes())
    closed.set()


def test_fifo_reader_returns_exact_frames(tmp_path):
    fifo = str(tmp_path / "cap.fifo")
    os.mkfifo(fifo)
    wave = np.sin(np.arange(2000) * 0.1).astype("<f4")
    ready = threading.Event()
    closed = threading.Event()
    t = threading.Thread(target=_writer, args=(fifo, wave, ready, closed))
    t.start()
    ready.wait()
    reader = RawFifoReader(fifo, 1)
    first = reader.read_block(480)
    second = reader.read_block(480)
    t.join()
    assert closed.is_set()
    assert first.shape == (480,)
    assert second.shape == (480,)
    assert np.allclose(first, wave[:480], atol=1e-6)


def test_live_engine_mix_path_is_fault_free():
    # Drive the exact engine the live backend uses with synthetic blocks.
    engine = MonitorEngine(4800, 2400, 1e-5, 1e-6, "linear")
    master = np.sin(np.arange(480) * 0.05)
    engine.buffer.push(np.zeros(2400))
    mixed, slave = engine.process_block(master, np.zeros(482))
    assert mixed.shape == (480,)
    assert engine.buffer.overruns == 0
    assert engine.resampler.underrun_samples == 0
