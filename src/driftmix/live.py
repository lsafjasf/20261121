"""Live PulseAudio/PipeWire monitoring backend.

PipeWire ships a PulseAudio-compatible client, so the same ``parec`` /
``paplay`` tools drive both stacks.  The master device is the playback
sound card: its callback clock defines block timing and the master input
is never re-clocked.  The slave device is captured independently, pushed
through the same drift buffer + async resampler as the simulation.

The two capture streams are external processes writing raw little-endian
f32 frames into FIFOs; this module only does the mixing math.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

from .engine import MonitorEngine
from .scenario import SAMPLE_RATE


class RawFifoReader:
    """Blocking reader that always returns exactly ``n`` f32 frames."""

    def __init__(self, path: str, channels: int) -> None:
        self.path = path
        self.channels = channels
        self._fh = open(path, "rb")

    def read_block(self, frames: int) -> np.ndarray:
        needed = frames * self.channels * 4
        chunks = []
        got = 0
        while got < needed:
            chunk = self._fh.read(needed - got)
            if not chunk:
                raise EOFError(f"capture FIFO {self.path} closed")
            chunks.append(chunk)
            got += len(chunk)
        raw = np.frombuffer(b"".join(chunks), dtype="<f4")
        # Downmix to mono by averaging channels.
        return raw.reshape(frames, self.channels).mean(axis=1)


def start_capture(
    fifo_path: str, device: str | None, rate: int, channels: int
) -> subprocess.Popen:
    cmd = [
        "parec",
        "--format=float32le",
        "--rate",
        str(rate),
        "--channels",
        str(channels),
        "--raw",
    ]
    if device:
        cmd += ["--device", device]
    cmd.append(fifo_path)
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def run_live(args: argparse.Namespace) -> int:
    block = args.block
    target = int(args.target_seconds * SAMPLE_RATE)
    capacity = int(args.buffer_seconds * SAMPLE_RATE)
    engine = MonitorEngine(
        capacity, target, args.kp, args.ki, args.kind
    )

    tmp = tempfile.mkdtemp(prefix="driftmix-")
    master_fifo = os.path.join(tmp, "master.fifo")
    slave_fifo = os.path.join(tmp, "slave.fifo")
    os.mkfifo(master_fifo)
    os.mkfifo(slave_fifo)

    procs = [
        start_capture(master_fifo, args.master_device, SAMPLE_RATE, args.channels),
        start_capture(slave_fifo, args.slave_device, SAMPLE_RATE, args.channels),
    ]
    play = subprocess.Popen(
        [
            "paplay",
            "--format=float32le",
            "--rate",
            str(SAMPLE_RATE),
            "--channels=1",
            "--raw",
        ],
        stdin=subprocess.PIPE,
    )

    master = RawFifoReader(master_fifo, args.channels)
    slave = RawFifoReader(slave_fifo, args.channels)
    blocks = 0
    try:
        while True:
            master_block = master.read_block(block)
            # The slave runs on its own clock; the drift buffer absorbs the
            # per-block surplus/deficit between the two devices.
            slave_block = slave.read_block(block)
            mixed, _ = engine.process_block(master_block, slave_block)
            play.stdin.write(mixed.astype("<f4").tobytes())
            blocks += 1
            if args.max_blocks and blocks >= args.max_blocks:
                break
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        for proc in procs:
            proc.terminate()
        play.terminate()
    print(
        f"processed {blocks} master blocks; "
        f"overruns={engine.buffer.overruns} "
        f"underruns={engine.resampler.underrun_samples}"
    )
    return 0
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Live dual-input drift mixer")
    p.add_argument("--master-device", default=None)
    p.add_argument("--slave-device", default=None)
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--block", type=int, default=480)
    p.add_argument("--buffer-seconds", type=float, default=0.1)
    p.add_argument("--target-seconds", type=float, default=0.05)
    p.add_argument("--kp", type=float, default=1e-5)
    p.add_argument("--ki", type=float, default=1e-6)
    p.add_argument("--kind", choices=["linear", "cubic"], default="linear")
    p.add_argument("--max-blocks", type=int, default=0)
    return p


def main() -> int:
    return run_live(build_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
