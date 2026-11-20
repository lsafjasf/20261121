"""Render buffer/ratio/energy evidence for the drift-mix scenario."""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from driftmix.analysis import block_rms  # noqa: E402
from driftmix.scenario import (  # noqa: E402
    SAMPLE_RATE,
    ScenarioConfig,
    run_adaptive,
    run_naive,
)

OUT = os.path.join(os.path.dirname(__file__), "..", "artifacts")


def main() -> int:
    cfg = ScenarioConfig()
    adaptive = run_adaptive(cfg)
    naive = run_naive(cfg)
    block_t = (np.arange(cfg.n_blocks) + 0.5) * cfg.block / SAMPLE_RATE
    target = int(cfg.target_seconds * SAMPLE_RATE)
    capacity = int(cfg.buffer_seconds * SAMPLE_RATE)

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)

    axes[0].plot(block_t, adaptive.fill, label="adaptive fill", lw=1.2)
    axes[0].plot(block_t, naive.fill, label="naive fill", lw=1.0, alpha=0.7)
    axes[0].axhline(target, color="k", ls=":", lw=1, label="target")
    axes[0].axhline(capacity, color="r", ls="--", lw=1, label="capacity")
    axes[0].axhline(0, color="r", ls="--", lw=1)
    axes[0].axvline(cfg.fast_seconds, color="grey", ls="-.", lw=1)
    axes[0].set_ylabel("slave buffer (frames)")
    axes[0].legend(loc="upper right", ncol=4, fontsize=8)
    axes[0].set_title("Buffer occupancy: adaptive stays bounded, naive fails")

    axes[1].plot(block_t, adaptive.ratios, lw=1.2, label="consume ratio")
    axes[1].axhline(1.0, color="k", ls=":", lw=1)
    axes[1].axvline(cfg.fast_seconds, color="grey", ls="-.", lw=1)
    axes[1].set_ylabel("resample ratio")
    axes[1].set_title("Adaptive consume ratio tracks the slave clock drift")

    adapt_rms = block_rms(adaptive.slave_out, cfg.block)
    naive_rms = block_rms(naive.slave_out, cfg.block)
    axes[2].plot(block_t, adapt_rms, lw=1.0, label="adaptive")
    axes[2].plot(block_t, naive_rms, lw=1.0, alpha=0.7, label="naive")
    axes[2].axvline(cfg.fast_seconds, color="grey", ls="-.", lw=1)
    axes[2].set_ylabel("block RMS")
    axes[2].set_title("Per-block energy: no periodic silence in adaptive path")
    axes[2].legend(loc="upper right", fontsize=8)

    naive_silent = naive_rms == 0.0
    axes[3].plot(block_t, adaptive.ratios * 0, lw=0)
    axes[3].fill_between(
        block_t, 0, naive_silent, where=naive_silent,
        color="red", alpha=0.5, label="naive silent blocks",
    )
    axes[3].axvline(cfg.fast_seconds, color="grey", ls="-.", lw=1)
    axes[3].set_ylabel("silence")
    axes[3].set_xlabel("master time (s)")
    axes[3].set_title("Naive baseline silence map (adaptive has none)")
    axes[3].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUT, "drift_mix_evidence.png")
    fig.savefig(path, dpi=130)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
