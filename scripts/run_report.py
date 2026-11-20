"""Run the fast-then-slow drift scenario and print sample-count evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from driftmix.analysis import analyze  # noqa: E402
from driftmix.scenario import (  # noqa: E402
    SAMPLE_RATE,
    ScenarioConfig,
    run_adaptive,
    run_naive,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--fast-seconds", type=float, default=20.0)
    ap.add_argument("--drift-ppm", type=float, default=5000.0)
    ap.add_argument("--kind", choices=["linear", "cubic"], default="linear")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cfg = ScenarioConfig(
        seconds=args.seconds,
        fast_seconds=args.fast_seconds,
        drift_ppm=args.drift_ppm,
        resampler_kind=args.kind,
    )
    adaptive = analyze(run_adaptive(cfg))
    naive = analyze(run_naive(cfg))

    if args.json:
        payload = {
            "config": cfg.__dict__,
            "adaptive": adaptive.__dict__,
            "naive": naive.__dict__,
        }
        payload["adaptive"]["fault_free"] = adaptive.fault_free
        payload["naive"]["fault_free"] = naive.fault_free
        print(
            json.dumps(
                payload,
                indent=2,
            )
        )
        return 0

    block_ms = cfg.block / SAMPLE_RATE * 1000
    print("=" * 68)
    print("Dual-clock drift-mix report")
    print(
        f"{cfg.seconds:.0f}s run, {cfg.drift_ppm:.0f} ppm drift "
        f"(fast {cfg.fast_seconds:.0f}s, then slow), "
        f"{block_ms:.1f} ms master blocks, {args.kind} resampler"
    )
    print("=" * 68)
    for rep in (adaptive, naive):
        print(f"\n[{rep.name}]")
        print(f"  master blocks                 : {rep.total_blocks}")
        print(f"  every block exactly {cfg.block} frames     : {rep.block_size_ok}")
        print(f"  buffer overruns (dropped)     : {rep.buffer_overruns}")
        print(f"  resampler underrun samples    : {rep.resampler_underruns}")
        print(f"  all-zero (silent) blocks      : {rep.silent_blocks}")
        print(f"  lowest block RMS              : {rep.lowest_block_rms:.6f}")
        print(f"  sample accounting residual    : {rep.accounting_residual}")
        print(f"  fractional slack              : {rep.fractional_slack:.6f}")
        print(f"  max buffer deviation (frames) : {rep.max_fill_deviation}")
        print(f"  slave interpolation RMS error : {rep.slave_rms_error:.6e}")
        print(f"  FAULT FREE                    : {rep.fault_free}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
