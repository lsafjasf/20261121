#!/usr/bin/env python3
"""Acceptance harness for delayed, noisy, unevenly buffered loopback runs."""

from __future__ import annotations

import argparse
import json
import math
import os
import pwd
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PERIOD = 2048
PROBE_LEN = 1023
CONFIDENCE_THRESHOLD = 0.03
SETTLE_FRAMES = 8
STEPS = ((85.0, 36), (160.0, 36), (55.0, 36))
FALSE_FRAME = 54
FALSE_LAG = 900
IRREGULAR_CHUNKS = (257, 31, 1024, 93, 511, 7, 333, 1, 191)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, help="prebuilt roundtrip-probe")
    parser.add_argument("--no-build", action="store_true", help="skip cargo build")
    parser.add_argument("--keep", type=Path, help="directory for emitted JSONL")
    return parser.parse_args()


def find_cargo() -> str:
    configured = os.environ.get("CARGO")
    if configured:
        return configured
    found = shutil.which("cargo")
    if found:
        return found
    user_home = Path(pwd.getpwuid(os.geteuid()).pw_dir)
    for candidate in (
        Path.home() / ".cargo" / "bin" / "cargo",
        user_home / ".cargo" / "bin" / "cargo",
    ):
        if candidate.exists():
            return str(candidate)
    raise SystemExit("cargo was not found; set CARGO or use --binary")


def toolchain_env() -> dict[str, str]:
    env = os.environ.copy()
    user_home = Path(pwd.getpwuid(os.geteuid()).pw_dir)
    rustup_home = user_home / ".rustup"
    cargo_home = user_home / ".cargo"
    if "RUSTUP_HOME" not in env and rustup_home.exists():
        env["RUSTUP_HOME"] = str(rustup_home)
    if "CARGO_HOME" not in env and cargo_home.exists():
        env["CARGO_HOME"] = str(cargo_home)
    return env


def build_binary(binary: Path | None, no_build: bool) -> Path:
    if binary is not None:
        return binary.resolve()
    if not no_build:
        subprocess.run(
            [find_cargo(), "build", "--release"],
            cwd=ROOT,
            check=True,
            env=toolchain_env(),
        )
    built = ROOT / "target" / "release" / "roundtrip-probe"
    if not built.exists():
        raise SystemExit(f"{built} does not exist; run cargo build --release")
    return built


def delay_plan() -> tuple[list[float], list[int]]:
    centers: list[float] = []
    for center, count in STEPS:
        centers.extend([center] * count)
    delays = [
        center
        + 0.65 * math.sin(index * 1.37)
        + 0.35 * math.sin(index * 0.31)
        for index, center in enumerate(centers)
    ]
    boundaries = []
    cursor = 0
    for _, count in STEPS:
        boundaries.append(cursor)
        cursor += count
    return delays, boundaries


def run_case(binary: Path, delays: list[float], chunks: tuple[int, ...], out: Path) -> dict:
    command = [
        str(binary),
        "run",
        "--delays",
        ",".join(f"{value:.6f}" for value in delays),
        "--chunks",
        ",".join(str(value) for value in chunks),
        "--noise-rms",
        "0.010",
        "--interference-rms",
        "0.050",
        "--amplitude",
        "0.008",
        "--confidence",
        str(CONFIDENCE_THRESHOLD),
        "--dropout-frames",
        str(FALSE_FRAME),
        "--false-burst",
        f"{FALSE_FRAME}:{FALSE_LAG}:0.30",
        "--jsonl",
        str(out),
        "--quiet",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    observations = [row for row in rows if row["record"] == "observation"]
    summaries = [row for row in rows if row["record"] == "summary"]
    if len(summaries) != 1:
        raise AssertionError(f"expected one summary, got {len(summaries)}")
    return {"observations": observations, "summary": summaries[0]}


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def validate_records(
    case: dict, delays: list[float], boundaries: list[int], require_uneven: bool
) -> None:
    observations = case["observations"]
    if len(observations) != len(delays):
        raise AssertionError(
            f"expected {len(delays)} observations, got {len(observations)}"
        )

    for index, row in enumerate(observations):
        if row["frame"] != index:
            raise AssertionError(f"frame order mismatch at {index}: {row['frame']}")
        start = index * PERIOD
        if (row["capture_start"], row["capture_end"]) != (start, start + PERIOD):
            raise AssertionError(f"bad capture window at frame {index}: {row}")
        if row["lag_min"] != 0 or row["lag_max"] != PERIOD - PROBE_LEN:
            raise AssertionError(f"bad lag search window at frame {index}: {row}")
        confidence = row["confidence"]
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise AssertionError(f"bad confidence at frame {index}: {confidence}")
        if row["observed_lag"] is not None and confidence < CONFIDENCE_THRESHOLD:
            raise AssertionError(f"sub-threshold observation at frame {index}")
        if row["accepted"]:
            if row["state"] != "locked":
                raise AssertionError(f"accepted row is not locked at frame {index}")
            if row["observed_lag"] is None or row["reported_lag"] is None:
                raise AssertionError(f"accepted row lacks lag at frame {index}")
            if row["gate_min"] is None or row["gate_max"] is None:
                raise AssertionError(f"accepted row lacks gate at frame {index}")
            if not row["gate_min"] <= row["reported_lag"] <= row["gate_max"]:
                raise AssertionError(f"reported lag outside gate at frame {index}")

    accepted = [row for row in observations if row["accepted"]]
    if any(
        abs(row["reported_lag"] - FALSE_LAG) < 10.0 for row in accepted
    ):
        raise AssertionError("tracker accepted the injected false-lag peak")

    errors = []
    for start, (_, count) in zip(boundaries, STEPS):
        segment = observations[start + SETTLE_FRAMES : start + count]
        settled = [row for row in segment if row["accepted"]]
        ratio = len(settled) / len(segment)
        if ratio < 0.80:
            raise AssertionError(
                f"segment at frame {start} only tracked {ratio:.1%} after settling"
            )
        segment_errors = [
            abs(row["reported_lag"] - delays[row["frame"]]) for row in settled
        ]
        if any(error > 2.0 for error in segment_errors):
            raise AssertionError(
                f"segment at frame {start} had an error above 2 samples"
            )
        p95 = percentile(segment_errors, 0.95)
        if p95 > 1.20:
            raise AssertionError(
                f"segment at frame {start} p95 error was {p95:.3f} samples"
            )
        errors.extend(segment_errors)

    for boundary in boundaries[1:]:
        recovered = None
        for row in observations[boundary : boundary + 8]:
            if (
                row["accepted"]
                and abs(row["reported_lag"] - delays[row["frame"]]) <= 1.5
            ):
                recovered = row["frame"]
                break
        if recovered is None or recovered - boundary > 6:
            raise AssertionError(f"delay step at frame {boundary} did not settle")

    false_row = observations[FALSE_FRAME]
    if false_row["accepted"]:
        raise AssertionError("the muted frame accepted a false detection")
    next_row = observations[FALSE_FRAME + 1]
    if not next_row["accepted"]:
        raise AssertionError("tracking did not recover on the frame after the false peak")
    if abs(next_row["reported_lag"] - delays[FALSE_FRAME + 1]) > 1.5:
        raise AssertionError("reported lag did not recover after the false peak")

    summary = case["summary"]
    if summary["accepted_frames"] / len(delays) < 0.80:
        raise AssertionError("summary accepted-frame ratio is too low")
    if summary["delay_error_p95_samples"] > 1.20:
        raise AssertionError("summary p95 tracking error is too high")
    if not 0.0 <= summary["jitter_rms_samples"] < 1.0:
        raise AssertionError("summary jitter RMS is outside the expected range")
    if summary["jitter_p95_samples"] > 1.2:
        raise AssertionError("summary p95 jitter is too high")
    if require_uneven and summary["min_chunk_size"] >= summary["max_chunk_size"]:
        raise AssertionError("the uneven-buffer run did not actually vary chunks")


def compare_runs(irregular: dict, regular: dict) -> None:
    left = irregular["observations"]
    right = regular["observations"]
    if len(left) != len(right):
        raise AssertionError("regular and irregular runs emitted different frame counts")
    fields = (
        "confidence",
        "observed_lag",
        "reported_lag",
        "accepted",
        "state",
        "reason",
    )
    for a, b in zip(left, right):
        for field in fields:
            if isinstance(a[field], float):
                if abs(a[field] - b[field]) > 1.0e-6:
                    raise AssertionError(
                        f"chunking changed {field} at frame {a['frame']}"
                    )
            elif a[field] != b[field]:
                raise AssertionError(
                    f"chunking changed {field} at frame {a['frame']}"
                )


def main() -> int:
    args = parse_args()
    binary = build_binary(args.binary, args.no_build)
    delays, boundaries = delay_plan()

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.keep:
        args.keep.mkdir(parents=True, exist_ok=True)
        output_dir = args.keep
    else:
        temporary = tempfile.TemporaryDirectory(prefix="roundtrip-acceptance-")
        output_dir = Path(temporary.name)

    try:
        irregular = run_case(
            binary, delays, IRREGULAR_CHUNKS, output_dir / "irregular.jsonl"
        )
        validate_records(irregular, delays, boundaries, require_uneven=True)
        print("PASS known delay steps, noise, false peak, and recovery")

        regular = run_case(binary, delays, (512,), output_dir / "regular.jsonl")
        validate_records(regular, delays, boundaries, require_uneven=False)
        compare_runs(irregular, regular)
        print("PASS regular and irregular capture buffers are observation-equivalent")

        summary = irregular["summary"]
        print(
            "PASS summary"
            f" accepted={summary['accepted_frames']}/{summary['frames']}"
            f" p95_error={summary['delay_error_p95_samples']:.3f}samples"
            f" chunks={summary['min_chunk_size']}..{summary['max_chunk_size']}"
        )
    finally:
        if temporary is not None:
            temporary.cleanup()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as error:
        print(f"FAIL {error}", file=sys.stderr)
        sys.exit(1)
