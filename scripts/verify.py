"""Verify the rendered output against expected timing and source colors.

Checks:
1. output frame count and key boundary-frame pixels;
2. V1 dissolve weights on the lossless intermediate;
3. A1 audio crossfade position located from 220/440 Hz band energies;
4. contribution manifest spot checks, including a manifest-vs-pixels compare.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

RED = np.array([200, 30, 30], dtype=float)
BLUE = np.array([30, 60, 220], dtype=float)
GREEN = np.array([30, 190, 60], dtype=float)
BLACK = np.array([0, 0, 0], dtype=float)
PIXEL_TOL = 8.0


def frame_rgb(path: str, frame: int) -> np.ndarray:
    command = [
        "ffmpeg", "-v", "error", "-i", path,
        "-vf", f"select=eq(n\\,{frame}),scale=1:1",
        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    raw = subprocess.run(command, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).astype(float)


def frame_count(path: str) -> int:
    command = [
        "ffprobe", "-v", "error", "-count_frames",
        "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0", path,
    ]
    return int(subprocess.run(command, capture_output=True, text=True,
                              check=True).stdout.strip())


def read_audio(path: str) -> tuple[np.ndarray, int]:
    command = [
        "ffmpeg", "-v", "error", "-i", path,
        "-f", "f32le", "-ac", "1", "-",
    ]
    raw = subprocess.run(command, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32), 48000


def band_energy(signal: np.ndarray, rate: int, center: float,
                window: float = 0.05) -> float:
    samples = round(window * rate)
    windowed = signal[:samples] * np.hanning(samples)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(samples, 1 / rate)
    idx = np.argmin(np.abs(freqs - center))
    return float(np.sum(spectrum[max(0, idx - 1): idx + 2] ** 2))


def energy_at(signal, rate, second, freq):
    start = round(second * rate)
    return band_energy(signal[start: start + rate], rate, freq)


def locate_midpoint(signal, rate, expected):
    times = np.arange(1.2, 2.8, 0.02)
    measured = None
    for index, t in enumerate(times[:-1]):
        now_220 = energy_at(signal, rate, t, 220)
        now_440 = energy_at(signal, rate, t, 440)
        next_220 = energy_at(signal, rate, times[index + 1], 220)
        next_440 = energy_at(signal, rate, times[index + 1], 440)
        if now_220 >= now_440 and next_440 > next_220:
            measured = float(times[index + 1])
            break
    if measured is None:
        measured = float(times[-1])
    return measured, abs(measured - expected)


def mad(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - expected)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("v1_intermediate")
    parser.add_argument("a1_intermediate")
    parser.add_argument("manifest")
    parser.add_argument("report_dir")
    args = parser.parse_args()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    failures = 0

    def check(label, ok, detail=""):
        nonlocal failures
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        line = f"[{status}] {label} {detail}".rstrip()
        print(line)
        lines.append(line)

    count = frame_count(args.output)
    check("rendered frame count", count == 105, f"actual={count} expected=105")

    print("== video boundary frames ==")
    lines.append("== video boundary frames ==")
    expectations = [
        (0, "V2 explicit black prefix", BLACK),
        (14, "last black-prefix frame", BLACK),
        (15, "first green media frame", GREEN),
        (44, "green cover pre-dissolve", GREEN),
        (75, "first full blue frame", BLUE),
        (104, "last timeline frame", BLUE),
    ]
    for frame, label, expected in expectations:
        actual = frame_rgb(args.output, frame)
        score = mad(actual, expected)
        check(f"f{frame} {label}", score <= PIXEL_TOL,
              f"expect={expected.astype(int).tolist()} "
              f"actual={np.round(actual, 1).tolist()} mad={score:.2f}")

    print("== V1 dissolve blend (lossless intermediate) ==")
    lines.append("== V1 dissolve blend ==")
    manifest = json.loads(Path(args.manifest).read_text())
    weights = {
        entry["frame"]: entry["tracks"][0]["contributors"]
        for entry in manifest["per_frame"]
    }
    for frame in (45, 59, 60, 73, 74):
        actual = frame_rgb(args.v1_intermediate, frame)
        mix = np.zeros(3)
        for contributor in weights[frame]:
            color = RED if contributor["source"] == "red" else BLUE
            mix += color * contributor["weight"]
        score = mad(actual, mix)
        check(f"f{frame} manifest-vs-pixels", score <= PIXEL_TOL,
              f"mad={score:.2f}")

    print("== audio crossfade position ==")
    lines.append("== audio crossfade position ==")
    signal, rate = read_audio(args.a1_intermediate)
    pre_220 = energy_at(signal, rate, 1.05, 220)
    pre_440 = energy_at(signal, rate, 1.05, 440)
    post_220 = energy_at(signal, rate, 2.7, 220)
    post_440 = energy_at(signal, rate, 2.7, 440)
    check("pre-crossfade dominated by 220 Hz", pre_220 > pre_440 * 5)
    check("post-crossfade dominated by 440 Hz", post_440 > post_220 * 5)
    mid_220 = energy_at(signal, rate, 2.0, 220)
    mid_440 = energy_at(signal, rate, 2.0, 440)
    check("cut point carries both tones",
          mid_220 > 0 and mid_440 > 0,
          f"e220={mid_220:.3e} e440={mid_440:.3e}")
    measured, error = locate_midpoint(signal, rate, expected=2.0)
    check("crossfade midpoint within one frame", error <= 1 / 30,
          f"measured={measured:.2f}s expected=2.00s err={error:.2f}s")

    print("== contribution manifest spot checks ==")
    lines.append("== contribution manifest spot checks ==")
    f0 = manifest["per_frame"][0]["tracks"][1]["contributors"][0]["source"]
    check("f0 V2 contributor is black:green", f0 == "black:green", f"got={f0}")
    f60 = {c["source"]: c["weight"] for c in weights[60]}
    check("f60 red/blue weights 50/50",
          abs(f60["red"] - 0.5) < 0.01 and abs(f60["blue"] - 0.5) < 0.01,
          f"red={f60['red']:.3f} blue={f60['blue']:.3f}")

    result = "ALL CHECKS PASSED" if failures == 0 else f"{failures} CHECK(S) FAILED"
    print(f"\nRESULT: {result}")
    lines.append(f"RESULT: {result}")
    (report_dir / "verify_output.txt").write_text("\n".join(lines))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
