from __future__ import annotations

import argparse
import subprocess
from itertools import chain
from pathlib import Path

import cv2
import numpy as np


WIDTH = 320
HEIGHT = 180


def build_scene() -> np.ndarray:
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    base = 18.0 + 10.0 * (x / WIDTH) + 6.0 * (y / HEIGHT)
    scene = np.dstack([base * 0.82, base * 0.94, base]).astype(np.uint8)
    cv2.rectangle(scene, (30, 34), (118, 155), (52, 61, 48), 3)
    cv2.rectangle(scene, (153, 45), (286, 137), (35, 44, 58), 3)
    cv2.line(scene, (34, 146), (112, 38), (72, 67, 45), 2)
    cv2.line(scene, (157, 132), (281, 51), (29, 58, 76), 2)
    cv2.circle(scene, (74, 91), 17, (44, 72, 73), 2)
    cv2.putText(
        scene,
        "CAM 01",
        (177, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (82, 86, 78),
        1,
        cv2.LINE_AA,
    )
    return scene


def normal_lowlight_frames(count: int, seed: int = 20261121):
    rng = np.random.default_rng(seed)
    scene = build_scene()
    for _ in range(count):
        noise = rng.normal(0.0, 1.8, scene.shape[:2]).astype(np.float32)
        frame = np.clip(scene.astype(np.float32) + noise[..., None], 0, 255).astype(np.uint8)
        yield frame


def repeated_frozen_frames(count: int, seed: int = 20261121):
    scene = build_scene()
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.2, scene.shape[:2]).astype(np.float32)
    frozen = np.clip(
        scene.astype(np.float32) + noise[..., None], 0, 255
    ).astype(np.uint8)
    for _ in range(count):
        yield frozen.copy()


def black_frames(count: int):
    frozen = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for _ in range(count):
        yield frozen.copy()


def encode(frames, output: Path, fps: int, bitrate: str, label: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-b:v",
        bitrate,
        "-maxrate",
        bitrate,
        "-bufsize",
        bitrate,
        "-g",
        str(fps * 10),
        "-keyint_min",
        str(fps * 10),
        "-sc_threshold",
        "0",
        "-bf",
        "0",
        "-threads",
        "1",
        "-movflags",
        "+faststart",
        "-metadata",
        f"comment={label}",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in frames:
            process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg failed for {output}: {stderr}")


def main() -> int:
    parser = argparse.ArgumentParser(description="生成冻结诊断三段样例")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()
    count = round(args.duration * args.fps)
    output = args.output_dir.resolve()
    cases = [
        (
            "normal_lowlight.mp4",
            normal_lowlight_frames(count),
            "400k",
            "fixed camera, true static scene, low light sensor noise",
        ),
        (
            "repeat_lowbitrate.mp4",
            repeated_frozen_frames(count),
            "45k",
            "capture repeat fault, identical decoded frame, nominal PTS",
        ),
        (
            "black_emergency.mp4",
            black_frames(count),
            "20k",
            "all black emergency input, no texture or temporal noise",
        ),
    ]
    for name, frames, bitrate, label in cases:
        target = output / name
        encode(frames, target, args.fps, bitrate, label)
        print(f"generated: {target}")

    mixed_target = output / "mixed_incident.mp4"
    mixed_frames = chain(
        normal_lowlight_frames(args.fps * 5),
        repeated_frozen_frames(args.fps * 10),
        black_frames(args.fps * 5),
    )
    encode(
        mixed_frames,
        mixed_target,
        args.fps,
        "400k",
        "normal 5s + repeat 10s + black 5s",
    )
    print(f"generated: {mixed_target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
