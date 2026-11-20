"""Generate ground-truth straight-alpha RGBA masters as PNG sequences.

Three synthetic cases that expose straight/premultiplied confusion:
  gradient          - smooth semi-transparent alpha ramp, tinted gradient
  edge_glow         - thin bright glow line at low alpha over dark plate
  transparent_color - fully transparent pixels that still carry RGB
"""
import argparse
from pathlib import Path

import numpy as np

from common import FPS, save_png_sequence, to_u8

W, H, N = 320, 180, 24


def _ramp_frames():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    x = xx / (W - 1)
    y = yy / (H - 1)
    frames = np.zeros((N, H, W, 4), dtype=np.float32)
    for t in range(N):
        shift = 0.15 * np.sin(2 * np.pi * t / N)
        a = np.clip((x - 0.15 - shift) / 0.7, 0.0, 1.0) ** 1.2
        frames[t, ..., 0] = 0.15 + 0.75 * y
        frames[t, ..., 1] = 0.75 - 0.55 * y
        frames[t, ..., 2] = 0.25 + 0.60 * (1.0 - x)
        frames[t, ..., 3] = a
    return frames


def _edge_glow_frames():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    frames = np.zeros((N, H, W, 4), dtype=np.float32)
    for t in range(N):
        frames[t, ..., 0] = 0.03
        frames[t, ..., 1] = 0.03
        frames[t, ..., 2] = 0.05
        frames[t, ..., 3] = 1.0
        cx = W * (0.30 + 0.40 * t / (N - 1))
        cy = H * (0.50 + 0.18 * np.sin(2 * np.pi * t / N))
        radius = H * 0.32
        dist = np.abs(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - radius)
        glow = np.exp(-(dist / 1.6) ** 2)          # thin ~2px bright core
        halo = 0.25 * np.exp(-(dist / 6.0) ** 2)   # faint wide halo
        a = np.clip(glow * 0.85 + halo, 0.0, 1.0)
        frames[t, ..., 0] = 0.20 + 0.80 * a
        frames[t, ..., 1] = 0.55 + 0.45 * a
        frames[t, ..., 2] = 0.90 + 0.10 * a
        frames[t, ..., 3] = np.maximum(a, 0.02)
    return frames


def _transparent_color_frames():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    frames = np.zeros((N, H, W, 4), dtype=np.float32)
    for t in range(N):
        # Hidden RGB painting: saturated stripes, alpha = 0 everywhere here.
        stripe = ((xx // 20 + t // 6) % 2).astype(np.float32)
        frames[t, ..., 0] = 0.20 + 0.80 * stripe
        frames[t, ..., 1] = 0.90 - 0.70 * stripe
        frames[t, ..., 2] = 0.30 + 0.40 * (yy / H)
        frames[t, ..., 3] = 0.0
        # One small opaque square so the frame is not visually empty.
        frames[t, 20:60, 20:60, 0] = 0.95
        frames[t, 20:60, 20:60, 1] = 0.85
        frames[t, 20:60, 20:60, 2] = 0.20
        frames[t, 20:60, 20:60, 3] = 1.0
    return frames


GENERATORS = {
    "gradient": _ramp_frames,
    "edge_glow": _edge_glow_frames,
    "transparent_color": _transparent_color_frames,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out/masters",
                    help="output root for PNG master sequences")
    ap.add_argument("--cases", nargs="*", default=list(GENERATORS),
                    choices=list(GENERATORS))
    args = ap.parse_args()
    for name in args.cases:
        frames = to_u8(np.clip(GENERATORS[name](), 0.0, 1.0))
        seq_dir = Path(args.out) / name
        save_png_sequence(frames, seq_dir)
        print(f"[master] {name}: {frames.shape[0]} frames "
              f"{frames.shape[2]}x{frames.shape[1]} @ {FPS}fps -> {seq_dir}")


if __name__ == "__main__":
    main()
