"""Generate test clips for the deflicker demo.

Outputs (into artifacts/):
  flicker_clean.mp4   - ground-truth clean clip (static bg + mover)
  flicker_input.mp4   - same clip with simulated strobe/rolling banding
  dimming_input.mp4   - clip with a genuine global dimming ramp plus
                        a small amount of flicker
"""

import os

import cv2
import numpy as np

W, H, N, FPS = 320, 240, 90, 30
ART = os.path.join(os.path.dirname(__file__), "artifacts")


def write_video(path, frames):
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H), False)
    for f in frames:
        writer.write(f)
    writer.release()


def base_scene(rng):
    """Static background: vertical gradient + texture + shapes."""
    yy = np.linspace(0.25, 0.75, H, dtype=np.float32)[:, None]
    xx = np.linspace(0.9, 1.1, W, dtype=np.float32)[None, :]
    bg = yy * xx
    bg += rng.normal(0, 0.02, (H, W)).astype(np.float32)
    cv2.rectangle(bg, (30, 30), (110, 90), 0.85, -1)
    cv2.circle(bg, (260, 60), 25, 0.35, -1)
    return np.clip(bg, 0, 1)


def mover_mask(t):
    """Bright rectangle sweeping left->right across the frame."""
    x = int(-60 + (W + 120) * t / N)
    mask = np.zeros((H, W), np.float32)
    cv2.rectangle(mask, (x, 130), (x + 50, 190), 1.0, -1)
    return mask


def clean_frames(bg):
    frames = []
    for t in range(N):
        m = mover_mask(t)
        frame = bg * (1 - m) + 0.95 * m
        frames.append((np.clip(frame, 0, 1) * 255).astype(np.uint8))
    return frames


def apply_flicker(frames, rng, frame_amp=0.18, row_amp=0.12):
    """Simulated strobe: per-frame gain + rolling row banding."""
    out = []
    rows = np.arange(H, dtype=np.float32)[:, None]
    for t, f in enumerate(frames):
        a = 1.0 + frame_amp * np.sin(2 * np.pi * 8 * t / FPS) \
                 + rng.normal(0, 0.02)
        b = 1.0 + row_amp * np.sin(2 * np.pi * (rows / H * 4 - t * 0.5))
        g = a * (1 + rng.normal(0, 0.01, (H, W)).astype(np.float32)) * b
        out.append((np.clip(f / 255.0 * g, 0, 1) * 255).astype(np.uint8))
    return out


def apply_dimming(frames, rng):
    """Genuine global dimming ramp 1.0 -> 0.55 plus mild flicker."""
    out = []
    for t, f in enumerate(frames):
        ramp = 1.0 - 0.45 * t / (N - 1)
        flick = 1.0 + 0.05 * np.sin(2 * np.pi * 8 * t / FPS)
        g = ramp * flick
        out.append((np.clip(f / 255.0 * g, 0, 1) * 255).astype(np.uint8))
    return out


def main():
    os.makedirs(ART, exist_ok=True)
    rng = np.random.default_rng(7)
    bg = base_scene(rng)
    clean = clean_frames(bg)

    write_video(os.path.join(ART, "flicker_clean.mp4"), clean)
    write_video(os.path.join(ART, "flicker_input.mp4"),
                apply_flicker(clean, rng))
    write_video(os.path.join(ART, "dimming_input.mp4"),
                apply_dimming(clean, rng))
    print("wrote flicker_clean.mp4, flicker_input.mp4, dimming_input.mp4")


if __name__ == "__main__":
    main()
