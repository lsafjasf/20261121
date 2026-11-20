"""Synthetic rolling-shutter LED-flicker video generator.

The generator simulates the exact forward model in :mod:`model`: every
sensor row integrates a known square-wave LED over its own exposure
window, then sensor rotation, vignetting and configurable noise are
applied.  The ground-truth parameters travel alongside the video so the
estimator can be scored against them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

from .model import exposure_fraction


@dataclass
class SynthConfig:
    """Parameters of the synthetic capture."""

    width: int = 480
    height: int = 360
    frames: int = 12
    fps: float = 60.0
    f_led_hz: float = 1000.0
    duty: float = 0.5
    t_line_us: float = 20.0          # per-row readout time
    t_exp_us: float = 250.0          # per-row exposure time
    scan_direction: int = 1          # +1 top-down, -1 bottom-up
    rotation_deg: float = 12.0       # sensor rotation vs scene vertical
    brightness: float = 220.0        # LED-on mean grey level
    black_level: float = 8.0
    gaussian_sigma: float = 4.0      # read noise std (grey levels)
    shot_noise: float = 0.0          # Poisson scale, 0 disables
    fpn_sigma: float = 0.0           # fixed-pattern row noise
    phase_jitter_us: float = 5.0     # per-frame random trigger jitter
    genlock: bool = False            # LED phase locked to frame trigger
    seed: int = 7

    @property
    def period_s(self) -> float:
        return 1.0 / self.f_led_hz

    @property
    def t_line_s(self) -> float:
        return self.t_line_us * 1e-6

    @property
    def t_exp_s(self) -> float:
        return self.t_exp_us * 1e-6

    def to_dict(self) -> dict:
        return asdict(self)


def render_frame(cfg: SynthConfig, frame_index: int, rng: np.random.Generator) -> np.ndarray:
    """Render one synthetic frame as an 8-bit single-channel image."""
    period = cfg.period_s
    t_line = cfg.t_line_s
    t_exp = cfg.t_exp_s

    # Free-running frame trigger: nominal spacing plus jitter.
    frame_time = frame_index / cfg.fps
    frame_time += rng.normal(0.0, cfg.phase_jitter_us * 1e-6)

    rows = np.arange(cfg.height, dtype=np.float64)
    start_times = frame_time - cfg.scan_direction * rows * t_line
    if cfg.genlock:
        # LED phase 0 at every frame trigger, so time relative to the
        # trigger drives the waveform.
        waveform_time = start_times - frame_time
    else:
        waveform_time = start_times
    fraction = exposure_fraction(waveform_time, t_exp, period, cfg.duty)

    # Smooth spatial shading (vignetting) in sensor coordinates.
    yy = rows - 0.5 * (cfg.height - 1)
    shade = 1.0 - 0.15 * (yy / (0.5 * cfg.height)) ** 2
    mean = cfg.black_level + (cfg.brightness - cfg.black_level) * fraction
    image = np.repeat((mean * shade)[:, None], cfg.width, axis=1)

    if cfg.shot_noise > 0.0:
        lam = np.clip(image / max(cfg.brightness, 1.0), 0.0, None)
        image = image + cfg.shot_noise * (rng.poisson(lam) - lam)
    if cfg.fpn_sigma > 0.0:
        pattern = rng.normal(0.0, cfg.fpn_sigma, size=cfg.height)
        image = image + pattern[:, None]
    if cfg.gaussian_sigma > 0.0:
        image = image + rng.normal(0.0, cfg.gaussian_sigma, size=image.shape)

    clean = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    return rotate_frame(clean, cfg.rotation_deg)


def rotate_frame(frame: np.ndarray, rotation_deg: float) -> np.ndarray:
    """Rotate a frame around its image center (forward sensor rotation)."""
    if rotation_deg == 0.0:
        return frame
    height, width = frame.shape[:2]
    matrix = cv2.getRotationMatrix2D((width * 0.5, height * 0.5), rotation_deg, 1.0)
    return cv2.warpAffine(
        frame, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )


def generate_frames(cfg: SynthConfig):
    """Yield ``(index, frame)`` pairs deterministically from ``cfg.seed``."""
    rng = np.random.default_rng(cfg.seed)
    for index in range(cfg.frames):
        yield index, render_frame(cfg, index, rng)


def write_video(cfg: SynthConfig, path: str | Path, truth_path: str | Path | None = None):
    """Render the whole clip and write it to ``path``.

    Uses the lossless-ish MJPG codec in an AVI container so round trips
    do not distort the faint high-order harmonics used for exposure
    fitting.  The ground truth is dumped next to the video as JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, cfg.fps, (cfg.width, cfg.height), 0)
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer for {path}")
    for _, frame in generate_frames(cfg):
        writer.write(frame)
    writer.release()

    if truth_path is None:
        truth_path = path.with_suffix(".truth.json")
    truth_path = Path(truth_path)
    truth = cfg.to_dict()
    truth.update(
        {
            "period_s": cfg.period_s,
            "t_line_s": cfg.t_line_s,
            "t_exp_s": cfg.t_exp_s,
        }
    )
    truth_path.write_text(json.dumps(truth, indent=2, sort_keys=True))
    return path, truth_path
