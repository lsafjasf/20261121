from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .io import write_video
from .types import SequenceInfo, apply_matrix, rigid_matrix


@dataclass
class SyntheticConfig:
    width: int = 640
    height: int = 480
    frames: int = 72
    fps: float = 24.0
    seed: int = 17
    gate_jitter_px: float = 3.2
    gate_rotation_deg: float = 1.15
    splice_frame: int | None = 47
    splice_step_px: tuple[float, float] = (9.0, -6.5)
    splice_step_deg: float = 1.6
    missing_left: set[int] = field(default_factory=set)
    missing_right: set[int] = field(default_factory=set)
    grain_sigma: float = 5.0
    sensor_noise_sigma: float = 3.0

    def resolved_missing(self) -> tuple[set[int], set[int]]:
        if self.missing_left or self.missing_right:
            return set(self.missing_left), set(self.missing_right)
        return (
            {11, 12, 23, 38, 39, self.frames // 2},
            {17, 18, 31, 32, 52, self.frames // 2},
        )


@dataclass
class SyntheticTruth:
    index: int
    original_frame: int
    matrix: np.ndarray
    gate_tx: float
    gate_ty: float
    gate_angle_deg: float
    left: tuple[float, float]
    right: tuple[float, float]
    actor: tuple[float, float]
    missing_left: bool
    missing_right: bool
    splice: bool


class SyntheticSequence:
    """Render a film scan whose perforations and picture share one gate pose."""

    def __init__(self, config: SyntheticConfig | None = None) -> None:
        self.config = config or SyntheticConfig()
        if self.config.width < 160 or self.config.height < 120:
            raise ValueError("synthetic frames must be at least 160x120")
        self.info = SequenceInfo(
            width=self.config.width,
            height=self.config.height,
            fps=self.config.fps,
            frame_count=self.config.frames,
            source="synthetic",
        )
        self.missing_left, self.missing_right = self.config.resolved_missing()
        self.left_canonical = (
            float(round(self.config.width * 0.060)),
            float(round(self.config.height * 0.185)),
        )
        self.right_canonical = (
            float(round(self.config.width * 0.940)),
            float(round(self.config.height * 0.185)),
        )
        self._base = self._build_base()
        self._truth = self._build_truth()

    @property
    def truth(self) -> list[SyntheticTruth]:
        return self._truth

    def _build_base(self) -> np.ndarray:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        height, width = cfg.height, cfg.width
        base = np.empty((height, width, 3), dtype=np.uint8)
        base[:] = (22, 20, 18)
        y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        texture = (
            24.0
            + 8.0 * np.sin(x * 31.0 + 0.4)
            + 5.0 * np.cos(y * 27.0)
            + rng.normal(0.0, 2.2, (height, width))
        )
        base[:] = np.clip(texture[..., None], 8, 45).astype(np.uint8)

        margin_x = int(round(width * 0.145))
        margin_y = int(round(height * 0.060))
        aperture = (margin_x, margin_y, width - margin_x, height - margin_y)
        ax0, ay0, ax1, ay1 = aperture
        interior = rng.normal(0.0, 1.0, (ay1 - ay0, ax1 - ax0)).astype(np.float32)
        interior = cv2.GaussianBlur(interior, (0, 0), 1.1)
        interior = 120.0 + 35.0 * interior
        interior += 23.0 * np.sin(
            np.linspace(0.0, 8.0, ax1 - ax0, dtype=np.float32)[None, :]
        )
        interior += 17.0 * np.cos(
            np.linspace(0.0, 6.0, ay1 - ay0, dtype=np.float32)[:, None]
        )
        interior = np.clip(interior, 30, 220).astype(np.uint8)
        base[ay0:ay1, ax0:ax1] = cv2.cvtColor(interior, cv2.COLOR_GRAY2BGR)

        for cx in (ax0 + 24, ax1 - 34):
            cv2.line(
                base,
                (cx, ay0 + 10),
                (cx, ay1 - 10),
                (82, 92, 98),
                2,
                cv2.LINE_AA,
            )
        for cy in (ay0 + 18, ay1 - 26):
            cv2.line(
                base,
                (ax0 + 8, cy),
                (ax1 - 8, cy),
                (72, 80, 84),
                1,
                cv2.LINE_AA,
            )
        return base

    def _gate_matrix(self, index: int) -> tuple[np.ndarray, float, float, float]:
        cfg = self.config
        t = float(index)
        tx = cfg.gate_jitter_px * (
            0.72 * np.sin(0.29 * t + 0.3) + 0.35 * np.sin(0.71 * t + 1.1)
        )
        ty = cfg.gate_jitter_px * (
            0.65 * np.sin(0.21 * t + 1.7) + 0.30 * np.sin(0.83 * t)
        )
        angle = cfg.gate_rotation_deg * (
            0.68 * np.sin(0.13 * t + 0.4) + 0.32 * np.sin(0.37 * t + 2.0)
        )
        splice = cfg.splice_frame is not None and index >= cfg.splice_frame
        if splice:
            tx += cfg.splice_step_px[0]
            ty += cfg.splice_step_px[1]
            angle += cfg.splice_step_deg
        angle_rad = np.radians(angle)
        center = np.array([cfg.width * 0.5, cfg.height * 0.5])
        rotation = rigid_matrix(angle_rad, 0.0, 0.0)[:2, :2]
        offset = center + np.array([tx, ty]) - rotation @ center
        matrix = rigid_matrix(angle_rad, float(offset[0]), float(offset[1]))
        return matrix, float(tx), float(ty), float(angle)

    def _actor_center(self, index: int) -> tuple[float, float]:
        cfg = self.config
        t = float(index)
        ax0 = cfg.width * 0.112 + 28.0
        ax1 = cfg.width * 0.888 - 28.0
        ay0 = cfg.height * 0.060 + 30.0
        ay1 = cfg.height * 0.940 - 30.0
        x = cfg.width * 0.50 + cfg.width * 0.25 * np.sin(0.47 * t + 0.1)
        y = cfg.height * 0.48 + cfg.height * 0.22 * np.cos(0.31 * t + 0.8)
        x = float(np.clip(x, ax0, ax1))
        y = float(np.clip(y, ay0, ay1))
        return x, y

    def _build_truth(self) -> list[SyntheticTruth]:
        truth: list[SyntheticTruth] = []
        for index in range(self.config.frames):
            matrix, tx, ty, angle = self._gate_matrix(index)
            left = apply_matrix(matrix, [self.left_canonical])[0]
            right = apply_matrix(matrix, [self.right_canonical])[0]
            actor = apply_matrix(matrix, [self._actor_center(index)])[0]
            truth.append(
                SyntheticTruth(
                    index=index,
                    original_frame=index,
                    matrix=matrix,
                    gate_tx=tx,
                    gate_ty=ty,
                    gate_angle_deg=angle,
                    left=(float(left[0]), float(left[1])),
                    right=(float(right[0]), float(right[1])),
                    actor=(float(actor[0]), float(actor[1])),
                    missing_left=index in self.missing_left,
                    missing_right=index in self.missing_right,
                    splice=(
                        self.config.splice_frame is not None
                        and index == self.config.splice_frame
                    ),
                )
            )
        return truth

    def _draw_perforations(
        self, frame: np.ndarray, truth: SyntheticTruth, rng: np.random.Generator
    ) -> None:
        cfg = self.config
        hole_width = max(7, int(round(cfg.width * 0.032)) | 1)
        hole_height = max(9, int(round(cfg.height * 0.061)) | 1)
        bevel = max(1, int(round(min(hole_width, hole_height) * 0.10)))
        for side, center, missing in (
            ("left", self.left_canonical, truth.missing_left),
            ("right", self.right_canonical, truth.missing_right),
        ):
            if missing:
                continue
            cx = int(round(center[0]))
            cy = int(round(center[1]))
            x0 = cx - hole_width // 2
            y0 = cy - hole_height // 2
            x1 = x0 + hole_width - 1
            y1 = y0 + hole_height - 1
            brightness = int(np.clip(rng.normal(238, 4), 218, 252))
            cv2.rectangle(frame, (x0, y0), (x1, y1), (brightness,) * 3, -1)
            inset = bevel
            inner = int(np.clip(brightness + 12, 230, 255))
            cv2.rectangle(
                frame,
                (x0 + inset, y0 + inset),
                (x1 - inset, y1 - inset),
                (inner,) * 3,
                -1,
            )
            cv2.rectangle(
                frame,
                (x0, y0),
                (x1, y1),
                (int(brightness * 0.62),) * 3,
                1,
            )

    def _draw_actor(self, frame: np.ndarray, index: int) -> None:
        cx, cy = self._actor_center(index)
        cx_i, cy_i = int(round(cx)), int(round(cy))
        radius = max(10, int(round(min(self.config.width, self.config.height) * 0.048)))
        phase = 0.45 * index
        angle = int((phase * 57.0) % 360)
        cv2.ellipse(
            frame,
            (cx_i, cy_i),
            (radius, int(radius * 0.72)),
            angle,
            0,
            360,
            (38, 170, 72),
            -1,
            cv2.LINE_AA,
        )
        cv2.circle(
            frame,
            (cx_i, cy_i),
            max(3, radius // 3),
            (175, 235, 210),
            -1,
            cv2.LINE_AA,
        )
        cv2.circle(
            frame,
            (cx_i, cy_i),
            radius,
            (20, 80, 35),
            2,
            cv2.LINE_AA,
        )

    def _draw_splice(self, frame: np.ndarray, truth: SyntheticTruth) -> None:
        if self.config.splice_frame is None or truth.index < self.config.splice_frame:
            return
        width = self.config.width
        x = int(round(width * 0.122))
        overlay = frame.copy()
        cv2.line(
            overlay,
            (x, 0),
            (x + 5, self.config.height - 1),
            (86, 105, 142),
            max(2, width // 190),
            cv2.LINE_AA,
        )
        cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)

    def iter_frames(self) -> Iterator[tuple[int, str, int, np.ndarray]]:
        cfg = self.config
        for truth in self._truth:
            rng = np.random.default_rng(cfg.seed * 1009 + truth.index)
            canonical = self._base.copy()
            grain = rng.normal(0.0, cfg.grain_sigma, canonical.shape[:2])
            canonical = np.clip(
                canonical.astype(np.float32) + grain[..., None], 0, 255
            ).astype(np.uint8)
            self._draw_actor(canonical, truth.index)
            self._draw_perforations(canonical, truth, rng)
            self._draw_splice(canonical, truth)

            scanned = cv2.warpAffine(
                canonical,
                truth.matrix[:2, :],
                (cfg.width, cfg.height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(8, 8, 8),
            )
            sensor = rng.normal(
                0.0, cfg.sensor_noise_sigma, (cfg.height, cfg.width, 1)
            )
            scanned = np.clip(
                scanned.astype(np.float32) + sensor, 0, 255
            ).astype(np.uint8)
            cv2.rectangle(
                scanned,
                (2, 2),
                (cfg.width - 3, cfg.height - 3),
                (14, 14, 14),
                3,
            )
            yield (
                truth.index,
                f"frame_{truth.original_frame:06d}.png",
                truth.original_frame,
                scanned,
            )

    def write(self, out_dir: Path, write_images: bool = True) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = out_dir / "frames"
        if write_images:
            frames_dir.mkdir(parents=True, exist_ok=True)
        raw_frames: list[np.ndarray] = []
        for index, name, _, frame in self.iter_frames():
            raw_frames.append(frame)
            if write_images:
                cv2.imwrite(str(frames_dir / name), frame)
        write_video(
            out_dir / "synthetic.mp4",
            iter(raw_frames),
            self.config.fps,
            (self.config.width, self.config.height),
        )
        self._write_truth(out_dir / "ground_truth.csv")
        return frames_dir if write_images else out_dir / "synthetic.mp4"

    def _write_truth(self, path: Path) -> None:
        fields = [
            "original_frame",
            "gate_tx",
            "gate_ty",
            "gate_angle_deg",
            "left_x",
            "left_y",
            "right_x",
            "right_y",
            "actor_x",
            "actor_y",
            "missing_left",
            "missing_right",
            "splice",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for item in self._truth:
                writer.writerow(
                    {
                        "original_frame": item.original_frame,
                        "gate_tx": f"{item.gate_tx:.5f}",
                        "gate_ty": f"{item.gate_ty:.5f}",
                        "gate_angle_deg": f"{item.gate_angle_deg:.5f}",
                        "left_x": f"{item.left[0]:.4f}",
                        "left_y": f"{item.left[1]:.4f}",
                        "right_x": f"{item.right[0]:.4f}",
                        "right_y": f"{item.right[1]:.4f}",
                        "actor_x": f"{item.actor[0]:.4f}",
                        "actor_y": f"{item.actor[1]:.4f}",
                        "missing_left": int(item.missing_left),
                        "missing_right": int(item.missing_right),
                        "splice": int(item.splice),
                    }
                )


def _parse_pair(value: str) -> tuple[float, float]:
    parts = value.replace(",", " ").split()
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected two numbers, e.g. 9,-6.5")
    return float(parts[0]), float(parts[1])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic 16 mm scan")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--splice-frame", type=int, default=47)
    parser.add_argument("--splice-step", type=_parse_pair, default=(9.0, -6.5))
    parser.add_argument(
        "--missing-both",
        default="",
        help="comma-separated frame indices, e.g. 20,21",
    )
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args(argv)
    missing_both = {
        int(item) for item in args.missing_both.split(",") if item.strip()
    }
    config = SyntheticConfig(
        width=args.width,
        height=args.height,
        frames=args.frames,
        fps=args.fps,
        seed=args.seed,
        splice_frame=args.splice_frame,
        splice_step_px=args.splice_step,
        missing_left=missing_both,
        missing_right=missing_both,
    )
    output = SyntheticSequence(config).write(args.out, not args.no_images)
    print(output)


if __name__ == "__main__":
    main()
