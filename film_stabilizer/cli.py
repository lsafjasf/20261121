from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from .crop import apply_crop
from .detector import DetectorConfig
from .io import open_source, write_manifest
from .stabilizer import StabilizerConfig, stabilize
from .tracker import TrackerConfig
from .types import matrix_angle_deg, matrix_translation


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="film-stabilize",
        description="Perforation-based stabilizer for scanned 16 mm film",
    )
    parser.add_argument("input", type=Path, help="image directory or video")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--fps", type=float, default=None, help="output frame rate")
    parser.add_argument(
        "--frame-number-offset",
        type=int,
        default=0,
        help="added to decoded video frame indices for the original frame map",
    )
    parser.add_argument("--side-band", type=float, default=0.12)
    parser.add_argument("--top-margin", type=float, default=0.03)
    parser.add_argument("--bottom-margin", type=float, default=0.03)
    parser.add_argument("--polarity", choices=("bright", "dark"), default="bright")
    parser.add_argument("--min-area", type=float, default=1.2e-4)
    parser.add_argument("--max-area", type=float, default=8e-3)
    parser.add_argument("--spike-jump", type=float, default=4.0)
    parser.add_argument("--splice-jump", type=float, default=6.0)
    parser.add_argument("--splice-angle", type=float, default=0.85)
    parser.add_argument(
        "--write-frames",
        action="store_true",
        help="also write stabilized PNGs using original source names",
    )
    parser.add_argument(
        "--burn-frame-number",
        action="store_true",
        help="burn the original frame number into video and PNG output",
    )
    parser.add_argument("--no-full-video", action="store_true")
    parser.add_argument("--no-cropped-video", action="store_true")
    return parser


def _write_crop_csv(path: Path, result) -> None:
    fields = [
        "output_frame",
        "original_frame",
        "source_name",
        "crop_x",
        "crop_y",
        "crop_width",
        "crop_height",
        "confidence",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in result.records:
            x, y, width, height = record.crop
            writer.writerow(
                {
                    "output_frame": record.index,
                    "original_frame": record.original_frame,
                    "source_name": record.source_name,
                    "crop_x": x,
                    "crop_y": y,
                    "crop_width": width,
                    "crop_height": height,
                    "confidence": record.confidence,
                    "status": record.status,
                }
            )


def _write_track_csv(path: Path, result) -> None:
    fields = [
        "output_frame",
        "original_frame",
        "left_observed",
        "right_observed",
        "confidence",
        "status",
        "tx",
        "ty",
        "rotation_deg",
        "is_splice",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in result.records:
            tx, ty = matrix_translation(record.matrix)
            writer.writerow(
                {
                    "output_frame": record.index,
                    "original_frame": record.original_frame,
                    "left_observed": int(record.left_observed),
                    "right_observed": int(record.right_observed),
                    "confidence": record.confidence,
                    "status": record.status,
                    "tx": f"{tx:.6f}",
                    "ty": f"{ty:.6f}",
                    "rotation_deg": f"{matrix_angle_deg(record.matrix):.6f}",
                    "is_splice": int(record.is_splice),
                }
            )


def _label_frame(frame: np.ndarray, original_frame: int) -> np.ndarray:
    labeled = frame.copy()
    cv2.putText(
        labeled,
        f"orig {original_frame:06d}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        labeled,
        f"orig {original_frame:06d}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (24, 24, 24),
        1,
        cv2.LINE_AA,
    )
    return labeled


def _writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create video writer: {path}")
    return writer


def _render_outputs(
    source,
    result,
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    full_path = out_dir / "stabilized.mp4"
    crop_x, crop_y, crop_w, crop_h = result.common_crop
    cropped_path = out_dir / "stabilized_cropped.mp4"
    full_writer = (
        None
        if args.no_full_video
        else _writer(full_path, result.info.fps, (result.info.width, result.info.height))
    )
    cropped_writer = (
        None
        if args.no_cropped_video or crop_w <= 0 or crop_h <= 0
        else _writer(cropped_path, result.info.fps, (crop_w, crop_h))
    )
    frames_dir = out_dir / "frames_stabilized"
    if args.write_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    try:
        for record, (_, _, _, frame) in zip(result.records, source.iter_frames()):
            stable = cv2.warpAffine(
                frame,
                record.matrix[:2, :],
                (result.info.width, result.info.height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            if args.burn_frame_number:
                stable = _label_frame(stable, record.original_frame)
            if full_writer is not None:
                full_writer.write(stable)
            if cropped_writer is not None:
                cropped_writer.write(apply_crop(stable, result.common_crop))
            if args.write_frames:
                cv2.imwrite(str(frames_dir / record.source_name), stable)
    finally:
        if full_writer is not None:
            full_writer.release()
        if cropped_writer is not None:
            cropped_writer.release()


def _manifest(result, input_path: Path, args: argparse.Namespace) -> dict:
    return {
        "tool": "film-stabilizer",
        "input": str(input_path),
        "frame_count": len(result.records),
        "fps": result.info.fps,
        "canvas_width": result.info.width,
        "canvas_height": result.info.height,
        "crop_coordinate_system": "top-left origin in the full stabilized canvas",
        "common_crop": {
            "x": result.common_crop[0],
            "y": result.common_crop[1],
            "width": result.common_crop[2],
            "height": result.common_crop[3],
        },
        "reference_left": result.track.reference_left,
        "reference_right": result.track.reference_right,
        "reference_angle_deg": result.track.reference_angle_deg,
        "splice_frames": result.splice_frames,
        "frame_number_policy": (
            "Numeric image filename suffixes are preserved; otherwise the "
            "sequence index is used. Video indices are offsets from decode index."
        ),
        "outputs": {
            "full_video": None if args.no_full_video else "stabilized.mp4",
            "cropped_video": (
                None if args.no_cropped_video else "stabilized_cropped.mp4"
            ),
            "crop_ranges": "crop_ranges.csv",
            "track": "track.csv",
            "stabilized_frames": "frames_stabilized/" if args.write_frames else None,
        },
    }


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    source = open_source(args.input, frame_number_offset=args.frame_number_offset)
    if args.fps is not None:
        source.info.fps = args.fps
    if not source.info.fps or not np.isfinite(source.info.fps):
        source.info.fps = 24.0

    config = StabilizerConfig(
        detector=DetectorConfig(
            side_band_fraction=args.side_band,
            top_margin_fraction=args.top_margin,
            bottom_margin_fraction=args.bottom_margin,
            polarity=args.polarity,
            min_area_fraction=args.min_area,
            max_area_fraction=args.max_area,
        ),
        tracker=TrackerConfig(
            spike_jump_px=args.spike_jump,
            splice_jump_px=args.splice_jump,
            splice_angle_deg=args.splice_angle,
        ),
    )
    result = stabilize(source, config)
    args.out.mkdir(parents=True, exist_ok=True)
    _render_outputs(source, result, args.out, args)
    _write_crop_csv(args.out / "crop_ranges.csv", result)
    _write_track_csv(args.out / "track.csv", result)
    write_manifest(args.out / "manifest.json", _manifest(result, args.input, args))

    both = sum(record.confidence == 2 for record in result.records)
    single = sum(record.confidence == 1 for record in result.records)
    print(
        f"{len(result.records)} frames: {both} dual-perf, {single} single-perf, "
        f"{len(result.records) - both - single} filled"
    )
    print(
        f"common crop: {result.common_crop[2]}x{result.common_crop[3]} "
        f"at ({result.common_crop[0]}, {result.common_crop[1]})"
    )
    print(f"splices: {result.splice_frames}")
    print(f"outputs: {args.out}")


if __name__ == "__main__":
    main()
