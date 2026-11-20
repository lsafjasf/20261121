from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

import cv2
import imageio_ffmpeg
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .calibration import CalibrationResult
from .config import LabConfig
from .restoration import RestorationResult
from .synth import CalibrationClip, DemoClip


BG = (7, 10, 16)
PANEL = (18, 24, 36)
TEXT = (224, 232, 240)
MUTED = (124, 143, 163)
CYAN = (82, 220, 225)
LIME = (166, 232, 132)
PINK = (232, 91, 182)
AMBER = (245, 177, 74)
RED = (244, 82, 94)


def _ffmpeg_executable() -> str:
    local = shutil.which("ffmpeg")
    if local:
        return local
    return imageio_ffmpeg.get_ffmpeg_exe()


def save_video(path: Path, frames: np.ndarray, fps: int, crf: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must have shape (T, H, W, 3)")
    height, width = frames.shape[1:3]
    command = [
        _ffmpeg_executable(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in frames:
            process.stdin.write(
                np.clip(frame * 255.0, 0, 255).astype(np.uint8).tobytes()
            )
        process.stdin.close()
        return_code = process.wait()
    except Exception:
        process.kill()
        raise
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with status {return_code}")


def _to_u8(frame: np.ndarray) -> np.ndarray:
    return np.clip(frame * 255.0, 0, 255).astype(np.uint8)


def _resize(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _label(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.55,
    color: tuple[int, int, int] = TEXT,
    thickness: int = 1,
) -> None:
    cv2.putText(
        canvas,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _panel(
    canvas: np.ndarray,
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    accent: tuple[int, int, int],
) -> None:
    cv2.rectangle(canvas, (x, y), (x + width - 1, y + height - 1), accent, 1)
    resized = _resize(_to_u8(image), width, height)
    canvas[y : y + height, x : x + width] = resized
    cv2.rectangle(canvas, (x, y), (x + width - 1, y + 30), (8, 12, 20), -1)
    cv2.rectangle(canvas, (x, y), (x + width - 1, y + 30), accent, 1)
    _label(canvas, title, (x + 12, y + 21), 0.49, accent, 1)


def _crop_center(image: np.ndarray, center: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    frame_height, frame_width = image.shape[:2]
    center_x = int(round(float(center[0])))
    center_y = int(round(float(center[1])))
    x0 = int(np.clip(center_x - width // 2, 0, frame_width - width))
    y0 = int(np.clip(center_y - height // 2, 0, frame_height - height))
    return image[y0 : y0 + height, x0 : x0 + width]


def render_comparison(
    clip: DemoClip,
    restoration: RestorationResult,
    metrics: dict[str, Any],
    config: LabConfig,
    path: Path,
) -> None:
    canvas_width, canvas_height = 1200, 910
    panel_width, panel_height = 520, 390
    left_x, right_x = 32, 648
    top_y, bottom_y = 72, 496
    crop_width, crop_height = 174, 130
    frames = np.empty((len(clip.source), canvas_height, canvas_width, 3), np.uint8)

    for frame_index in range(len(clip.source)):
        canvas = np.empty((canvas_height, canvas_width, 3), dtype=np.uint8)
        canvas[:] = BG
        cv2.line(canvas, (32, 57), (1168, 57), (31, 43, 58), 1)
        _label(canvas, "LCD PERSISTENCE / SYNCHRONIZED RESTORATION", (32, 31), 0.72, TEXT, 2)
        time_value = frame_index / config.fps
        _label(
            canvas,
            f"FRAME {frame_index:03d}   T+{time_value:05.2f}s",
            (930, 31),
            0.52,
            MUTED,
            1,
        )
        _label(
            canvas,
            f"GHOST  {metrics['ghost']['restored']['rms_8bit']:04.1f} RMS / 255",
            (615, 31),
            0.45,
            LIME,
            1,
        )
        restored_zoom = _crop_center(
            restoration.frames[frame_index],
            clip.zoom_centers[frame_index],
            (crop_width, crop_height),
        )
        _panel(
            canvas,
            clip.captured[frame_index],
            left_x,
            top_y,
            panel_width,
            panel_height,
            "ORIGINAL RECORDING",
            AMBER,
        )
        _panel(
            canvas,
            restoration.frames[frame_index],
            right_x,
            top_y,
            panel_width,
            panel_height,
            "RESTORED",
            CYAN,
        )
        _panel(
            canvas,
            _crop_center(
                clip.captured[frame_index],
                clip.zoom_centers[frame_index],
                (crop_width, crop_height),
            ),
            left_x,
            bottom_y,
            panel_width,
            panel_height,
            "SYNCHRONIZED ZOOM 3x / ORIGINAL",
            AMBER,
        )
        _panel(
            canvas,
            restored_zoom,
            right_x,
            bottom_y,
            panel_width,
            panel_height,
            "SYNCHRONIZED ZOOM 3x / RESTORED",
            CYAN,
        )
        frame_flags = int(
            metrics["diagnostics"]["unstable_frame_pixels"][frame_index]
        )
        sat_flags = int(
            metrics["diagnostics"]["saturated_frame_pixels"][frame_index]
        )
        _label(
            canvas,
            f"UNCERTAIN {frame_flags:5d} px   CLIPPED {sat_flags:5d} px",
            (648, 477),
            0.45,
            MUTED,
            1,
        )
        frames[frame_index] = canvas

    save_video(path, frames.astype(np.float32) / 255.0, config.fps, crf=18)


def _plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#091018",
            "axes.facecolor": "#0e1722",
            "axes.edgecolor": "#304255",
            "axes.labelcolor": "#c7d4df",
            "text.color": "#d9e4ed",
            "xtick.color": "#8799aa",
            "ytick.color": "#8799aa",
            "grid.color": "#243444",
            "grid.alpha": 0.7,
            "font.family": "DejaVu Sans",
        }
    )


def render_calibration_plots(
    calibration: CalibrationResult,
    config: LabConfig,
    out_dir: Path,
) -> None:
    _plot_style()
    colors = {"R": "#ff6b6b", "G": "#72df93", "B": "#62a9ff"}
    fig, axes = plt.subplots(3, 2, figsize=(12.2, 8.2), sharex="col")
    for row, channel in enumerate(config.channels):
        for col, transition in enumerate(("rise", "fall")):
            axis = axes[row, col]
            measured = calibration.mean_step_curves[transition][channel]
            fitted = calibration.curves[transition][channel]
            time_ms = np.arange(1, len(measured) + 1) / config.fps * 1000.0
            time_fit = np.arange(1, len(fitted) + 1) / config.fps * 1000.0
            axis.plot(time_ms, measured, color=colors[channel], lw=2.2, label="measured")
            axis.plot(
                time_fit,
                fitted,
                color="#f4f8fb",
                lw=1.2,
                ls="--",
                label="fitted model",
            )
            tau = (
                calibration.params.tau_rise[row]
                if transition == "rise"
                else calibration.params.tau_fall[row]
            )
            axis.axhline(1.0, color="#516273", lw=0.7, ls=":")
            axis.grid(True, lw=0.6)
            axis.set_ylim(-0.03, 1.05)
            axis.set_title(
                f"{channel} / {transition.upper()}   tau = {tau:.2f} frames "
                f"({tau / config.fps * 1000:.0f} ms)",
                loc="left",
                fontsize=10,
            )
            if row == 2:
                axis.set_xlabel("time after step (ms)")
            if col == 0:
                axis.set_ylabel("normalized response")
            if row == 0 and col == 0:
                axis.legend(frameon=False, loc="lower right", fontsize=8)
    fig.suptitle(
        "Black-white step calibration: measured vs switched first-order fit",
        x=0.06,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "calibration_response.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    for col, transition in enumerate(("rise", "fall")):
        matrix = np.empty((3, 4), dtype=np.float64)
        directions = ("right", "left", "down", "up")
        for channel_index, channel in enumerate(config.channels):
            for direction_index, direction in enumerate(directions):
                fit = next(
                    fit
                    for fit in calibration.fits
                    if fit.scope == "directional"
                    and fit.channel == channel
                    and fit.transition == transition
                    and fit.direction == direction
                )
                matrix[channel_index, direction_index] = fit.tau_frames
        image = axes[col].imshow(matrix, cmap="viridis", aspect="auto")
        axes[col].set_xticks(range(4), [value.upper() for value in directions])
        axes[col].set_yticks(range(3), list(config.channels))
        axes[col].set_title(f"{transition.upper()} tau (frames) by sweep direction", loc="left")
        for row in range(3):
            for direction_index in range(4):
                axes[col].text(
                    direction_index,
                    row,
                    f"{matrix[row, direction_index]:.2f}",
                    ha="center",
                    va="center",
                    color="#f5f8fa" if matrix[row, direction_index] < matrix.max() * 0.88 else "#081018",
                    fontsize=9,
                    fontweight="bold",
                )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.83, bottom=0.27, wspace=0.16)
    color_axis = fig.add_axes((0.38, 0.105, 0.28, 0.032))
    fig.colorbar(
        image,
        cax=color_axis,
        orientation="horizontal",
        label="tau (frames)",
    )
    fig.suptitle(
        "Directional calibration data (synthetic mild panel bias)",
        x=0.06,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(out_dir / "calibration_directions.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_diagnostics(
    clip: DemoClip,
    restoration: RestorationResult,
    metrics: dict[str, Any],
    config: LabConfig,
    out_dir: Path,
) -> None:
    _plot_style()
    sample_frame = min(78, len(clip.source) - 1)
    unstable = restoration.unstable[sample_frame]
    saturated = restoration.saturated[sample_frame]
    overlay = _to_u8(clip.captured[sample_frame]).copy()
    overlay[unstable] = (
        overlay[unstable] * 0.22 + np.array(RED, dtype=np.float32) * 0.78
    ).astype(np.uint8)
    overlay[saturated] = (
        overlay[saturated] * 0.10 + np.array(AMBER, dtype=np.float32) * 0.90
    ).astype(np.uint8)

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.3))
    axes[0, 0].imshow(overlay)
    axes[0, 0].set_title(
        f"Diagnostic overlay / frame {sample_frame}: red=unstable, amber=clipped",
        loc="left",
        fontsize=10,
    )
    axes[0, 0].axis("off")
    axes[0, 0].legend(
        handles=[
            Patch(facecolor="#f4525e", label="unstable inverse"),
            Patch(facecolor="#f5b14a", label="out-of-range estimate"),
        ],
        loc="lower left",
        frameon=True,
        facecolor="#101722",
        edgecolor="#354658",
        fontsize=8,
    )

    unstable_limit = max(
        0.004, float(np.percentile(restoration.unstable_frequency, 99.9))
    )
    unstable_image = axes[0, 1].imshow(
        restoration.unstable_frequency,
        cmap="magma",
        vmin=0.0,
        vmax=unstable_limit,
    )
    axes[0, 1].set_title(
        f"Unstable frequency (gain > {config.instability_gain:.1f})",
        loc="left",
        fontsize=10,
    )
    axes[0, 1].axis("off")
    fig.colorbar(unstable_image, ax=axes[0, 1], fraction=0.046, pad=0.025)

    saturated_limit = max(
        0.004, float(np.percentile(restoration.saturated_frequency, 99.9))
    )
    saturated_image = axes[1, 0].imshow(
        restoration.saturated_frequency,
        cmap="inferno",
        vmin=0.0,
        vmax=saturated_limit,
    )
    axes[1, 0].set_title(
        "Clipped / out-of-range frequency",
        loc="left",
        fontsize=10,
    )
    axes[1, 0].axis("off")
    fig.colorbar(saturated_image, ax=axes[1, 0], fraction=0.046, pad=0.025)

    unstable_counts = np.asarray(metrics["diagnostics"]["unstable_frame_pixels"])
    saturated_counts = np.asarray(metrics["diagnostics"]["saturated_frame_pixels"])
    time_axis = np.arange(len(unstable_counts)) / config.fps
    axes[1, 1].fill_between(
        time_axis, unstable_counts, color="#f4525e", alpha=0.35, label="unstable"
    )
    axes[1, 1].plot(time_axis, unstable_counts, color="#f4525e", lw=1.4)
    axes[1, 1].fill_between(
        time_axis, saturated_counts, color="#f5b14a", alpha=0.35, label="clipped"
    )
    axes[1, 1].plot(time_axis, saturated_counts, color="#f5b14a", lw=1.4)
    axes[1, 1].axvline(72 / config.fps, color="#62a9ff", lw=1.0, ls="--")
    axes[1, 1].text(
        72 / config.fps + 0.08,
        axes[1, 1].get_ylim()[1] * 0.90,
        "checker flip",
        color="#62a9ff",
        fontsize=8,
    )
    axes[1, 1].set_title("Flagged pixels over time", loc="left", fontsize=10)
    axes[1, 1].set_xlabel("time (s)")
    axes[1, 1].set_ylabel("pixels")
    axes[1, 1].grid(True, lw=0.6)
    axes[1, 1].legend(frameon=False, loc="upper right", fontsize=8)

    fig.suptitle(
        "Saturation and inversion-stability diagnosis",
        x=0.055,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "diagnostics.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_metric_chart(metrics: dict[str, Any], out_dir: Path) -> None:
    _plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    captured = metrics["ghost"]["captured"]
    restored = metrics["ghost"]["restored"]
    channels = ("R", "G", "B")
    positions = np.arange(3)
    axes[0].bar(
        positions - 0.19,
        captured["per_channel_rms_8bit"],
        0.38,
        color="#f5b14a",
        label="original",
    )
    axes[0].bar(
        positions + 0.19,
        restored["per_channel_rms_8bit"],
        0.38,
        color="#52dce1",
        label="restored",
    )
    axes[0].set_xticks(positions, channels)
    axes[0].set_ylabel("RMS / 255")
    axes[0].set_title(
        f"Ghost energy on overlapping motion contours  "
        f"({metrics['ghost']['improvement_db']:.1f} dB better)",
        loc="left",
        fontsize=10,
    )
    axes[0].grid(True, axis="y", lw=0.6)
    axes[0].legend(frameon=False)

    capture_over = metrics["overshoot"]["captured"]
    restored_over = metrics["overshoot"]["restored"]
    labels = ("RMS excursion", "mean excursion", "peak excursion")
    captured_values = (
        capture_over["rms_excursion_8bit"],
        capture_over["mean_excursion_8bit"],
        capture_over["peak_excursion_8bit"],
    )
    restored_values = (
        restored_over["rms_excursion_8bit"],
        restored_over["mean_excursion_8bit"],
        restored_over["peak_excursion_8bit"],
    )
    positions = np.arange(3)
    axes[1].bar(
        positions - 0.19, captured_values, 0.38, color="#f5b14a", label="original"
    )
    axes[1].bar(
        positions + 0.19, restored_values, 0.38, color="#52dce1", label="restored"
    )
    axes[1].set_xticks(positions, labels, rotation=12)
    axes[1].set_ylabel("level / 255")
    axes[1].set_title(
        f"Checkerboard post-flip overshoot  "
        f"({metrics['overshoot']['improvement_db']:.1f} dB better)",
        loc="left",
        fontsize=10,
    )
    axes[1].grid(True, axis="y", lw=0.6)
    axes[1].legend(frameon=False)
    fig.suptitle(
        "Measured residual error",
        x=0.055,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / "metrics.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_contact_sheet(
    clip: DemoClip,
    restoration: RestorationResult,
    config: LabConfig,
    out_dir: Path,
) -> None:
    _plot_style()
    sample_frames = [18, 52, 74, 112]
    rows = ("Original", "Restored", "Reference", "Abs. error x3")
    fig, axes = plt.subplots(4, len(sample_frames), figsize=(14.0, 9.0))
    for column, frame_index in enumerate(sample_frames):
        error = np.abs(
            restoration.frames[frame_index] - clip.source[frame_index]
        ).mean(axis=-1)
        images = (
            clip.captured[frame_index],
            restoration.frames[frame_index],
            clip.source[frame_index],
            np.clip(error * 3.0, 0.0, 1.0),
        )
        for row, image in enumerate(images):
            axis = axes[row, column]
            if row == 3:
                axis.imshow(image, cmap="inferno", vmin=0.0, vmax=1.0)
            else:
                axis.imshow(np.clip(image, 0.0, 1.0))
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(
                    f"frame {frame_index} / {frame_index / config.fps:.2f}s",
                    fontsize=9,
                )
            if column == 0:
                axis.set_ylabel(rows[row], fontsize=10, color="#d9e4ed")
    fig.suptitle(
        "Restoration stills: overlap crossing and checkerboard phase transition",
        x=0.045,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.03, 0, 1, 0.95))
    fig.savefig(out_dir / "comparison_stills.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_metadata(
    clip: DemoClip,
    calibration: CalibrationClip,
    config: LabConfig,
    out_dir: Path,
) -> None:
    zoom_track = {
        "fps": config.fps,
        "crop_size": [174, 130],
        "centers": clip.zoom_centers.round(2).tolist(),
    }
    (out_dir / "zoom_track.json").write_text(
        json.dumps(zoom_track, separators=(",", ":")), encoding="utf-8"
    )
    segments = [segment.to_dict() for segment in calibration.segments]
    (out_dir / "calibration_segments.json").write_text(
        json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
    )
