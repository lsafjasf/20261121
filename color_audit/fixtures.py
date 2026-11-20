from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .ffmpeg import Tools, run_capture
from .mp4 import patch_mp4_colr, read_mp4_container_tags

WIDTH = 640
HEIGHT = 360
FPS = 12
FRAMES = 24
SOURCE = {"primaries": 1, "transfer": 1, "matrix": 1, "range": 0}
MISTAG = {"primaries": 9, "transfer": 16, "matrix": 9, "range": 1}

GRAY_LEVELS = (16, 32, 64, 96, 128, 160, 192, 224, 235)
COLOR_PATCHES = (
    ("red", (63, 102, 240)),
    ("green", (173, 42, 26)),
    ("blue", (32, 240, 118)),
    ("cyan", (188, 154, 16)),
    ("magenta", (78, 94, 230)),
    ("yellow", (219, 16, 138)),
    ("black", (16, 128, 128)),
    ("white", (235, 128, 128)),
)


def _rect(x: int, y: int, width: int, height: int) -> dict[str, int]:
    return {"x": x, "y": y, "width": width, "height": height}


def chart_definition() -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    bars: list[dict[str, Any]] = []
    gray_width = WIDTH // len(GRAY_LEVELS)
    for index, value in enumerate(GRAY_LEVELS):
        x = index * gray_width
        bars.append(
            {
                "kind": "grayscale",
                "label": f"gray_{value}",
                "value": value,
                "rect": _rect(x, 0, gray_width, HEIGHT // 2),
            }
        )
        samples.append(
            {
                "id": f"gray_{value}",
                "kind": "grayscale",
                "x": x + gray_width // 2,
                "y": HEIGHT // 4,
                "expected_yuv": {"y": value, "u": 128, "v": 128},
            }
        )

    color_width = WIDTH // len(COLOR_PATCHES)
    for index, (label, yuv) in enumerate(COLOR_PATCHES):
        x = index * color_width
        bars.append(
            {
                "kind": "color",
                "label": label,
                "yuv": {"y": yuv[0], "u": yuv[1], "v": yuv[2]},
                "rect": _rect(x, HEIGHT // 2, color_width, HEIGHT // 2),
            }
        )
        samples.append(
            {
                "id": label,
                "kind": "color" if label not in {"black", "white"} else "neutral",
                "x": x + color_width // 2,
                "y": HEIGHT * 3 // 4,
                "expected_yuv": {"y": yuv[0], "u": yuv[1], "v": yuv[2]},
            }
        )
    return {
        "schema": "ffmpeg-color-audit.chart/v1",
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FPS,
        "frames": FRAMES,
        "pix_fmt": "yuv420p",
        "source_tags": SOURCE,
        "bars": bars,
        "samples": samples,
    }


def _render_yuv420_frame(definition: dict[str, Any]) -> bytes:
    width = definition["width"]
    height = definition["height"]
    chroma_width = width // 2
    chroma_height = height // 2
    y_plane = bytearray([16]) * (width * height)
    u_plane = bytearray([128]) * (chroma_width * chroma_height)
    v_plane = bytearray([128]) * (chroma_width * chroma_height)
    for bar in definition["bars"]:
        rect = bar["rect"]
        x0 = rect["x"]
        y0 = rect["y"]
        x1 = x0 + rect["width"]
        y1 = y0 + rect["height"]
        if bar["kind"] == "grayscale":
            value = bytes([bar["value"]])
            for y in range(y0, y1):
                start = y * width + x0
                y_plane[start : start + rect["width"]] = value * rect["width"]
            continue
        yuv = bar["yuv"]
        for y in range(y0, y1):
            start = y * width + x0
            y_plane[start : start + rect["width"]] = bytes([yuv["y"]]) * rect["width"]
        for y in range(y0 // 2, y1 // 2):
            start = y * chroma_width + x0 // 2
            count = rect["width"] // 2
            u_plane[start : start + count] = bytes([yuv["u"]]) * count
            v_plane[start : start + count] = bytes([yuv["v"]]) * count
    return bytes(y_plane + u_plane + v_plane)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_fixtures(tools: Tools, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    definition = chart_definition()
    frame = _render_yuv420_frame(definition)
    reference = output / "rec709_limited_reference.mp4"
    mistagged = output / "rec709_mistagged_container.mp4"
    command = [
        tools.ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "pipe:0",
        "-frames:v",
        str(FRAMES),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-qp",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-colorspace",
        "bt709",
        "-color_range",
        "tv",
        "-movflags",
        "+faststart",
        str(reference),
    ]
    completed = run_capture(command, input_data=frame * FRAMES, timeout=180)
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"fixture encoding failed: {detail}")
    patch_mp4_colr(reference, mistagged, MISTAG)

    preview = output / "test-card.png"
    preview_cmd = [
        tools.ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(reference),
        "-frames:v",
        "1",
        "-vf",
        "scale=in_range=limited:out_range=limited,format=rgb24",
        str(preview),
    ]
    preview_completed = run_capture(preview_cmd, timeout=60)
    if preview_completed.returncode != 0:
        preview = None

    manifest = {
        **definition,
        "assets": {
            "reference": reference.name,
            "mistagged_container": mistagged.name,
            "preview": preview.name if preview else None,
        },
        "expected": {
            "container": MISTAG,
            "codec": SOURCE,
            "truth": "codec",
        },
        "files": {},
        "generator": {
            "command": command,
            "raw_frame_sha256": hashlib.sha256(frame).hexdigest(),
            "mistag_patch": {
                "box": "colr/nclx",
                "primaries": MISTAG["primaries"],
                "transfer": MISTAG["transfer"],
                "matrix": MISTAG["matrix"],
                "range": MISTAG["range"],
            },
        },
    }
    for path in (reference, mistagged):
        manifest["files"][path.name] = {
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        layer = read_mp4_container_tags(path)
        manifest["files"][path.name]["container_tags"] = layer.to_dict()["values"]
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest
