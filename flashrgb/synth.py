from __future__ import annotations

from pathlib import Path

from .imageio import GrayImage, RGBImage, write_pgm, write_ppm
from .sync import CHANNELS, write_json


BACKGROUND_PALETTE = (
    (72, 94, 128),
    (150, 88, 64),
    (68, 132, 92),
    (142, 128, 70),
    (104, 78, 142),
    (154, 150, 142),
    (54, 118, 138),
    (118, 58, 74),
)
TARGET_PALETTE = (
    (238, 44, 48),
    (34, 222, 82),
    (48, 72, 238),
    (238, 216, 44),
    (222, 52, 210),
    (38, 218, 218),
    (242, 118, 32),
    (214, 222, 238),
)
DEFAULT_MISSING = {3: {"G"}, 6: {"G", "B"}, 8: {"R"}, 10: {"R", "B"}}
FLASH_ROI = (0, 0, 96, 32)


def _fill(image: RGBImage, x: int, y: int, width: int, height: int, color) -> None:
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(image.width, x + width)
    y1 = min(image.height, y + height)
    if x0 >= x1 or y0 >= y1:
        return
    row = bytes(color) * (x1 - x0)
    for yy in range(y0, y1):
        start = (yy * image.width + x0) * 3
        image.pixels[start : start + len(row)] = row


def _material_texture(
    image: RGBImage,
    x: int,
    y: int,
    width: int,
    height: int,
    seed: int,
    amplitude: int = 20,
) -> None:
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(image.width, x + width)
    y1 = min(image.height, y + height)
    for yy in range(y0, y1):
        for xx in range(x0, x1):
            value = (
                (xx * 374761393)
                ^ (yy * 668265263)
                ^ ((seed + x) * 2246822519)
                ^ ((seed + y) * 3266489917)
            ) & 0xFFFFFFFF
            value = ((value ^ (value >> 13)) * 1274126177) & 0xFFFFFFFF
            delta = ((value >> 8) % (2 * amplitude + 1)) - amplitude
            offset = (yy * image.width + xx) * 3
            for channel in range(3):
                image.pixels[offset + channel] = max(
                    0, min(255, image.pixels[offset + channel] + delta)
                )


def _background(width: int, height: int) -> RGBImage:
    image = bytearray(width * height * 3)
    for y in range(height):
        for x in range(width):
            cell_x = x // 24
            cell_y = y // 24
            hashed = (cell_x * 73856093) ^ (cell_y * 19349663)
            color = BACKGROUND_PALETTE[hashed % len(BACKGROUND_PALETTE)]
            texture = 10 if ((x // 6) + (y // 6)) % 2 else -10
            color = tuple(max(0, min(255, value + texture)) for value in color)
            offset = (y * width + x) * 3
            image[offset : offset + 3] = bytes(color)
    result = RGBImage(width, height, image)
    _fill(result, 288, 0, 16, height, (36, 36, 40))
    _fill(result, 290, 0, 3, height, (92, 96, 104))
    _material_texture(result, 0, 0, width, height, seed=11)
    return result


def _draw_target(image: RGBImage, x: int, y: int) -> None:
    width = 72
    height = 60
    _fill(image, x, y, width, height, (18, 18, 20))
    inner_x = x + 3
    inner_y = y + 3
    inner_width = width - 6
    inner_height = height - 6
    columns = 6
    rows = 5
    cell_width = inner_width // columns
    cell_height = inner_height // rows
    for row in range(rows):
        for column in range(columns):
            color = TARGET_PALETTE[(row * 5 + column * 3 + row * column) % len(TARGET_PALETTE)]
            _fill(
                image,
                inner_x + column * cell_width,
                inner_y + row * cell_height,
                cell_width + (1 if column == columns - 1 else 0),
                cell_height + (1 if row == rows - 1 else 0),
                color,
            )
    _fill(image, inner_x + 4, inner_y + 4, 8, inner_height - 8, (244, 244, 240))
    _fill(image, inner_x + inner_width - 12, inner_y + 4, 8, inner_height - 8, (12, 12, 14))
    _material_texture(image, x, y, width, height, seed=101)


def _draw_occluder(image: RGBImage, box: tuple[int, int, int, int]) -> None:
    x, y, width, height = box
    _fill(image, x, y, width, height, (48, 42, 38))
    for stripe in range(3, width, 7):
        _fill(image, x + stripe, y, 2, height, (86, 78, 70))
    _material_texture(image, x, y, width, height, seed=211)


def group_geometry(group_id: int) -> dict:
    base_x = 28 + (group_id * 17) % 148
    base_y = 78 + (group_id * 11) % 46
    red_to_green = (7, 2)
    green_to_blue = (7, 2)
    positions = {
        "R": (base_x, base_y),
        "G": (base_x + red_to_green[0], base_y + red_to_green[1]),
        "B": (
            base_x + red_to_green[0] + green_to_blue[0],
            base_y + red_to_green[1] + green_to_blue[1],
        ),
    }
    occluders: dict[str, list[tuple[int, int, int, int]]] = {"R": [], "G": [], "B": []}
    blue_x, blue_y = positions["B"]
    occluders["B"].append((blue_x + 31, blue_y + 12, 24, 30))
    if group_id % 3 == 1:
        green_x, green_y = positions["G"]
        occluders["G"].append((green_x - 17, green_y + 24, 17, 20))
    return {
        "positions": positions,
        "occluders": occluders,
        "known_motion": {"R_to_G": red_to_green, "B_to_G": (-7, -2)},
    }


def _flash(gray: bytearray, width: int, height: int, channel: str) -> None:
    x, y, roi_width, roi_height = FLASH_ROI
    for yy in range(y, min(height, y + roi_height)):
        start = yy * width + x
        gray[start : start + roi_width] = b"\x04" * roi_width
    cell = CHANNELS.index(channel)
    x0 = min(width, x + cell * (roi_width // 3) + 2)
    x1 = min(width, x + (cell + 1) * (roi_width // 3) - 2)
    for yy in range(y + 2, min(height - 1, y + roi_height - 2)):
        start = yy * width + x0
        gray[start : start + (x1 - x0)] = b"\xf4" * (x1 - x0)


def _noise(x: int, y: int, channel: str, group_id: int) -> int:
    value = (x * 374761393 + y * 668265263 + group_id * 2246822519 + ord(channel) * 3266489917) & 0xFFFFFFFF
    value = (value ^ (value >> 13)) * 1274126177 & 0xFFFFFFFF
    return (value & 7) - 3


def _exposure(
    reflectance: RGBImage,
    channel: str,
    gain: float,
    group_id: int,
) -> GrayImage:
    width = reflectance.width
    height = reflectance.height
    red_index = CHANNELS.index(channel)
    output = bytearray(width * height)
    center_x = width / 2
    center_y = height / 2
    max_distance = center_x * center_x + center_y * center_y
    for y in range(height):
        for x in range(width):
            index = y * width + x
            source = index * 3
            distance = (x - center_x) ** 2 + (y - center_y) ** 2
            vignette = 0.92 + 0.08 * (1.0 - distance / max_distance)
            value = int(
                reflectance.pixels[source + red_index] * gain * vignette
                + _noise(x, y, channel, group_id)
                + 0.5
            )
            output[index] = max(0, min(255, value))
    _flash(output, width, height, channel)
    return GrayImage(width, height, output)


def _edge_mask(
    truth: RGBImage, region: tuple[int, int, int, int]
) -> GrayImage:
    width = truth.width
    height = truth.height
    mask = bytearray(width * height)
    x0 = max(1, region[0])
    y0 = max(1, region[1])
    x1 = min(width - 1, region[0] + region[2])
    y1 = min(height - 1, region[1] + region[3])
    for y in range(y0, y1):
        for x in range(x0, x1):
            index = (y * width + x) * 3
            score = 0
            if x + 1 < width:
                score = max(
                    score,
                    sum(
                        abs(truth.pixels[index + c] - truth.pixels[index + 3 + c])
                        for c in range(3)
                    ),
                )
            if y + 1 < height:
                below = index + width * 3
                score = max(
                    score,
                    sum(
                        abs(truth.pixels[index + c] - truth.pixels[below + c])
                        for c in range(3)
                    ),
                )
            if score >= 55:
                mask[y * width + x] = 255
    dilated = bytearray(mask)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            index = y * width + x
            if mask[index] == 255:
                dilated[index - 1] = 255
                dilated[index + 1] = 255
                dilated[index - width] = 255
                dilated[index + width] = 255
    return GrayImage(width, height, dilated)


def _target_mask(
    width: int, height: int, region: tuple[int, int, int, int]
) -> GrayImage:
    mask = bytearray(width * height)
    x0 = max(0, region[0])
    y0 = max(0, region[1])
    x1 = min(width, region[0] + region[2])
    y1 = min(height, region[1] + region[3])
    for y in range(y0, y1):
        start = y * width + x0
        mask[start : start + (x1 - x0)] = b"\xff" * (x1 - x0)
    return GrayImage(width, height, mask)


def _visible_and_occlusion_masks(
    width: int,
    height: int,
    geometry: dict,
) -> tuple[GrayImage, GrayImage, list[tuple[int, int, int, int]]]:
    reference_x, reference_y = geometry["positions"]["G"]
    masks: list[tuple[int, int, int, int]] = []
    for channel in CHANNELS:
        source_x, source_y = geometry["positions"][channel]
        motion_x = reference_x - source_x
        motion_y = reference_y - source_y
        for box in geometry["occluders"][channel]:
            masks.append(
                (
                    box[0] + motion_x,
                    box[1] + motion_y,
                    box[2],
                    box[3],
                )
            )
    occlusion = bytearray(width * height)
    visible = bytearray((255,) * (width * height))
    for x, y, box_width, box_height in masks:
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(width, x + box_width)
        y1 = min(height, y + box_height)
        for yy in range(y0, y1):
            start = yy * width + x0
            occlusion[start : start + (x1 - x0)] = b"\xff" * (x1 - x0)
            visible[start : start + (x1 - x0)] = b"\x00" * (x1 - x0)
    return (
        GrayImage(width, height, visible),
        GrayImage(width, height, occlusion),
        masks,
    )


def generate_dataset(
    output_dir: str | Path,
    group_count: int = 12,
    width: int = 320,
    height: int = 240,
    missing: dict[int, set[str]] | None = None,
) -> dict:
    root = Path(output_dir)
    if width < FLASH_ROI[2] or height < FLASH_ROI[3]:
        raise ValueError("synthetic frames must be at least 96x32 pixels")
    frames_dir = root / "frames"
    truth_dir = root / "truth"
    frames_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frames_dir.glob("frame_*.pgm"):
        old_frame.unlink()
    for old_truth in truth_dir.glob("*"):
        if old_truth.is_file():
            old_truth.unlink()

    missing_map = DEFAULT_MISSING if missing is None else missing
    for group_id in missing_map:
        if len(missing_map[group_id]) >= 3:
            raise ValueError(f"group {group_id} would have no exposures")

    background = _background(width, height)
    frame_counter = 0
    truth_groups: list[dict] = []
    for group_id in range(group_count):
        geometry = group_geometry(group_id)
        missing_channels = missing_map.get(group_id, set())
        files: dict[str, str] = {}
        for channel in CHANNELS:
            if channel in missing_channels:
                continue
            reflectance = RGBImage(width, height, background.pixels)
            target_x, target_y = geometry["positions"][channel]
            _draw_target(reflectance, target_x, target_y)
            for box in geometry["occluders"][channel]:
                _draw_occluder(reflectance, box)
            gain = {"R": 0.92, "G": 1.0, "B": 0.84}[channel]
            frame = _exposure(reflectance, channel, gain, group_id)
            filename = f"frame_{frame_counter:04d}.pgm"
            write_pgm(frames_dir / filename, frame)
            files[channel] = filename
            frame_counter += 1

        reference = RGBImage(width, height, background.pixels)
        reference_x, reference_y = geometry["positions"]["G"]
        _draw_target(reference, reference_x, reference_y)
        truth_name = f"truth_{group_id:03d}.ppm"
        visible_name = f"visible_{group_id:03d}.pgm"
        occlusion_name = f"occlusion_{group_id:03d}.pgm"
        edge_name = f"edge_{group_id:03d}.pgm"
        target_name = f"target_{group_id:03d}.pgm"
        write_ppm(truth_dir / truth_name, reference)
        visible, occlusion, projected_boxes = _visible_and_occlusion_masks(
            width, height, geometry
        )
        write_pgm(truth_dir / visible_name, visible)
        write_pgm(truth_dir / occlusion_name, occlusion)
        target_region = (reference_x - 2, reference_y - 2, 76, 64)
        write_pgm(truth_dir / edge_name, _edge_mask(reference, target_region))
        write_pgm(
            truth_dir / target_name,
            _target_mask(width, height, target_region),
        )
        missing_sorted = [channel for channel in CHANNELS if channel in missing_channels]
        status = "complete" if not missing_sorted else (
            f"missing_{len(missing_sorted)}:" + "".join(missing_sorted)
        )
        truth_groups.append(
            {
                "group_id": group_id,
                "status": status,
                "missing": missing_sorted,
                "frames": files,
                "reference_channel": "G",
                "truth_image": f"truth/{truth_name}",
                "visible_mask": f"truth/{visible_name}",
                "occlusion_mask": f"truth/{occlusion_name}",
                "edge_mask": f"truth/{edge_name}",
                "target_mask": f"truth/{target_name}",
                "target_positions": {
                    channel: list(position)
                    for channel, position in geometry["positions"].items()
                },
                "known_motion": geometry["known_motion"],
                "occluders_projected_to_reference": [list(box) for box in projected_boxes],
            }
        )

    metadata = {
        "schema_version": 1,
        "width": width,
        "height": height,
        "flash_roi": list(FLASH_ROI),
        "flash_encoding": "left=R, center=G, right=B",
        "illumination_gains": {"R": 0.92, "G": 1.0, "B": 0.84},
        "groups": truth_groups,
    }
    write_json(root / "truth.json", metadata)
    manifest_lines = [
        "file,group_id,channel,expected_status",
    ]
    for group in truth_groups:
        for channel, filename in group["frames"].items():
            manifest_lines.append(
                f"frames/{filename},{group['group_id']},{channel},{group['status']}"
            )
    (root / "manifest.csv").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    return metadata
