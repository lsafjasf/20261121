from __future__ import annotations

import bisect
import math
from collections import Counter
from dataclasses import dataclass

from .imageio import GrayImage


@dataclass
class TileMotion:
    cx: int
    cy: int
    dx: int
    dy: int
    valid: bool
    score: int
    zero_score: int

    @property
    def magnitude(self) -> float:
        return math.hypot(self.dx, self.dy)


@dataclass
class MotionField:
    width: int
    height: int
    columns: int
    rows: int
    centers_x: list[int]
    centers_y: list[int]
    vectors: list[tuple[float, float]]
    tiles: list[TileMotion]
    source_channel: str
    reference_channel: str
    rigid_mask: GrayImage | None = None
    rigid_dx: int = 0
    rigid_dy: int = 0

    def at(self, x: float, y: float) -> tuple[float, float]:
        if not self.centers_x or not self.centers_y:
            return (0.0, 0.0)
        x = min(max(x, self.centers_x[0]), self.centers_x[-1])
        y = min(max(y, self.centers_y[0]), self.centers_y[-1])
        ix = bisect.bisect_right(self.centers_x, x) - 1
        iy = bisect.bisect_right(self.centers_y, y) - 1
        ix = min(max(ix, 0), self.columns - 1)
        iy = min(max(iy, 0), self.rows - 1)
        ix1 = min(ix + 1, self.columns - 1)
        iy1 = min(iy + 1, self.rows - 1)
        fx = 0.0 if ix1 == ix else (x - self.centers_x[ix]) / (
            self.centers_x[ix1] - self.centers_x[ix]
        )
        fy = 0.0 if iy1 == iy else (y - self.centers_y[iy]) / (
            self.centers_y[iy1] - self.centers_y[iy]
        )
        x0y0 = self.vectors[iy * self.columns + ix]
        x1y0 = self.vectors[iy * self.columns + ix1]
        x0y1 = self.vectors[iy1 * self.columns + ix]
        x1y1 = self.vectors[iy1 * self.columns + ix1]
        dx = (
            x0y0[0] * (1 - fx) * (1 - fy)
            + x1y0[0] * fx * (1 - fy)
            + x0y1[0] * (1 - fx) * fy
            + x1y1[0] * fx * fy
        )
        dy = (
            x0y0[1] * (1 - fx) * (1 - fy)
            + x1y0[1] * fx * (1 - fy)
            + x0y1[1] * (1 - fx) * fy
            + x1y1[1] * fx * fy
        )
        return dx, dy

    @property
    def valid_ratio(self) -> float:
        return sum(tile.valid for tile in self.tiles) / max(1, len(self.tiles))

    @property
    def mean_motion(self) -> float:
        moving = [tile.magnitude for tile in self.tiles if tile.valid]
        return sum(moving) / max(1, len(moving))

    def to_dict(self) -> dict:
        moving = [tile for tile in self.tiles if tile.valid and (tile.dx or tile.dy)]
        payload = {
            "source_channel": self.source_channel,
            "reference_channel": self.reference_channel,
            "tile_columns": self.columns,
            "tile_rows": self.rows,
            "valid_tile_ratio": round(self.valid_ratio, 4),
            "mean_motion_px": round(self.mean_motion, 3),
            "moving_tile_count": len(moving),
            "max_motion_px": round(
                max((tile.magnitude for tile in moving), default=0.0), 3
            ),
        }
        if self.rigid_mask is not None:
            payload["refined_target_motion_px"] = [self.rigid_dx, self.rigid_dy]
            payload["refined_foreground_pixel_ratio"] = round(
                sum(value > 0 for value in self.rigid_mask.pixels)
                / (self.rigid_mask.width * self.rigid_mask.height),
                4,
            )
        return payload


def _threshold_from_histogram(histogram: list[int], fraction: float) -> int:
    total = sum(histogram)
    target = max(1, int(total * fraction))
    running = 0
    for value, count in enumerate(histogram):
        running += count
        if running >= target:
            return value
    return len(histogram) - 1


def _edge_bits(
    image: GrayImage,
    roi: tuple[int, int, int, int] | None,
    quantile: float = 0.72,
) -> tuple[list[int], int]:
    width = image.width
    height = image.height
    pixels = image.pixels
    magnitudes = [0] * (width * height)
    histogram = [0] * 1024
    for y in range(1, height - 1):
        row = y * width
        for x in range(1, width - 1):
            index = row + x
            magnitude = abs(pixels[index + 1] - pixels[index - 1]) + abs(
                pixels[index + width] - pixels[index - width]
            )
            magnitudes[index] = magnitude
            histogram[magnitude] += 1
    threshold = max(5, _threshold_from_histogram(histogram, quantile))
    bits = [1 if value >= threshold else 0 for value in magnitudes]
    if roi is not None:
        x, y, roi_width, roi_height = roi
        x0 = max(0, x)
        x1 = min(width, x + roi_width)
        y1 = min(height, y + roi_height)
        for yy in range(max(0, y), y1):
            start = yy * width + x0
            bits[start : start + (x1 - x0)] = [0] * (x1 - x0)
    return bits, threshold


def _signature_grid(
    bits: list[int], width: int, height: int, block: int
) -> tuple[list[int], int, int]:
    starts_x = width - block + 1
    starts_y = height - block + 1
    if starts_x <= 0 or starts_y <= 0:
        return [], 0, 0
    row_windows = [0] * (height * starts_x)
    for y in range(height):
        base = y * width
        for x in range(starts_x):
            value = 0
            for k in range(block):
                value |= bits[base + x + k] << k
            row_windows[y * starts_x + x] = value
    signatures = [0] * (starts_x * starts_y)
    for y in range(starts_y):
        for x in range(starts_x):
            value = 0
            for j in range(block):
                value |= row_windows[(y + j) * starts_x + x] << (block * j)
            signatures[y * starts_x + x] = value
    return signatures, starts_x, starts_y


def _median(values: list[int]) -> int:
    ordered = sorted(values)
    return ordered[len(ordered) // 2] if ordered else 0


def _refine_target_mask(
    source_bits: list[int],
    reference_bits: list[int],
    width: int,
    height: int,
    dx: int,
    dy: int,
) -> GrayImage | None:
    block = 8
    stride = 4
    source_sig, sig_width, sig_height = _signature_grid(
        source_bits, width, height, block
    )
    reference_sig, _, _ = _signature_grid(reference_bits, width, height, block)
    if not source_sig:
        return None
    block_bits = block * block
    mask = bytearray(width * height)
    for ty in range(block, sig_height - block, stride):
        for tx in range(block, sig_width - block, stride):
            reference_x = tx + dx
            reference_y = ty + dy
            if not (0 <= reference_x < sig_width and 0 <= reference_y < sig_height):
                continue
            source_value = source_sig[ty * sig_width + tx]
            zero_distance = (
                source_value ^ reference_sig[ty * sig_width + tx]
            ).bit_count()
            shifted_distance = (
                source_value ^ reference_sig[reference_y * sig_width + reference_x]
            ).bit_count()
            if (
                shifted_distance <= int(block_bits * 0.35)
                and shifted_distance + max(4, block_bits // 12) < zero_distance
            ):
                x1 = min(width, tx + block)
                y1 = min(height, ty + block)
                for yy in range(ty, y1):
                    start = yy * width + tx
                    mask[start : start + (x1 - tx)] = b"\x01" * (x1 - tx)
    if not any(mask):
        return None

    for _ in range(2):
        expanded = bytearray(mask)
        for y in range(1, height - 1):
            row = y * width
            for x in range(1, width - 1):
                index = row + x
                if mask[index]:
                    continue
                if (
                    mask[index - 1]
                    or mask[index + 1]
                    or mask[index - width]
                    or mask[index + width]
                ):
                    expanded[index] = 1
        mask = expanded

    eroded = bytearray(mask)
    for y in range(1, height - 1):
        row = y * width
        for x in range(1, width - 1):
            index = row + x
            if mask[index] and not (
                mask[index - 1]
                and mask[index + 1]
                and mask[index - width]
                and mask[index + width]
            ):
                eroded[index] = 0
    count = sum(eroded)
    if count < 32 or count > width * height * 0.3:
        return None
    return GrayImage(
        width,
        height,
        bytes(255 if value else 0 for value in eroded),
    )


def estimate_motion(
    source: GrayImage,
    reference: GrayImage,
    source_channel: str,
    reference_channel: str,
    search_radius: int = 16,
    roi: tuple[int, int, int, int] | None = None,
) -> MotionField:
    if (source.width, source.height) != (reference.width, reference.height):
        raise ValueError("motion estimation requires equal frame sizes")
    block = 12
    stride = 8
    width = source.width
    height = source.height
    source_bits, _ = _edge_bits(source, roi)
    reference_bits, _ = _edge_bits(reference, roi)
    source_sig, sig_width, sig_height = _signature_grid(source_bits, width, height, block)
    reference_sig, _, _ = _signature_grid(reference_bits, width, height, block)
    if not source_sig:
        return MotionField(
            source.width,
            source.height,
            0,
            0,
            [],
            [],
            [],
            [],
            source_channel,
            reference_channel,
        )

    radius = max(1, int(round(search_radius)))
    usable_width = sig_width - 2 * radius
    usable_height = sig_height - 2 * radius
    if usable_width <= 0 or usable_height <= 0:
        radius = 1
        usable_width = max(1, min(sig_width, width - block + 1))
        usable_height = max(1, min(sig_height, height - block + 1))
    tile_starts_x = list(range(radius, sig_width - radius, stride))
    tile_starts_y = list(range(radius, sig_height - radius, stride))
    if not tile_starts_x:
        tile_starts_x = [0]
    if not tile_starts_y:
        tile_starts_y = [0]

    block_bits = block * block
    accept_limit = int(block_bits * 0.36)
    margin = max(5, block_bits // 20)
    tiles: list[TileMotion] = []
    for ty in tile_starts_y:
        for tx in tile_starts_x:
            source_value = source_sig[ty * sig_width + tx]
            zero_value = reference_sig[ty * sig_width + tx]
            zero_distance = (source_value ^ zero_value).bit_count()
            best_dx = 0
            best_dy = 0
            best_distance = zero_distance
            for dy in range(-radius, radius + 1):
                row = (ty + dy) * sig_width
                for dx in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    distance = (source_value ^ reference_sig[row + tx + dx]).bit_count()
                    if distance < best_distance:
                        best_distance = distance
                        best_dx = dx
                        best_dy = dy
            if best_distance + margin >= zero_distance:
                best_dx = 0
                best_dy = 0
            if best_dx == 0 and best_dy == 0:
                valid = zero_distance <= int(block_bits * 0.55)
            else:
                valid = best_distance <= accept_limit
            if not valid:
                best_dx = 0
                best_dy = 0
            tiles.append(
                TileMotion(
                    cx=tx + block // 2,
                    cy=ty + block // 2,
                    dx=best_dx,
                    dy=best_dy,
                    valid=valid,
                    score=best_distance,
                    zero_score=zero_distance,
                )
            )

    columns = len(tile_starts_x)
    rows = len(tile_starts_y)
    for row in range(rows):
        for column in range(columns):
            tile = tiles[row * columns + column]
            if not tile.valid or (tile.dx == 0 and tile.dy == 0):
                continue
            supporters = 0
            for oy in range(-2, 3):
                for ox in range(-2, 3):
                    if ox == 0 and oy == 0:
                        continue
                    yy = row + oy
                    xx = column + ox
                    if not (0 <= yy < rows and 0 <= xx < columns):
                        continue
                    neighbor = tiles[yy * columns + xx]
                    if (
                        neighbor.valid
                        and math.hypot(neighbor.dx - tile.dx, neighbor.dy - tile.dy)
                        <= 2.5
                    ):
                        supporters += 1
            if supporters < 2:
                tile.valid = False
                tile.dx = 0
                tile.dy = 0

    rigid_mask = None
    rigid_dx = 0
    rigid_dy = 0
    nonzero = [
        (tile.dx, tile.dy)
        for tile in tiles
        if tile.valid and (tile.dx != 0 or tile.dy != 0)
    ]
    if len(nonzero) >= 4:
        candidate, support = Counter(nonzero).most_common(1)[0]
        if support >= 3:
            rigid_mask = _refine_target_mask(
                source_bits,
                reference_bits,
                width,
                height,
                candidate[0],
                candidate[1],
            )
            if rigid_mask is not None:
                rigid_dx, rigid_dy = candidate

    vectors: list[tuple[float, float]] = []
    for row in range(rows):
        for column in range(columns):
            tile = tiles[row * columns + column]
            if tile.valid:
                vectors.append((float(tile.dx), float(tile.dy)))
                continue
            neighbor_dx: list[int] = []
            neighbor_dy: list[int] = []
            for oy in (-1, 0, 1):
                for ox in (-1, 0, 1):
                    if ox == 0 and oy == 0:
                        continue
                    yy = row + oy
                    xx = column + ox
                    if 0 <= yy < rows and 0 <= xx < columns:
                        neighbor = tiles[yy * columns + xx]
                        if neighbor.valid:
                            neighbor_dx.append(neighbor.dx)
                            neighbor_dy.append(neighbor.dy)
            vectors.append((float(_median(neighbor_dx)), float(_median(neighbor_dy))))

    centers_x = [tx + block // 2 for tx in tile_starts_x]
    centers_y = [ty + block // 2 for ty in tile_starts_y]
    return MotionField(
        source.width,
        source.height,
        columns,
        rows,
        centers_x,
        centers_y,
        vectors,
        tiles,
        source_channel,
        reference_channel,
        rigid_mask,
        rigid_dx,
        rigid_dy,
    )


def warp_gray(source: GrayImage, field: MotionField) -> GrayImage:
    width = source.width
    height = source.height
    output = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            if field.rigid_mask is not None and field.rigid_mask.pixels[y * width + x]:
                dx, dy = float(field.rigid_dx), float(field.rigid_dy)
            else:
                dx, dy = field.at(x, y)
            sx = x - dx
            sy = y - dy
            if sx < 0 or sy < 0 or sx > width - 1 or sy > height - 1:
                continue
            x0 = int(sx)
            y0 = int(sy)
            x1 = min(width - 1, x0 + 1)
            y1 = min(height - 1, y0 + 1)
            fx = sx - x0
            fy = sy - y0
            p00 = source.pixels[y0 * width + x0]
            p10 = source.pixels[y0 * width + x1]
            p01 = source.pixels[y1 * width + x0]
            p11 = source.pixels[y1 * width + x1]
            top = p00 * (1 - fx) + p10 * fx
            bottom = p01 * (1 - fx) + p11 * fx
            output[y * width + x] = int(top * (1 - fy) + bottom * fy + 0.5)
    return GrayImage(width, height, output)


def zero_field(width: int, height: int, channel: str, reference: str) -> MotionField:
    return MotionField(
        width,
        height,
        0,
        0,
        [],
        [],
        [],
        [],
        channel,
        reference,
    )
