from __future__ import annotations

from .imageio import RGBImage


_FONT = {
    " ": ("000", "000", "000", "000", "000"),
    "A": ("010", "101", "111", "101", "101"),
    "B": ("110", "101", "110", "101", "110"),
    "C": ("011", "100", "100", "100", "011"),
    "D": ("110", "101", "101", "101", "110"),
    "E": ("111", "100", "110", "100", "111"),
    "F": ("111", "100", "110", "100", "100"),
    "G": ("011", "100", "101", "101", "011"),
    "H": ("101", "101", "111", "101", "101"),
    "I": ("111", "010", "010", "010", "111"),
    "J": ("001", "001", "001", "101", "010"),
    "K": ("101", "101", "110", "101", "101"),
    "L": ("100", "100", "100", "100", "111"),
    "M": ("101", "111", "111", "101", "101"),
    "N": ("101", "111", "111", "111", "101"),
    "O": ("010", "101", "101", "101", "010"),
    "P": ("110", "101", "110", "100", "100"),
    "Q": ("010", "101", "101", "111", "011"),
    "R": ("110", "101", "110", "101", "101"),
    "S": ("011", "100", "010", "001", "110"),
    "T": ("111", "010", "010", "010", "010"),
    "U": ("101", "101", "101", "101", "111"),
    "V": ("101", "101", "101", "101", "010"),
    "W": ("101", "101", "111", "111", "101"),
    "X": ("101", "101", "010", "101", "101"),
    "Y": ("101", "101", "010", "010", "010"),
    "Z": ("111", "001", "010", "100", "111"),
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("110", "001", "010", "100", "111"),
    "3": ("110", "001", "010", "001", "110"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "110", "001", "110"),
    "6": ("011", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "110"),
    "-": ("000", "000", "111", "000", "000"),
    "_": ("000", "000", "000", "000", "111"),
    ":": ("000", "010", "000", "010", "000"),
    ".": ("000", "000", "000", "000", "010"),
    "/": ("001", "001", "010", "100", "100"),
    "+": ("000", "010", "111", "010", "000"),
    "?": ("110", "001", "010", "000", "010"),
}


def fill_rect(
    image: RGBImage,
    x: int,
    y: int,
    width: int,
    height: int,
    color: tuple[int, int, int],
) -> None:
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


def draw_border(
    image: RGBImage,
    x: int,
    y: int,
    width: int,
    height: int,
    thickness: int,
    color: tuple[int, int, int],
) -> None:
    fill_rect(image, x, y, width, thickness, color)
    fill_rect(image, x, y + height - thickness, width, thickness, color)
    fill_rect(image, x, y, thickness, height, color)
    fill_rect(image, x + width - thickness, y, thickness, height, color)


def text_width(text: str, scale: int = 1) -> int:
    return max(0, len(text) * 4 * scale - scale)


def draw_text(
    image: RGBImage,
    x: int,
    y: int,
    text: str,
    color: tuple[int, int, int] = (255, 255, 255),
    scale: int = 1,
    background: tuple[int, int, int] | None = None,
) -> None:
    text = text.upper()
    if background is not None:
        fill_rect(image, x, y, text_width(text, scale) + 2 * scale, 6 * scale, background)
    cursor = x
    for char in text:
        glyph = _FONT.get(char)
        if glyph is None:
            glyph = _FONT["?"]
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit == "1":
                    fill_rect(image, cursor + gx * scale, y + gy * scale, scale, scale, color)
        cursor += 4 * scale


def side_by_side(images: list[RGBImage], gap: int = 0, background=(0, 0, 0)) -> RGBImage:
    width = sum(image.width for image in images) + gap * (len(images) - 1)
    height = max(image.height for image in images)
    canvas = RGBImage(width, height, bytes(background) * (width * height))
    x = 0
    for image in images:
        for row in range(image.height):
            src = row * image.width * 3
            dst = (row * width + x) * 3
            canvas.pixels[dst : dst + image.width * 3] = image.pixels[src : src + image.width * 3]
        x += image.width + gap
    return canvas


def stack_vertical(images: list[RGBImage], gap: int = 0, background=(0, 0, 0)) -> RGBImage:
    width = max(image.width for image in images)
    height = sum(image.height for image in images) + gap * (len(images) - 1)
    canvas = RGBImage(width, height, bytes(background) * (width * height))
    y = 0
    for image in images:
        start = y * width * 3
        canvas.pixels[start : start + len(image.pixels)] = image.pixels
        y += image.height + gap
    return canvas
