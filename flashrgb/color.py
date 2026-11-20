from __future__ import annotations

from .imageio import GrayImage, RGBImage


def blank_rgb(width: int, height: int, value: int = 0) -> RGBImage:
    return RGBImage(width, height, bytes((value, value, value)) * (width * height))


def compose_channels(channels: dict[str, GrayImage], width: int, height: int) -> RGBImage:
    output = bytearray(width * height * 3)
    red = channels.get("R")
    green = channels.get("G")
    blue = channels.get("B")
    for index in range(width * height):
        output[index * 3] = red.pixels[index] if red is not None else 0
        output[index * 3 + 1] = green.pixels[index] if green is not None else 0
        output[index * 3 + 2] = blue.pixels[index] if blue is not None else 0
    return RGBImage(width, height, output)


def clear_rect(
    image: RGBImage, x: int, y: int, width: int, height: int, value=(0, 0, 0)
) -> None:
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(image.width, x + width)
    y1 = min(image.height, y + height)
    if x0 >= x1 or y0 >= y1:
        return
    row = bytes(value) * (x1 - x0)
    for yy in range(y0, y1):
        start = (yy * image.width + x0) * 3
        image.pixels[start : start + len(row)] = row


def rgb_chroma(red: int, green: int, blue: int) -> tuple[float, float]:
    cb = 128.0 - 0.168736 * red - 0.331264 * green + 0.5 * blue
    cr = 128.0 + 0.5 * red - 0.418688 * green - 0.081312 * blue
    return cb, cr


def image_chroma(image: RGBImage) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    pixels = image.pixels
    for index in range(0, len(pixels), 3):
        result.append(rgb_chroma(pixels[index], pixels[index + 1], pixels[index + 2]))
    return result


def heatmap_image(
    errors: list[float],
    width: int,
    height: int,
    gain: float = 4.0,
) -> RGBImage:
    output = bytearray(width * height * 3)
    for index, error in enumerate(errors):
        value = min(255, int(error * gain + 0.5))
        if value < 64:
            red, green, blue = 0, value, min(255, value * 3)
        elif value < 160:
            red, green, blue = value, value // 2, 0
        else:
            red, green, blue = 255, max(0, 255 - (value - 160) * 2), 0
        output[index * 3] = red
        output[index * 3 + 1] = green
        output[index * 3 + 2] = blue
    return RGBImage(width, height, output)
