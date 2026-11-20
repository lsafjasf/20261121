"""把多组积分结果渲染成剪辑师可读的接触表（contact sheet）。

每行一个配置（如不同快门角/曲线/CFR vs VFR），每列一个输出帧；
单元格顶部标注帧标签与时间重心，重复采样帧加红框，
部分捕获闪光的帧加黄色边框提示能量差异。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .color import linear_to_srgb
from .integrator import IntegrationResult
from .shutter import Shutter

_PAD = 10
_HEADER_H = 46
_ROW_LABEL_W = 150
_FONT_CANDIDATES = (
    "/home/hj/.local/share/fonts/NotoSansCJK-Regular.ttc",
    "/home/hj/.local/share/fonts/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


@dataclass
class ContactRow:
    """接触表中的一行：一个配置对应的整段结果。"""

    title: str
    result: IntegrationResult
    shutter: Shutter
    highlight_energy: bool = True
    energy_floor: float = 0.0


def _to_thumb(exposure: np.ndarray, thumb_w: int, thumb_h: int) -> Image.Image:
    encoded = linear_to_srgb(exposure)
    uint8 = np.rint(encoded * 255.0).astype(np.uint8)
    image = Image.fromarray(uint8, mode="RGB")
    return image.resize((thumb_w, thumb_h), Image.Resampling.NEAREST)


def _draw_waveform_inset(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    w: int,
    h: int,
    shutter: Shutter,
) -> None:
    """在缩略图右下角画一个快门开合曲线迷你示意。"""
    phases = np.linspace(-0.15, 1.15, 120)
    weights = shutter.weights(phases, frame_start=0.0, frame_duration=1.0)
    pts = []
    for phase, weight in zip(phases, weights):
        px = x0 + (phase - phases.min()) / (phases.max() - phases.min()) * w
        py = y0 + h - float(weight) * h
        pts.append((px, py))
    draw.rectangle([x0, y0, x0 + w, y0 + h], fill=(18, 18, 18))
    draw.line(pts, fill=(120, 220, 255), width=1)


def _energy_baseline(row: ContactRow) -> float:
    energies = [frame.energy for frame in row.result.frames]
    return max(float(np.max(energies)) if energies else 0.0, 1e-12)


def render_contact_sheet(
    rows: list[ContactRow],
    output_path: str,
    thumb_w: int = 150,
    thumb_h: int = 84,
    columns: int | None = None,
    title: str = "快门积分接触表",
) -> str:
    """渲染并保存接触表 PNG，返回文件路径。"""
    if not rows:
        raise ValueError("至少需要一行数据")
    n_cols = columns or max(row.result.frames.__len__() for row in rows)
    font = _load_font(11)
    title_font = _load_font(15)
    cell_w = thumb_w + _PAD
    cell_h = thumb_h + _HEADER_H + _PAD
    sheet_w = _ROW_LABEL_W + n_cols * cell_w + _PAD
    sheet_h = 40 + len(rows) * cell_h + _PAD
    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    draw.text((_PAD, 10), title, fill=(20, 20, 20), font=title_font)

    for row_idx, row in enumerate(rows):
        y_base = 40 + row_idx * cell_h
        draw.text(
            (_PAD, y_base + thumb_h // 2),
            row.title,
            fill=(30, 30, 30),
            font=font,
        )
        baseline = _energy_baseline(row)
        for col_idx, frame in enumerate(row.result.frames):
            if col_idx >= n_cols:
                break
            x0 = _ROW_LABEL_W + col_idx * cell_w
            y0 = y_base
            thumb = _to_thumb(frame.exposure, thumb_w, thumb_h)
            sheet.paste(thumb, (x0, y0 + _HEADER_H))
            header = (
                f"#{frame.index} {frame.label}\n"
                f"t*={frame.centroid_t:.4f}s E={frame.energy:.3f}"
            )
            draw.text((x0, y0), header, fill=(40, 40, 40), font=font)
            border_color = None
            if frame.duplicate_of is not None:
                border_color = (220, 40, 40)
            elif (
                row.highlight_energy
                and frame.energy < 0.66 * baseline
                and frame.energy > row.energy_floor
            ):
                border_color = (210, 160, 20)
            if border_color is not None:
                draw.rectangle(
                    [x0 - 1, y0 + _HEADER_H - 1, x0 + thumb_w, y0 + _HEADER_H + thumb_h],
                    outline=border_color,
                    width=2,
                )
            _draw_waveform_inset(
                draw,
                x0 + thumb_w - 34,
                y0 + _HEADER_H + thumb_h - 22,
                30,
                18,
                row.shutter,
            )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sheet.save(output_path)
    return output_path
