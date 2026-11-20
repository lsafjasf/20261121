"""对积分结果做可复现的量化测量，供报告与单元测试断言。"""
from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import numpy as np

from .integrator import IntegratedFrame, IntegrationResult


def frame_energy(exposure: np.ndarray) -> float:
    """单帧全图平均线性光能量。"""
    return float(np.mean(exposure))


def spatial_centroid_x(exposure: np.ndarray) -> float:
    """按亮度加权的画面 x 重心（像素）。"""
    luminance = np.mean(exposure, axis=2)
    weight = luminance.sum(axis=0)
    total = float(weight.sum())
    if total <= 0.0:
        return float("nan")
    xs = np.arange(exposure.shape[1], dtype=np.float64)
    return float(np.sum(xs * weight) / total)


def edge_spread_width(exposure: np.ndarray, row_ratio: float = 0.5) -> float:
    """测量指定行上 10%->90% 上升沿宽度（像素），量化拖影模糊。

    取画面中部一行的亮度曲线，找最大斜率区间，返回其 10-90 跨越宽度。
    理想点采样的锐利边沿宽度接近 1px；积分拖影会显著变宽。
    """
    row = int(exposure.shape[0] * row_ratio)
    line = np.mean(exposure[row, :, :], axis=1)
    span = float(line.max() - line.min())
    if span <= 1e-9:
        return float("nan")
    low = line.min() + 0.1 * span
    high = line.min() + 0.9 * span
    above_low = np.flatnonzero(line >= low)
    above_high = np.flatnonzero(line >= high)
    if above_low.size == 0 or above_high.size == 0:
        return float("nan")
    return float(above_high[0] - above_low[0])


def summarize_frames(result: IntegrationResult) -> list[dict]:
    """把每帧的时间/能量/重心/模糊量整理成可序列化字典。"""
    records = []
    for frame in result.frames:
        records.append(
            {
                "index": frame.index,
                "label": frame.label,
                "frame_start_s": frame.frame_start,
                "frame_duration_s": frame.frame_duration,
                "window_start_s": frame.window_start,
                "window_end_s": frame.window_end,
                "exposure_centroid_s": frame.centroid_t,
                "energy_mean_linear": frame.energy,
                "spatial_centroid_x_px": spatial_centroid_x(frame.exposure),
                "edge_width_px": edge_spread_width(frame.exposure),
                "sample_count": frame.sample_count,
                "duplicate_of": frame.duplicate_of,
            }
        )
    return records


def count_duplicates(frames: Iterable[IntegratedFrame]) -> int:
    """统计点采样模式下的重复采样帧数。"""
    return sum(1 for frame in frames if frame.duplicate_of is not None)


def expected_motion_centroid(
    source, shutter: Shutter, frame_start: float, frame_duration: float
) -> float:
    """移动条场景下曝光时间重心对应的理论边沿 x 位置。"""
    w0, w1 = shutter.window(frame_start, frame_duration)
    # 矩形快门的时间重心即窗口中点；其它曲线在报告中以实测重心为准
    center_t = 0.5 * (w0 + w1)
    return float(source.edge_position(center_t))


def frame_to_basic_dict(frame: IntegratedFrame) -> dict:
    """精简版帧信息（不含图像数组）。"""
    data = asdict(frame)
    data.pop("exposure", None)
    return data
