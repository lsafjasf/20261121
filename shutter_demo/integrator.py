"""快门积分核心。

对每个输出帧，在快门开放窗口内对连续光源做加权时间积分：

    E(x) = ∫ w(t) · L(t, x) dt / ∫ w(t) dt

其中 ``w(t)`` 是快门透过率（开合曲线），``L`` 是线性光辐照度。
积分用细网格中点法则；网格密度由 ``samples_per_second`` 控制，
对闪光等瞬态事件应取足够高的采样率。

另提供点采样模式（模拟“零曝光时间”的采样保持），
当多个输出帧落在同一个输入采样间隔上时会检出重复采样。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .shutter import Shutter
from .sources import RadianceSource
from .vfr import FrameSchedule


@dataclass
class IntegratedFrame:
    """单帧积分结果与诊断量。"""

    index: int
    label: str
    frame_start: float
    frame_duration: float
    window_start: float
    window_end: float
    exposure: np.ndarray  # 归一化后的线性光曝光图 (H, W, 3)
    energy: float  # 全图平均的曝光能量（未归一化积分 / 像素数）
    centroid_t: float  # 曝光时间重心（绝对秒）
    window_weight_sum: float  # 快门权重积分 ∫w dt
    sample_count: int
    duplicate_of: int | None = None  # 点采样模式下与哪一帧取到同一时刻


@dataclass
class IntegrationResult:
    frames: list[IntegratedFrame]
    mode: str
    samples_per_second: float

    @property
    def exposures(self) -> np.ndarray:
        return np.stack([frame.exposure for frame in frames], axis=0)


def _weighted_centroid(times: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= 0.0:
        return float("nan")
    return float(np.sum(times * weights) / total)


def _evaluate_source(source: RadianceSource, times: np.ndarray) -> np.ndarray:
    """优先向量化批量求值，不支持时回退到逐时刻调用。"""
    batch = getattr(source, "radiance_batch", None)
    if callable(batch):
        return batch(times)
    return np.stack([source.radiance(float(t)) for t in times], axis=0)


def _build_time_grid(
    w0: float,
    w1: float,
    samples_per_second: float,
    source: RadianceSource,
) -> tuple[np.ndarray, np.ndarray]:
    """构造积分时间中点与每个中点对应的严格小区间宽度。

    把窗口切成互不重叠的若干段：瞬态事件区间（如闪光）按
    ``samples_per_second`` 细分，其余部分按上限 4kHz 的粗网格细分。
    每段独立做中点矩形积分，因此采样宽度严格等于段宽/采样数，
    不会在粗细网格拼接处产生能量误差。
    """
    coarse_rate = min(samples_per_second, 4000.0)
    intervals = getattr(source, "event_intervals", None)
    if not callable(intervals):
        coarse_count = max(int(round((w1 - w0) * coarse_rate)), 8)
        dt = (w1 - w0) / coarse_count
        times = w0 + (np.arange(coarse_count, dtype=np.float64) + 0.5) * dt
        return times, np.full(coarse_count, dt)

    # 事件区间与窗口求交，按时间排序并裁剪到 [w0, w1)
    events = sorted(
        (max(lo, w0), min(hi, w1)) for lo, hi in intervals() if min(hi, w1) > max(lo, w0)
    )
    segments: list[tuple[float, float, float]] = []
    cursor = w0
    for ev_lo, ev_hi in events:
        if ev_lo > cursor:
            segments.append((cursor, ev_lo, coarse_rate))
        segments.append((ev_lo, ev_hi, samples_per_second))
        cursor = ev_hi
    if cursor < w1:
        segments.append((cursor, w1, coarse_rate))

    times_list = []
    widths_list = []
    for seg_lo, seg_hi, rate in segments:
        seg_len = seg_hi - seg_lo
        count = max(int(round(seg_len * rate)), 2)
        step = seg_len / count
        midpoints = seg_lo + (np.arange(count, dtype=np.float64) + 0.5) * step
        times_list.append(midpoints)
        widths_list.append(np.full(count, step))
    times = np.concatenate(times_list)
    cell_widths = np.concatenate(widths_list)
    return times, cell_widths


def integrate_frame(
    source: RadianceSource,
    shutter: Shutter,
    frame_start: float,
    frame_duration: float,
    samples_per_second: float,
) -> IntegratedFrame:
    """积分单个输出帧。"""
    w0, w1 = shutter.window(frame_start, frame_duration)
    window_len = w1 - w0
    if window_len <= 0.0:
        shape = (source.height, source.width, 3)
        empty = np.zeros(shape, dtype=np.float64)
        return IntegratedFrame(
            index=-1,
            label="",
            frame_start=frame_start,
            frame_duration=frame_duration,
            window_start=w0,
            window_end=w1,
            exposure=empty,
            energy=0.0,
            centroid_t=float("nan"),
            window_weight_sum=0.0,
            sample_count=0,
        )

    times, cell_widths = _build_time_grid(w0, w1, samples_per_second, source)
    sample_count = int(times.size)
    weights = shutter.weights(times, frame_start, frame_duration)

    radiance_stack = _evaluate_source(source, times)  # (T, H, W, 3)
    w = weights[:, None, None, None]
    widths = cell_widths[:, None, None, None]
    weighted_energy = np.sum(w * widths * radiance_stack, axis=0)
    weight_sum = float(np.sum(weights * cell_widths))
    if weight_sum > 0.0:
        exposure = weighted_energy / weight_sum
    else:
        exposure = np.zeros_like(weighted_energy)

    energy = float(np.mean(weighted_energy))
    centroid_t = _weighted_centroid(times, weights)
    return IntegratedFrame(
        index=-1,
        label="",
        frame_start=frame_start,
        frame_duration=frame_duration,
        window_start=w0,
        window_end=w1,
        exposure=exposure,
        energy=energy,
        centroid_t=centroid_t,
        window_weight_sum=weight_sum,
        sample_count=sample_count,
    )
def integrate_schedule(
    source: RadianceSource,
    shutter: Shutter,
    schedule: FrameSchedule,
    samples_per_second: float = 20000.0,
) -> IntegrationResult:
    """按给定帧调度积分整段输出。"""
    frames: list[IntegratedFrame] = []
    for idx, (start, duration, label) in enumerate(
        zip(schedule.start_times, schedule.durations, schedule.labels)
    ):
        frame = integrate_frame(
            source, shutter, float(start), float(duration), samples_per_second
        )
        frame.index = idx
        frame.label = label
        frames.append(frame)
    return IntegrationResult(
        frames=frames,
        mode="integrated",
        samples_per_second=samples_per_second,
    )


def point_sample_schedule(
    source: RadianceSource,
    schedule: FrameSchedule,
    hfr_fps: float,
) -> IntegrationResult:
    """点采样模式：每帧只取快门窗中心一个时刻，并按 HFR 网格量化。

    量化到输入帧中点网格后，若两个输出帧取到同一个输入时刻，
    后者标记 ``duplicate_of`` 指向首次出现的帧——这就是重复采样。
    """
    dt = 1.0 / hfr_fps
    frames: list[IntegratedFrame] = []
    seen_grid: dict[int, int] = {}
    for idx, (start, duration, label) in enumerate(
        zip(schedule.start_times, schedule.durations, schedule.labels)
    ):
        sample_t = float(start) + 0.5 * float(duration)
        grid_index = int(np.floor(sample_t / dt + 0.5))
        quantized_t = (grid_index + 0.5) * dt
        exposure = source.radiance(quantized_t).copy()
        duplicate = seen_grid.get(grid_index)
        if duplicate is None:
            seen_grid[grid_index] = idx
        frame = IntegratedFrame(
            index=idx,
            label=label,
            frame_start=float(start),
            frame_duration=float(duration),
            window_start=sample_t,
            window_end=sample_t,
            exposure=exposure,
            energy=float(np.mean(exposure) * dt),
            centroid_t=quantized_t,
            window_weight_sum=dt,
            sample_count=1,
            duplicate_of=duplicate,
        )
        frames.append(frame)
    return IntegrationResult(
        frames=frames,
        mode=f"point-sample@{hfr_fps:g}fps",
        samples_per_second=hfr_fps,
    )
