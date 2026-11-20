"""输出帧时间基（PTS 调度），支持 CFR 与 VFR。

调度以“帧起点时间戳 + 帧间隔”描述，帧间隔允许逐帧变化（VFR）。
剪辑语境中常见的 2:1 下拉（film 24fps 包进 60fps 容器）也在这里实现：
相邻帧按 2 长 1 短的节奏重复（或等效地让帧间隔交替）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class FrameSchedule:
    """输出帧调度。

    start_times: 每帧帧间隔起点（秒），长度 N。
    durations: 每帧帧间隔时长（秒），长度 N；最后一帧用前一间隔补齐。
    labels: 每帧可读标签，供接触表标注（如 "A", "B(dup)"）。
    """

    start_times: np.ndarray
    durations: np.ndarray
    labels: list[str]
    mode: str

    @property
    def count(self) -> int:
        return int(self.start_times.size)

    def describe(self) -> dict:
        unique_durations = np.unique(np.round(self.durations, 12))
        return {
            "mode": self.mode,
            "frame_count": self.count,
            "unique_frame_durations_s": [float(x) for x in unique_durations],
            "mean_fps": float(self.count / float(np.sum(self.durations))),
        }


def cfr_schedule(fps: float, frame_count: int, start: float = 0.0) -> FrameSchedule:
    """恒定帧率调度。"""
    if fps <= 0.0 or frame_count <= 0:
        raise ValueError("fps 与 frame_count 必须为正数")
    dt = 1.0 / fps
    starts = start + np.arange(frame_count, dtype=np.float64) * dt
    durations = np.full(frame_count, dt, dtype=np.float64)
    labels = [f"{idx:02d}" for idx in range(frame_count)]
    return FrameSchedule(starts, durations, labels, mode=f"CFR {fps:g}fps")


def explicit_schedule(
    pts_seconds: Sequence[float],
    labels: Sequence[str] | None = None,
) -> FrameSchedule:
    """显式 PTS 调度；帧间隔由相邻 PTS 差分得到，末帧沿用最后一个间隔。"""
    pts = np.asarray(pts_seconds, dtype=np.float64)
    if pts.ndim != 1 or pts.size < 2:
        raise ValueError("VFR 至少需要两个 PTS")
    if np.any(np.diff(pts) <= 0.0):
        raise ValueError("PTS 必须严格递增")
    diffs = np.diff(pts)
    durations = np.concatenate([diffs, diffs[-1:]])
    if labels is None:
        labels = [f"{idx:02d}" for idx in range(pts.size)]
    if len(labels) != pts.size:
        raise ValueError("labels 数量必须与 PTS 数量一致")
    return FrameSchedule(pts, durations, list(labels), mode="VFR explicit PTS")


def pulldown_2_1_schedule(
    base_fps: float,
    cycle_count: int,
    start: float = 0.0,
) -> FrameSchedule:
    """2:1 节奏的 VFR 调度。

    每个 3 帧循环中，帧 *保持时长* 按 2:2:1 单位分配：
    前两帧保持 ``2/(5*base_fps)``，第三帧保持 ``1/(5*base_fps)``，
    循环总时长恰为 ``1/base_fps``。短帧在接触表中体现为更窄的曝光窗，
    配合点采样模式即可看到重复采样。
    """
    if base_fps <= 0.0 or cycle_count <= 0:
        raise ValueError("base_fps 与 cycle_count 必须为正数")
    long_dt = 2.0 / (5.0 * base_fps)
    short_dt = 1.0 / (5.0 * base_fps)
    pattern = [long_dt, long_dt, short_dt]
    durations = np.array(pattern * cycle_count, dtype=np.float64)
    starts = start + np.concatenate([[0.0], np.cumsum(durations[:-1])])
    labels = []
    for cycle in range(cycle_count):
        labels.extend([f"{cycle}:L", f"{cycle}:L", f"{cycle}:S"])
    return FrameSchedule(starts, durations, labels, mode=f"VFR 2:1 @{base_fps:g} base")


def parse_pts_text(text: str) -> list[float]:
    """解析逗号/空白分隔的 PTS 文本，例如 "0, 0.04, 0.08, 0.1"。"""
    tokens = text.replace(",", " ").split()
    return [float(token) for token in tokens]
