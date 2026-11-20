"""快门开合曲线模型。

快门以“相对于输出帧间隔的归一化相位”描述：相位 0 为帧间隔起点，1 为终点。
快门角（shutter angle, 度）在 360 度时覆盖整个帧间隔，因此
``开放时长占比 = shutter_angle / 360``。

所有波形接收归一化相位数组（任意实数值，函数需按周期 1 处理），
返回 0..1 的相对透过率权重；窗口外权重必须为 0。
"""
from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np


Waveform = Callable[[np.ndarray], np.ndarray]


def _rectangular(phase: np.ndarray) -> np.ndarray:
    """理想瞬时开合：窗口内 1，窗口外 0。"""
    return np.where((phase >= 0.0) & (phase < 1.0), 1.0, 0.0)


def make_trapezoid(rise_ratio: float = 0.2, fall_ratio: float = 0.2) -> Waveform:
    """梯形开合：rise/fall 占开放窗口的比例，模拟卷帘/机械叶片缓开缓闭。"""
    rise_ratio = float(np.clip(rise_ratio, 0.0, 0.5))
    fall_ratio = float(np.clip(fall_ratio, 0.0, 0.5))

    def waveform(phase: np.ndarray) -> np.ndarray:
        p = np.asarray(phase, dtype=np.float64)
        weight = np.zeros_like(p)
        inside = (p >= 0.0) & (p < 1.0)
        pi = p[inside]
        if rise_ratio > 0.0:
            opening = pi / rise_ratio
        else:
            opening = np.full_like(pi, 1.0)
        if fall_ratio > 0.0:
            closing = (1.0 - pi) / fall_ratio
        else:
            closing = np.full_like(pi, 1.0)
        weight[inside] = np.clip(np.minimum(opening, closing), 0.0, 1.0)
        return weight

    return waveform


def make_gaussian(edge_sigma: float = 0.12) -> Waveform:
    """高斯软快门：在窗口内取中心高斯曲线并在边缘截断为 0。"""
    sigma = max(float(edge_sigma), 1e-3)

    def waveform(phase: np.ndarray) -> np.ndarray:
        p = np.asarray(phase, dtype=np.float64)
        weight = np.zeros_like(p)
        inside = (p >= 0.0) & (p < 1.0)
        pi = p[inside]
        g = np.exp(-0.5 * ((pi - 0.5) / sigma) ** 2)
        weight[inside] = g
        return weight

    return waveform


def make_double_slot(slot_width: float = 0.25, gap: float = 0.2) -> Waveform:
    """双狭缝快门（如某些双叶旋转快门），窗口内两个等宽矩形缝。"""
    slot_width = float(np.clip(slot_width, 0.05, 0.45))
    gap = float(np.clip(gap, 0.0, 1.0 - 2.0 * slot_width))
    center = 0.5
    left0 = center - gap / 2.0 - slot_width
    left1 = center - gap / 2.0
    right0 = center + gap / 2.0
    right1 = center + gap / 2.0 + slot_width

    def waveform(phase: np.ndarray) -> np.ndarray:
        p = np.asarray(phase, dtype=np.float64)
        slot_a = (p >= left0) & (p < left1)
        slot_b = (p >= right0) & (p < right1)
        return np.where(slot_a | slot_b, 1.0, 0.0)

    return waveform


WAVEFORMS: Dict[str, Waveform] = {
    "rect": _rectangular,
    "trapezoid": make_trapezoid(),
    "gaussian": make_gaussian(),
    "double_slot": make_double_slot(),
}


@dataclass
class Shutter:
    """一次输出序列的快门配置。

    属性
    ----
    shutter_angle_deg: 0..360 度，360 度等于整个输出帧间隔全开。
    phase_offset_ratio: 窗口起点相对帧间隔起点的偏移（0..1），
        例如 0.5 表示窗口居中（前后各留 (1-duty)/2 的遮光时间）。
    waveform: 窗口内部的归一化开合曲线。
    """

    shutter_angle_deg: float = 180.0
    phase_offset_ratio: float = 0.0
    waveform_name: str = "rect"
    waveform: Waveform = _rectangular

    @property
    def duty(self) -> float:
        """开放时长占帧间隔的比例。"""
        return float(np.clip(self.shutter_angle_deg, 0.0, 360.0)) / 360.0

    def window(self, frame_start: float, frame_duration: float) -> tuple[float, float]:
        """返回某帧快门窗口的绝对 (起点, 终点) 时间。"""
        offset = self.phase_offset_ratio * (1.0 - self.duty)
        start = frame_start + offset * frame_duration
        return start, start + self.duty * frame_duration

    def weights(self, times: np.ndarray, frame_start: float, frame_duration: float) -> np.ndarray:
        """给定绝对时间数组，返回快门透过率权重。"""
        w0, w1 = self.window(frame_start, frame_duration)
        if self.duty <= 0.0 or frame_duration <= 0.0:
            return np.zeros_like(np.asarray(times, dtype=np.float64))
        phase = (np.asarray(times, dtype=np.float64) - w0) / (w1 - w0)
        return self.waveform(phase)

    def describe(self) -> dict:
        return {
            "shutter_angle_deg": self.shutter_angle_deg,
            "duty": self.duty,
            "phase_offset_ratio": self.phase_offset_ratio,
            "waveform": self.waveform_name,
        }


def make_shutter(
    shutter_angle_deg: float = 180.0,
    waveform: str = "rect",
    phase_offset_ratio: float = 0.0,
) -> Shutter:
    """按名称构造内置快门。"""
    if waveform not in WAVEFORMS:
        raise KeyError(f"未知快门曲线 {waveform!r}，可选: {sorted(WAVEFORMS)}")
    return Shutter(
        shutter_angle_deg=shutter_angle_deg,
        phase_offset_ratio=phase_offset_ratio,
        waveform_name=waveform,
        waveform=WAVEFORMS[waveform],
    )


def load_custom_waveform(path: str, function_name: str = "shutter") -> Waveform:
    """从 Python 文件加载自定义时间函数 ``function_name(phase_ndarray) -> weight_ndarray``。"""
    path = os.path.abspath(os.path.expanduser(path))
    spec = importlib.util.spec_from_file_location("custom_shutter_waveform", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载自定义波形文件: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, function_name):
        raise AttributeError(f"{path} 中未找到函数 {function_name}()")
    func = getattr(module, function_name)
    probe = np.linspace(-0.2, 1.2, 11)
    result = np.asarray(func(probe), dtype=np.float64)
    if result.shape != probe.shape:
        raise ValueError("自定义波形返回数组形状必须与输入相位数组一致")
    if np.any(result < 0.0) or np.any(result > 1.0 + 1e-9):
        raise ValueError("自定义波形权重必须落在 0..1")
    return func
