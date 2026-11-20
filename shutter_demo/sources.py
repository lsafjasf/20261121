"""连续时间的合成辐照光源（高帧率输入由这些函数采样得到）。

所有光源返回 *线性光* RGB 数组，形状 ``(H, W, 3)``。
与“先离散成视频再积分”不同，光源可在任意时间求值，
因此闪光这类瞬态事件能按真实瞬时能量建模，用于验证能量与边界。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .color import blend_gamma, blend_linear, linear_to_srgb, srgb_to_linear


class RadianceSource(Protocol):
    """连续时间光源协议。"""

    height: int
    width: int

    def radiance(self, t: float) -> np.ndarray:
        """返回 t 时刻的线性光 RGB 图像。"""
        raise NotImplementedError

    def radiance_batch(self, times: np.ndarray) -> np.ndarray:
        """可选：一次返回 T 个时刻的辐照度，形状 (T, H, W, 3)。"""
        raise NotImplementedError


@dataclass
class MovingBars:
    """水平移动的黑白条（在 x 方向平动），用于观察拖影与曝光重心。

    条带为固定空间周期的方波，整体以 ``speed_px_s`` 像素/秒平移。
    图像左右各留一条参考标记列，便于接触表中读取位移。
    """

    width: int = 160
    height: int = 90
    bar_period_px: float = 40.0
    bar_duty: float = 0.5
    speed_px_s: float = 120.0
    white_radiance: float = 1.0
    initial_offset_px: float = 0.0

    def edge_position(self, t: float) -> float:
        """t 时刻第一条“黑->白”上升沿的 x 位置（浮点，可超出画面）。"""
        return self.initial_offset_px + self.speed_px_s * t

    def radiance(self, t: float) -> np.ndarray:
        x = np.arange(self.width, dtype=np.float64)[None, :]
        phase = ((x - self.edge_position(t)) / self.bar_period_px) % 1.0
        stripe = np.where(phase < self.bar_duty, self.white_radiance, 0.0)
        image = np.repeat(stripe, self.height, axis=0)
        image = np.repeat(image[..., None], 3, axis=2)
        return image.copy()

    def radiance_batch(self, times: np.ndarray) -> np.ndarray:
        """向量化求值：(T,) 时刻 -> (T, H, W, 3)。"""
        ts = np.asarray(times, dtype=np.float64)
        x = np.arange(self.width, dtype=np.float64)[None, None, :]
        edges = self.initial_offset_px + self.speed_px_s * ts
        phase = ((x - edges[:, None, None]) / self.bar_period_px) % 1.0
        stripe = np.where(phase < self.bar_duty, self.white_radiance, 0.0)
        frames = np.repeat(stripe, self.height, axis=1)
        return np.repeat(frames[..., None], 3, axis=3)


@dataclass
class FlashField:
    """稀疏脉冲闪光场：在给定时刻出现极短矩形脉冲。

    脉冲持续时间默认 1ms。空间上为左白右黑的分界图案，
    闪光命中时可在接触表中清楚看到“定格”的锐利边界，
    未命中帧则只剩底光，用于验证闪光能量与快门边界。
    """

    width: int = 160
    height: int = 90
    flash_times_s: tuple[float, ...] = (0.1, 0.3, 0.5)
    flash_duration_s: float = 0.001
    flash_radiance: float = 4.0
    base_radiance: float = 0.05
    split_ratio: float = 0.5

    def _spatial_pattern(self) -> np.ndarray:
        pattern = np.full((self.height, self.width), self.base_radiance)
        split = int(round(self.width * self.split_ratio))
        pattern[:, :split] = self.base_radiance
        pattern[:, split:] = self.base_radiance + self.flash_radiance
        return np.repeat(pattern[..., None], 3, axis=2)

    def is_active(self, t: float) -> bool:
        for flash_t in self.flash_times_s:
            if flash_t <= t < flash_t + self.flash_duration_s:
                return True
        return False

    def event_intervals(self) -> list[tuple[float, float]]:
        """供积分器做自适应细分的瞬态事件区间列表。"""
        return [
            (flash_t, flash_t + self.flash_duration_s)
            for flash_t in self.flash_times_s
        ]

    def radiance(self, t: float) -> np.ndarray:
        base = np.full((self.height, self.width, 3), self.base_radiance)
        if self.is_active(t):
            base = np.maximum(base, self._spatial_pattern())
        return base

    def radiance_batch(self, times: np.ndarray) -> np.ndarray:
        """向量化求值：用半开区间判定每个时刻是否落在任一闪光脉冲内。"""
        ts = np.asarray(times, dtype=np.float64)
        active = np.zeros(ts.shape, dtype=bool)
        for flash_t in self.flash_times_s:
            active |= (ts >= flash_t) & (ts < flash_t + self.flash_duration_s)
        frames = np.full(
            (ts.size, self.height, self.width, 3), self.base_radiance, dtype=np.float64
        )
        if active.any():
            pattern = self._spatial_pattern()
            frames[active] = np.maximum(frames[active], pattern[None, ...])
        return frames

    def pulse_energy(self) -> float:
        """单次闪光在亮区贡献的积分能量（辐照度 × 时长）。"""
        return self.flash_radiance * self.flash_duration_s


@dataclass
class BlendingDemo:
    """线性光混合 vs 直接在 sRGB 伽马值上混合的对照场景。

    场景中一个亮斑随时间从位置 A 移动到位置 B。
    ``radiance`` 始终给出物理正确的线性光叠加结果；
    ``gamma_blended_radiance`` 给出把两帧 sRGB 值直接平均后的“错误”结果，
    供接触表并排比较中间调与高光的差别。
    """

    width: int = 160
    height: int = 90
    spot_radius_px: float = 14.0
    spot_radiance: float = 1.0
    start_x_ratio: float = 0.25
    end_x_ratio: float = 0.75

    def _spot(self, center_x: float) -> np.ndarray:
        x = np.arange(self.width, dtype=np.float64)[None, :]
        y = np.arange(self.height, dtype=np.float64)[:, None]
        center_y = self.height / 2.0
        dist2 = (x - center_x) ** 2 + (y - center_y) ** 2
        spot = np.where(dist2 <= self.spot_radius_px**2, self.spot_radiance, 0.0)
        return np.repeat(spot[..., None], 3, axis=2)

    def _position(self, t: float, duration: float) -> float:
        u = float(np.clip(t / max(duration, 1e-9), 0.0, 1.0))
        x0 = self.start_x_ratio * self.width
        x1 = self.end_x_ratio * self.width
        return x0 + (x1 - x0) * u

    def radiance_at(self, t: float, duration: float) -> np.ndarray:
        return self._spot(self._position(t, duration))

    def linear_blend(self, t_a: float, t_b: float, duration: float, alpha: float) -> np.ndarray:
        """线性域混合（物理正确）。"""
        return blend_linear(
            self.radiance_at(t_a, duration),
            self.radiance_at(t_b, duration),
            alpha,
        )

    def gamma_blend(self, t_a: float, t_b: float, duration: float, alpha: float) -> np.ndarray:
        """sRGB 伽马域直接混合后再解码（对照用的非物理结果）。"""
        srgb_a = linear_to_srgb(self.radiance_at(t_a, duration))
        srgb_b = linear_to_srgb(self.radiance_at(t_b, duration))
        return blend_gamma(srgb_a, srgb_b, alpha)

    def radiance(self, t: float) -> np.ndarray:
        # 协议兼容：默认按 1 秒行程求值
        return self.radiance_at(t, 1.0)


def sample_hfr_sequence(source: RadianceSource, hfr_fps: float, duration_s: float) -> np.ndarray:
    """把连续光源采样成高帧率输入序列（帧中点采样），形状 (N, H, W, 3)。"""
    dt = 1.0 / hfr_fps
    count = int(round(duration_s * hfr_fps))
    frames = [source.radiance((idx + 0.5) * dt) for idx in range(count)]
    return np.stack(frames, axis=0)


def moving_pulse_source(width: int = 160, height: int = 90) -> MovingBars:
    """构造一个便于做重心/能量解析校验的单脉冲移动条场景。"""
    return MovingBars(
        width=width,
        height=height,
        bar_period_px=float(width) * 4.0,
        bar_duty=0.25,
        speed_px_s=100.0,
        white_radiance=1.0,
    )


__all__ = [
    "RadianceSource",
    "MovingBars",
    "FlashField",
    "BlendingDemo",
    "sample_hfr_sequence",
    "moving_pulse_source",
    "srgb_to_linear",
]
