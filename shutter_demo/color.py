"""线性光与 sRGB 显示编码之间的转换。

整个快门积分管线在 *线性光* 域完成，最后编码回 sRGB 供显示与接触表使用。
所有函数都支持 numpy 数组，输入输出均为 0..1 浮点。
"""
from __future__ import annotations

import numpy as np

_EPS = 0.04045
_KAPPA = 12.92
_GAMMA = 2.4
_LINEAR_CUT = 0.0031308


def srgb_to_linear(value: np.ndarray) -> np.ndarray:
    """sRGB 显示编码值 -> 线性光辐照度。"""
    arr = np.asarray(value, dtype=np.float64)
    linear = np.where(
        arr <= _EPS,
        arr / _KAPPA,
        ((arr + 0.055) / 1.055) ** _GAMMA,
    )
    return linear


def linear_to_srgb(value: np.ndarray) -> np.ndarray:
    """线性光辐照度 -> sRGB 显示编码值。"""
    arr = np.clip(np.asarray(value, dtype=np.float64), 0.0, None)
    encoded = np.where(
        arr <= _LINEAR_CUT,
        _KAPPA * arr,
        1.055 * np.power(arr, 1.0 / _GAMMA) - 0.055,
    )
    return np.clip(encoded, 0.0, 1.0)


def linear_to_uint8(value: np.ndarray) -> np.ndarray:
    """线性光值直接转为 sRGB 8bit 图像（含编码与伽马）。"""
    encoded = linear_to_srgb(value)
    return np.rint(encoded * 255.0).astype(np.uint8)


def uint8_srgb_to_linear(image: np.ndarray) -> np.ndarray:
    """sRGB 8bit 图像解码为线性光浮点数组。"""
    return srgb_to_linear(np.asarray(image, dtype=np.float64) / 255.0)


def blend_linear(base: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    """线性域 alpha 混合，alpha 为 0..1 的 overlay 占比。"""
    a = float(np.clip(alpha, 0.0, 1.0))
    return base * (1.0 - a) + overlay * a


def blend_gamma(base_srgb: np.ndarray, overlay_srgb: np.ndarray, alpha: float) -> np.ndarray:
    """直接在 sRGB 编码值上混合（作为“错误做法”对照，返回线性光数组）。"""
    a = float(np.clip(alpha, 0.0, 1.0))
    mixed_srgb = base_srgb * (1.0 - a) + overlay_srgb * a
    return srgb_to_linear(mixed_srgb)
