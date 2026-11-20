"""自定义快门开合曲线示例。

函数接收归一化相位 numpy 数组（窗口内为 0..1，窗口外为其它值），
返回 0..1 的透过率权重。窗口外必须返回 0；窗口内可以是任意时间函数，
这里演示一个“先快速打开、再缓慢关闭”的非对称叶片曲线。
"""
import numpy as np


def shutter(phase: np.ndarray) -> np.ndarray:
    p = np.asarray(phase, dtype=np.float64)
    weight = np.zeros_like(p)
    inside = (p >= 0.0) & (p < 1.0)
    pi = p[inside]
    # 前 15% 相位线性开到最大，之后按余弦平方缓慢关合
    opened = np.clip(pi / 0.15, 0.0, 1.0)
    tail_phase = np.clip((pi - 0.15) / 0.85, 0.0, 1.0)
    closing = np.cos(tail_phase * np.pi / 2.0) ** 2
    weight[inside] = np.minimum(opened, closing + (pi < 0.15))
    return np.clip(weight, 0.0, 1.0)
