"""逐样本包络跟随器与静态增益曲线（压缩/限幅核心）。"""

from __future__ import annotations

import math


class EnvelopeFollower:
    """分支峰值包络：攻击用快系数、释放用慢系数，避免咔哒声。"""

    __slots__ = ("at", "rt", "env")

    def __init__(self, attack_ms: float, release_ms: float, fs: int):
        self.at = math.exp(-1.0 / (attack_ms * 0.001 * fs))
        self.rt = math.exp(-1.0 / (release_ms * 0.001 * fs))
        self.env = 0.0

    def reset(self):
        self.env = 0.0

    def process(self, x):
        v = abs(x)
        coef = self.at if v > self.env else self.rt
        self.env = coef * self.env + (1.0 - coef) * v
        return self.env


def db_to_gain(db: float) -> float:
    return 10.0 ** (db / 20.0)


def gain_to_db(g: float) -> float:
    return 20.0 * math.log10(max(g, 1e-12))


def static_gain_db(level_db: float, knee_db: float, ratio: float) -> float:
    """软膝压缩静态增益（输出相对输入的增益，dB）。

    level_db 低于 knee-5 为 1:1；knee 上下各 5 dB 为二次软膝过渡；
    高于 knee+5 按 ratio 压缩。返回要施加的增益 dB（<=0）。
    """
    lo = knee_db - 5.0
    hi = knee_db + 5.0
    if level_db <= lo:
        return 0.0
    if level_db >= hi:
        return (1.0 / ratio - 1.0) * (level_db - knee_db)
    # 软膝二次段
    t = (level_db - lo) / 10.0
    return (1.0 / ratio - 1.0) * 5.0 * t * t

