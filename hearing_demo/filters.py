"""一阶互补分频（纯标准库，双线性变换 + 频率预畸变）。

每个分频节点由共享分母的一阶 LP/HP 组成：
    w   = tan(pi*fc/fs)
    u   = w/(1+w),  v = (w-1)/(1+w),  k = 1/(1+w)
    Hlp(z) = u(1+z^-1) / (1+v z^-1)
    Hhp(z) = k(1-z^-1) / (1+v z^-1)
代数上 Hlp+Hhp 恒等于 1，因此嵌套成 4 频段树后，四带求和在全频带
严格平坦（实测重组误差受浮点精度限制），分频点处相邻两带各 -3 dB。

选择一阶而非 LR4 的原因：嵌套 4 带 LR4 在分频点存在约 -3.5 dB 的
固有叠加凹陷，会污染"旁路 vs 压缩"电平对比；一阶互补树无此问题，
且实现简单、对原型检查足够透明。
"""

from __future__ import annotations

import math


class Biquad:
    """通用一阶/二阶节（TDF-II）；二阶字段 b2/a2 缺省为 0。"""

    __slots__ = ("b0", "b1", "b2", "a1", "a2", "z1", "z2")

    def __init__(self, b0, b1, b2, a1, a2):
        self.b0, self.b1, self.b2 = b0, b1, b2
        self.a1, self.a2 = a1, a2
        self.z1 = self.z2 = 0.0

    def reset(self):
        self.z1 = self.z2 = 0.0

    def process(self, x):
        y = self.b0 * x + self.z1
        self.z1 = self.b1 * x - self.a1 * y + self.z2
        self.z2 = self.b2 * x - self.a2 * y
        return y


def first_order_pair(fc: float, fs: float):
    """返回 (LP 一阶节, HP 一阶节)，二者传递函数之和恒为 1。"""
    w = math.tan(math.pi * fc / fs)
    u = w / (1.0 + w)
    v = (w - 1.0) / (1.0 + w)
    k = 1.0 / (1.0 + w)
    lp = Biquad(u, u, 0.0, v, 0.0)
    hp = Biquad(k, -k, 0.0, v, 0.0)
    return lp, hp


def _run(stage, x):
    return stage.process(x)


class Crossover4:
    """4 频段一阶互补分频树。

    分频点 fc1/fc2/fc3（默认 500/2000/6000 Hz）。根节点按 fc2 分
    低/高两支，低支再按 fc1 分 band0/band1，高支按 fc3 分 band2/band3。
    """

    def __init__(self, fc1, fc2, fc3, fs):
        self.fs = fs
        self.crossovers = (fc1, fc2, fc3)
        self.root_lp, self.root_hp = first_order_pair(fc2, fs)
        self.lo_lp, self.lo_hp = first_order_pair(fc1, fs)
        self.hi_lp, self.hi_hp = first_order_pair(fc3, fs)

    def reset(self):
        for st in (self.root_lp, self.root_hp, self.lo_lp,
                   self.lo_hp, self.hi_lp, self.hi_hp):
            st.reset()

    def split_sample(self, x):
        low2 = _run(self.root_lp, x)
        high2 = _run(self.root_hp, x)
        band0 = _run(self.lo_lp, low2)
        band1 = _run(self.lo_hp, low2)
        band2 = _run(self.hi_lp, high2)
        band3 = _run(self.hi_hp, high2)
        return band0, band1, band2, band3

    def split_block(self, samples):
        self.reset()
        b0, b1, b2, b3 = [], [], [], []
        for x in samples:
            p = self.split_sample(x)
            b0.append(p[0]); b1.append(p[1])
            b2.append(p[2]); b3.append(p[3])
        return [b0, b1, b2, b3]

    def recombine(self, bands):
        return [bands[0][i] + bands[1][i] + bands[2][i] + bands[3][i]
                for i in range(len(bands[0]))]

