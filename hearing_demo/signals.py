"""合成测试信号：正弦、 gated 阶跃、扫频、噪声与演示节目。"""

from __future__ import annotations

import math
import random


def sine(freq_hz: float, dur_s: float, fs: int, level_dbfs: float = -20.0,
         phase: float = 0.0):
    amp = 10.0 ** (level_dbfs / 20.0)
    n = int(dur_s * fs)
    w = 2.0 * math.pi * freq_hz / fs
    return [amp * math.sin(w * i + phase) for i in range(n)]


def gate(samples, on_ranges, fs: int = 48000):
    """on_ranges: [(start_s, end_s), ...]，其余时刻置零。"""
    out = [0.0] * len(samples)
    for a, b in on_ranges:
        ia, ib = int(a * fs), int(b * fs)
        for i in range(max(0, ia), min(len(samples), ib)):
            out[i] = samples[i]
    return out


def level_step(freq_hz: float, schedule, fs: int):
    """schedule: [(start_s, level_dbfs), ...] 分段恒定电平的纯音阶跃。"""
    total = schedule[-1][0] + 1.0
    n = int(total * fs)
    out = [0.0] * n
    w = 2.0 * math.pi * freq_hz / fs
    for k, (start, lvl) in enumerate(schedule):
        end = schedule[k + 1][0] if k + 1 < len(schedule) else total
        amp = 10.0 ** (lvl / 20.0)
        for i in range(int(start * fs), int(end * fs)):
            out[i] = amp * math.sin(w * i)
    return out


def white_noise(dur_s: float, fs: int, level_dbfs: float, seed: int = 7):
    rng = random.Random(seed)
    amp = 10.0 ** (level_dbfs / 20.0)
    n = int(dur_s * fs)
    # 均匀白噪声峰值约 -4.7 dBFS，RMS 与设定值偏差在此标注
    return [amp * (rng.random() * 2.0 - 1.0) for _ in range(n)]


def sweep(dur_s: float, fs: int, f0=100.0, f1=10000.0, level_dbfs=-20.0):
    amp = 10.0 ** (level_dbfs / 20.0)
    n = int(dur_s * fs)
    k = (f1 / f0) ** (1.0 / n)
    out, phase, f = [], 0.0, f0
    for _ in range(n):
        out.append(amp * math.sin(phase))
        phase += 2.0 * math.pi * f / fs
        f *= k
    return out


def concat(*parts, gap_s=0.0, fs=48000):
    gap = [0.0] * int(gap_s * fs)
    out = []
    for p in parts:
        out.extend(p)
        out.extend(gap)
    return out


def demo_program(fs: int = 48000):
    """8 秒合成演示节目（双耳共用同一源），覆盖各频段与快速电平变化。"""
    seg = []
    seg.append(sine(250, 1.5, fs, -24.0))                 # 0.0-1.5s 低频稳态
    seg.append(sine(1000, 1.5, fs, -18.0))                # 1.5-3.0s 中频
    seg.append(level_step(2000, [(0.0, -30.0), (0.4, -12.0),
                                 (0.8, -30.0), (1.2, -8.0)], fs))  # 快速跃迁
    seg.append(sine(4000, 1.6, fs, -16.0))                # 高频
    seg.append(white_noise(1.4, fs, -20.0, seed=11))      # 宽带
    return concat(*seg, gap_s=0.15, fs=fs)

