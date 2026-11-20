"""分块电平表与测量工具：RMS/峰值 dBFS、分段电平时序、稳态电平。"""

from __future__ import annotations

import math


def rms_db(samples) -> float:
    if not samples:
        return -math.inf
    acc = 0.0
    for x in samples:
        acc += x * x
    return 10.0 * math.log10(acc / len(samples) + 1e-14)


def peak_db(samples) -> float:
    peak = 0.0
    for x in samples:
        a = abs(x)
        if a > peak:
            peak = a
    return 20.0 * math.log10(peak + 1e-14)


def block_meters(samples, fs: int, block_ms: float = 100.0):
    """按固定块长返回 [(t_start_s, rms_db, peak_db), ...]。"""
    size = max(1, int(fs * block_ms / 1000.0))
    rows = []
    for start in range(0, len(samples), size):
        chunk = samples[start:start + size]
        rows.append((start / fs, rms_db(chunk), peak_db(chunk)))
    return rows


def steady_level(samples, fs: int, from_s: float, to_s: float) -> float:
    """取指定时间窗（跳过起振）的 RMS dBFS，用于纯音稳态测量。"""
    a, b = int(from_s * fs), int(to_s * fs)
    return rms_db(samples[a:b])


def band_balance(bands, fs: int, from_s=0.3, to_s=0.9):
    """各频段在时间窗内的稳态 RMS，用于交叉频带能量分配检查。"""
    return [steady_level(b, fs, from_s, to_s) for b in bands]


def ascii_level_bar(db: float, floor=-80.0) -> str:
    if db <= floor:
        return "[" + " " * 40 + "]"
    fill = int(round((db - floor) / (-floor) * 40))
    fill = max(0, min(40, fill))
    return "[" + "#" * fill + " " * (40 - fill) + "]"

