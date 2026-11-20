"""听阈曲线加载、空白测点插值与压缩处方映射。

测点标准频率（Hz）：125, 250, 500, 1000, 2000, 4000, 8000。
空白测点规则：
  1) 位于两个实测点之间 -> 在对数频率轴上线性插值（听阈 dB 线性变化）；
  2) 位于实测范围之外 -> 钳位到最近的实测值（不外推趋势），并标记 clamped；
  3) 全部为空 -> 抛错。
处方为简化的线性规则，用于原型检查，不等同 NAL-NL2 等临床公式。
"""

from __future__ import annotations

import math

STD_FREQS = (125, 250, 500, 1000, 2000, 4000, 8000)

# 频段边缘频率（与 500/2000/6000 Hz 分频点对应，末端用 8000 代表高频段上沿）
BAND_EDGES = (125, 500, 2000, 6000, 8000)
BAND_NAMES = ("低 (<500Hz)", "中低 (500-2k)", "中高 (2k-6k)", "高 (>6k)")


def interpolate_threshold(raw: dict):
    """返回 (thresholds: dict[Hz->dB], info: dict[Hz->'measured'|...] )。"""
    measured = []
    for f in STD_FREQS:
        v = raw.get(str(f), raw.get(f))
        if v is not None:
            measured.append((f, float(v)))
    if not measured:
        raise ValueError("该耳没有任何实测听阈点，无法插值")
    measured.sort()
    log_measured = [(math.log10(f), hl) for f, hl in measured]
    thresholds, info = {}, {}
    f_min, f_max = measured[0][0], measured[-1][0]
    hl_min, hl_max = measured[0][1], measured[-1][1]
    for f in STD_FREQS:
        v = raw.get(str(f), raw.get(f))
        if v is not None:
            thresholds[f], info[f] = float(v), "measured"
            continue
        lf = math.log10(f)
        if f < f_min:
            thresholds[f], info[f] = hl_min, "clamped_low"
        elif f > f_max:
            thresholds[f], info[f] = hl_max, "clamped_high"
        else:
            for k in range(len(log_measured) - 1):
                x0, y0 = log_measured[k]
                x1, y1 = log_measured[k + 1]
                if x0 <= lf <= x1:
                    t = (lf - x0) / (x1 - x0)
                    thresholds[f] = y0 + t * (y1 - y0)
                    info[f] = "interpolated"
                    break
    return thresholds, info


def threshold_at(freq: float, thresholds: dict) -> float:
    """任意频率处的听阈：标准点之间对数频率线性插值，范围外钳位。"""
    pts = sorted(thresholds.items())
    if freq <= pts[0][0]:
        return pts[0][1]
    if freq >= pts[-1][0]:
        return pts[-1][1]
    lf = math.log10(freq)
    for k in range(len(pts) - 1):
        f0, h0 = pts[k]
        f1, h1 = pts[k + 1]
        if f0 <= freq <= f1:
            t = (lf - math.log10(f0)) / (math.log10(f1) - math.log10(f0))
            return h0 + t * (h1 - h0)
    return pts[-1][1]


def prescribe_band(hl_edge_lo: float, hl_edge_hi: float) -> dict:
    """由频段两端听阈生成单段压缩参数（简化线性处方）。"""
    hl = 0.5 * (hl_edge_lo + hl_edge_hi)
    insertion_gain = max(0.0, min(40.0, 0.5 * hl))           # 半增益规则，封顶 40 dB
    ratio = max(1.0, min(4.0, 1.0 + hl / 25.0))              # HL 0->1:1, 75->4:1
    knee = max(-65.0, min(-40.0, -60.0 + 0.5 * hl))          # 软膝中点 dBFS
    ceiling = -3.0                                           # 输出限幅顶 dBFS
    return {
        "hl_band_db": round(hl, 1),
        "insertion_gain_db": round(insertion_gain, 1),
        "ratio": round(ratio, 2),
        "knee_dbfs": round(knee, 1),
        "ceiling_dbfs": ceiling,
        "attack_ms": 5.0,
        "release_ms": 80.0,
    }


def build_prescription(thresholds: dict) -> list:
    params = []
    for b in range(4):
        hl_lo = threshold_at(BAND_EDGES[b], thresholds)
        hl_hi = threshold_at(BAND_EDGES[b + 1], thresholds)
        params.append(prescribe_band(hl_lo, hl_hi))
    return params

