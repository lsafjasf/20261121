"""零依赖自检：python3 tests/selftest.py，全部通过退出码为 0。"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hearing_demo.audiogram import interpolate_threshold, build_prescription
from hearing_demo.filters import Crossover4
from hearing_demo.chain import ProcessingChain
from hearing_demo.signals import sine, demo_program
from hearing_demo.measurements import level_transition_measure
from hearing_demo.wavio import peak_dbfs

FS = 48000
FAILS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILS.append(name)


def test_interpolation():
    raw = {"500": 20, "1000": 40, "8000": 60}
    thr, info = interpolate_threshold(raw)
    check("对数频率插值 1k 间点", abs(thr[1000] - 40) < 1e-9)
    # 250 低于最低实测 500 -> 钳位
    check("低端钳位", info[250] == "clamped_low" and thr[250] == 20)
    check("高端钳位", info[4000] == "interpolated")
    params = build_prescription(thr)
    check("四段处方数量", len(params) == 4)
    check("增益封顶 40dB", all(p["insertion_gain_db"] <= 40 for p in params))


def test_crossover():
    xo = Crossover4(500, 2000, 6000, FS)
    sig = sine(1000, 0.5, FS, -20) + sine(3000, 0.5, FS, -20)
    bands = xo.split_block(sig)
    rec = xo.recombine(bands)
    err = math.sqrt(sum((a - b) ** 2 for a, b in zip(sig, rec))
                    / sum(a * a for a in sig))
    check("分频重组误差 < -60dB", 20 * math.log10(err) < -60,
          f"({20 * math.log10(err):.1f} dB)")
    # 分频点处相邻两带约 -3dB 均分
    tone = sine(2000, 1.0, FS, -20)
    b = xo.split_block(tone)
    from hearing_demo.meters import steady_level
    lvls = [steady_level(x, FS, 0.4, 0.9) for x in b]
    check("2k 分频点邻带均分", abs(lvls[1] - lvls[2]) < 1.0,
          f"({lvls[1]:.1f}/{lvls[2]:.1f})")


def test_bypass_and_limiter():
    params = build_prescription(interpolate_threshold(
        {"125": 25, "250": 30, "500": 35, "1000": 45,
         "2000": 55, "4000": 62.5, "8000": 70})[0])
    demo = demo_program(FS)
    bp = ProcessingChain(params, FS, bypass=True)
    out_bp, bin_b, bout_b = bp.process_block(demo)
    identity = all(
        abs(bin_b[i][k] - bout_b[i][k]) < 1e-12
        for i in range(4) for k in range(len(demo)))
    check("旁路分段严格恒等", identity)
    ac = ProcessingChain(params, FS, bypass=False)
    out_ac, _, _ = ac.process_block(demo)
    check("压缩输出峰值不超过 -3dBFS", peak_dbfs(out_ac) <= -2.9,
          f"(peak={peak_dbfs(out_ac):.1f})")


def test_envelope_timing():
    params = build_prescription(interpolate_threshold(
        {"500": 30, "1000": 40, "2000": 50, "4000": 60, "8000": 65})[0])
    t = level_transition_measure(params, FS)
    check("攻击时间接近 5ms 配置",
          t["attack_ms_measured"] is not None
          and abs(t["attack_ms_measured"] - 5.0) < 2.0,
          f"({t['attack_ms_measured']} ms)")
    check("释放时间接近 80ms 配置",
          t["release_ms_measured"] is not None
          and abs(t["release_ms_measured"] - 80.0) < 25.0,
          f"({t['release_ms_measured']} ms)")


if __name__ == "__main__":
    test_interpolation()
    test_crossover()
    test_bypass_and_limiter()
    test_envelope_timing()
    print("\n结果:", "全部通过" if not FAILS else f"失败 {len(FAILS)} 项: {FAILS}")
    sys.exit(1 if FAILS else 0)

