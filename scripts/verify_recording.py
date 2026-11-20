#!/usr/bin/env python3
"""校验 recorder 录下的 float32 raw 文件:

1. 有效输出: 全部样本有限、幅度 <= 1、非全静音;
2. 样本顺序: 每个非静音片段的中段, 相邻样本差分恒等于斜坡步进
   (单调递增、无丢样/重复/乱序)。淡化迁移是合法的过渡区:
   异常差分必须聚集成少量短促的"过渡带"(长度 <= MAX_TRANS),
   过渡带之外不允许任何乱序。

退出码 0 = PASS, 1 = FAIL。
"""
import array
import math
import sys

SILENCE = 1e-5
TRIM = 768          # 片段头尾裁剪样本数(覆盖淡化 480 + 若干周期)
MIN_SEG = 4096      # 参与差分校验的最短片段
MAX_TRANS = 1200    # 单次淡化过渡带允许的最大长度(25ms @48k)
MAX_TRANS_RUNS = 16 # 允许的最大过渡带个数


def main() -> int:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <file.raw>")
        return 2
    with open(sys.argv[1], "rb") as fh:
        x = array.array("f")
        x.frombytes(fh.read())
    n = len(x)
    ok = True

    def check(cond: bool, msg: str) -> None:
        nonlocal ok
        print(("PASS" if cond else "FAIL"), msg)
        ok = ok and cond

    check(n > 48000, f"样本数 {n} (>1s)")
    check(all_finite(x), "全部样本有限 (无 NaN/Inf)")
    peak = max((abs(v) for v in x), default=0.0)
    check(peak <= 1.0 + 1e-6, f"幅度限幅 peak={peak:.4f} <= 1")

    voice = [abs(v) > SILENCE for v in x]
    ratio = sum(voice) / n if n else 0.0
    check(ratio > 0.05, f"非静音占比 {ratio:.2%} (>5%)")

    segs = segments(voice)
    long_segs = [(a, b) for a, b in segs if b - a >= MIN_SEG]
    check(len(long_segs) >= 1, f"有效片段数 {len(long_segs)} (>= {MIN_SEG} 样本)")
    if not long_segs:
        print("RESULT: FAIL")
        return 1

    a, b = max(long_segs, key=lambda s: s[1] - s[0])
    diffs = [x[i + 1] - x[i] for i in range(a + TRIM, b - TRIM - 1)]
    diffs.sort()
    step = diffs[len(diffs) // 2]
    check(step > 0, f"斜坡单调递增, step={step:.3e}")
    tol = max(5e-7, 0.2 * step)

    # 收集全部异常差分位置, 聚成连续过渡带(间隔 <=4 视为同一带)
    bad_idx = []
    for a, b in long_segs:
        for i in range(a + TRIM, b - TRIM - 1):
            if abs(x[i + 1] - x[i] - step) > tol:
                bad_idx.append(i)
    runs = []
    for i in bad_idx:
        if runs and i - runs[-1][1] <= 4:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    for r in runs[:8]:
        print(f"  过渡带 @{r[0]} 长度 {r[1] - r[0] + 1}")
    check(len(runs) <= MAX_TRANS_RUNS,
          f"过渡带个数 {len(runs)} (<= {MAX_TRANS_RUNS})")
    longest = max((r[1] - r[0] + 1 for r in runs), default=0)
    check(longest <= MAX_TRANS,
          f"过渡带均为短淡化 (最长 {longest} <= {MAX_TRANS} 样本)")

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def all_finite(x) -> bool:
    return all(math.isfinite(v) for v in x)


def segments(voice):
    segs = []
    start = None
    for i, v in enumerate(voice):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segs.append((start, i))
            start = None
    if start is not None:
        segs.append((start, len(voice)))
    return segs


if __name__ == "__main__":
    sys.exit(main())
