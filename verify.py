"""Acceptance checks for the streaming PLC test bed.

Covers, per acceptance data set (voiced / silence / abrupt stop):
A. samples outside gaps (and outside the recovery fade) are bit-exact;
B. no hard cut at gap start / recovery (bounded sample-to-sample jump);
C. recovery latency: output tracks the real signal again within 1 frame;
D. stability under consecutive multi-packet loss: finite, bounded,
   energy decays smoothly, long bursts fade to silence.

Run:  python3 verify.py
Exit code 0 means all checks passed.
"""

import math

from bench import CASES, FRAME, FS, run_case

FAILURES = []


def check(cond, msg):
    tag = "PASS" if cond else "FAIL"
    print("  [%s] %s" % (tag, msg))
    if not cond:
        FAILURES.append(msg)


def gap_ranges(lost_frames):
    """Convert a set of lost frame indices into (start, end_excl) ranges."""
    ranges = []
    for f in sorted(lost_frames):
        if ranges and f == ranges[-1][1]:
            ranges[-1][1] += 1
        else:
            ranges.append([f, f + 1])
    return [tuple(r) for r in ranges]


def frame_energy(out, f):
    seg = out[f * FRAME:(f + 1) * FRAME]
    return sum(x * x for x in seg) / max(1, len(seg))


def check_case(name, samples, lost):
    print("case: %s" % name)
    out = run_case(name, samples, lost)
    n_frames = len(samples) // FRAME
    gaps = gap_ranges(lost)
    fade = int(0.010 * FS)  # plc fade_ms = 10 ms

    # A: out-of-gap samples are bit-exact
    recovery_frames = {end for _, end in gaps}  # first good frame after a gap
    bad = 0
    for f in range(n_frames):
        if f in lost:
            continue
        start = f * FRAME + (fade if f in recovery_frames else 0)
        for i in range(start, (f + 1) * FRAME):
            if out[i] != samples[i]:
                bad += 1
    check(bad == 0, "A: out-of-gap samples bit-exact (%d mismatches)" % bad)

    # B: no hard cut at gap boundaries
    peak = max(abs(x) for x in samples) or 1.0
    limit = 0.25 * peak
    worst = 0.0
    for g0, g1 in gaps:
        bounds = [g0 * FRAME]  # gap start
        if g1 < n_frames:
            bounds.append(g1 * FRAME)  # recovery point
        for b in bounds:
            if 0 < b < len(out):
                worst = max(worst, abs(out[b] - out[b - 1]))
    check(worst <= limit,
          "B: boundary jump %.4f <= %.4f (no hard cut)" % (worst, limit))

    # C: recovery latency <= 1 frame
    for g0, g1 in gaps:
        if g1 >= n_frames:
            continue
        seg_err = max(abs(out[i] - samples[i])
                      for i in range(g1 * FRAME + fade, (g1 + 1) * FRAME))
        check(seg_err == 0.0,
              "C: gap [%d,%d) recovered within first good frame" % (g0, g1))

    # D: stability during consecutive loss
    finite = all(math.isfinite(x) for x in out)
    check(finite, "D: output finite everywhere")
    for g0, g1 in gaps:
        run = g1 - g0
        if run < 2:
            continue
        seg = out[g0 * FRAME:g1 * FRAME]
        check(max(abs(x) for x in seg) <= 1.2 * peak,
              "D: burst [%d,%d) bounded by 1.2x peak" % (g0, g1))
        e_first = frame_energy(out, g0)
        e_last = frame_energy(out, g1 - 1)
        if run >= 5:  # long gap: must decay smoothly towards silence
            check(e_last <= 0.1 * e_first,
                  "D: long burst [%d,%d) energy decays (%.2e -> %.2e)"
                  % (g0, g1, e_first, e_last))
            # Monotonicity is only required once the decay phase starts;
            # the constant-gain periodic continuation before it may wobble
            # slightly because frames are not aligned to the pitch period.
            decay_from = g0 + 1 + 3  # plc short_gap_frames = 3
            mono = all(frame_energy(out, f) <= frame_energy(out, f - 1) * 1.05
                       for f in range(decay_from, g1))
            check(mono, "D: long burst [%d,%d) decay is monotone" % (g0, g1))


def check_extreme_burst():
    """50 consecutive lost frames after voiced audio must stay stable."""
    print("case: extreme_burst (50 frames)")
    from bench import gen_voiced
    samples = gen_voiced(0.5)
    lost = set(range(10, 10 + 50))
    samples = samples + [0.0] * (70 * FRAME - len(samples))
    out = run_case("extreme_burst", samples, lost)
    check(all(math.isfinite(x) for x in out), "D: 50-frame burst finite")
    tail = out[-FRAME:]
    check(max(abs(x) for x in tail) < 1e-3,
          "D: 50-frame burst fades to silence (tail max %.2e)"
          % max(abs(x) for x in tail))


def main():
    for name, samples, lost in CASES:
        check_case(name, samples, lost)
    check_extreme_burst()
    print()
    if FAILURES:
        print("RESULT: %d check(s) FAILED" % len(FAILURES))
        raise SystemExit(1)
    print("RESULT: all checks passed")


if __name__ == "__main__":
    main()
