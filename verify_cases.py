"""Re-check all synthetic cases and assert the expected diagnosis.

Key guarantees under test:
  - normal horizontal parallax (caseA) is NOT reported as misalignment
  - a one-frame-late eye (caseB) IS reported as a start offset of 1
"""

import check_stereo as cs

EXPECT = {
    "caseA": {"offset": 0, "drops_l": [], "drops_r": [],
              "repeats_l": 0, "repeats_r": 0},
    "caseB": {"offset": -1, "drops_l": [], "drops_r": [],
              "repeats_l": 0, "repeats_r": 1},
    "caseC": {"offset": 0, "drops_l": [12, 27], "drops_r": [],
              "repeats_l": 0, "repeats_r": 0},
    "caseD": {"offset": 0, "drops_l": [], "drops_r": [],
              "repeats_l": 0, "repeats_r": 2},
}


def main():
    failed = False
    for name, exp in EXPECT.items():
        _fl, idx_l, pl = cs.load_eye("cases/%s_left.mp4" % name, "L")
        _fr, idx_r, pr = cs.load_eye("cases/%s_right.mp4" % name, "R")
        ana = cs.analyze(idx_l, idx_r)
        got = {
            "offset": ana["start_offset"],
            "drops_l": ana["drops_l"],
            "drops_r": ana["drops_r"],
            "repeats_l": len(ana["repeats_l"]),
            "repeats_r": len(ana["repeats_r"]),
        }
        ok = got == exp and not pl and not pr
        failed |= not ok
        print("%s: %s  got=%s" % (name, "PASS" if ok else "FAIL", got))
    print("ALL PASS" if not failed else "SOME FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
