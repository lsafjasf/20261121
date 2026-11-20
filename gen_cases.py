"""Generate synthetic per-eye test videos (separately encoded h264).

Each case writes cases/<name>_left.mp4 and cases/<name>_right.mp4.
The right eye always includes the normal horizontal PARALLAX shift.

Cases:
  caseA  clean, in sync, parallax only        -> expect: no defect
  caseB  right eye one frame late (start hold)-> expect: offset +1
  caseC  left eye drops source frames 12, 27  -> expect: 2 drops in L
  caseD  right eye repeats src 8 (x2), 30 (x3)-> expect: repeats in R
"""

import os
import stereokit as sk

N = 40


def build(seq, eye):
    return [sk.render_source_frame(i, eye) for i in seq]


def main():
    os.makedirs("cases", exist_ok=True)
    base = list(range(N))

    cases = {
        "caseA": (base, base),
        "caseB": (base, [0] + list(range(N - 1))),
        "caseC": ([i for i in base if i not in (12, 27)], base),
        "caseD": (base, [i for i in base for _ in
                         range(3 if i == 30 else (2 if i == 8 else 1))]),
    }

    for name, (lseq, rseq) in cases.items():
        sk.encode_video(build(lseq, "L"), "cases/%s_left.mp4" % name)
        sk.encode_video(build(rseq, "R"), "cases/%s_right.mp4" % name)
        print("%s: L=%d frames R=%d frames" % (name, len(lseq), len(rseq)))


if __name__ == "__main__":
    main()
