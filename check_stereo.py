"""Stereo material checker for dual-projector setups.

Usage: python3 check_stereo.py <case_name|left.mp4> [right.mp4]

Decodes separately encoded left/right eye videos, reads the geometric
barcode (source frame index) and the eye marker of every frame, then
reports:
  - start offset between the two eyes (in frames)
  - dropped source frames per eye
  - repeated (held) frames per eye
Sync judgement is based on the decoded frame indices only. Normal
horizontal parallax between the eyes changes pixels but not indices,
so it is never flagged as misalignment.
Outputs (written to out/):
  <name>_report.txt          human/machine readable report
  <name>_sbs.mp4             side-by-side per-frame view (as recorded)
  <name>_corrected_left.mp4  index-aligned, projector-ready left eye
  <name>_corrected_right.mp4 index-aligned, projector-ready right eye
  <name>_corrected_sbs.mp4   side-by-side view of the corrected output
"""

import os
import sys

import stereokit as sk

MAX_SHIFT = 8


def load_eye(path, expect_eye):
    frames = sk.decode_video(path)
    indices = []
    problems = []
    for t, buf in enumerate(frames):
        eye = sk.read_eye_marker(buf)
        idx = sk.read_barcode(buf)
        if eye != expect_eye:
            problems.append("frame %d: eye marker reads %r" % (t, eye))
        if idx is None:
            problems.append("frame %d: barcode unreadable" % t)
        indices.append(idx)
    return frames, indices, problems


def find_repeats(indices):
    out = []
    t = 0
    while t < len(indices) - 1:
        if indices[t + 1] == indices[t]:
            run = 1
            while t + run < len(indices) and indices[t + run] == indices[t]:
                run += 1
            out.append((indices[t], t, run))
            t += run
        else:
            t += 1
    return out


def find_drops(indices):
    have = set(i for i in indices if i is not None)
    if not have:
        return []
    return [i for i in range(min(have), max(have) + 1) if i not in have]


def first_positions(indices):
    pos = {}
    for t, idx in enumerate(indices):
        if idx is not None and idx not in pos:
            pos[idx] = t
    return pos


def mean_abs_diff(buf_a, buf_b):
    total = 0
    n = len(buf_a)
    for i in range(0, n, 16):
        total += abs(buf_a[i] - buf_b[i])
    return total / (n / 16)


def analyze(idx_l, idx_r):
    pos_l = first_positions(idx_l)
    pos_r = first_positions(idx_r)
    common = sorted(set(pos_l) & set(pos_r))
    start_offset = 0
    repeated = {r[0] for r in find_repeats(idx_l) + find_repeats(idx_r)}
    for s in common:
        if s not in repeated:
            start_offset = pos_l[s] - pos_r[s]
            break
    drift = {}
    for s in common:
        d = pos_l[s] - pos_r[s]
        drift.setdefault(d, []).append(s)
    return {
        "start_offset": start_offset,
        "common": common,
        "pos_l": pos_l,
        "pos_r": pos_r,
        "drift": drift,
        "drops_l": find_drops(idx_l),
        "drops_r": find_drops(idx_r),
        "repeats_l": find_repeats(idx_l),
        "repeats_r": find_repeats(idx_r),
    }


class Canvas:
    def __init__(self, w, h, color=(16, 16, 16)):
        self.w = w
        self.h = h
        self.buf = bytearray(bytes(color) * (w * h))

    def rect(self, x0, y0, w, h, color):
        row = bytes(color) * max(0, w)
        for y in range(max(0, y0), min(self.h, y0 + h)):
            xa, xb = max(0, x0), min(self.w, x0 + w)
            if xa < xb:
                i = (y * self.w + xa) * 3
                self.buf[i:i + (xb - xa) * 3] = row[:(xb - xa) * 3]

    def blit(self, frame, x0, y0):
        for y in range(sk.H):
            dst = ((y0 + y) * self.w + x0) * 3
            src = y * sk.W * 3
            self.buf[dst:dst + sk.W * 3] = frame[src:src + sk.W * 3]

    def text(self, x, y, s, color, scale=2):
        cx = x
        for ch in s.upper():
            glyph = sk.FONT.get(ch, sk.FONT[" "])
            for gy, row in enumerate(glyph):
                for gx, bit in enumerate(row):
                    if bit == "1":
                        self.rect(cx + gx * scale, y + gy * scale,
                                  scale, scale, color)
            cx += 4 * scale


def encode_canvas(frames, path, w, h):
    import subprocess
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", "%dx%d" % (w, h), "-r", str(sk.FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", path],
        stdin=subprocess.PIPE)
    for c in frames:
        proc.stdin.write(bytes(c.buf if isinstance(c, Canvas) else c))
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("encode failed: " + path)


def make_sbs(frames_l, frames_r, idx_l, idx_r, path):
    w, h = sk.W * 2 + 8, sk.H + 22
    n = max(len(frames_l), len(frames_r))
    out = []
    for t in range(n):
        c = Canvas(w, h)
        if t < len(frames_l):
            c.blit(frames_l[t], 0, 0)
        if t < len(frames_r):
            c.blit(frames_r[t], sk.W + 8, 0)
        a = idx_l[t] if t < len(idx_l) else None
        b = idx_r[t] if t < len(idx_r) else None
        ok = a is not None and a == b
        c.rect(sk.W, 0, 8, sk.H, (40, 200, 40) if ok else (220, 40, 40))
        c.text(4, sk.H + 4, "L T=%02d SRC=%s" % (t, a), (255, 220, 80))
        c.text(sk.W + 12, sk.H + 4, "R T=%02d SRC=%s" % (t, b), (255, 220, 80))
        c.text(w - 96, sk.H + 4, "SYNC" if ok else "SKEW",
               (40, 220, 40) if ok else (240, 60, 60))
        out.append(c)
    encode_canvas(out, path, w, h)


def make_corrected(frames_l, frames_r, ana, name):
    pos_l, pos_r = ana["pos_l"], ana["pos_r"]
    lo = min(set(pos_l) | set(pos_r))
    hi = max(set(pos_l) | set(pos_r))
    w, h = sk.W * 2 + 8, sk.H + 22
    corr_l, corr_r, sbs = [], [], []
    last = {"L": None, "R": None}
    for s in range(lo, hi + 1):
        pair = []
        for eye, frames, pos in (("L", frames_l, pos_l),
                                 ("R", frames_r, pos_r)):
            if s in pos:
                buf = bytearray(frames[pos[s]])
                last[eye] = buf
                held = False
            else:
                buf = bytearray(last[eye]) if last[eye] else sk.new_frame()
                held = True
            if held:
                for k in range(4):
                    sk.fill_rect(buf, k, k, sk.W - 2 * k, 1, (230, 40, 40))
                    sk.fill_rect(buf, k, sk.H - 1 - k, sk.W - 2 * k, 1,
                                 (230, 40, 40))
                sk.draw_barcode(buf, s)
                sk.fill_rect(buf, 40, sk.H - 16, 100, 12, (40, 40, 48))
                sk.draw_text(buf, 40, sk.H - 16, "SRC " + str(s),
                             (220, 220, 220), scale=2)
                sk.draw_text(buf, sk.W - 120, sk.H - 16,
                             "HELD SRC %d" % s, (240, 80, 80))
            pair.append(buf)
        corr_l.append(pair[0])
        corr_r.append(pair[1])
        c = Canvas(w, h)
        c.blit(pair[0], 0, 0)
        c.blit(pair[1], sk.W + 8, 0)
        c.rect(sk.W, 0, 8, sk.H, (40, 200, 40))
        c.text(4, sk.H + 4, "L SRC=%d" % s, (255, 220, 80))
        c.text(sk.W + 12, sk.H + 4, "R SRC=%d" % s, (255, 220, 80))
        c.text(w - 96, sk.H + 4, "ALIGNED", (40, 220, 40))
        sbs.append(c)
    encode_canvas(corr_l, "out/%s_corrected_left.mp4" % name, sk.W, sk.H)
    encode_canvas(corr_r, "out/%s_corrected_right.mp4" % name, sk.W, sk.H)
    encode_canvas(sbs, "out/%s_corrected_sbs.mp4" % name, w, h)


def build_report(name, idx_l, idx_r, ana, frames_l, frames_r, problems):
    lines = []
    lines.append("stereo check: %s" % name)
    lines.append("frames decoded: L=%d R=%d" % (len(idx_l), len(idx_r)))
    off = ana["start_offset"]
    if off == 0:
        lines.append("start offset: 0 (eyes start together)")
    elif off > 0:
        lines.append("start offset: LEFT eye starts %d frame(s) late" % off)
    else:
        lines.append("start offset: RIGHT eye starts %d frame(s) late" % -off)
    lines.append("dropped src frames L: %s" % (ana["drops_l"] or "none"))
    lines.append("dropped src frames R: %s" % (ana["drops_r"] or "none"))
    rep_l = ["src %d x%d @t%d" % (r[0], r[2], r[1]) for r in ana["repeats_l"]]
    rep_r = ["src %d x%d @t%d" % (r[0], r[2], r[1]) for r in ana["repeats_r"]]
    lines.append("repeated frames L: %s" % (rep_l or "none"))
    lines.append("repeated frames R: %s" % (rep_r or "none"))
    if len(ana["drift"]) > 1:
        drift_desc = ["shift %+d for src %s" % (d, v[:6])
                      for d, v in sorted(ana["drift"].items())]
        lines.append("position drift (L-R file pos by src): %s" % drift_desc)
    diffs = []
    for s in ana["common"][:: max(1, len(ana["common"]) // 8)]:
        diffs.append(mean_abs_diff(frames_l[ana["pos_l"][s]],
                                   frames_r[ana["pos_r"][s]]))
    if diffs:
        lines.append("mean abs pixel diff of index-aligned pairs: %.1f "
                     "(normal horizontal parallax; NOT used for sync "
                     "verdict)" % (sum(diffs) / len(diffs)))
    verdict = []
    verdict.append("OFFSET" if off else "no offset")
    verdict.append("DROPS" if (ana["drops_l"] or ana["drops_r"]) else "no drops")
    verdict.append("REPEATS" if (ana["repeats_l"] or ana["repeats_r"])
                   else "no repeats")
    lines.append("verdict: " + ", ".join(verdict))
    if problems:
        lines.append("decode warnings: %s" % problems)
    return "\n".join(lines)


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1
    if len(args) == 1:
        name = args[0]
        left = "cases/%s_left.mp4" % name
        right = "cases/%s_right.mp4" % name
    else:
        left, right = args
        name = os.path.basename(left).replace("_left.mp4", "")
    os.makedirs("out", exist_ok=True)
    frames_l, idx_l, prob_l = load_eye(left, "L")
    frames_r, idx_r, prob_r = load_eye(right, "R")
    ana = analyze(idx_l, idx_r)
    report = build_report(name, idx_l, idx_r, ana, frames_l, frames_r,
                          prob_l + prob_r)
    print(report)
    with open("out/%s_report.txt" % name, "w") as fh:
        fh.write(report + "\n")
    make_sbs(frames_l, frames_r, idx_l, idx_r, "out/%s_sbs.mp4" % name)
    make_corrected(frames_l, frames_r, ana, name)
    print("wrote out/%s_{report.txt,sbs.mp4,corrected_left.mp4,"
          "corrected_right.mp4,corrected_sbs.mp4}" % name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
