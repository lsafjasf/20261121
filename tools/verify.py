"""Verify a migrated IMSC document against its TTML source using real
rendered frames (ffmpeg/libass), plus model-driven timing checks.

Checks per document pair:
  1. frame diff   - per-frame pixel difference between source and migrated
  2. overlap      - pixel intersection of simultaneously active cues
  3. flicker      - rendered presence vs expected active frames per cue
  4. fractional   - first rendered frame for cues with off-grid in-points
"""
import json
import math
import os
from fractions import Fraction

import ttml_model as M
import ass_render as R

PIXEL_THRESHOLD = 40
DIFF_RATIO_FLAG = 0.0005   # fraction of pixels changed to flag a frame
OVERLAP_PIXELS_FLAG = 20   # intersecting pixels to flag an overlap

_TABLE = bytes([0] * (PIXEL_THRESHOLD + 1)) + bytes([1] * (255 - PIXEL_THRESHOLD))


def _mask_int(frame_bytes):
    return int.from_bytes(frame_bytes.translate(_TABLE), "big")


def _active(cue, frame_n, fps):
    """True if the cue is active at the frame's presentation instant n/fps
    (point sampling: begin <= n/fps < end), matching how real renderers
    (libass, imscJS, players) sample timed text."""
    end = cue.end if cue.end is not None else Fraction(10**9)
    return cue.begin * fps <= frame_n < end * fps


def _ranges(frames):
    """[1,2,3,7,8] -> '1-3, 7-8'"""
    if not frames:
        return ""
    out, start, prev = [], frames[0], frames[0]
    for f in frames[1:]:
        if f == prev + 1:
            prev = f
            continue
        out.append("%d-%d" % (start, prev) if start != prev else str(start))
        start = prev = f
    out.append("%d-%d" % (start, prev) if start != prev else str(start))
    return ", ".join(out)


def analyze(src_path, imsc_path, workdir):
    os.makedirs(workdir, exist_ok=True)
    frames_dir = os.path.join(workdir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    src = M.parse_document(src_path)
    imsc = M.parse_document(imsc_path)
    tp = imsc.params
    fps = tp.frame_rate
    W, H = tp.root_extent

    ends = [c.end for c in src.cues + imsc.cues if c.end is not None]
    duration = max(ends) if ends else Fraction(5)
    n_frames = int(math.ceil(float(duration) * float(fps))) + 2

    src_ass = os.path.join(workdir, "source.ass")
    imsc_ass = os.path.join(workdir, "imsc.ass")
    with open(src_ass, "w") as fh:
        fh.write(R.build_ass(src))
    with open(imsc_ass, "w") as fh:
        fh.write(R.build_ass(imsc))

    report = {"source": src_path, "imsc": imsc_path, "fps": str(fps),
              "frames": n_frames, "findings": [], "checks": {}}

    def find(check, message, severity="error", **extra):
        entry = {"check": check, "severity": severity, "message": message}
        entry.update(extra)
        report["findings"].append(entry)

    # ---- 1. frame diff (source vs migrated) ------------------------------
    src_gray = os.path.join(workdir, "src.gray")
    imsc_gray = os.path.join(workdir, "imsc.gray")
    R.render_gray(src_ass, tp, n_frames, src_gray)
    R.render_gray(imsc_ass, tp, n_frames, imsc_gray)
    changed = []
    total = W * H
    for n, (fa, fb) in enumerate(zip(R.read_gray_frames(src_gray, W, H),
                                     R.read_gray_frames(imsc_gray, W, H))):
        diff = (_mask_int(fa) ^ _mask_int(fb)).bit_count()
        if diff > total * DIFF_RATIO_FLAG:
            changed.append(n)
    report["checks"]["frame_diff"] = {
        "changed_frames": _ranges(changed) or "none",
        "changed_count": len(changed),
    }
    if changed:
        find("frame_diff",
             "migrated render differs from source on frames: %s"
             % _ranges(changed),
             severity="warning", frames=_ranges(changed))
    os.remove(src_gray)
    os.remove(imsc_gray)

    # ---- per-cue masks of the migrated document --------------------------
    cue_masks = {}
    for cue in imsc.cues:
        ass_path = os.path.join(workdir, "cue_%s.ass" % cue.id)
        with open(ass_path, "w") as fh:
            fh.write(R.build_ass(imsc, only_cue=cue))
        gray = os.path.join(workdir, "cue_%s.gray" % cue.id)
        R.render_gray(ass_path, tp, n_frames, gray)
        cue_masks[cue.id] = [_mask_int(f)
                             for f in R.read_gray_frames(gray, W, H)]
        os.remove(gray)
        os.remove(ass_path)

    # ---- 2. overlap -------------------------------------------------------
    cues = imsc.cues
    overlap_groups = {}
    for n in range(n_frames):
        active = [c for c in cues if _active(c, n, fps)]
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                a, b = active[i], active[j]
                inter = (cue_masks[a.id][n] & cue_masks[b.id][n]).bit_count()
                if inter > OVERLAP_PIXELS_FLAG:
                    key = (a.id, b.id, a.region == b.region)
                    overlap_groups.setdefault(key, []).append((n, inter))
    for (aid, bid, same_region), lst in sorted(overlap_groups.items()):
        frames = [n for n, _ in lst]
        peak = max(i for _, i in lst)
        kind = "same-region stacking" if same_region else "cross-region overlap"
        find("overlap",
             "%s: cues %s and %s intersect (%d px peak) on frames %s"
             % (kind, aid, bid, peak, _ranges(frames)),
             severity="info" if same_region else "error",
             frames=_ranges(frames))
    report["checks"]["overlap"] = {"incident_groups": len(overlap_groups)}

    # ---- 3. flicker / presence -------------------------------------------
    flicker = []
    for cue in cues:
        masks = cue_masks[cue.id]
        expected = [_active(cue, n, fps) for n in range(n_frames)]
        rendered = [m != 0 for m in masks]
        mismatch = [n for n in range(n_frames)
                    if expected[n] != rendered[n]]
        if mismatch:
            find("flicker",
                 "cue %s: rendered presence differs from timing on frames %s"
                 % (cue.id, _ranges(mismatch)),
                 severity="error", frames=_ranges(mismatch))
        act_len = 0
        for n in range(n_frames):
            if expected[n]:
                act_len += 1
            elif act_len:
                if act_len == 1:
                    flicker.append((cue.id, n - 1))
                act_len = 0
        if act_len == 1:
            flicker.append((cue.id, n_frames - 1))
    for cid, n in flicker:
        find("flicker",
             "cue %s is active for a single frame (frame %d): flash-frame risk"
             % (cid, n),
             severity="warning", frames=str(n))
    report["checks"]["flicker"] = {"single_frame_cues":
                                   ["%s@%d" % t for t in flicker]}

    # ---- 4. fractional in-points ------------------------------------------
    frac_checks = []
    for cue in cues:
        pos = cue.begin * fps
        if pos.denominator == 1:
            continue
        # first frame whose presentation instant is >= begin: ceil(begin*fps)
        first_expected = -(-pos.numerator // pos.denominator)
        masks = cue_masks[cue.id]
        first_rendered = next((n for n in range(n_frames) if masks[n]), None)
        ok = first_rendered == first_expected
        frac_checks.append({"cue": cue.id, "begin_s": float(cue.begin),
                            "expected_first_frame": first_expected,
                            "rendered_first_frame": first_rendered,
                            "ok": ok})
        find("fractional",
             "cue %s fractional in-point %s: first frame %s, expected %d [%s]"
             % (cue.id, float(cue.begin), first_rendered, first_expected,
                "PASS" if ok else "FAIL"),
             severity="info" if ok else "error")
    report["checks"]["fractional"] = frac_checks
    return report, src, imsc, n_frames, find


def export_keyframes(report, src, imsc, workdir):
    """PNG stills: multi-line cue, simultaneous multi-region, fractional
    in-point (frame before / at / after) for the migrated document."""
    tp = imsc.params
    fps = tp.frame_rate
    frames_dir = os.path.join(workdir, "frames")
    imsc_ass = os.path.join(workdir, "imsc.ass")
    src_ass = os.path.join(workdir, "source.ass")
    shots = []

    multi = next((c for c in imsc.cues if len(c.lines) > 1), None)
    if multi and multi.end is not None:
        n = int((multi.begin + multi.end) / 2 * fps)
        shots.append(("multiline", n, "imsc"))
        shots.append(("multiline", n, "src"))

    # simultaneous activity in >=2 distinct regions
    ends = sorted({c.begin for c in imsc.cues} |
                  {c.end for c in imsc.cues if c.end is not None})
    for i in range(len(ends) - 1):
        mid = (ends[i] + ends[i + 1]) / 2
        regs = {c.region for c in imsc.cues
                if c.begin <= mid and (c.end is None or mid < c.end)}
        if len(regs) >= 2:
            n = int(mid * fps)
            shots.append(("multiregion", n, "imsc"))
            shots.append(("multiregion", n, "src"))
            break

    for chk in report["checks"].get("fractional", []):
        n = chk["expected_first_frame"]
        for tag, fn in (("pre", n - 1), ("at", n), ("post", n + 1)):
            if fn >= 0:
                shots.append(("fractional_%s_cue%s" % (tag, chk["cue"]),
                              fn, "imsc"))
        break  # one representative fractional cue is enough

    for f in report["findings"]:
        if f["check"] == "overlap" and f["severity"] == "error":
            first = int(str(f.get("frames", "0")).split("-")[0].split(",")[0])
            shots.append(("overlap", first, "imsc"))
            break

    written = []
    for tag, n, which in shots:
        ass = imsc_ass if which == "imsc" else src_ass
        out = os.path.join(frames_dir, "%s_%s_f%d.png" % (tag, which, n))
        R.render_frame_png(ass, tp, n, out)
        written.append(out)
    return written


def verify_pair(src_path, imsc_path, workdir):
    report, src, imsc, n_frames, _ = analyze(src_path, imsc_path, workdir)
    report["keyframes"] = export_keyframes(report, src, imsc, workdir)
    with open(os.path.join(workdir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    return report


if __name__ == "__main__":
    import sys
    rep = verify_pair(sys.argv[1], sys.argv[2], sys.argv[3])
    for f in rep["findings"]:
        print("[%s] %s: %s" % (f["severity"].upper(), f["check"], f["message"]))
