"""Independent verification of reconstructed frames.

Two checks:

1. provenance: every output frame must cite exactly two distinct input fields
   of opposite parity, and the cited buffers must equal the woven pixels;
2. combing: a correctly matched weave has a low comb ratio at moving edges.
   As a negative control we deliberately weave neighbouring *mismatched*
   fields; the same metric must flag them strongly.
"""

from __future__ import annotations

from .imageio import weave
from .metrics import comb_energy
from .reconstruct import ReconstructedFrame


COMB_RATIO_LIMIT = 2.0
BOUNDARY_LIMIT = 2.0


def verify_reconstruction(analysis, reconstructed: list[ReconstructedFrame]):
    fields = analysis.fields
    rows = []
    passed = True
    for item in reconstructed:
        top = fields[item.top_field]
        bottom = fields[item.bottom_field]
        provenance_ok = (
            top.parity == "top" and bottom.parity == "bottom"
            and top.index != bottom.index
        )
        rewoven = weave(top, bottom, item.frame.index, {})
        pixels_ok = rewoven.buf == item.frame.buf
        stats = comb_energy(item.frame)
        comb_ok = (stats["comb_ratio"] < COMB_RATIO_LIMIT
                   and stats["boundary_mismatch"] < BOUNDARY_LIMIT)
        ok = provenance_ok and pixels_ok and comb_ok
        passed = passed and ok
        rows.append({
            "output_frame": item.frame.index,
            "top_field": item.top_field,
            "bottom_field": item.bottom_field,
            "provenance_ok": provenance_ok,
            "pixels_match_manifest": pixels_ok,
            "comb_ratio": round(stats["comb_ratio"], 4),
            "boundary_mismatch": round(stats["boundary_mismatch"], 4),
            "passed": ok,
        })

    # negative control: cross-weave two adjacent frames from a pulldown run
    control = negative_control(analysis)
    return {
        "format": "ivtc-verification-v1",
        "comb_ratio_limit": COMB_RATIO_LIMIT,
        "boundary_limit": BOUNDARY_LIMIT,
        "all_passed": passed,
        "frames": rows,
        "negative_control": control,
    }


def negative_control(analysis):
    """Weave fields from two different time instants on purpose."""
    fields = analysis.fields
    # find two consecutive top fields inside a pulldown segment
    seg = next((s for s in analysis.segments if s.kind == "pulldown"
                and len(s.anchors) >= 3), None)
    if seg is None:
        return {"available": False}
    top_a = fields[seg.field_start]
    bottom_foreign = None
    for idx in range(seg.field_start + 3, seg.field_end):
        if fields[idx].parity == "bottom":
            bottom_foreign = fields[idx]
            break
    if bottom_foreign is None:
        return {"available": False}
    bad = weave(top_a, bottom_foreign, 0, {"note": "intentional mismatch"})
    stats_bad = comb_energy(bad)
    good = weave(top_a, fields[seg.field_start + 1], 1, {})
    stats_good = comb_energy(good)
    return {
        "available": True,
        "matched_weave": {
            "top_field": top_a.index,
            "bottom_field": seg.field_start + 1,
            **{k: round(v, 4) if isinstance(v, float) else v
               for k, v in stats_good.items()},
        },
        "mismatched_weave": {
            "top_field": top_a.index,
            "bottom_field": bottom_foreign.index,
            **{k: round(v, 4) if isinstance(v, float) else v
               for k, v in stats_bad.items()},
        },
        "mismatch_detected": stats_bad["comb_ratio"] > COMB_RATIO_LIMIT,
    }
