"""Progressive reconstruction from confident pulldown segments.

Given a low-residue anchor at field ``a`` (field a repeats field a-5), each
5-field block ending at the anchor holds one complete film frame in two
complementary fields.  Global field indices alternate top/bottom parity, so
the correct weave pair is derived directly from parity -- no phase table
needed:

* ``a`` repeats ``a-5``.  Enumerating the four cadence entry phases gives a
  parity-determined pairing (global indices alternate top/bottom):

  - ``a`` even (top):    top = a-4 (even), bottom = a-3 (odd)
  - ``a`` odd (bottom):  top = a-1 (even), bottom = a-2 (odd)

  Both members of each pair belong to the same film frame under every phase;
  no heuristic search is needed, and phase-shifted segments work unchanged.

Every output frame carries the exact two input field indices it was built
from (provenance), so the manifest is independently auditable.
"""

from __future__ import annotations

from dataclasses import dataclass

from .detector import Analysis, Segment
from .imageio import Field, VideoFrame, weave


@dataclass
class ReconstructedFrame:
    frame: VideoFrame
    top_field: int
    bottom_field: int
    anchor: int
    confidence: float
    segment_start: int


def _pair_for_anchor(fields: list[Field], anchor: int,
                     seg_start: int, seg_end: int):
    if anchor >= len(fields):
        return None
    if anchor % 2 == 0:
        top_index, bottom_index = anchor - 4, anchor - 3
    else:
        top_index, bottom_index = anchor - 1, anchor - 2
    # never reach across a cut / segment boundary
    if top_index < seg_start or bottom_index >= seg_end:
        return None
    top, bottom = fields[top_index], fields[bottom_index]
    return top, bottom


def reconstruct(analysis: Analysis, min_confidence: float = 0.9):
    """Weave progressive frames for every confident pulldown segment."""
    out: list[ReconstructedFrame] = []
    fields = analysis.fields
    out_index = 0
    for seg in analysis.segments:
        if seg.kind != "pulldown" or seg.confidence < min_confidence:
            continue
        for anchor in seg.anchors:
            pair = _pair_for_anchor(fields, anchor,
                                    seg.field_start, seg.field_end)
            if pair is None:
                continue
            top, bottom = pair
            source = {
                "mode": "ivtc_weave",
                "top_field": top.index,
                "bottom_field": bottom.index,
                "input_frame_top": top.source.get("frame"),
                "input_frame_bottom": bottom.source.get("frame"),
                "anchor": anchor,
                "segment_field_start": seg.field_start,
                "confidence": seg.confidence,
            }
            frame = weave(top, bottom, out_index, source)
            out.append(ReconstructedFrame(
                frame=frame,
                top_field=top.index,
                bottom_field=bottom.index,
                anchor=anchor,
                confidence=seg.confidence,
                segment_start=seg.field_start,
            ))
            out_index += 1
    return out


def manifest(frames: list[ReconstructedFrame], fps: float = 24.0 / 1.001) -> dict:
    """Build the auditable manifest mapping output frames to input fields."""
    return {
        "format": "ivtc-manifest-v1",
        "output_fps": fps,
        "frame_count": len(frames),
        "frames": [
            {
                "output_frame": item.frame.index,
                "pts_seconds": round(item.frame.index / fps, 6),
                "top_field_index": item.top_field,
                "bottom_field_index": item.bottom_field,
                "anchor_field": item.anchor,
                "source_interlaced_frames": sorted({
                    item.top_field // 2, item.bottom_field // 2}),
                "confidence": item.confidence,
                "segment_field_start": item.segment_start,
            }
            for item in frames
        ],
    }
