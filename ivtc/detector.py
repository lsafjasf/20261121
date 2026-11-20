"""3:2 pulldown / native-interlace / cut detector.

Pipeline:

1. Split every interlaced frame into top/bottom fields.
2. Compute same-parity residuals r[n] = MAD(field n, field n-2).
3. Classify residues into "low" (repeated field) vs "high" with an adaptive
   threshold derived from the residue distribution.
4. Chain low residues at exact 5-field spacing: that chain *is* the pulldown
   cadence.  A chain of k anchors spans k-1 film frames (each 5-field block
   reconstructs one film frame between two repeats).
5. Anything with no supported chain is labelled native interlaced.  Gaps
   shorter than a full cadence cycle are "uncertain" and never reconstructed.
6. Cuts come from two independent signals: frame-diff spikes, and cadence
   breaks (a cut landing inside a locked pulldown run destroys the 5 spacing).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from .imageio import Field, VideoFrame, split_fields
from .metrics import adjacent_frame_diffs, same_parity_residues


@dataclass
class Segment:
    kind: str                  # "pulldown" | "native" | "uncertain"
    field_start: int
    field_end: int             # exclusive
    confidence: float
    anchors: list[int] = field(default_factory=list)
    phase: int | None = None   # low-residue phase class, 0..4
    note: str = ""

    @property
    def frame_start(self) -> int:
        return self.field_start // 2

    @property
    def frame_end(self) -> int:
        return (self.field_end + 1) // 2


@dataclass
class Cut:
    field_index: int
    frame_index: int
    frame_diff: float
    residue_left: float | None
    residue_right: float | None
    reason: str


@dataclass
class Analysis:
    fields: list[Field]
    residues: list
    frame_diffs: list
    threshold: float
    segments: list[Segment]
    cuts: list[Cut]


def _adaptive_threshold(residues: list) -> float:
    vals = sorted(r for r in residues if r is not None)
    if not vals:
        return 1.0
    median = statistics.median(vals)
    # Repeated fields differ only by compression/grain noise; active fields
    # differ by real motion.  A factor of 0.25 sits well inside that gap, and
    # an absolute floor keeps near-static streams from over-segmenting.
    return max(0.5, 0.25 * median)


def _build_anchor_chains(low: dict[int, float]) -> list[list[int]]:
    """Group low-residue indices connected by exact 5-field spacing."""
    anchors = sorted(low)
    chains: list[list[int]] = []
    used: set[int] = set()
    for anchor in anchors:
        if anchor in used:
            continue
        chain = [anchor]
        used.add(anchor)
        cursor = anchor
        while cursor + 5 in low and cursor + 5 not in used:
            cursor += 5
            chain.append(cursor)
            used.add(cursor)
        # extend backwards as well, in case iteration started mid-chain
        cursor = anchor
        while cursor - 5 in low and cursor - 5 not in used:
            cursor -= 5
            chain.append(cursor)
            used.add(cursor)
        chains.append(sorted(chain))
    return sorted(chains, key=lambda c: c[0])


def _phase_of(anchor: int) -> int:
    return anchor % 5


MIN_CYCLES = 2  # need >=2 cadence cycles (10 fields) before we reconstruct


def analyze(frames: list[VideoFrame], min_confidence: float = 0.9) -> Analysis:
    fields: list[Field] = []
    for frame in frames:
        top, bottom = split_fields(frame)
        fields.extend((top, bottom))

    residues = same_parity_residues(fields)
    frame_diffs = adjacent_frame_diffs(frames)
    threshold = _adaptive_threshold(residues)
    low = {n: r for n, r in enumerate(residues)
           if r is not None and r <= threshold}
    chains = _build_anchor_chains(low)

    segments: list[Segment] = []
    cuts: list[Cut] = []
    n_fields = len(fields)

    # ---- frame-diff derived cut candidates (robust median-MAD gate) ----
    diff_vals = [d for d in frame_diffs if d is not None]
    diff_med = statistics.median(diff_vals) if diff_vals else 0.0
    diff_mad = statistics.median([abs(d - diff_med) for d in diff_vals]) \
        if diff_vals else 0.0
    cut_gate = max(diff_med + 6.0 * (diff_mad + 1e-9), 3.0 * diff_med + 0.5)
    diff_cuts = {i for i, d in enumerate(frame_diffs)
                 if d is not None and d >= cut_gate}

    # ---- pulldown segments from anchor chains ---------------------------
    # Cut fields break the timeline first: a pulldown run can never cross a
    # scene change, so chain-derived ranges are clipped to cut boundaries.
    cut_fields = sorted({2 * fi for fi in diff_cuts})
    covered: set[int] = set()
    for chain in chains:
        if len(chain) < MIN_CYCLES:
            continue
        start = max(0, chain[0] - 5)
        end = min(n_fields, chain[-1] + 5)
        for cut in cut_fields:
            if start < cut <= end:
                if cut <= chain[0]:
                    start = max(start, cut)
                else:
                    end = min(end, cut)
        # guard against partial blocks left after clipping
        if end - start < 10:
            continue
        ratios = [low[a] / threshold if threshold else 0.0 for a in chain]
        spacing_ok = all(b - a == 5 for a, b in zip(chain, chain[1:]))
        confidence = round(
            max(0.0, min(1.0,
                         1.0 - statistics.mean(ratios) * 0.5
                         - (0.0 if spacing_ok else 0.2))), 3)
        seg = Segment(
            kind="pulldown",
            field_start=start,
            field_end=end,
            confidence=confidence,
            anchors=list(chain),
            phase=_phase_of(chain[0]),
            note=("reconstructed" if confidence >= min_confidence
                  else "below confidence floor, not reconstructed"),
        )
        segments.append(seg)
        covered.update(range(start, end))

        # cuts inside an otherwise-locked run: cadence break at chain edge
        for boundary, side in ((start, "left"), (end, "right")):
            if 0 < boundary < n_fields and boundary // 2 in diff_cuts:
                cuts.append(Cut(
                    field_index=boundary,
                    frame_index=boundary // 2,
                    frame_diff=frame_diffs[boundary // 2],
                    residue_left=residues[boundary - 1]
                    if boundary - 1 >= 0 else None,
                    residue_right=residues[boundary]
                    if boundary < n_fields else None,
                    reason=f"frame-diff spike + cadence break ({side} edge)",
                ))

    # ---- frame-diff cuts that did not coincide with a cadence edge ------
    for fi in sorted(diff_cuts):
        field_i = 2 * fi
        if all(abs(c.field_index - field_i) > 1 for c in cuts):
            cuts.append(Cut(
                field_index=field_i,
                frame_index=fi,
                frame_diff=frame_diffs[fi],
                residue_left=residues[field_i - 1] if field_i >= 1 else None,
                residue_right=residues[field_i]
                if field_i < n_fields else None,
                reason="frame-diff spike (no local pulldown lock)",
            ))

    # ---- fill uncovered ranges as native / uncertain --------------------
    cursor = 0
    filled: list[Segment] = []
    for seg in sorted(segments, key=lambda s: s.field_start):
        if seg.field_start > cursor:
            filled.append(_classify_gap(cursor, seg.field_start, residues,
                                        threshold))
        filled.append(seg)
        cursor = max(cursor, seg.field_end)
    if cursor < n_fields:
        filled.append(_classify_gap(cursor, n_fields, residues, threshold))

    # merge adjacent same-kind native runs for a cleaner timeline
    merged: list[Segment] = []
    for seg in filled:
        if (merged and merged[-1].kind == seg.kind
                and merged[-1].field_end == seg.field_start
                and seg.kind != "pulldown"):
            prev = merged.pop()
            merged.append(Segment(seg.kind, prev.field_start, seg.field_end,
                                  min(prev.confidence, seg.confidence),
                                  note=prev.note))
        else:
            merged.append(seg)

    cuts.sort(key=lambda c: c.field_index)
    return Analysis(fields, residues, frame_diffs, threshold, merged, cuts)


def _classify_gap(start: int, end: int, residues: list,
                  threshold: float) -> Segment:
    vals = [r for r in residues[start:end] if r is not None]
    if not vals:
        return Segment("uncertain", start, end, 0.0,
                       note="not enough fields to measure")
    low_ratio = sum(1 for r in vals if r <= threshold) / len(vals)
    span = end - start
    if span < 6:
        return Segment("uncertain", start, end, round(1.0 - low_ratio, 3),
                       note="shorter than a full cadence cycle")
    # native 60i: residuals stay high and show no 5-field structure
    if low_ratio <= 0.1:
        return Segment("native", start, end, round(1.0 - low_ratio, 3),
                       note="same-parity fields always differ: native 60i")
    return Segment("uncertain", start, end, round(1.0 - low_ratio, 3),
                   note="mixed residue pattern around a transition/cut")
