"""Cadence and combing metrics.

The whole detector rests on one observation about 3:2 pulldown.  Film frames
are emitted as field pairs over 4 interlaced frames (10 fields), and two of
the five pairs contain a repeated field.  Comparing fields two temporal steps
apart (same parity), the stream therefore produces near-zero residuals at a
perfect 5-field periodicity::

    fields      f0 f1 f2 f3 f4 f5 f6 f7 f8 f9
    same parity  .  .  R0 .  .  R5 .  .  .  .

Native 60i material has genuinely different content in every field, so no
periodic low residuals appear.  A hard cut breaks the rhythm: residuals around
the cut spike regardless of cadence.
"""

from __future__ import annotations

from . import backend
from .imageio import Field, VideoFrame, weave


def same_parity_residues(fields: list[Field]) -> list[float]:
    """MAD between field n and field n-2 (same parity), for n >= 2.

    Returned list is aligned to global field indices; entries 0 and 1 are
    ``None`` so callers can index directly with field indices.
    """
    residues: list[float | None] = [None, None]
    arrays = [f.array() for f in fields]
    for n in range(2, len(fields)):
        residues.append(backend.mean_abs_subset(arrays[n], arrays[n - 2]))
    return residues  # type: ignore[return-value]


def adjacent_frame_diffs(frames: list[VideoFrame]) -> list[float]:
    """Full-frame MAD between consecutive interlaced frames; cut signal."""
    diffs: list[float | None] = [None]
    arrays = [f.array() for f in frames]
    for n in range(1, len(frames)):
        diffs.append(backend.mean_abs_subset(arrays[n], arrays[n - 1]))
    return diffs  # type: ignore[return-value]


def comb_energy(frame: VideoFrame, region_rows: int | None = None) -> dict:
    """Measure interlace mismatch of a woven frame.

    For a correctly woven progressive frame each even scan line sits close to
    both neighbouring odd lines (vertical interpolation error is small).  Two
    fields from different time instants create comb teeth at moving edges:
    the even line deviates strongly from the local odd-line average.

    Returns mean boundary mismatch, mean within-field smoothness (control),
    and their ratio (the comb ratio; ~1 for matched, >>1 for mismatched).
    """
    w, h = frame.width, frame.height
    buf = frame.buf
    r0, r1 = (1, h - 1) if region_rows is None else (
        max(1, h // 2 - region_rows), min(h - 1, h // 2 + region_rows))
    boundary_total = 0
    smooth_total = 0
    count = 0
    for r in range(r0, r1):
        row = r * w
        row_up = (r - 1) * w
        row_dn = (r + 1) * w
        for x in range(w):
            vert_mean = (buf[row_up + x] + buf[row_dn + x]) / 2
            boundary_total += abs(buf[row + x] - vert_mean)
            # within-field neighbours are two rows away (same parity)
            if r + 2 < h:
                smooth_total += abs(buf[row + x] - buf[(r + 2) * w + x])
            count += 1
    boundary = boundary_total / count
    smooth = smooth_total / max(1, count - w)
    ratio = boundary / smooth if smooth > 1e-9 else float("inf")
    return {
        "boundary_mismatch": boundary,
        "within_field_smoothness": smooth,
        "comb_ratio": ratio,
        "rows_scanned": r1 - r0,
    }


def weave_pair_energy(top: Field, bottom: Field) -> dict:
    """Convenience: comb energy of a candidate top/bottom weave."""
    frame = weave(top, bottom, 0, {"note": "probe"})
    return comb_energy(frame)
