"""Grayscale image/video IO and field manipulation.

Archive pipelines usually hand us raw Y' (``.yuv`` / ``.gray``) or per-frame
images.  We support:

* binary ``P5`` PGM (per frame or a concatenated stream),
* raw gray8 little/raw streams with explicit geometry,
* in-memory frame lists built by the sample generator.

Everything is single-plane luma, which is exactly what cadence analysis and
combing checks need.  ffmpeg can convert any real container to this format.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from . import backend


@dataclass(frozen=True)
class Field:
    """One temporal field (half the scan lines of a frame)."""

    index: int          # global field index in the stream, 0-based
    parity: str         # "top" (even source rows) or "bottom" (odd rows)
    width: int
    height: int         # height of this field (frame height // 2)
    buf: bytes          # row-major gray8, length width*height
    source: dict        # provenance metadata, always preserved

    def array(self):
        return backend.as_array(self.buf, self.width, self.height)


@dataclass(frozen=True)
class VideoFrame:
    """One full-height interlaced or progressive frame."""

    index: int
    width: int
    height: int
    buf: bytes
    source: dict

    def array(self):
        return backend.as_array(self.buf, self.width, self.height)


def split_fields(frame: VideoFrame) -> tuple[Field, Field]:
    """Split a frame into top (even rows) and bottom (odd rows) fields.

    Field 0 of the stream is conventionally the top field of frame 0.
    Global field parity therefore alternates T, B, T, B ...
    """
    w, h = frame.width, frame.height
    if h % 2:
        raise ValueError("frame height must be even for field splitting")
    fh = h // 2
    top = bytearray(w * fh)
    bot = bytearray(w * fh)
    for r in range(fh):
        base_even = (2 * r) * w
        base_odd = (2 * r + 1) * w
        out = r * w
        top[out:out + w] = frame.buf[base_even:base_even + w]
        bot[out:out + w] = frame.buf[base_odd:base_odd + w]
    t_idx = 2 * frame.index
    src = frame.source
    # nested per-parity provenance (synthetic clip) vs flat source (real IO)
    top_nested = src.get("top")
    bot_nested = src.get("bottom")
    if top_nested is not None and bot_nested is not None:
        top_src = {"frame": frame.index, **dict(top_nested)}
        bot_src = {"frame": frame.index, **dict(bot_nested)}
    else:
        top_src = {"origin": "split", "frame": frame.index,
                   "parity": "top", **dict(src)}
        bot_src = {"origin": "split", "frame": frame.index,
                   "parity": "bottom", **dict(src)}
    return (
        Field(t_idx, "top", w, fh, bytes(top), top_src),
        Field(t_idx + 1, "bottom", w, fh, bytes(bot), bot_src),
    )


def weave(top: Field, bottom: Field, out_index: int, source: dict) -> VideoFrame:
    """Weave two complementary fields back into a full-height frame."""
    if top.parity != "top" or bottom.parity != "bottom":
        raise ValueError("weave expects (top, bottom) in that order")
    if top.width != bottom.width or top.height != bottom.height:
        raise ValueError("field geometry mismatch")
    w, fh = top.width, top.height
    out = bytearray(w * 2 * fh)
    for r in range(fh):
        off = r * w
        out[(2 * r) * w:(2 * r) * w + w] = top.buf[off:off + w]
        out[(2 * r + 1) * w:(2 * r + 1) * w + w] = bottom.buf[off:off + w]
    return VideoFrame(out_index, w, 2 * fh, bytes(out), source)
