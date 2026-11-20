"""Surgical ISO-BMFF re-writer.

Rebuilds the moov box after replacing edit lists and duration fields, while
leaving mdat sample bytes untouched. The fixture is written with moov last
(default non-faststart ffmpeg output), so growing moov does not invalidate
stco chunk offsets.
"""
from __future__ import annotations

import struct

from . import mp4box


def _build_elst(edits: list[tuple[int, int]], version: int = 0) -> bytes:
    # edits: (segment_duration_movie_units, media_time_units or -1)
    payload = struct.pack(">I", len(edits))
    for segment, media_time in edits:
        if version == 1:
            mt = media_time if media_time >= 0 else (1 << 64) - 1
            payload += struct.pack(">Q", segment)
            payload += struct.pack(">Q", mt)
            payload += struct.pack(">H", 1) + struct.pack(">H", 0)
        else:
            payload += struct.pack(">I", segment)
            payload += struct.pack(">i", media_time)
            payload += struct.pack(">H", 1) + struct.pack(">H", 0)
    body = bytes([version, 0, 0, 0]) + payload
    return struct.pack(">I4s", 8 + len(body), b"elst") + body


def _build_edts(edits: list[tuple[int, int]]) -> bytes:
    elst = _build_elst(edits)
    return struct.pack(">I4s", 8 + len(elst), b"edts") + elst


def _box(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(body), kind) + body


def _patch_tkhd_bytes(trak_bytes: bytes, duration: int) -> bytes:
    out = bytearray(trak_bytes)
    idx = out.find(b"tkhd")
    if idx < 0:
        raise ValueError("tkhd not found")
    payload = idx + 4  # version byte
    if out[payload] == 1:
        p = payload + 4 + 8 + 16 + 4
        struct.pack_into(">Q", out, p, duration)
    else:
        p = payload + 4 + 8 + 8 + 4
        struct.pack_into(">I", out, p, duration)
    return bytes(out)


def _mvhd_box(mvhd: mp4box.Box, data: bytes, duration: int) -> bytes:
    body = bytearray(data[mvhd.payload_start:mvhd.end])
    if body[0] == 1:
        p = 4 + 16
        struct.pack_into(">Q", body, p + 4, duration)
    else:
        p = 4 + 8
        struct.pack_into(">I", body, p + 4, duration)
    return _box(b"mvhd", bytes(body))


def _serialize(box: mp4box.Box, data: bytes,
               edts_override: bytes | None) -> bytes:
    """Re-emit a container verbatim, optionally replacing/inserting edts."""
    if not box.children:
        return data[box.start:box.end]
    payload = bytearray()
    inserted = edts_override is None
    for child in box.children:
        if child.type == "edts":
            continue
        payload += _serialize(child, data, None)
        if box.type == "trak" and child.type == "tkhd" and not inserted:
            payload += edts_override
            inserted = True
    if box.type == "trak" and not inserted:
        payload = edts_override + bytes(payload)
    return _box(box.type.encode(), bytes(payload))


def _handler_of(trak: mp4box.Box, data: bytes) -> str:
    hdlr = trak.find("mdia", "hdlr")
    return data[hdlr.payload_start + 8:hdlr.payload_start + 12].decode("latin1")


def _rebuild_trak(trak: mp4box.Box, data: bytes, plan: dict) -> bytes:
    handler = _handler_of(trak, data)
    if handler not in plan:
        return data[trak.start:trak.end]
    spec = plan[handler]
    edts = _build_edts(spec["edits"])
    trak_bytes = _serialize(trak, data, edts)
    return _patch_tkhd_bytes(trak_bytes, spec["track_duration_movie_units"])


def rewrite_movie(data: bytes, plan: dict) -> bytes:
    """Replace edit lists and durations, returning a new MP4 byte string.

    plan maps handler type ('vide'/'soun') to:
      {"edits": [(segment_duration, media_time_or_-1), ...],
       "track_duration_movie_units": int}
    The mvhd duration becomes the maximum edited track duration.
    """
    roots = mp4box.scan_boxes(data, 0, len(data))
    moov = next(b for b in roots if b.type == "moov")
    mvhd = moov.find("mvhd")

    max_duration = max(
        (spec["track_duration_movie_units"] for spec in plan.values()),
        default=0,
    )

    children = bytearray()
    for child in moov.children:
        if child.type == "mvhd":
            children += _mvhd_box(mvhd, data, max_duration)
        elif child.type == "trak":
            children += _rebuild_trak(child, data, plan)
        else:
            children += data[child.start:child.end]

    new_moov = _box(b"moov", bytes(children))
    return data[:moov.start] + new_moov + data[moov.end:]
