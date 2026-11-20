from __future__ import annotations

from pathlib import Path

from .model import LayerTags, tags_from_codes

EBML_HEADER = 0x1A45DFA3
SEGMENT = 0x18538067
TRACKS = 0x1654AE6B
TRACK_ENTRY = 0xAE
TRACK_TYPE = 0x83
VIDEO = 0xE0
COLOUR = 0x55B0
MATRIX = 0x55B1
RANGE = 0x55B9
TRANSFER = 0x55BA
PRIMARIES = 0x55BB


def _vint(data: bytes, position: int, *, is_id: bool) -> tuple[int, int, bool]:
    if position >= len(data):
        raise ValueError("truncated EBML variable integer")
    first = data[position]
    marker = 0x80
    length = 1
    while length <= 8 and not (first & marker):
        marker >>= 1
        length += 1
    if length > 8 or position + length > len(data):
        raise ValueError("invalid EBML variable integer")
    if is_id:
        return int.from_bytes(data[position : position + length], "big"), length, False
    value = first & (marker - 1)
    for offset in range(1, length):
        value = (value << 8) | data[position + offset]
    unknown = value == (1 << (7 * length)) - 1
    return value, length, unknown


def _elements(data: bytes, start: int, end: int):
    position = start
    while position < end:
        element_id, id_length, _ = _vint(data, position, is_id=True)
        size, size_length, unknown = _vint(data, position + id_length, is_id=False)
        payload_start = position + id_length + size_length
        payload_end = end if unknown else payload_start + size
        if payload_start > end or payload_end > end or payload_end < payload_start:
            return
        yield element_id, payload_start, payload_end
        if payload_end == position:
            return
        position = payload_end


def _children(data: bytes, start: int, end: int) -> dict[int, tuple[int, int]]:
    found: dict[int, tuple[int, int]] = {}
    for element_id, payload_start, payload_end in _elements(data, start, end):
        found.setdefault(element_id, (payload_start, payload_end))
    return found


def _uint(data: bytes, bounds: tuple[int, int]) -> int:
    start, end = bounds
    return int.from_bytes(data[start:end], "big")


def _find_colour(
    data: bytes,
    start: int,
    end: int,
    path: str,
    found: list[tuple[dict[str, int | None], str]],
    depth: int = 0,
) -> None:
    if depth > 8:
        return
    children = _children(data, start, end)
    if TRACKS in children:
        tracks_start, tracks_end = children[TRACKS]
        track_index = 0
        for element_id, item_start, item_end in _elements(data, tracks_start, tracks_end):
            if element_id != TRACK_ENTRY:
                continue
            index = track_index
            track_index += 1
            entry_children = _children(data, item_start, item_end)
            track_type = entry_children.get(TRACK_TYPE)
            if track_type and _uint(data, track_type) != 1:
                continue
            if TRACK_TYPE in entry_children and VIDEO in entry_children:
                video_start, video_end = entry_children[VIDEO]
                video_children = _children(data, video_start, video_end)
                if COLOUR not in video_children:
                    continue
                colour_start, colour_end = video_children[COLOUR]
                colour_children = _children(data, colour_start, colour_end)
                codes: dict[str, int | None] = {
                    "primaries": None,
                    "transfer": None,
                    "matrix": None,
                    "range": None,
                }
                if PRIMARIES in colour_children:
                    codes["primaries"] = _uint(data, colour_children[PRIMARIES])
                if TRANSFER in colour_children:
                    codes["transfer"] = _uint(data, colour_children[TRANSFER])
                if MATRIX in colour_children:
                    codes["matrix"] = _uint(data, colour_children[MATRIX])
                if RANGE in colour_children:
                    raw_range = _uint(data, colour_children[RANGE])
                    if raw_range == 1:
                        codes["range"] = 0
                    elif raw_range == 2:
                        codes["range"] = 1
                rendered = ", ".join(f"{key}={value}" for key, value in codes.items())
                found.append(
                    (
                        codes,
                        f"{path}/Tracks/TrackEntry[{index}]/Video/Colour "
                        f"({rendered})",
                    )
                )


def read_matroska_container_tags(path: str | Path) -> LayerTags:
    data = Path(path).read_bytes()
    found: list[tuple[dict[str, int | None], str]] = []
    try:
        for element_id, start, end in _elements(data, 0, len(data)):
            if element_id == SEGMENT:
                _find_colour(data, start, end, "Segment", found)
    except ValueError as exc:
        return LayerTags(
            layer="container",
            status="unsupported",
            evidence=[f"Matroska parse failed: {exc}"],
        )
    if not found:
        return LayerTags(
            layer="container",
            status="absent",
            evidence=["no TrackEntry/Video/Colour element found"],
        )
    codes, first_evidence = found[0]
    evidence = [item[1] for item in found]
    status = "parsed"
    if len(found) > 1 and any(item[0] != codes for item in found[1:]):
        status = "ambiguous"
        evidence.append("multiple Colour elements disagree; first video track shown")
    return LayerTags(
        layer="container",
        values=tags_from_codes(codes),
        evidence=evidence,
        status=status,
    )
