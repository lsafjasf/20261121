from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .model import LayerTags, tags_from_codes

CONTAINER_BOXES = {
    b"moov",
    b"trak",
    b"mdia",
    b"minf",
    b"stbl",
    b"edts",
    b"dinf",
    b"mvex",
    b"moof",
    b"traf",
}


@dataclass(frozen=True)
class Box:
    kind: bytes
    start: int
    end: int
    payload_start: int

    @property
    def size(self) -> int:
        return self.end - self.start


def iter_boxes(data: bytes, start: int = 0, end: int | None = None):
    if end is None:
        end = len(data)
    position = start
    while position + 8 <= end:
        size = int.from_bytes(data[position : position + 4], "big")
        kind = bytes(data[position + 4 : position + 8])
        header = 8
        if size == 1:
            if position + 16 > end:
                return
            size = int.from_bytes(data[position + 8 : position + 16], "big")
            header = 16
        elif size == 0:
            size = end - position
        if size < header or position + size > end:
            return
        yield Box(kind, position, position + size, position + header)
        position += size


def _parse_colr(data: bytes, box: Box, path: str) -> tuple[dict[str, int | None], str] | None:
    payload = data[box.payload_start : box.end]
    if len(payload) < 10:
        return None
    colour_type = bytes(payload[:4])
    if colour_type not in {b"nclx", b"nclc"}:
        return None
    codes: dict[str, int | None] = {
        "primaries": int.from_bytes(payload[4:6], "big"),
        "transfer": int.from_bytes(payload[6:8], "big"),
        "matrix": int.from_bytes(payload[8:10], "big"),
        "range": None,
    }
    if colour_type == b"nclx" and len(payload) >= 11:
        codes["range"] = 1 if payload[10] & 0x80 else 0
    rendered = ", ".join(f"{key}={value}" for key, value in codes.items())
    return codes, f"{path} ({colour_type.decode('ascii')}: {rendered})"


def _walk_moov(
    data: bytes,
    start: int,
    end: int,
    path: str,
    found: list[tuple[dict[str, int | None], str]],
    depth: int = 0,
) -> None:
    if depth > 16:
        return
    for box in iter_boxes(data, start, end):
        child_path = f"{path}/{box.kind.decode('ascii', 'replace')}"
        if box.kind == b"colr":
            parsed = _parse_colr(data, box, child_path)
            if parsed:
                found.append(parsed)
        elif box.kind in CONTAINER_BOXES:
            _walk_moov(data, box.payload_start, box.end, child_path, found, depth + 1)
        elif box.kind == b"stsd":
            if box.payload_start + 8 > box.end:
                continue
            entries_start = box.payload_start + 8
            for entry in iter_boxes(data, entries_start, box.end):
                entry_path = f"{child_path}/{entry.kind.decode('ascii', 'replace')}"
                children_start = entry.payload_start + 78
                if children_start <= entry.end:
                    _walk_moov(
                        data, children_start, entry.end, entry_path, found, depth + 1
                    )


def read_mp4_container_tags(path: str | Path) -> LayerTags:
    data = Path(path).read_bytes()
    if len(data) < 12 or data[4:8] != b"ftyp":
        return LayerTags(
            layer="container",
            status="unsupported",
            evidence=["not an ISO BMFF/MP4 file (missing ftyp box)"],
        )
    found: list[tuple[dict[str, int | None], str]] = []
    for box in iter_boxes(data):
        if box.kind == b"moov":
            _walk_moov(data, box.payload_start, box.end, "moov", found)
    if not found:
        return LayerTags(
            layer="container",
            status="absent",
            evidence=["no colr/nclx or colr/nclc box found under moov/stsd"],
        )
    codes, first_evidence = found[0]
    evidence = [item[1] for item in found]
    status = "parsed"
    if len(found) > 1 and any(item[0] != codes for item in found[1:]):
        status = "ambiguous"
        evidence.append("multiple colr boxes disagree; first box shown")
    return LayerTags(
        layer="container",
        values=tags_from_codes(codes),
        evidence=evidence,
        status=status,
    )


def patch_mp4_colr(
    source: str | Path,
    destination: str | Path,
    codes: dict[str, int],
) -> None:
    """Rewrite the first nclx colr box. Used only to build a mistagged fixture."""
    data = bytearray(Path(source).read_bytes())
    found: list[tuple[dict[str, int | None], str]] = []
    for box in iter_boxes(data):
        if box.kind == b"moov":
            _walk_moov(data, box.payload_start, box.end, "moov", found)
    if not found:
        raise ValueError("input has no MP4 colr box")

    colour_type = None
    for box in iter_boxes(data):
        if box.kind != b"moov":
            continue
        stack = [(box.payload_start, box.end)]
        while stack:
            start, end = stack.pop()
            for child in iter_boxes(data, start, end):
                if child.kind == b"colr":
                    payload = data[child.payload_start : child.end]
                    if payload[:4] == b"nclx":
                        colour_type = b"nclx"
                        base = child.payload_start + 4
                        break
                    if payload[:4] == b"nclc" and colour_type is None:
                        colour_type = b"nclc"
                        base = child.payload_start + 4
                        break
                elif child.kind in CONTAINER_BOXES:
                    stack.append((child.payload_start, child.end))
                elif child.kind == b"stsd" and child.payload_start + 8 <= child.end:
                    for entry in iter_boxes(data, child.payload_start + 8, child.end):
                        if entry.payload_start + 78 <= entry.end:
                            stack.append((entry.payload_start + 78, entry.end))
            if colour_type:
                break
        if colour_type:
            break
    if not colour_type:
        raise ValueError("input has no nclx/nclc colr box")

    data[base : base + 2] = int(codes["primaries"]).to_bytes(2, "big")
    data[base + 2 : base + 4] = int(codes["transfer"]).to_bytes(2, "big")
    data[base + 4 : base + 6] = int(codes["matrix"]).to_bytes(2, "big")
    if colour_type == b"nclx":
        data[base + 6] = 0x80 if codes.get("range") == 1 else 0
    Path(destination).write_bytes(data)
