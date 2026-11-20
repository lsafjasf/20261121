"""Minimal, dependency-free ISO-BMFF (MP4/MOV) box reader.

Only the boxes needed for timing investigation are interpreted; unknown boxes
are kept as raw byte ranges so the rest of the file can be re-emitted unchanged.
All multi-byte integers in ISO-BMFF are big-endian.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass


CONTAINERS = {"moov", "trak", "mdia", "minf", "stbl", "edts", "udta", "meta"}


@dataclass
class Box:
    type: str
    start: int          # first byte of the size field
    header: int         # 8 for normal, 16 for largesize
    size: int           # total box size including header
    depth: int
    children: list["Box"]

    @property
    def payload_start(self) -> int:
        return self.start + self.header

    @property
    def end(self) -> int:
        return self.start + self.size

    def fullbox(self, data: bytes) -> tuple[int, int]:
        """Return (version, flags) for a FullBox, payload relative."""
        p = self.payload_start
        version = data[p]
        flags = int.from_bytes(data[p + 1:p + 4], "big")
        return version, flags

    def find(self, *path: str) -> "Box | None":
        node = self
        for name in path:
            node = next((c for c in node.children if c.type == name), None)
            if node is None:
                return None
        return node


def scan_boxes(data: bytes, start: int, end: int, depth: int = 0) -> list[Box]:
    boxes: list[Box] = []
    p = start
    while p + 8 <= end:
        size32 = struct.unpack(">I", data[p:p + 4])[0]
        btype = data[p + 4:p + 8].decode("latin1")
        header = 8
        if size32 == 1:  # largesize
            size = struct.unpack(">Q", data[p + 8:p + 16])[0]
            header = 16
        elif size32 == 0:  # box runs to end of buffer
            size = end - p
        else:
            size = size32
        box = Box(btype, p, header, size, depth, [])
        if btype in CONTAINERS:
            child_start = p + header
            # 'meta' is a FullBox: skip its version/flags before children.
            if btype == "meta":
                child_start += 4
            box.children = scan_boxes(data, child_start, p + size, depth + 1)
        boxes.append(box)
        p += size
        if size32 == 0:
            break
    return boxes


def u8(data: bytes, p: int) -> int:
    return data[p]


def u16(data: bytes, p: int) -> int:
    return struct.unpack(">H", data[p:p + 2])[0]


def u32(data: bytes, p: int) -> int:
    return struct.unpack(">I", data[p:p + 4])[0]


def u64(data: bytes, p: int) -> int:
    return struct.unpack(">Q", data[p:p + 8])[0]


def i32(data: bytes, p: int) -> int:
    return struct.unpack(">i", data[p:p + 4])[0]


def u32_or_64(data: bytes, p: int, version: int) -> int:
    if version == 1:
        return u64(data, p)
    return u32(data, p)
