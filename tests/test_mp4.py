from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from color_audit.mp4 import read_mp4_container_tags


def box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", len(payload) + 8, kind) + payload


def visual_sample_entry(colr_payload: bytes) -> bytes:
    visual_header = bytes(78)
    return box(b"avc1", visual_header + box(b"colr", colr_payload))


def make_mp4(colr_payload: bytes) -> bytes:
    stsd = box(b"stsd", bytes(4) + struct.pack(">I", 1) + visual_sample_entry(colr_payload))
    stbl = box(b"stbl", stsd)
    minf = box(b"minf", stbl)
    mdia = box(b"mdia", minf)
    trak = box(b"trak", mdia)
    moov = box(b"moov", trak)
    return box(b"ftyp", b"isom" + bytes(4) + b"isom") + moov


class Mp4Tests(unittest.TestCase):
    def test_reads_nclx_without_using_probe(self) -> None:
        payload = b"nclx" + struct.pack(">HHHB", 9, 16, 9, 0x80)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.mp4"
            path.write_bytes(make_mp4(payload))
            layer = read_mp4_container_tags(path)
        self.assertEqual(layer.status, "parsed")
        self.assertEqual(layer.values["primaries"].code, 9)
        self.assertEqual(layer.values["transfer"].code, 16)
        self.assertEqual(layer.values["matrix"].code, 9)
        self.assertEqual(layer.values["range"].code, 1)

    def test_reports_absent_colour_box(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.mp4"
            path.write_bytes(box(b"ftyp", b"isom" + bytes(4) + b"isom") + box(b"moov", b""))
            layer = read_mp4_container_tags(path)
        self.assertEqual(layer.status, "absent")
        self.assertIsNone(layer.values.get("primaries"))


if __name__ == "__main__":
    unittest.main()
