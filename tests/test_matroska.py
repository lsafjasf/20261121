from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from color_audit.matroska import read_matroska_container_tags


def element(element_id: bytes, payload: bytes) -> bytes:
    if len(payload) > 126:
        raise ValueError("test helper only supports one-byte sizes")
    return element_id + bytes([0x80 | len(payload)]) + payload


def make_mkv(colour_payload: bytes) -> bytes:
    header = element(b"\x1a\x45\xdf\xa3", b"")
    track = element(
        b"\xae",
        element(b"\x83", b"\x01")
        + element(b"\xe0", element(b"\x55\xb0", colour_payload)),
    )
    tracks = element(b"\x16\x54\xae\x6b", track)
    segment = element(b"\x18\x53\x80\x67", tracks)
    return header + segment


class MatroskaTests(unittest.TestCase):
    def test_reads_colour_element(self) -> None:
        colour = (
            element(b"\x55\xb1", b"\x09")
            + element(b"\x55\xba", b"\x10")
            + element(b"\x55\xbb", b"\x09")
            + element(b"\x55\xb9", b"\x02")
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.mkv"
            path.write_bytes(make_mkv(colour))
            layer = read_matroska_container_tags(path)
        self.assertEqual(layer.status, "parsed")
        self.assertEqual(layer.values["primaries"].code, 9)
        self.assertEqual(layer.values["transfer"].code, 16)
        self.assertEqual(layer.values["matrix"].code, 9)
        self.assertEqual(layer.values["range"].code, 1)
        self.assertIn("Video/Colour", layer.evidence[0])


if __name__ == "__main__":
    unittest.main()
