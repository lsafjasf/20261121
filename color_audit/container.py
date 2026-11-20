from __future__ import annotations

from pathlib import Path

from .matroska import read_matroska_container_tags
from .model import LayerTags
from .mp4 import read_mp4_container_tags


def read_container_tags(path: str | Path) -> LayerTags:
    with Path(path).open("rb") as handle:
        head = handle.read(12)
    if len(head) >= 8 and head[4:8] == b"ftyp":
        return read_mp4_container_tags(path)
    if head.startswith(b"\x1a\x45\xdf\xa3"):
        return read_matroska_container_tags(path)
    return LayerTags(
        layer="container",
        status="unsupported",
        evidence=["unrecognised container (supported: ISO BMFF/MP4/MOV, Matroska/WebM)"],
    )
