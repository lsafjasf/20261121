"""Minimal Broadcast Wave Format (BWF / RIFF WAVE) I/O, stdlib only.

Supports PCM integer (16/24/32-bit) and IEEE float32, any channel count.
Unknown chunks are preserved on read and written back verbatim.
Samples are kept interleaved: ints for PCM, floats for float32.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003


class BwfError(Exception):
    """Raised on malformed or unsupported BWF/WAVE data."""


@dataclass
class BextChunk:
    """Broadcast extension (`bext`) chunk, version-0/1 fixed layout."""

    description: str = ""
    originator: str = ""
    originator_reference: str = ""
    origination_date: str = "1970-01-01"  # YYYY-MM-DD
    origination_time: str = "00:00:00"    # HH:MM:SS
    time_reference: int = 0               # samples since midnight
    version: int = 1
    umid: bytes = bytes(64)
    coding_history: str = ""

    def pack(self) -> bytes:
        def fixed(text: str, size: int) -> bytes:
            raw = text.encode("ascii", "replace")[:size]
            return raw + bytes(size - len(raw))

        return b"".join(
            [
                fixed(self.description, 256),
                fixed(self.originator, 32),
                fixed(self.originator_reference, 32),
                fixed(self.origination_date, 10),
                fixed(self.origination_time, 8),
                struct.pack("<Q", self.time_reference),
                struct.pack("<H", self.version),
                self.umid[:64].ljust(64, b"\x00"),
                bytes(190),  # reserved loudness fields, kept zero
                self.coding_history.encode("ascii", "replace"),
            ]
        )

    @classmethod
    def unpack(cls, raw: bytes) -> "BextChunk":
        if len(raw) < 602:
            raise BwfError("bext chunk too small: %d bytes" % len(raw))

        def text(blob: bytes) -> str:
            return blob.split(b"\x00", 1)[0].decode("ascii", "replace").strip()

        return cls(
            description=text(raw[0:256]),
            originator=text(raw[256:288]),
            originator_reference=text(raw[288:320]),
            origination_date=text(raw[320:330]) or "1970-01-01",
            origination_time=text(raw[330:338]) or "00:00:00",
            time_reference=struct.unpack("<Q", raw[338:346])[0],
            version=struct.unpack("<H", raw[346:348])[0],
            umid=raw[348:412],
            coding_history=text(raw[602:]),
        )


def decode_samples(raw: bytes, audio_format: int, bit_depth: int) -> list:
    if audio_format == WAVE_FORMAT_IEEE_FLOAT and bit_depth == 32:
        return list(struct.unpack("<%df" % (len(raw) // 4), raw))
    if audio_format != WAVE_FORMAT_PCM:
        raise BwfError("unsupported audio format: 0x%04x" % audio_format)
    if bit_depth == 16:
        return list(struct.unpack("<%dh" % (len(raw) // 2), raw))
    if bit_depth == 24:
        return [
            int.from_bytes(raw[i : i + 3], "little", signed=True)
            for i in range(0, len(raw), 3)
        ]
    if bit_depth == 32:
        return list(struct.unpack("<%di" % (len(raw) // 4), raw))
    raise BwfError("unsupported bit depth: %d" % bit_depth)


def encode_samples(samples: list, audio_format: int, bit_depth: int) -> bytes:
    if audio_format == WAVE_FORMAT_IEEE_FLOAT and bit_depth == 32:
        return struct.pack("<%df" % len(samples), *samples)
    if bit_depth == 16:
        return struct.pack("<%dh" % len(samples), *samples)
    if bit_depth == 24:
        return b"".join(
            int(v).to_bytes(3, "little", signed=True) for v in samples
        )
    if bit_depth == 32:
        return struct.pack("<%di" % len(samples), *samples)
    raise BwfError("unsupported bit depth: %d" % bit_depth)


@dataclass
class BwfFile:
    sample_rate: int
    channels: int
    bit_depth: int
    audio_format: int
    samples: list  # interleaved; int for PCM, float for float32
    bext: BextChunk | None = None
    extra_chunks: list = field(default_factory=list)  # [(chunk_id, payload)]

    @property
    def frame_count(self) -> int:
        return len(self.samples) // self.channels

    @property
    def duration(self) -> float:
        return self.frame_count / self.sample_rate

    def slice_frames(self, start: int, end: int) -> "BwfFile":
        """Bit-exact frame slice [start, end); samples are copied verbatim."""
        start = max(0, min(start, self.frame_count))
        end = max(start, min(end, self.frame_count))
        ch = self.channels
        return BwfFile(
            sample_rate=self.sample_rate,
            channels=self.channels,
            bit_depth=self.bit_depth,
            audio_format=self.audio_format,
            samples=self.samples[start * ch : end * ch],
            bext=self.bext,
            extra_chunks=list(self.extra_chunks),
        )


def read_bwf(path: str) -> BwfFile:
    with open(path, "rb") as fh:
        riff = fh.read()
    if len(riff) < 12 or riff[0:4] != b"RIFF" or riff[8:12] != b"WAVE":
        raise BwfError("not a RIFF/WAVE file: %s" % path)
    fmt = None
    data = None
    bext = None
    extra = []
    pos = 12
    while pos + 8 <= len(riff):
        chunk_id = riff[pos : pos + 4]
        size = struct.unpack("<I", riff[pos + 4 : pos + 8])[0]
        payload = riff[pos + 8 : pos + 8 + size]
        pos += 8 + size + (size & 1)  # chunks are padded to even sizes
        if chunk_id == b"fmt ":
            fmt = payload
        elif chunk_id == b"data":
            data = payload
        elif chunk_id == b"bext":
            bext = BextChunk.unpack(payload)
        else:
            extra.append((chunk_id, payload))
    if fmt is None or data is None:
        raise BwfError("missing fmt or data chunk: %s" % path)
    audio_format, channels, sample_rate, _br, _ba, bits = struct.unpack(
        "<HHIIHH", fmt[:16]
    )
    return BwfFile(
        sample_rate=sample_rate,
        channels=channels,
        bit_depth=bits,
        audio_format=audio_format,
        samples=decode_samples(data, audio_format, bits),
        bext=bext,
        extra_chunks=extra,
    )


def write_bwf(path: str, bwf: BwfFile) -> None:
    block_align = bwf.channels * bwf.bit_depth // 8
    fmt = struct.pack(
        "<HHIIHH",
        bwf.audio_format,
        bwf.channels,
        bwf.sample_rate,
        bwf.sample_rate * block_align,
        block_align,
        bwf.bit_depth,
    )
    chunks = []
    if bwf.bext is not None:
        chunks.append((b"bext", bwf.bext.pack()))
    chunks.append((b"fmt ", fmt))
    chunks.append(
        (b"data", encode_samples(bwf.samples, bwf.audio_format, bwf.bit_depth))
    )
    chunks.extend(bwf.extra_chunks)
    body = bytearray()
    for chunk_id, payload in chunks:
        body += chunk_id
        body += struct.pack("<I", len(payload))
        body += payload
        if len(payload) & 1:
            body += b"\x00"
    with open(path, "wb") as fh:
        fh.write(b"RIFF")
        fh.write(struct.pack("<I", 4 + len(body)))
        fh.write(b"WAVE")
        fh.write(bytes(body))
