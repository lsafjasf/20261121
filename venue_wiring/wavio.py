"""Small dependency-free WAV reader and writer."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from pathlib import Path
import struct
import sys


class WavError(ValueError):
    """Raised when a WAV file is malformed or unsupported."""


@dataclass(frozen=True)
class WavData:
    sample_rate: int
    channels: list[list[float]]
    source_format: str

    @property
    def channel_count(self) -> int:
        return len(self.channels)

    @property
    def frame_count(self) -> int:
        return len(self.channels[0]) if self.channels else 0


def _read_chunks(payload: bytes) -> dict[bytes, list[bytes]]:
    chunks: dict[bytes, list[bytes]] = {}
    offset = 0
    while offset + 8 <= len(payload):
        chunk_id = payload[offset : offset + 4]
        size = struct.unpack_from("<I", payload, offset + 4)[0]
        start = offset + 8
        end = start + size
        if end > len(payload):
            raise WavError("WAV chunk extends beyond the end of the file")
        chunks.setdefault(chunk_id, []).append(payload[start:end])
        offset = end + (size & 1)
    return chunks


def _format_details(fmt: bytes) -> tuple[int, int, int, int]:
    if len(fmt) < 16:
        raise WavError("WAV fmt chunk is too short")
    tag, channel_count, sample_rate, _, block_align, bits = struct.unpack_from(
        "<HHIIHH", fmt, 0
    )
    if tag == 0xFFFE:
        if len(fmt) < 40:
            raise WavError("WAVE_FORMAT_EXTENSIBLE fmt chunk is too short")
        tag = struct.unpack_from("<H", fmt, 24)[0]
    if channel_count < 1:
        raise WavError("WAV file has no channels")
    width = max(1, block_align // channel_count)
    if bits == 0:
        bits = width * 8
    return tag, channel_count, sample_rate, bits


def _integer_samples(raw: bytes, bits: int, source_format: str) -> list[float]:
    if bits == 8:
        return [(value - 128) / 128.0 for value in raw]
    if bits == 16:
        values = array("h")
        values.frombytes(raw)
        if sys.byteorder != "little":
            values.byteswap()
        return [value / 32768.0 for value in values]
    if bits == 24:
        result: list[float] = []
        for offset in range(0, len(raw) - 2, 3):
            value = raw[offset] | (raw[offset + 1] << 8) | (raw[offset + 2] << 16)
            if value & 0x800000:
                value -= 0x1000000
            result.append(value / 8388608.0)
        return result
    if bits == 32:
        values = array("i")
        values.frombytes(raw)
        if sys.byteorder != "little":
            values.byteswap()
        return [value / 2147483648.0 for value in values]
    raise WavError(f"unsupported PCM sample width: {bits} bits ({source_format})")


def _float_samples(raw: bytes, bits: int, source_format: str) -> list[float]:
    code = "f" if bits == 32 else "d"
    if bits not in (32, 64):
        raise WavError(f"unsupported IEEE-float sample width: {bits} bits")
    values = array(code)
    values.frombytes(raw[: len(raw) - (len(raw) % values.itemsize)])
    if sys.byteorder != "little":
        values.byteswap()
    return [float(value) for value in values]


def read_wav(path: str | Path) -> WavData:
    """Read PCM 8/16/24/32-bit or IEEE float 32/64-bit WAV data."""

    raw_file = Path(path).read_bytes()
    if len(raw_file) < 12 or raw_file[:4] != b"RIFF" or raw_file[8:12] != b"WAVE":
        raise WavError(f"{path}: not a RIFF/WAVE file")
    chunks = _read_chunks(raw_file[12:])
    if b"fmt " not in chunks or b"data" not in chunks:
        raise WavError(f"{path}: missing fmt or data chunk")
    tag, channel_count, sample_rate, bits = _format_details(chunks[b"fmt "][0])
    raw_audio = chunks[b"data"][0]
    sample_bytes = max(1, bits // 8)
    frame_bytes = sample_bytes * channel_count
    raw_audio = raw_audio[: len(raw_audio) - (len(raw_audio) % frame_bytes)]
    if tag == 1:
        samples = _integer_samples(raw_audio, bits, "PCM")
        source_format = f"pcm{bits}"
    elif tag == 3:
        samples = _float_samples(raw_audio, bits, "IEEE float")
        source_format = f"float{bits}"
    else:
        raise WavError(f"{path}: unsupported WAVE format tag 0x{tag:04x}")
    channels = [samples[index::channel_count] for index in range(channel_count)]
    return WavData(sample_rate, channels, source_format)


def _little_endian_bytes(values: array) -> bytes:
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


def write_wav(
    path: str | Path,
    channels: list[list[float]],
    sample_rate: int,
    subtype: str = "pcm16",
) -> None:
    """Write deinterleaved channels, clipping integer formats to [-1, 1]."""

    if not channels:
        raise WavError("cannot write a WAV file with no channels")
    frame_count = len(channels[0])
    if any(len(channel) != frame_count for channel in channels):
        raise WavError("all channels must have the same length")
    if sample_rate <= 0:
        raise WavError("sample rate must be positive")

    if subtype == "pcm16":
        tag, bits = 1, 16
        interleaved = array("h")
        for frame in range(frame_count):
            for channel in channels:
                value = max(-1.0, min(1.0, channel[frame]))
                interleaved.append(round(value * 32767.0))
        payload = _little_endian_bytes(interleaved)
    elif subtype == "float32":
        tag, bits = 3, 32
        interleaved = array("f")
        for frame in range(frame_count):
            for channel in channels:
                interleaved.append(channel[frame])
        payload = _little_endian_bytes(interleaved)
    else:
        raise WavError(f"unsupported output subtype: {subtype}")

    channel_count = len(channels)
    block_align = channel_count * bits // 8
    fmt = struct.pack(
        "<HHIIHH",
        tag,
        channel_count,
        sample_rate,
        sample_rate * block_align,
        block_align,
        bits,
    )
    body = (
        b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(payload))
        + payload
        + (b"\x00" if len(payload) & 1 else b"")
    )
    Path(path).write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
