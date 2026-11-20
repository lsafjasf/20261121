"""Strict RIFF/WAVE access that preserves bytes outside edited PCM samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

from .errors import WavError


_PCM_GUID = b"\x01\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
_FLOAT_GUID = b"\x03\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"


@dataclass(frozen=True)
class WavFormat:
    encoding: str
    channels: int
    sample_rate: int
    block_align: int
    bits_per_sample: int

    @property
    def sample_width(self) -> int:
        return self.bits_per_sample // 8

    def as_dict(self) -> dict[str, int | str]:
        return {
            "encoding": self.encoding,
            "channels": self.channels,
            "sample_rate_hz": self.sample_rate,
            "bits_per_sample": self.bits_per_sample,
            "block_align_bytes": self.block_align,
        }


@dataclass(frozen=True)
class WavAudio:
    raw: bytes
    format: WavFormat
    data_offset: int
    data_size: int

    @property
    def frame_count(self) -> int:
        return self.data_size // self.format.block_align

    @property
    def data(self) -> bytes:
        return self.raw[self.data_offset : self.data_offset + self.data_size]

    def with_data(self, data: bytes) -> bytes:
        if len(data) != self.data_size:
            raise WavError("replacement PCM data changed the WAV length")
        output = bytearray(self.raw)
        output[self.data_offset : self.data_offset + self.data_size] = data
        return bytes(output)


def _parse_format(payload: bytes) -> WavFormat:
    if len(payload) < 16:
        raise WavError("WAV fmt chunk is shorter than 16 bytes")
    tag, channels, rate, byte_rate, block_align, bits = struct.unpack(
        "<HHIIHH", payload[:16]
    )
    if tag == 0xFFFE:
        if len(payload) < 40:
            raise WavError("WAVE_FORMAT_EXTENSIBLE fmt chunk is truncated")
        valid_bits = struct.unpack_from("<H", payload, 18)[0]
        guid = payload[24:40]
        if guid == _PCM_GUID:
            tag = 1
        elif guid == _FLOAT_GUID:
            tag = 3
        else:
            raise WavError("unsupported WAVE_FORMAT_EXTENSIBLE sub-format")
        if valid_bits and valid_bits > bits:
            raise WavError("valid bits exceed the container sample width")
    if tag not in (1, 3):
        raise WavError(f"unsupported WAV encoding tag {tag}; PCM or IEEE float required")
    if channels < 1 or rate < 1:
        raise WavError("WAV must have at least one channel and a positive sample rate")
    if bits < 8 or bits % 8:
        raise WavError(f"unsupported WAV sample width: {bits} bits")
    sample_width = bits // 8
    if block_align != channels * sample_width:
        raise WavError("WAV block alignment does not match channel/sample width")
    if byte_rate != rate * block_align:
        raise WavError("WAV byte rate does not match sample rate and block alignment")
    return WavFormat(
        encoding="pcm" if tag == 1 else "float",
        channels=channels,
        sample_rate=rate,
        block_align=block_align,
        bits_per_sample=bits,
    )


def read_wav(path: Path) -> WavAudio:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise WavError(f"cannot read WAV {path}: {exc}") from exc
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise WavError(f"{path} is not a little-endian RIFF/WAVE file")
    declared_end = 8 + struct.unpack_from("<I", raw, 4)[0]
    if declared_end > len(raw):
        raise WavError(f"{path} is truncated: RIFF size exceeds file size")

    fmt_payload: bytes | None = None
    data_offset: int | None = None
    data_size: int | None = None
    offset = 12
    while offset + 8 <= declared_end:
        chunk_id = raw[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", raw, offset + 4)[0]
        payload_offset = offset + 8
        payload_end = payload_offset + chunk_size
        if payload_end > declared_end:
            raise WavError(f"{path} has a chunk that exceeds the RIFF boundary")
        if chunk_id == b"fmt ":
            if fmt_payload is not None:
                raise WavError(f"{path} contains multiple fmt chunks")
            fmt_payload = raw[payload_offset:payload_end]
        elif chunk_id == b"data":
            if data_offset is not None:
                raise WavError(f"{path} contains multiple data chunks")
            data_offset = payload_offset
            data_size = chunk_size
        offset = payload_end + (chunk_size & 1)
    if fmt_payload is None:
        raise WavError(f"{path} has no fmt chunk")
    if data_offset is None or data_size is None:
        raise WavError(f"{path} has no data chunk")
    fmt = _parse_format(fmt_payload)
    if data_size % fmt.block_align:
        raise WavError(f"{path} PCM data is not frame-aligned")
    return WavAudio(raw=raw, format=fmt, data_offset=data_offset, data_size=data_size)


def validate_room_tone(source: WavAudio, room_tone: WavAudio) -> None:
    """Require raw-compatible room tone while allowing mono-to-stereo mapping."""
    left, right = source.format, room_tone.format
    if left.encoding != right.encoding:
        raise WavError("room tone encoding differs from the source WAV")
    if left.sample_rate != right.sample_rate:
        raise WavError("room tone sample rate differs from the source WAV")
    if left.bits_per_sample != right.bits_per_sample:
        raise WavError("room tone sample width differs from the source WAV")
    if room_tone.frame_count < 1:
        raise WavError("room tone WAV contains no sample frames")


def extract_channel_frames(
    audio: WavAudio, channel_index: int, start_frame: int, frame_count: int
) -> bytes:
    """Return channel samples with no interleaved samples from other channels."""
    if channel_index < 0 or channel_index >= audio.format.channels:
        raise WavError(f"channel index {channel_index} is out of range")
    if start_frame < 0 or frame_count < 0 or start_frame + frame_count > audio.frame_count:
        raise WavError("requested room-tone frames are outside the WAV")
    width = audio.format.sample_width
    step = audio.format.block_align
    first = start_frame * step + channel_index * width
    source = audio.data
    if frame_count == 0:
        return b""
    result = bytearray(frame_count * width)
    source_pos = first
    target_pos = 0
    for _ in range(frame_count):
        result[target_pos : target_pos + width] = source[source_pos : source_pos + width]
        source_pos += step
        target_pos += width
    return bytes(result)


def write_channel_frames(
    output_data: bytearray,
    audio: WavAudio,
    channel_index: int,
    start_frame: int,
    channel_data: bytes,
) -> None:
    """Write contiguous channel samples back into an interleaved PCM buffer."""
    width = audio.format.sample_width
    step = audio.format.block_align
    if len(channel_data) % width:
        raise WavError("channel replacement data is not sample-aligned")
    frame_count = len(channel_data) // width
    position = start_frame * step + channel_index * width
    source_pos = 0
    for _ in range(frame_count):
        output_data[position : position + width] = channel_data[source_pos : source_pos + width]
        position += step
        source_pos += width
