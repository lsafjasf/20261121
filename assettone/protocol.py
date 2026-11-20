"""Marker framing and checksum definitions shared by synth and detector."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Protocol:
    name: str = "MUSE-FSK/1"
    leader_hz: float = 2000.0
    space_hz: float = 1500.0
    mark_hz: float = 2500.0
    leader_seconds: float = 0.100
    gap_seconds: float = 0.018
    symbol_seconds: float = 0.010
    sync_word: int = 0xB6D3
    sync_bits: int = 16
    version: int = 1
    version_bits: int = 4
    asset_id_bits: int = 16
    meta_bits: int = 4
    meta: int = 0b0011
    crc_bits: int = 16

    @property
    def header_bits(self) -> int:
        return self.version_bits + self.asset_id_bits + self.meta_bits

    @property
    def body_bytes(self) -> int:
        return self.header_bits // 8

    @property
    def body_bits(self) -> int:
        return self.header_bits + self.crc_bits

    @property
    def frame_bits(self) -> int:
        return self.sync_bits + self.body_bits

    @property
    def frame_seconds(self) -> float:
        return (
            self.leader_seconds
            + self.gap_seconds
            + self.frame_bits * self.symbol_seconds
        )

    @property
    def sync_to_marker_seconds(self) -> float:
        return self.leader_seconds + self.gap_seconds


PROTOCOL = Protocol()


@dataclass(frozen=True)
class Asset:
    asset_id: int
    label: str
    title: str

    @property
    def hex_id(self) -> str:
        return f"{self.asset_id:04X}"


KNOWN_ASSETS = (
    Asset(0x1042, "CN-1042", "青釉莲纹罐"),
    Asset(0x7781, "CN-7781", "铜鎏金瑞兽镇"),
)
ASSET_BY_ID = {asset.asset_id: asset for asset in KNOWN_ASSETS}


def int_to_bits(value: int, width: int) -> list[int]:
    if value < 0 or value >= (1 << width):
        raise ValueError(f"{value} does not fit in {width} bits")
    return [(value >> shift) & 1 for shift in range(width - 1, -1, -1)]


def bits_to_int(bits: list[int] | tuple[int, ...]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | (1 if bit else 0)
    return value


def bits_to_hex(bits: list[int] | tuple[int, ...]) -> str:
    value = bits_to_int(bits)
    width = max(1, (len(bits) + 3) // 4)
    return f"{value:0{width}X}"


def crc16_ccitt(data: bytes, initial: int = 0xFFFF) -> int:
    crc = initial
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def build_header(asset_id: int, meta: int | None = None) -> tuple[list[int], bytes]:
    proto = PROTOCOL
    if not 0 <= asset_id < (1 << proto.asset_id_bits):
        raise ValueError("asset id must fit in 16 bits")
    if meta is None:
        meta = proto.meta
    if not 0 <= meta < (1 << proto.meta_bits):
        raise ValueError("meta must fit in 4 bits")
    bits = (
        int_to_bits(proto.version, proto.version_bits)
        + int_to_bits(asset_id, proto.asset_id_bits)
        + int_to_bits(meta, proto.meta_bits)
    )
    return bits, bytes(bits_to_int(bits[i : i + 8]) for i in range(0, len(bits), 8))


def build_body_bits(asset_id: int, meta: int | None = None) -> tuple[list[int], int]:
    header_bits, header = build_header(asset_id, meta)
    checksum = crc16_ccitt(header)
    return header_bits + int_to_bits(checksum, PROTOCOL.crc_bits), checksum


def build_frame_bits(asset_id: int, meta: int | None = None) -> tuple[list[int], int]:
    body, checksum = build_body_bits(asset_id, meta)
    return int_to_bits(PROTOCOL.sync_word, PROTOCOL.sync_bits) + body, checksum


def decode_body_bits(bits: list[int]) -> dict:
    proto = PROTOCOL
    header_bits = bits[: proto.header_bits]
    crc_bits = bits[proto.header_bits : proto.header_bits + proto.crc_bits]
    header = bytes(
        bits_to_int(header_bits[i : i + 8])
        for i in range(0, len(header_bits), 8)
    )
    return {
        "version": bits_to_int(header_bits[: proto.version_bits]),
        "asset_id": bits_to_int(
            header_bits[proto.version_bits : proto.version_bits + proto.asset_id_bits]
        ),
        "meta": bits_to_int(header_bits[-proto.meta_bits :]),
        "header_hex": header.hex().upper(),
        "crc_expected": crc16_ccitt(header),
        "crc_received": bits_to_int(crc_bits),
        "header_bits": header_bits,
        "crc_bits": crc_bits,
    }


def sync_distance(bits: list[int]) -> int:
    sync = int_to_bits(PROTOCOL.sync_word, PROTOCOL.sync_bits)
    return sum(a != b for a, b in zip(bits, sync))
