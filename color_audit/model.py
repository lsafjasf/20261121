from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

FIELDS = ("primaries", "transfer", "matrix", "range")

PRIMARIES = {
    1: "bt709",
    2: "unspecified",
    4: "bt470m",
    5: "bt470bg",
    6: "smpte170m",
    7: "smpte240m",
    8: "film",
    9: "bt2020",
    10: "smpte428",
    11: "smpte431",
    12: "smpte432",
    22: "jedec-p22",
}
TRANSFER = {
    1: "bt709",
    2: "unspecified",
    4: "gamma22",
    5: "gamma28",
    6: "smpte170m",
    7: "smpte240m",
    8: "linear",
    9: "log100",
    10: "log316",
    11: "iec61966-2-4",
    12: "bt1361e",
    13: "iec61966-2-1",
    14: "bt2020-10",
    15: "bt2020-12",
    16: "smpte2084",
    17: "smpte428",
    18: "arib-std-b67",
}
MATRIX = {
    0: "gbr",
    1: "bt709",
    2: "unspecified",
    4: "fcc",
    5: "bt470bg",
    6: "smpte170m",
    7: "smpte240m",
    8: "ycgco",
    9: "bt2020nc",
    10: "bt2020c",
    11: "smpte2085",
    12: "chroma-derived-nc",
    13: "chroma-derived-c",
    14: "ictcp",
}
RANGE = {0: "limited", 1: "full"}

MAPPINGS = {
    "primaries": PRIMARIES,
    "transfer": TRANSFER,
    "matrix": MATRIX,
    "range": RANGE,
}

# Names accepted by zscale. Numeric H.273 codes are used by the generated
# filter graph, but keeping the names makes the audit report readable.
ZSCALE_PRIMARIES = {
    1: "bt709",
    2: "unspecified",
    4: "bt470m",
    5: "bt470bg",
    6: "smpte170m",
    7: "smpte240m",
    8: "film",
    9: "bt2020",
    10: "smpte428",
    11: "smpte431",
    12: "smpte432",
    22: "jedec-p22",
}
ZSCALE_TRANSFER = {
    1: "bt709",
    2: "unspecified",
    4: "bt470m",
    5: "bt470bg",
    6: "smpte170m",
    7: "smpte240m",
    8: "linear",
    9: "log100",
    10: "log316",
    11: "iec61966-2-4",
    13: "iec61966-2-1",
    14: "bt2020-10",
    15: "bt2020-12",
    16: "smpte2084",
    18: "arib-std-b67",
}
ZSCALE_MATRIX = {
    0: "gbr",
    1: "bt709",
    2: "unspecified",
    4: "fcc",
    5: "bt470bg",
    6: "smpte170m",
    7: "smpte240m",
    8: "ycgco",
    9: "bt2020nc",
    10: "bt2020c",
    12: "chroma-derived-nc",
    13: "chroma-derived-c",
    14: "ictcp",
}
ZSCALE_RANGE = {0: "limited", 1: "full"}
ZSCALE_NAMES = {
    "primaries": ZSCALE_PRIMARIES,
    "transfer": ZSCALE_TRANSFER,
    "matrix": ZSCALE_MATRIX,
    "range": ZSCALE_RANGE,
}


@dataclass(frozen=True)
class TagValue:
    code: int | None
    name: str | None

    @classmethod
    def from_code(cls, field_name: str, code: int | None) -> "TagValue":
        if code is None:
            return cls(None, None)
        name = MAPPINGS[field_name].get(code, f"unknown({code})")
        return cls(code, name)

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "name": self.name}


@dataclass
class LayerTags:
    layer: str
    values: dict[str, TagValue] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    status: str = "parsed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "status": self.status,
            "values": {name: value.to_dict() for name, value in self.values.items()},
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class Conflict:
    field: str
    container: TagValue
    codec: TagValue

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "container": self.container.to_dict(),
            "codec": self.codec.to_dict(),
        }


@dataclass(frozen=True)
class ResolvedField:
    value: TagValue
    origin: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value.to_dict(),
            "origin": self.origin,
            "reason": self.reason,
        }


def find_conflicts(container: LayerTags, codec: LayerTags) -> list[Conflict]:
    conflicts: list[Conflict] = []
    for name in FIELDS:
        left = container.values.get(name, TagValue(None, None))
        right = codec.values.get(name, TagValue(None, None))
        if left.code is not None and right.code is not None and left.code != right.code:
            conflicts.append(Conflict(name, left, right))
    return conflicts


def resolve_tags(
    container: LayerTags,
    codec: LayerTags,
    decisions: dict[str, str],
) -> tuple[dict[str, ResolvedField], list[str]]:
    """Resolve each field without silently picking a conflicting layer."""
    conflicts = {item.field for item in find_conflicts(container, codec)}
    resolved: dict[str, ResolvedField] = {}
    errors: list[str] = []
    for name in FIELDS:
        left = container.values.get(name, TagValue(None, None))
        right = codec.values.get(name, TagValue(None, None))
        if name in conflicts:
            source = decisions.get(name)
            if source not in {"container", "codec"}:
                errors.append(
                    f"{name}: container={left.name}({left.code}) and "
                    f"codec={right.name}({right.code}); choose container or codec"
                )
                continue
            chosen = left if source == "container" else right
            resolved[name] = ResolvedField(
                chosen, source, f"user confirmed {source} declaration"
            )
            continue
        if left.code is not None and right.code is not None:
            resolved[name] = ResolvedField(left, "agreed", "both layers agree")
        elif left.code is not None:
            resolved[name] = ResolvedField(left, "container", "only container declares it")
        elif right.code is not None:
            resolved[name] = ResolvedField(right, "codec", "only codec declares it")
        else:
            errors.append(f"{name}: neither layer declares a value")
    return resolved, errors


def parse_decisions(items: list[str]) -> dict[str, str]:
    decisions: dict[str, str] = {}
    for item in items:
        for part in item.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"expected FIELD=SOURCE, got {part!r}")
            name, source = (value.strip() for value in part.split("=", 1))
            name = name.lower()
            source = source.lower()
            if name not in FIELDS:
                raise ValueError(f"unknown field {name!r}")
            if source in {"container", "codec"}:
                decisions[name] = source
            else:
                raise ValueError(f"source for {name} must be container or codec")
    return decisions


def parse_tag_code(field_name: str, text: str) -> int:
    normalized = text.strip().lower()
    aliases = {
        "range": {
            "tv": 0,
            "limited": 0,
            "pc": 1,
            "full": 1,
            "video": 0,
        },
        "transfer": {"pq": 16, "hlg": 18, "sdr": 1, "rec709": 1},
        "matrix": {
            "bt2020-ncl": 9,
            "bt2020-nc": 9,
            "bt2020-cl": 10,
            "bt2020-c": 10,
        },
        "primaries": {"rec709": 1, "rec2020": 9},
    }
    if normalized in aliases.get(field_name, {}):
        return aliases[field_name][normalized]
    if normalized.isdigit():
        code = int(normalized)
        if code in MAPPINGS[field_name]:
            return code
    for code, name in MAPPINGS[field_name].items():
        if normalized == name:
            return code
    raise ValueError(f"unsupported {field_name} value {text!r}")


def tags_from_codes(codes: dict[str, int | None]) -> dict[str, TagValue]:
    return {
        name: TagValue.from_code(name, codes.get(name))
        for name in FIELDS
    }
