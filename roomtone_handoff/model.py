"""Configuration data model and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

from .errors import ConfigError


LEFT = "left"
RIGHT = "right"
_CHANNEL_ALIASES = {
    "left": LEFT,
    "l": LEFT,
    "右": RIGHT,
    "right": RIGHT,
    "r": RIGHT,
    "both": "both",
    "双": "both",
}


@dataclass(frozen=True)
class Region:
    rule_id: str
    channel: str
    start_frame: int
    end_frame: int
    priority: int
    noise_offset: int = 0

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame

    def stable_key(self) -> tuple[int, int, int, str, int]:
        """The complete precedence key, independent of config array order."""
        return (
            self.priority,
            self.start_frame,
            self.end_frame,
            self.rule_id,
            self.noise_offset,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.rule_id,
            "channel": self.channel,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "priority": self.priority,
            "noise_offset_frames": self.noise_offset,
        }


@dataclass(frozen=True)
class Version:
    name: str
    regions: tuple[Region, ...]


@dataclass(frozen=True)
class HandoffConfig:
    source_path: Path
    room_tone_path: Path
    output_dir: Path
    versions: tuple[Version, ...]


def _expect_object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be a JSON object")
    return value


def _expect_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{where} must be a non-empty string")
    return value


def _expect_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where} must be an integer")
    return value


def _check_keys(
    obj: dict[str, Any], allowed: set[str], where: str
) -> None:
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise ConfigError(f"{where} has unknown field(s): {', '.join(unknown)}")


def _parse_channel(value: Any, where: str) -> str:
    raw = _expect_str(value, where).strip().lower()
    channel = _CHANNEL_ALIASES.get(raw)
    if channel is None:
        raise ConfigError(f"{where} must be left, right, or both")
    return channel


def _parse_region(raw_region: dict[str, Any], where: str) -> tuple[Region, ...]:
    _check_keys(
        raw_region,
        {
            "id",
            "channel",
            "start_frame",
            "end_frame",
            "start",
            "end",
            "priority",
            "noise_offset",
            "noise_offset_frames",
        },
        where,
    )
    channel = _parse_channel(raw_region.get("channel"), f"{where}.channel")
    if "start_frame" in raw_region and "start" in raw_region:
        raise ConfigError(f"{where} must not set both start_frame and start")
    if "end_frame" in raw_region and "end" in raw_region:
        raise ConfigError(f"{where} must not set both end_frame and end")
    start_key = "start_frame" if "start_frame" in raw_region else "start"
    end_key = "end_frame" if "end_frame" in raw_region else "end"
    if start_key not in raw_region or end_key not in raw_region:
        raise ConfigError(f"{where} needs start_frame and end_frame")
    start = _expect_int(raw_region[start_key], f"{where}.{start_key}")
    end = _expect_int(raw_region[end_key], f"{where}.{end_key}")
    if start < 0 or end <= start:
        raise ConfigError(f"{where} must satisfy 0 <= start_frame < end_frame")
    priority = _expect_int(raw_region.get("priority", 0), f"{where}.priority")
    if "noise_offset" in raw_region and "noise_offset_frames" in raw_region:
        raise ConfigError(f"{where} must not set both noise_offset fields")
    offset_value = raw_region.get(
        "noise_offset_frames", raw_region.get("noise_offset", 0)
    )
    noise_offset = _expect_int(offset_value, f"{where}.noise_offset_frames")
    if noise_offset < 0:
        raise ConfigError(f"{where}.noise_offset_frames must be non-negative")
    explicit_id = raw_region.get("id")
    if explicit_id is not None:
        _expect_str(explicit_id, f"{where}.id")
    channels = (LEFT, RIGHT) if channel == "both" else (channel,)
    regions = []
    for item in channels:
        if explicit_id is None:
            rule_id = f"auto-{item[0]}-{start}-{end}-p{priority}-n{noise_offset}"
        elif len(channels) == 1:
            rule_id = explicit_id
        else:
            rule_id = f"{explicit_id}@{item}"
        regions.append(
            Region(
                rule_id=rule_id,
                channel=item,
                start_frame=start,
                end_frame=end,
                priority=priority,
                noise_offset=noise_offset,
            )
        )
    return tuple(regions)


def _parse_version(raw_version: dict[str, Any], index: int) -> Version:
    where = f"versions[{index}]"
    _check_keys(raw_version, {"name", "regions"}, where)
    name = _expect_str(raw_version.get("name"), f"{where}.name")
    if name in {".", ".."} or any(char in name for char in ("/", "\\", "\x00")):
        raise ConfigError(f"{where}.name must be a safe file basename")
    raw_regions = raw_version.get("regions")
    if not isinstance(raw_regions, list):
        raise ConfigError(f"{where}.regions must be an array")
    parsed: list[Region] = []
    for region_index, raw_region in enumerate(raw_regions):
        region_where = f"{where}.regions[{region_index}]"
        parsed.extend(_parse_region(_expect_object(raw_region, region_where), region_where))
    seen: set[str] = set()
    for region in parsed:
        if region.rule_id in seen:
            raise ConfigError(f"{where} has duplicate rule id {region.rule_id!r}")
        seen.add(region.rule_id)
    return Version(name=name, regions=tuple(parsed))


def load_config(path: Path, output_override: Path | None = None) -> HandoffConfig:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {path}: {exc}") from exc
    root = _expect_object(payload, "config")
    _check_keys(root, {"schema", "source", "room_tone", "output_dir", "versions"}, "config")
    if "schema" in root and root["schema"] != "roomtone-handoff.config.v1":
        raise ConfigError(f"unsupported config schema: {root['schema']!r}")
    base = path.resolve().parent
    source = (base / _expect_str(root.get("source"), "config.source")).resolve()
    room_tone = (base / _expect_str(root.get("room_tone"), "config.room_tone")).resolve()
    configured_output = _expect_str(
        root.get("output_dir", "renders"), "config.output_dir"
    )
    output_dir = output_override or (base / configured_output)
    raw_versions = root.get("versions")
    if not isinstance(raw_versions, list) or not raw_versions:
        raise ConfigError("config.versions must be a non-empty array")
    versions = tuple(
        _parse_version(_expect_object(item, f"versions[{i}]"), i)
        for i, item in enumerate(raw_versions)
    )
    names = [version.name for version in versions]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ConfigError(f"duplicate version name(s): {', '.join(duplicates)}")
    return HandoffConfig(
        source_path=source,
        room_tone_path=room_tone,
        output_dir=Path(output_dir).resolve(),
        versions=versions,
    )
