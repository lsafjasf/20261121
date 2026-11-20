"""Pure rendering of one version from the pristine source WAV."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from .errors import ConfigError, WavError
from .model import LEFT, Version
from .planner import ChannelPlan, ResolvedSpan, build_channel_plans
from .wav import (
    WavAudio,
    extract_channel_frames,
    validate_room_tone,
    write_channel_frames,
)


@dataclass(frozen=True)
class NoiseMapping:
    output_start_frame: int
    output_end_frame: int
    noise_start_frame: int
    noise_end_frame: int


@dataclass(frozen=True)
class Replacement:
    span: ResolvedSpan
    noise_channel_index: int
    effective_noise_offset: int
    mappings: tuple[NoiseMapping, ...]
    channel_data: bytes
    channel_sha256: str


@dataclass(frozen=True)
class RenderResult:
    version: Version
    source: WavAudio
    room_tone: WavAudio
    plans: dict[str, ChannelPlan]
    replacements: tuple[Replacement, ...]
    output_bytes: bytes
    output_sha256: str


def _noise_channel(channel: str, room_tone: WavAudio) -> int:
    if channel == LEFT:
        return 0
    if room_tone.format.channels == 1:
        return 0
    return 1


def _replacement_for(
    source: WavAudio,
    room_tone: WavAudio,
    span: ResolvedSpan,
) -> Replacement:
    noise_channel = _noise_channel(span.channel, room_tone)
    noise_frames = room_tone.frame_count
    region = span.region
    effective_offset = (
        region.noise_offset + span.start_frame - region.start_frame
    ) % noise_frames
    cursor = effective_offset
    remaining = span.frame_count
    consumed = 0
    parts: list[bytes] = []
    mappings: list[NoiseMapping] = []
    while remaining:
        count = min(remaining, noise_frames - cursor)
        parts.append(
            extract_channel_frames(room_tone, noise_channel, cursor, count)
        )
        mappings.append(
            NoiseMapping(
                output_start_frame=span.start_frame + consumed,
                output_end_frame=span.start_frame + consumed + count,
                noise_start_frame=cursor,
                noise_end_frame=cursor + count,
            )
        )
        cursor = 0
        consumed += count
        remaining -= count
    channel_data = b"".join(parts)
    expected_bytes = span.frame_count * source.format.sample_width
    if len(channel_data) != expected_bytes:
        raise WavError("internal room-tone mapping produced the wrong byte count")
    return Replacement(
        span=span,
        noise_channel_index=noise_channel,
        effective_noise_offset=effective_offset,
        mappings=tuple(mappings),
        channel_data=channel_data,
        channel_sha256=hashlib.sha256(channel_data).hexdigest(),
    )


def render_version(
    source: WavAudio,
    room_tone: WavAudio,
    version: Version,
) -> RenderResult:
    if source.format.channels < 2:
        raise WavError("the source WAV must have left and right channels")
    validate_room_tone(source, room_tone)
    for region in version.regions:
        if region.end_frame > source.frame_count:
            raise ConfigError(
                f"rule {region.rule_id!r} ends at frame {region.end_frame}, "
                f"but the source has {source.frame_count} frames"
            )
        if region.noise_offset >= room_tone.frame_count:
            raise ConfigError(
                f"rule {region.rule_id!r} noise offset is outside the "
                f"{room_tone.frame_count}-frame room tone"
            )
    plans = build_channel_plans(version.regions)
    all_spans = sorted(
        (span for plan in plans.values() for span in plan.spans),
        key=lambda span: (span.channel != LEFT, span.start_frame, span.end_frame),
    )
    replacements = tuple(
        _replacement_for(source, room_tone, span) for span in all_spans
    )
    output_data = bytearray(source.data)
    for replacement in replacements:
        channel_index = 0 if replacement.span.channel == LEFT else 1
        write_channel_frames(
            output_data,
            source,
            channel_index,
            replacement.span.start_frame,
            replacement.channel_data,
        )
    output_bytes = source.with_data(bytes(output_data))
    return RenderResult(
        version=version,
        source=source,
        room_tone=room_tone,
        plans=plans,
        replacements=replacements,
        output_bytes=output_bytes,
        output_sha256=hashlib.sha256(output_bytes).hexdigest(),
    )
