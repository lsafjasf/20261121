"""Resolve possibly overlapping, prioritized channel intervals."""

from __future__ import annotations

from dataclasses import dataclass

from .model import LEFT, RIGHT, Region


@dataclass(frozen=True)
class ResolvedSpan:
    channel: str
    start_frame: int
    end_frame: int
    region: Region

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame


@dataclass(frozen=True)
class ChannelPlan:
    spans: tuple[ResolvedSpan, ...]
    shadowed: dict[str, tuple[tuple[int, int], ...]]


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    ordered = sorted(ranges)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _resolve_channel(regions: list[Region]) -> ChannelPlan:
    boundaries = sorted(
        {boundary for region in regions for boundary in (region.start_frame, region.end_frame)}
    )
    spans: list[ResolvedSpan] = []
    for start, end in zip(boundaries, boundaries[1:]):
        candidates = [
            region
            for region in regions
            if region.start_frame <= start and end <= region.end_frame
        ]
        if not candidates:
            continue
        winner = max(candidates, key=Region.stable_key)
        if (
            spans
            and spans[-1].region == winner
            and spans[-1].end_frame == start
        ):
            previous = spans[-1]
            spans[-1] = ResolvedSpan(
                channel=previous.channel,
                start_frame=previous.start_frame,
                end_frame=end,
                region=previous.region,
            )
        else:
            spans.append(
                ResolvedSpan(
                    channel=winner.channel,
                    start_frame=start,
                    end_frame=end,
                    region=winner,
                )
            )

    shadowed: dict[str, list[tuple[int, int]]] = {region.rule_id: [] for region in regions}
    for region in regions:
        for span in spans:
            overlap_start = max(region.start_frame, span.start_frame)
            overlap_end = min(region.end_frame, span.end_frame)
            if overlap_start < overlap_end and span.region != region:
                shadowed[region.rule_id].append((overlap_start, overlap_end))
    return ChannelPlan(
        spans=tuple(spans),
        shadowed={key: tuple(_merge_ranges(value)) for key, value in shadowed.items()},
    )


def build_channel_plans(regions: tuple[Region, ...]) -> dict[str, ChannelPlan]:
    by_channel = {
        LEFT: [region for region in regions if region.channel == LEFT],
        RIGHT: [region for region in regions if region.channel == RIGHT],
    }
    return {
        channel: _resolve_channel(channel_regions)
        for channel, channel_regions in by_channel.items()
    }
