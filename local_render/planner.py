"""Turn an OTIO timeline into a renderer-independent render plan.

The plan keeps every timing value in timeline frames (integer) while source
rate information is retained per clip so FFmpeg can apply the right
``fps`` conversion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import opentimelineio as otio
from opentimelineio.opentime import RationalTime

from local_render.errors import MissingHandleError


def _frames(value: RationalTime, rate: float) -> int:
    return round(value.rescaled_to(rate).to_frames())


def _timecode(frame: int, rate: float) -> str:
    return RationalTime(frame, rate).to_timecode()


@dataclass
class ClipPlan:
    name: str
    track: str
    kind: str
    target_url: str
    source_rate: float
    timeline_start: int
    timeline_end: int
    media_start: int
    source_in: int
    source_out: int
    black_prefix: int
    head_handle: int
    tail_handle: int
    head_handle_tl: int
    tail_handle_tl: int


@dataclass
class TransitionPlan:
    track: str
    kind: str
    cut_frame: int
    start_frame: int
    end_frame: int
    duration_frames: int
    outgoing: str
    incoming: str


@dataclass
class TrackPlan:
    name: str
    kind: str
    clips: list[ClipPlan] = field(default_factory=list)
    transitions: list[TransitionPlan] = field(default_factory=list)


@dataclass
class RenderPlan:
    path: Path
    timeline_rate: float
    width: int
    height: int
    total_frames: int
    video_tracks: list[TrackPlan]
    audio_tracks: list[TrackPlan]
    manifest: dict


def _require_media(clip: otio.schema.Clip) -> otio.schema.ExternalReference:
    ref = clip.media_reference
    if not isinstance(ref, otio.schema.ExternalReference) or not ref.target_url:
        raise ValueError(f"clip {clip.name!r} needs an external media reference")
    return ref


def _build_clip(clip, track_name, kind, timeline_rate, cursor_frame):
    ref = _require_media(clip)
    source_rate = ref.available_range.start_time.rate
    src = clip.source_range
    src_start = _frames(src.start_time, timeline_rate)
    src_duration = _frames(src.duration, timeline_rate)
    black_prefix = max(0, -src_start)
    media_start = cursor_frame + black_prefix
    source_in = max(0, src_start)
    source_out = source_in + (src_duration - black_prefix)
    avail = ref.available_range
    avail_start = _frames(avail.start_time, source_rate)
    avail_end = avail_start + _frames(avail.duration, source_rate)
    head_handle = source_in - avail_start
    tail_handle = avail_end - source_out
    head_handle_tl = round(head_handle * timeline_rate / source_rate)
    tail_handle_tl = round(tail_handle * timeline_rate / source_rate)
    return ClipPlan(
        name=clip.name,
        track=track_name,
        kind=kind,
        target_url=ref.target_url,
        source_rate=source_rate,
        timeline_start=cursor_frame,
        timeline_end=cursor_frame + src_duration,
        media_start=media_start,
        source_in=source_in,
        source_out=source_out,
        black_prefix=black_prefix,
        head_handle=head_handle,
        tail_handle=tail_handle,
        head_handle_tl=head_handle_tl,
        tail_handle_tl=tail_handle_tl,
    )


def _check_transitions(track_plan: TrackPlan, items, rate, errors):
    clips = {c.name: c for c in track_plan.clips}
    cursor = 0
    for index, item in enumerate(items):
        if isinstance(item, otio.schema.Clip):
            cursor += _frames(item.source_range.duration, rate)
            continue
        if not isinstance(item, otio.schema.Transition):
            continue
        in_frames = _frames(item.in_offset, rate)
        out_frames = _frames(item.out_offset, rate)
        cut_frame = cursor
        prev_clip = items[index - 1]
        next_clip = items[index + 1]
        prev = clips[prev_clip.name]
        nxt = clips[next_clip.name]
        need_prev = round(in_frames * prev.source_rate / rate)
        need_next = round(out_frames * nxt.source_rate / rate)
        if prev.tail_handle < need_prev:
            errors.append(
                f"{track_plan.name} :: {prev.name} needs {need_prev} frame(s) of "
                f"tail (in_offset) handle, only {prev.tail_handle} available "
                f"(transition at {_timecode(cut_frame, rate)})"
            )
        if nxt.head_handle < need_next:
            errors.append(
                f"{track_plan.name} :: {nxt.name} needs {need_next} frame(s) of "
                f"head (out_offset) handle, only {nxt.head_handle} available "
                f"(transition at {_timecode(cut_frame, rate)})"
            )
        track_plan.transitions.append(
            TransitionPlan(
                track=track_plan.name,
                kind=track_plan.kind,
                cut_frame=cut_frame,
                start_frame=cut_frame - in_frames,
                end_frame=cut_frame + out_frames,
                duration_frames=in_frames + out_frames,
                outgoing=prev.name,
                incoming=nxt.name,
            )
        )


def build_plan(timeline_path: str | Path) -> RenderPlan:
    """Read an OTIO file, validate handles and produce a :class:`RenderPlan`."""
    path = Path(timeline_path)
    timeline = otio.adapters.read_from_file(str(path))
    rate = timeline.duration().rate
    total_frames = _frames(timeline.duration(), rate)
    width = height = 0
    video_tracks: list[TrackPlan] = []
    audio_tracks: list[TrackPlan] = []
    errors: list[str] = []
    for track in timeline.video_tracks() + timeline.audio_tracks():
        kind = track.kind
        track_plan = TrackPlan(name=track.name, kind=kind)
        cursor = 0
        for item in track.find_clips():
            clip_plan = _build_clip(item, track.name, kind, rate, cursor)
            track_plan.clips.append(clip_plan)
            if kind == otio.schema.TrackKind.Video and clip_plan.source_out > 0:
                ref = item.media_reference
                meta = ref.metadata.get("local_render", {})
                width = int(meta.get("width", width))
                height = int(meta.get("height", height))
            cursor = clip_plan.timeline_end
        _check_transitions(track_plan, list(track), rate, errors)
        if kind == otio.schema.TrackKind.Video:
            video_tracks.append(track_plan)
        else:
            audio_tracks.append(track_plan)
    if errors:
        raise MissingHandleError(errors)
    from local_render.manifest import build_manifest

    manifest = build_manifest(rate, total_frames, video_tracks, audio_tracks)
    return RenderPlan(
        path=path,
        timeline_rate=rate,
        width=width,
        height=height,
        total_frames=total_frames,
        video_tracks=video_tracks,
        audio_tracks=audio_tracks,
        manifest=manifest,
    )
