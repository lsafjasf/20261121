"""Contribution manifest: per-frame source weights and segment summary."""

from __future__ import annotations

from opentimelineio.opentime import RationalTime


def _tc(frame: int, rate: float) -> str:
    return RationalTime(frame, rate).to_timecode()


def _track_contributors(track_plan, total_frames):
    """Return ``{frame: [(contributor, weight), ...]}`` for one track."""
    by_frame: dict[int, list[tuple[str, float]]] = {}
    for clip in track_plan.clips:
        for frame in range(clip.timeline_start, clip.timeline_end):
            if frame < clip.media_start:
                by_frame[frame] = [(f"black:{clip.name}", 1.0)]
            else:
                by_frame[frame] = [(clip.name, 1.0)]
    for transition in track_plan.transitions:
        n = transition.duration_frames
        for index, frame in enumerate(range(transition.start_frame, transition.end_frame)):
            # Match FFmpeg xfade's linear fade: weight steps uniformly from
            # 0 (first transition frame) to 1 - 1/n across the overlap.
            incoming_weight = index / n
            outgoing_weight = 1.0 - incoming_weight
            by_frame[frame] = [
                (transition.outgoing, round(outgoing_weight, 6)),
                (transition.incoming, round(incoming_weight, 6)),
            ]
    return by_frame


def _segments(per_frame_tracks, total_frames, rate):
    """Group consecutive frames with identical contributor signatures."""
    signatures = []
    for frame in range(total_frames):
        signature = tuple(
            (track_name, tuple(per_frame.get(frame, ())))
            for track_name, per_frame in per_frame_tracks
        )
        signatures.append(signature)
    segments = []
    start = 0
    for frame in range(1, total_frames + 1):
        if frame == total_frames or signatures[frame] != signatures[start]:
            segments.append(
                {
                    "start_frame": start,
                    "end_frame": frame - 1,
                    "start_timecode": _tc(start, rate),
                    "end_timecode": _tc(frame, rate),
                    "tracks": [
                        {
                            "track": track_name,
                            "contributors": [
                                {"source": name, "weight": weight}
                                for name, weight in contributors
                            ],
                        }
                        for track_name, contributors in signatures[start]
                    ],
                }
            )
            start = frame
    return segments


def build_manifest(rate, total_frames, video_tracks, audio_tracks):
    per_frame_tracks = [
        (track.name, _track_contributors(track, total_frames))
        for track in video_tracks
    ]
    per_frame = []
    for frame in range(total_frames):
        entry = {"frame": frame, "timecode": _tc(frame, rate), "tracks": []}
        for track_name, contributors_by_frame in per_frame_tracks:
            entry["tracks"].append(
                {
                    "track": track_name,
                    "contributors": [
                        {"source": name, "weight": weight}
                        for name, weight in contributors_by_frame.get(frame, [])
                    ],
                }
            )
        per_frame.append(entry)
    sources = [
        {
            "clip": clip.name,
            "track": clip.track,
            "kind": clip.kind,
            "source_fps": clip.source_rate,
            "target_url": clip.target_url,
            "timeline_start": clip.timeline_start,
            "timeline_end": clip.timeline_end,
            "black_prefix_frames": clip.black_prefix,
            "head_handle_frames": clip.head_handle,
            "tail_handle_frames": clip.tail_handle,
        }
        for track in [*video_tracks, *audio_tracks]
        for clip in track.clips
    ]
    return {
        "timeline_rate": rate,
        "total_frames": total_frames,
        "duration_seconds": total_frames / rate,
        "sources": sources,
        "segments": _segments(per_frame_tracks, total_frames, rate),
        "per_frame": per_frame,
    }


def render_markdown(manifest: dict) -> str:
    lines = [
        "# Contribution manifest",
        "",
        f"- Timeline rate: {manifest['timeline_rate']} fps",
        f"- Total frames: {manifest['total_frames']}",
        f"- Duration: {manifest['duration_seconds']:.3f} s",
        "",
        "## Sources",
        "",
        "| Clip | Track | Source fps | Black prefix | Head handle | Tail handle |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for source in manifest["sources"]:
        lines.append(
            f"| {source['clip']} | {source['track']} | {source['source_fps']} | "
            f"{source['black_prefix_frames']} | {source['head_handle_frames']} | "
            f"{source['tail_handle_frames']} |"
        )
    lines.extend(["", "## Per-segment contributors (video)", ""])
    for segment in manifest["segments"]:
        lines.append(
            f"### Frames {segment['start_frame']}–{segment['end_frame']} "
            f"({segment['start_timecode']}–{segment['end_timecode']})"
        )
        for track in segment["tracks"]:
            contributors = track["contributors"]
            if not contributors:
                lines.append(f"- {track['track']}: (empty)")
                continue
            text = ", ".join(
                f"{item['source']} w={item['weight']:.3f}" for item in contributors
            )
            lines.append(f"- {track['track']}: {text}")
        lines.append("")
    return "\n".join(lines)
