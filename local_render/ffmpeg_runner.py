"""Build FFmpeg filter graphs from a :class:`RenderPlan` and run the render.

Each clip produces a *visible* stream (with an explicit black/silence prefix
when the source start is negative). Transition participants additionally
produce real media handle streams. xfade/acrossfade operate on
handle-inclusive streams while a leading black/silence buffer keeps the
visible cut aligned with the timeline; handle-only padding is trimmed off
before the track intermediate is written.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from local_render.errors import LocalRenderError

SAMPLE_RATE = 48000


def _run(command: list[str]):
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        raise LocalRenderError(
            f"ffmpeg failed ({proc.returncode}):\n"
            f"command: {' '.join(command)}\n{proc.stderr[-3000:]}"
        )


def _input_index(clip, url_to_index):
    if clip.target_url not in url_to_index:
        url_to_index[clip.target_url] = len(url_to_index)
    return url_to_index[clip.target_url]


def _timeline_frame_bounds(clip, rate):
    start = round(clip.source_in * rate / clip.source_rate)
    end = round(clip.source_out * rate / clip.source_rate)
    return start, end


def _source_time_bounds(clip):
    return (
        clip.source_in / clip.source_rate,
        clip.source_out / clip.source_rate,
    )


def _sample(frame, rate):
    return round(frame * SAMPLE_RATE / rate)


def _visible_video(clip, rate, width, height, index, chains):
    start, end = _timeline_frame_bounds(clip, rate)
    base = (
        f"[{index}:v]fps={rate}:round=near,scale={width}:{height},"
        f"format=rgb48,setsar=1"
    )
    media = f"v_{clip.name}_media"
    chains.append(
        base
        + f",trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS,"
        f"fps={rate}:round=near[{media}]"
    )
    if not clip.black_prefix:
        return media
    black = f"v_{clip.name}_black"
    chains.append(
        f"color=c=black:s={width}x{height}:r={rate}:"
        f"d={clip.black_prefix / rate:.6f},format=rgb48[{black}]"
    )
    out = f"v_{clip.name}_vis"
    chains.append(
        f"[{black}][{media}]concat=n=2:v=1:a=0,fps={rate}:round=near[{out}]"
    )
    return out


def _handle_video(clip, rate, width, height, index, kind, chains):
    start, end = _timeline_frame_bounds(clip, rate)
    if kind == "head":
        lo, hi = start - clip.head_handle_tl, start
    else:
        lo, hi = end, end + clip.tail_handle_tl
    label = f"v_{clip.name}_{kind}handle"
    chains.append(
        f"[{index}:v]fps={rate}:round=near,scale={width}:{height},"
        f"format=rgb48,setsar=1,trim=start_frame={lo}:end_frame={hi},"
        f"setpts=PTS-STARTPTS,fps={rate}:round=near[{label}]"
    )
    return label


def _visible_audio(clip, rate, index, chains):
    start_t, end_t = _source_time_bounds(clip)
    base = (
        f"[{index}:a]aresample={SAMPLE_RATE},aformat=sample_fmts=fltp:"
        f"channel_layouts=stereo,asetpts=N/SR/TB"
    )
    media = f"a_{clip.name}_media"
    chains.append(
        base
        + f",atrim=start_sample={round(start_t * SAMPLE_RATE)}:"
        f"end_sample={round(end_t * SAMPLE_RATE)},"
        f"asetpts=PTS-STARTPTS[{media}]"
    )
    if not clip.black_prefix:
        return media
    silence = f"a_{clip.name}_silence"
    chains.append(
        f"anullsrc=r={SAMPLE_RATE}:cl=stereo:"
        f"d={clip.black_prefix / rate:.6f}[{silence}]"
    )
    out = f"a_{clip.name}_vis"
    chains.append(
        f"[{silence}][{media}]concat=n=2:v=0:a=1[{out}]"
    )
    return out

def _audio_track_filtergraph(track, plan, url_to_index, chains):
    """Build a full-length audio track with explicit afade crossfades.

    Every clip is padded with leading silence so it starts at its timeline
    position; transition participants get complementary gain ramps placed
    exactly over the transition window, then all full-length layers are mixed.
    """
    rate = plan.timeline_rate
    total_frames = track.clips[-1].timeline_end
    layers = []
    transition_by_cut = {t.cut_frame: t for t in track.transitions}
    for clip in track.clips:
        media = _visible_audio(
            clip, rate, _input_index(clip, url_to_index), chains
        )
        label = f"al_{clip.name}"
        incoming_cut = next(
            (cut for cut, t in transition_by_cut.items()
             if t.incoming == clip.name), None
        )
        fade_preroll = (
            transition_by_cut[incoming_cut].duration_frames
            if incoming_cut is not None else 0
        )
        lead_frames = clip.timeline_start - fade_preroll
        if lead_frames:
            silence = f"as_{clip.name}"
            chains.append(
                f"anullsrc=r={SAMPLE_RATE}:cl=stereo:"
                f"d={lead_frames / rate:.6f}[{silence}]"
            )
            chains.append(
                f"[{silence}][{media}]concat=n=2:v=0:a=1[{label}]"
            )
        else:
            label = media
        fade = f"af_{clip.name}"
        filters = []
        outgoing_cut = next(
            (cut for cut, t in transition_by_cut.items()
             if t.outgoing == clip.name), None
        )
        if outgoing_cut is not None:
            transition = transition_by_cut[outgoing_cut]
            start = transition.start_frame / rate
            filters.append(
                f"afade=t=out:st={start:.6f}:d="
                f"{transition.duration_frames / rate:.6f}"
            )
        if incoming_cut is not None:
            transition = transition_by_cut[incoming_cut]
            start = transition.start_frame / rate
            filters.append(
                f"afade=t=in:st={start:.6f}:d="
                f"{transition.duration_frames / rate:.6f}"
            )
        if filters:
            chains.append(f"[{label}]" + ",".join(filters) + f"[{fade}]")
            label = fade
        layers.append(label)
    mixer_inputs = "".join(f"[{name}]" for name in layers)
    chains.append(
        f"{mixer_inputs}amix=inputs={len(layers)}:normalize=0:dropout_transition=0,"
        f"apad=whole_dur={total_frames / rate:.6f},"
        f"atrim=start_sample=0:end_sample={_sample(total_frames, rate)},"
        f"asetpts=PTS-STARTPTS[aout]"
    )
    return ";".join(chains)


def _track_filtergraph(track, plan, url_to_index):
    is_video = track.kind == "Video"
    rate = plan.timeline_rate
    chains: list[str] = []
    if not is_video:
        return _audio_track_filtergraph(track, plan, url_to_index, chains), "aout"
    visible = lambda clip: (
        _visible_video(clip, rate, plan.width, plan.height,
                       _input_index(clip, url_to_index), chains)
    )
    handle = lambda clip, kind: (
        _handle_video(clip, rate, plan.width, plan.height,
                      _input_index(clip, url_to_index), kind, chains)
    )
    transitions = {t.cut_frame: t for t in track.transitions}
    current = visible(track.clips[0])
    visible_output_frames = track.clips[0].timeline_end - track.clips[0].timeline_start
    serial = 0
    for clip in track.clips[1:]:
        next_visible = visible(clip)
        transition = transitions.get(clip.timeline_start)
        if transition is None:
            merged = f"m{serial}"
            joiner = "concat=n=2:v=1:a=0" if is_video else "concat=n=2:v=0:a=1"
            chains.append(f"[{current}][{next_visible}]{joiner}[{merged}]")
            current, serial = merged, serial + 1
            visible_output_frames += clip.timeline_end - clip.timeline_start
            continue
        outgoing = track.clips[track.clips.index(clip) - 1]
        stream_a = f"xa{serial}"
        stream_b = f"xb{serial}"
        target_visible = visible_output_frames + (
            clip.timeline_end - clip.timeline_start
        )
        head = clip.head_handle_tl
        tail = outgoing.tail_handle_tl
        a_len = (
            outgoing.timeline_end - outgoing.timeline_start
        ) + tail
        b_len = (
            head
            + (clip.timeline_end - clip.timeline_start)
            + clip.tail_handle_tl
        )
        target_len = max(a_len, b_len)
        a_parts = [current]
        if tail:
            a_parts.append(handle(outgoing, "tail"))
        b_parts = []
        if head:
            b_parts.append(handle(clip, "head"))
        b_parts.append(next_visible)
        b_real_tail = 0
        if clip.tail_handle_tl:
            b_parts.append(handle(clip, "tail"))
            b_real_tail = clip.tail_handle_tl
        suffix = f"ext{serial}"
        if is_video:
            build = lambda parts, label: chains.append(
                "".join(f"[{p}]" for p in parts)
                + f"concat=n={len(parts)}:v=1:a=0,fps={rate}:round=near[{label}]"
            )
        else:
            build = lambda parts, label: chains.append(
                "".join(f"[{p}]" for p in parts)
                + f"concat=n={len(parts)}:v=0:a=1[{label}]"
            )
        build(a_parts, stream_a)
        build(b_parts, stream_b)
        if a_len < target_len:
            extra = target_len - a_len
            pad = f"apad{serial}"
            if is_video:
                chains.append(
                    f"color=c=black:s={plan.width}x{plan.height}:r={rate}:"
                    f"d={extra / rate:.6f},format=rgb48[{pad}]"
                )
                chains.append(
                    f"[{stream_a}][{pad}]concat=n=2:v=1:a=0,"
                    f"fps={rate}:round=near[{suffix}]"
                )
            else:
                chains.append(
                    f"anullsrc=r={SAMPLE_RATE}:cl=stereo:"
                    f"d={extra / rate:.6f}[{pad}]"
                )
                chains.append(
                    f"[{stream_a}][{pad}]concat=n=2:v=0:a=1[{suffix}]"
                )
            stream_a = suffix
        if b_len < target_len:
            extra = target_len - b_len
            pad = f"bpad{serial}"
            if is_video:
                chains.append(
                    f"color=c=black:s={plan.width}x{plan.height}:r={rate}:"
                    f"d={extra / rate:.6f},format=rgb48[{pad}]"
                )
                chains.append(
                    f"[{stream_b}][{pad}]concat=n=2:v=1:a=0,"
                    f"fps={rate}:round=near[{suffix}b]"
                )
            else:
                chains.append(
                    f"anullsrc=r={SAMPLE_RATE}:cl=stereo:"
                    f"d={extra / rate:.6f}[{pad}]"
                )
                chains.append(
                    f"[{stream_b}][{pad}]concat=n=2:v=0:a=1[{suffix}b]"
                )
            stream_b = suffix + "b"
        cut = visible_output_frames
        duration = transition.duration_frames / rate
        # Streams are equal length; the overlap starts a transition duration
        # before the visible cut on the shared timeline.
        offset_frames = cut - transition.duration_frames
        merged = f"m{serial}"
        if is_video:
            chains.append(
                f"[{stream_a}][{stream_b}]xfade=transition=fade:"
                f"offset={offset_frames / rate:.6f}:duration={duration:.6f}"
                f"[{merged}]"
            )
        else:
            chains.append(
                f"[{stream_a}][{stream_b}]acrossfade=d={duration:.6f}:"
                f"c1=tri:c2=tri[{merged}]"
            )
        trimmed = f"t{serial}"
        target_visible = visible_output_frames + (
            clip.timeline_end - clip.timeline_start
        )
        if is_video:
            chains.append(
                f"[{merged}]trim=start_frame=0:end_frame={target_visible},"
                f"setpts=PTS-STARTPTS,fps={rate}:round=near[{trimmed}]"
            )
        else:
            chains.append(
                f"[{merged}]atrim=start_sample=0:end_sample="
                f"{_sample(target_visible, rate)},"
                f"asetpts=PTS-STARTPTS[{trimmed}]"
            )
        current, serial = trimmed, serial + 1
        visible_output_frames = target_visible
    output_label = "vout" if is_video else "aout"
    chains.append(
        f"[{current}]null[{output_label}]"
        if is_video else f"[{current}]anull[{output_label}]"
    )
    return ";".join(chains), output_label


def _input_args(url_to_index):
    ordered = [url for url, _ in sorted(url_to_index.items(), key=lambda x: x[1])]
    return [arg for url in ordered for arg in ("-i", url)]


def _render_track(track, plan, workdir, url_to_index):
    filtergraph, label = _track_filtergraph(track, plan, url_to_index)
    if track.kind == "Video":
        output_path = workdir / f"track_{track.name}.mkv"
        command = (
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
            + _input_args(url_to_index)
            + ["-filter_complex", filtergraph, "-map", f"[{label}]",
               "-c:v", "ffv1", "-pix_fmt", "yuv420p", "-r", str(plan.timeline_rate),
               str(output_path)]
        )
    else:
        output_path = workdir / f"track_{track.name}.wav"
        command = (
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
            + _input_args(url_to_index)
            + ["-filter_complex", filtergraph, "-map", f"[{label}]",
               "-c:a", "pcm_s16le", str(output_path)]
        )
    _run(command)
    return output_path


def _composite_command(plan, video_paths, audio_paths, output_path):
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for path in [*video_paths, *audio_paths]:
        command += ["-i", str(path)]
    filters = []
    video_count = len(video_paths)
    if video_count == 1:
        filters.append("[0:v]format=yuv420p[vout]")
    else:
        previous = "0:v"
        for index in range(1, video_count):
            out_label = "vout" if index == video_count - 1 else f"vstack{index}"
            filters.append(
                f"[{previous}][{index}:v]overlay=eof_action=pass:format=auto"
                f"[{out_label}]"
            )
            previous = out_label
    if audio_paths:
        mixer = "".join(f"[{video_count + i}:a]" for i in range(len(audio_paths)))
        filters.append(f"{mixer}amix=inputs={len(audio_paths)}:normalize=0[aout]")
    command += ["-filter_complex", ";".join(filters), "-map", "[vout]"]
    if audio_paths:
        command += ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]
    command += ["-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p", "-r", str(plan.timeline_rate),
                str(output_path)]
    return command


def render(plan, output_path, workdir):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    paths = {}
    for track in [*plan.video_tracks, *plan.audio_tracks]:
        url_to_index = {}
        paths[track.name] = str(_render_track(track, plan, workdir, url_to_index))
    video_paths = [Path(paths[t.name]) for t in plan.video_tracks]
    audio_paths = [Path(paths[t.name]) for t in plan.audio_tracks]
    _run(_composite_command(plan, video_paths, audio_paths, output_path))
    manifest_path = workdir / "contrib.json"
    manifest_path.write_text(json.dumps(plan.manifest, indent=2, ensure_ascii=False))
    from local_render.manifest import render_markdown

    (workdir / "contrib.md").write_text(render_markdown(plan.manifest))
    return {"output": str(output_path), "intermediates": paths,
            "manifest": str(manifest_path)}
