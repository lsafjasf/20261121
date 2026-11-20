"""High-level render pipeline and machine-readable manifest."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from .audio import (
    DEFAULT_CUE_SECONDS,
    DEFAULT_CUE_HZ,
    build_cue_audio,
    cues_for_events,
    write_wav,
)
from .encode import encode_mp4
from .model import Timeline, TraceData, capture_bounds
from .render import RED, RenderConfig, TraceRenderer

AUDIO_SAMPLE_RATE = 48000


def render_trace_video(
    data: TraceData,
    output_path: str | Path,
    *,
    fps: int,
    window_seconds: float,
    duration_seconds: float | None = None,
    width: int = 960,
    height: int = 540,
    supersample: int = 2,
    title: str = "OSCILLOSCOPE TRACE",
    subtitle: str = "",
    y_min: float | None = None,
    y_max: float | None = None,
    cue_seconds: float = DEFAULT_CUE_SECONDS,
    crf: int = 18,
    preset: str = "medium",
    progress: bool = False,
) -> dict[str, Any]:
    capture_start, capture_end = capture_bounds(data)
    timeline = Timeline.fit(
        capture_start,
        capture_end,
        window_seconds,
        fps,
        duration_seconds,
    )
    config = RenderConfig(
        width=width,
        height=height,
        supersample=supersample,
        title=title,
        subtitle=subtitle,
        y_min=y_min,
        y_max=y_max,
    )
    renderer = TraceRenderer(data, timeline, config)
    cues = cues_for_events(data.events, timeline, AUDIO_SAMPLE_RATE, cue_seconds=cue_seconds)
    playback_duration = timeline.frame_count / fps
    samples = build_cue_audio(playback_duration, cues, AUDIO_SAMPLE_RATE)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix(".manifest.json")

    with tempfile.TemporaryDirectory(prefix="scope-audio-") as temp_dir:
        wav_path = Path(temp_dir) / "cues.wav"
        write_wav(wav_path, samples, AUDIO_SAMPLE_RATE)
        encode_mp4(
            output,
            wav_path,
            renderer,
            timeline,
            width=width,
            height=height,
            supersample=supersample,
            crf=crf,
            preset=preset,
            progress=_progress_printer() if progress else None,
        )

    manifest = build_manifest(
        data,
        timeline,
        renderer,
        cues,
        output,
        width=width,
        height=height,
        supersample=supersample,
        cue_seconds=cue_seconds,
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_manifest(
    data: TraceData,
    timeline: Timeline,
    renderer: TraceRenderer,
    cues: list,
    output: Path,
    *,
    width: int,
    height: int,
    supersample: int,
    cue_seconds: float,
) -> dict[str, Any]:
    events = []
    for index, (event, cue) in enumerate(zip(data.events, cues)):
        first_frame = timeline.first_frame(event.time_s)
        events.append(
            {
                "index": index,
                "label": event.safe_label,
                "time_s": round(event.time_s, 9),
                "video_time_s": round(cue.video_time_s, 9),
                "first_frame": first_frame,
                "first_frame_pts_s": round(first_frame / timeline.fps, 9),
                "screen_x": _event_screen_x(renderer, timeline, event, first_frame, width),
                "audio_sample": cue.start_sample,
                "audio_sample_count": cue.sample_count,
                "cue_hz": cue.hz,
            }
        )

    return {
        "format": "oscilloscope-trace-video",
        "version": 1,
        "video": {
            "file": output.name,
            "width": width,
            "height": height,
            "supersample": supersample,
            "fps": timeline.fps,
            "frame_count": timeline.frame_count,
            "duration_s": round(timeline.frame_count / timeline.fps, 9),
            "codec": "h264",
            "pixel_format": "yuv420p",
            "constant_frame_rate": True,
        },
        "audio": {
            "codec": "aac",
            "sample_rate": AUDIO_SAMPLE_RATE,
            "channels": 1,
            "cue_seconds": cue_seconds,
            "default_cue_hz": DEFAULT_CUE_HZ,
        },
        "timeline": {
            "capture_start_s": round(timeline.capture_start_s, 9),
            "capture_end_s": round(timeline.capture_end_s, 9),
            "capture_span_s": round(timeline.capture_span_s, 9),
            "playback_duration_s": timeline.duration_s,
            "time_scale": round(timeline.time_scale, 9),
            "window_span_s": timeline.window_span_s,
            "window_mode": "centered-smooth-scroll",
            "event_first_frame_rule": "ceil(video_time_s * fps)",
        },
        "trace": {
            "source": data.source,
            "unit": data.unit,
            "sample_count": len(data.times_s),
            "valid_sample_count": len(data.valid_voltages),
            "gap_factor": data.gap_factor,
            "gap_threshold_s": round(data.gap_threshold(), 9),
            "unknown_intervals": [
                [round(start, 9), round(end, 9)]
                for start, end in data.effective_unknown_intervals()
            ],
            "explicit_unknown_intervals": [
                [round(start, 9), round(end, 9)]
                for start, end in data.unknown_intervals
            ],
        },
        "render": {
            "temporal_reduction": "per-column min/max envelope (preserves transients)",
            "spatial_antialiasing": "supersample + Lanczos downscale",
            "event_flash_frames": renderer.config.flash_frames,
            "event_flash_rgb": list(RED),
        },
        "events": events,
    }


def _event_screen_x(
    renderer: TraceRenderer,
    timeline: Timeline,
    event,
    first_frame: int,
    output_width: int,
) -> int:
    view_start = timeline.window_start(first_frame)
    fraction = (event.time_s - view_start) / timeline.window_span_s
    plot_x0 = renderer.plot_x0 / renderer.ss
    plot_width = renderer.plot_w / renderer.ss
    return round(plot_x0 + fraction * plot_width)


def _progress_printer():
    last_bucket = {"value": -1}

    def report(done: int, total: int) -> None:
        bucket = min(20, int(done * 20 / total))
        if bucket != last_bucket["value"]:
            last_bucket["value"] = bucket
            print(f"render {bucket * 5:3d}% ({done}/{total} frames)", flush=True)

    return report
