"""Build the reference fixture.

Produces a single H.264/AAC MP4 whose continuous media contains three visual
segments (simulating recordings that were paused and resumed), then rewrites
the edit lists to expose:

* an empty-edit pre-roll (movie time with no media),
* paused gaps between segments,
* an audio track that starts later than the picture,
* H.264 B-frames so decode order != display order.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from .boxpatcher import rewrite_movie
from .timing import parse_movie
from .ass import write_frame_labels


# Movie-timescale (1000 units) layout -------------------------------
PREROLL_MS = 700          # empty edit before any picture
GAP_MS = 400              # paused gap between two segments
SEGMENT_MS = 1000         # each visual segment is one second
AUDIO_DELAY_MS = 600      # audio starts this far after the first picture
# -------------------------------------------------------------------


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed:\n{' '.join(cmd)}\n{proc.stderr}")


def make_segment(path: Path, index: int, label: str, color: str) -> None:
    # testsrc2 gives a moving recognizable pattern; a colored band plus an ASS
    # subtitle stamps the segment name and a per-frame presentation counter.
    ass_path = path.with_suffix(".ass")
    write_frame_labels(ass_path, label, 25, 1 / 25)
    vf = (
        f"testsrc2=size=320x240:rate=25:duration=1,"
        f"drawbox=x=0:y=0:w=iw:h=30:color={color}@0.8:t=fill,"
        f"ass='{ass_path}'"
    )
    run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", vf,
        "-c:v", "libx264", "-bf", "3", "-g", "25",
        "-pix_fmt", "yuv420p", "-video_track_timescale", "12800",
        str(path),
    ])


def make_audio(path: Path, duration_seconds: float) -> None:
    run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", f"sine=frequency=880:duration={duration_seconds}",
        "-c:a", "aac", str(path),
    ])


def concat_video(segments: list[Path], path: Path) -> None:
    list_file = path.with_suffix(".txt")
    list_file.write_text(
        "".join(f"file '{s.resolve()}'\n" for s in segments)
    )
    run([
        "ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file), "-c", "copy", str(path),
    ])


def mux(video: Path, audio: Path, path: Path) -> None:
    run([
        "ffmpeg", "-v", "error", "-y",
        "-i", str(video), "-i", str(audio),
        "-c", "copy", "-map", "0:v:0", "-map", "1:a:0",
        str(path),
    ])


def build(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_fixture_build"
    work.mkdir(exist_ok=True)

    labels = [("SEG-A-take1", "red"),
              ("SEG-B-take2", "darkgreen"),
              ("SEG-C-take3", "navy")]
    segments = []
    for i, (label, color) in enumerate(labels):
        seg = work / f"seg{i}.mp4"
        make_segment(seg, i, label, color)
        segments.append(seg)

    video_only = work / "video.mp4"
    concat_video(segments, video_only)

    audio = work / "audio.m4a"
    make_audio(audio, 3.0)

    merged = work / "merged.mp4"
    mux(video_only, audio, merged)

    data = merged.read_bytes()
    movie_ts, tracks, _ = parse_movie(data)
    video = next(t for t in tracks if t.is_video)
    sound = next(t for t in tracks if t.handler == "soun")

    # Native AAC pre-roll that ffmpeg already encoded (media_time to skip).
    audio_native_preroll = next(
        (e.media_time for e in sound.edits if not e.is_empty), 0
    )
    # Video native pre-roll points at the first presented sample's PTS.
    video_native_preroll = next(
        (e.media_time for e in video.edits if not e.is_empty), 0
    )

    seg_units = video.media_timescale  # 12800 => 1 s per segment
    gap_media = GAP_MS * video.media_timescale // movie_ts

    # Video edits: empty pre-roll, then three media segments with empty gaps.
    video_edits = [
        (PREROLL_MS, -1),
        (SEGMENT_MS, video_native_preroll),
        (GAP_MS, -1),
        (SEGMENT_MS, video_native_preroll + seg_units),
        (GAP_MS, -1),
        (SEGMENT_MS, video_native_preroll + 2 * seg_units),
    ]
    video_track_duration = PREROLL_MS + 3 * SEGMENT_MS + 2 * GAP_MS

    # Audio edits: keep the native encoder pre-roll, then delay the start.
    audio_media_start = audio_native_preroll
    audio_present_ms = 3000
    audio_edits = [
        (PREROLL_MS + AUDIO_DELAY_MS, -1),
        (audio_present_ms, audio_media_start),
    ]
    audio_track_duration = PREROLL_MS + AUDIO_DELAY_MS + audio_present_ms

    patched = rewrite_movie(data, {
        "vide": {
            "edits": video_edits,
            "track_duration_movie_units": video_track_duration,
        },
        "soun": {
            "edits": audio_edits,
            "track_duration_movie_units": audio_track_duration,
        },
    })

    output = out_dir / "fixture.mp4"
    output.write_bytes(patched)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    output = build(args.out_dir)
    print(output)


if __name__ == "__main__":
    main()
