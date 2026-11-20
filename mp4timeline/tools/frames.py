"""Per-frame thumbnails and decode/display contact sheets.

No Pillow is available in this environment, so frame extraction, annotation
and tiling are all delegated to ffmpeg (ass/libass for text, tile for grids).
"""
from __future__ import annotations
import subprocess
import math
from pathlib import Path

from .timing import Track, presentation_map

FONT = "DejaVu Sans Mono"


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed:\n{' '.join(cmd)}\n{proc.stderr}")


def _ass_time(seconds: float) -> str:
    centis = round(seconds * 100)
    h, rem = divmod(centis, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def extract_decode_order(movie: Path, out_dir: Path) -> list[Path]:
    """Extract every frame in decode (DTS) order via -vsync 0."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("decode_*.png"):
        stale.unlink()
    pattern = out_dir / "decode_%03d.png"
    run([
        # ignore_editlist bypasses empty edits so we get the raw coded frames
        # (one per sample) rather than padded presentation repeats.
        "ffmpeg", "-v", "error", "-y", "-ignore_editlist", "1",
        "-i", str(movie),
        "-map", "0:v:0", "-vsync", "0", str(pattern),
    ])
    return sorted(out_dir.glob("decode_*.png"))


def _annotation_ass(path: Path, lines_per_frame: list[list[str]],
                    frame_period: float, width: int, height: int) -> None:
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Tag,{FONT},15,&H0000FFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,3,1,0,7,6,6,4,1
Style: Sub,{FONT},12,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,3,1,0,1,6,6,4,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    out = [header]
    for i, lines in enumerate(lines_per_frame):
        start = _ass_time(i * frame_period)
        end = _ass_time((i + 1) * frame_period)
        slot = f"{start},{end}"
        out.append(f"Dialogue: 0,{slot},Tag,,0,0,0,,{lines[0]}")
        for extra in lines[1:]:
            out.append(f"Dialogue: 1,{slot},Sub,,0,0,0,,{extra}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def build_sheets(movie: Path, track: Track, work: Path,
                 out_dir: Path, tile_columns: int = 8) -> dict:
    raw = work / "raw_decode"
    frames = extract_decode_order(movie, raw)
    if len(frames) != len(track.samples):
        raise ValueError(
            f"frame count {len(frames)} != sample count {len(track.samples)}"
        )

    presentation = presentation_map(track)
    movie_time_by_sample = {r["sample_index"]: r["movie_pts_seconds"]
                            for r in presentation}
    display_pos_by_sample = {r["sample_index"]: r["display_position"]
                             for r in presentation}

    # Decode-order annotations: file N corresponds to decode sample index N-1.
    decode_lines: list[list[str]] = []
    for sample in track.samples:
        decode_lines.append([
            f"DEC#{sample.index} -> DISP#{display_pos_by_sample.get(sample.index, '-')}",
            f"PTS {sample.pts / track.media_timescale:.3f}s "
            f"OFF {sample.composition_offset / track.media_timescale:+.3f}s"
            + ("  KEY" if sample.is_keyframe else ""),
        ])

    fps = track.media_timescale / track.samples[0].duration if track.samples else 25
    frame_period = 1 / fps
    rows = math.ceil(len(track.samples) / tile_columns)
    width = track.width or 320
    height = track.height or 240
    decode_ass = work / "decode_labels.ass"
    _annotation_ass(decode_ass, decode_lines, frame_period, width, height)

    decode_sheet = out_dir / "contact_decode_order.jpg"
    run([
        "ffmpeg", "-v", "error", "-y", "-ignore_editlist", "1",
        "-i", str(movie),
        "-map", "0:v:0", "-vsync", "0",
        "-vf", f"ass='{decode_ass}',scale={width}:{height},"
               f"tile={tile_columns}x{rows}",
        "-frames:v", "1", "-q:v", "3", str(decode_sheet),
    ])

    # Display-order annotations: reorder the same decode frames by PTS.
    display_lines: list[list[str]] = []
    for display_pos, sample_index in enumerate(track.display_order):
        sample = track.samples[sample_index]
        movie_s = movie_time_by_sample.get(sample_index)
        movie_txt = f"MOVIE {movie_s:.3f}s" if movie_s is not None else "SKIPPED"
        display_lines.append([
            f"DISP#{display_pos} <- DEC#{sample_index}",
            f"{movie_txt}  "
            + ("KEY" if sample.is_keyframe else "B/P"),
        ])
    display_ass = work / "display_labels.ass"
    _annotation_ass(display_ass, display_lines, frame_period, width, height)

    # Reorder frames for display by feeding an image2 sequence sorted by PTS.
    ordered_dir = work / "raw_display"
    ordered_dir.mkdir(exist_ok=True)
    for stale in ordered_dir.glob("display_*.png"):
        stale.unlink()
    for display_pos, sample_index in enumerate(track.display_order):
        src = raw / f"decode_{sample_index + 1:03d}.png"
        dst = ordered_dir / f"display_{display_pos:03d}.png"
        dst.write_bytes(src.read_bytes())

    display_sheet = out_dir / "contact_display_order.jpg"
    run([
        "ffmpeg", "-v", "error", "-y",
        "-framerate", str(fps), "-i",
        str(ordered_dir / "display_%03d.png"),
        "-vf", f"ass='{display_ass}',tile={tile_columns}x{rows}",
        "-frames:v", "1", "-q:v", "3", str(display_sheet),
    ])

    # Per-frame labeled thumbnails kept as individual files for third parties.
    thumbs = out_dir / "frames_labeled"
    thumbs.mkdir(exist_ok=True)
    for stale in thumbs.glob("frame_*.jpg"):
        stale.unlink()
    run([
        "ffmpeg", "-v", "error", "-y",
        "-framerate", str(fps), "-i",
        str(ordered_dir / "display_%03d.png"),
        "-vf", f"ass='{display_ass}'",
        str(thumbs / "frame_%03d.jpg"),
    ])

    return {
        "decode_sheet": decode_sheet,
        "display_sheet": display_sheet,
        "thumb_dir": thumbs,
        "frame_count": len(frames),
    }
