"""Generate ASS subtitle files for per-frame burn-in labels.

The static ffmpeg build here ships libass but not the drawtext filter, so all
on-screen text is rendered through the `ass` filter instead.
"""
from __future__ import annotations

from pathlib import Path

ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 320
PlayResY: 240

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Label,DejaVu Sans Mono,18,&H00FFFFFF,&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,1,0,7,8,8,6,1
Style: Frame,DejaVu Sans Mono,17,&H0000FFFF,&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,1,0,9,8,8,6,1
Style: Clock,DejaVu Sans Mono,13,&H00FFFFFF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,1,0,1,8,8,6,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _ass_time(seconds: float) -> str:
    centis = round(seconds * 100)
    h, rem = divmod(centis, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def write_frame_labels(path: Path, label: str, frame_count: int,
                       frame_period_seconds: float) -> None:
    lines = [ASS_HEADER]
    for frame in range(frame_count):
        start = frame * frame_period_seconds
        end = (frame + 1) * frame_period_seconds
        slot = f"{_ass_time(start)},{_ass_time(end)}"
        lines.append(
            f"Dialogue: 0,{slot},Label,,0,0,0,,{label}"
        )
        lines.append(
            f"Dialogue: 0,{slot},Frame,,0,0,0,,f{frame:02d}"
        )
        clock = f"{start:0.3f}s"
        lines.append(
            f"Dialogue: 0,{slot},Clock,,0,0,0,,{clock}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
