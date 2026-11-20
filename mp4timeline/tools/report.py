"""CSV / JSON / Markdown exporters for the three timing layers."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from .timing import Track, presentation_map


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def export_sample_table(track: Track, path: Path) -> None:
    """Layer 1 + 2: one row per sample in decode order."""
    display_position = {sample_index: pos
                        for pos, sample_index in enumerate(track.display_order)}
    rows: list[list] = []
    for sample in track.samples:
        rows.append([
            sample.index,
            display_position[sample.index],
            sample.dts,
            round(track.to_seconds(sample.dts), 6),
            sample.pts,
            round(track.to_seconds(sample.pts), 6),
            sample.composition_offset,
            round(track.to_seconds(sample.composition_offset), 6),
            sample.duration,
            round(track.to_seconds(sample.duration), 6),
            int(sample.is_keyframe),
            sample.size,
            sample.data_offset,
        ])
    _write_csv(path, [
        "decode_index", "display_position",
        "dts_media_units", "dts_seconds",
        "pts_media_units", "pts_seconds",
        "composition_offset_units", "composition_offset_seconds",
        "duration_units", "duration_seconds",
        "is_keyframe", "size_bytes", "file_offset",
    ], rows)


def export_edit_table(track: Track, path: Path) -> None:
    """Layer 3: one row per edit list entry."""
    rows: list[list] = []
    movie_cursor = 0
    for edit in track.edits:
        if edit.is_empty:
            media_time_seconds = ""
            start_media_seconds = ""
        else:
            media_time_seconds = round(track.to_seconds(edit.media_time), 6)
            start_media_seconds = media_time_seconds
        rows.append([
            edit.index,
            int(edit.is_empty),
            edit.segment_duration,
            round(track.to_movie_seconds(edit.segment_duration), 6),
            movie_cursor,
            round(track.to_movie_seconds(movie_cursor), 6),
            edit.media_time,
            media_time_seconds,
            start_media_seconds,
            edit.media_rate,
        ])
        movie_cursor += edit.segment_duration
    _write_csv(path, [
        "edit_index", "is_empty_edit",
        "segment_duration_movie_units", "segment_duration_seconds",
        "movie_start_units", "movie_start_seconds",
        "media_time_units", "media_time_seconds",
        "media_start_seconds", "media_rate",
    ], rows)


def export_presentation_table(track: Track, path: Path) -> None:
    """Edit list applied: display order with movie timeline placement."""
    rows = []
    for row in presentation_map(track):
        rows.append([
            row["edit_index"],
            row["display_position"],
            row["sample_index"],
            row["media_pts"],
            round(track.to_seconds(row["media_pts"]), 6),
            round(row["movie_pts_seconds"], 6),
        ])
    _write_csv(path, [
        "edit_index", "display_position", "sample_decode_index",
        "media_pts_units", "media_pts_seconds", "movie_pts_seconds",
    ], rows)


def build_summary(movie_timescale: int, tracks: list[Track]) -> dict:
    summary = {"movie_timescale": movie_timescale, "tracks": []}
    for track in tracks:
        reordered = any(
            track.samples[i].composition_offset != 0 for i in range(len(track.samples))
        )
        first_display = track.display_order[0] if track.display_order else None
        summary["tracks"].append({
            "track_id": track.track_id,
            "handler": track.handler,
            "media_timescale": track.media_timescale,
            "media_duration_units": track.media_duration,
            "media_duration_seconds": round(track.to_seconds(track.media_duration), 6),
            "sample_count": len(track.samples),
            "edit_count": len(track.edits),
            "has_reordering": reordered,
            "first_displayed_sample_index": first_display,
            "display_order_first_16": track.display_order[:16],
            "edits": [
                {
                    "index": e.index,
                    "empty": e.is_empty,
                    "segment_duration_units": e.segment_duration,
                    "segment_duration_seconds":
                        round(track.to_movie_seconds(e.segment_duration), 6),
                    "media_time_units": e.media_time,
                    "media_time_seconds":
                        None if e.is_empty
                        else round(track.to_seconds(e.media_time), 6),
                    "media_rate": e.media_rate,
                }
                for e in track.edits
            ],
        })
    return summary


def export_json(summary: dict, path: Path) -> None:
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def export_markdown(summary: dict, path: Path, source: str) -> None:
    out = io.StringIO()
    out.write("# MP4 片段时序调查报告\n\n")
    out.write(f"- 源文件: `{source}`\n")
    out.write(f"- movie timescale: {summary['movie_timescale']}\n")
    out.write(f"- 轨道数: {len(summary['tracks'])}\n\n")
    for track in summary["tracks"]:
        kind = "视频" if track["handler"] == "vide" else "音频"
        out.write(f"## 轨道 {track['track_id']}（{kind} / {track['handler']}）\n\n")
        out.write(f"- media timescale: {track['media_timescale']}\n")
        out.write(f"- 媒体时长: {track['media_duration_seconds']} s "
                  f"（{track['media_duration_units']} 单位）\n")
        out.write(f"- 采样数: {track['sample_count']}\n")
        out.write(f"- 含 B 帧重排: {'是' if track['has_reordering'] else '否'}\n")
        out.write(f"- 首个展示采样: {track['first_displayed_sample_index']}\n")
        out.write(f"- 展示顺序前 16: {track['display_order_first_16']}\n\n")
        out.write("### Edit list（第三层）\n\n")
        out.write("| # | 空编辑 | 段时长(s) | media_time(s) | rate |\n")
        out.write("|---|--------|-----------|---------------|------|\n")
        for edit in track["edits"]:
            mt = "—" if edit["empty"] else edit["media_time_seconds"]
            out.write(f"| {edit['index']} | {'是' if edit['empty'] else '否'} "
                      f"| {edit['segment_duration_seconds']} | {mt} "
                      f"| {edit['media_rate']} |\n")
        out.write("\n")
    path.write_text(out.getvalue(), encoding="utf-8")
