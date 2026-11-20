"""Command line entry: analyze an MP4 and emit all timing artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import report
from .frames import build_sheets
from .timing import parse_movie


def investigate(movie: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = out_dir / "tables"
    tables.mkdir(exist_ok=True)
    for stale in tables.glob("*.csv"):
        stale.unlink()
    for stale in tables.glob("*.json"):
        stale.unlink()
    for stale in tables.glob("*.md"):
        stale.unlink()

    data = movie.read_bytes()
    movie_timescale, tracks, _ = parse_movie(data)

    produced: list[Path] = []
    for track in tracks:
        kind = "video" if track.is_video else "audio"
        prefix = f"{kind}_id{track.track_id}"
        sample_csv = tables / f"{prefix}_layer12_samples.csv"
        edit_csv = tables / f"{prefix}_layer3_edits.csv"
        pres_csv = tables / f"{prefix}_presentation.csv"
        report.export_sample_table(track, sample_csv)
        report.export_edit_table(track, edit_csv)
        report.export_presentation_table(track, pres_csv)
        produced.extend([sample_csv, edit_csv, pres_csv])

    summary = report.build_summary(movie_timescale, tracks)
    summary_json = tables / "summary.json"
    report.export_json(summary, summary_json)
    report.export_markdown(summary, tables / "report.md", str(movie))
    produced.append(summary_json)

    sheet_info = None
    video = next((t for t in tracks if t.is_video), None)
    if video is not None:
        sheet_info = build_sheets(movie, video, out_dir / "_frames_work",
                                  out_dir)

    return {
        "tables": produced,
        "summary": summary,
        "sheets": sheet_info,
        "out_dir": out_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MP4 sample-duration / composition-offset / edit-list "
                    "timing investigation tool."
    )
    parser.add_argument("movie", type=Path, help="input .mp4")
    parser.add_argument("out_dir", type=Path, help="artifact output directory")
    args = parser.parse_args()
    result = investigate(args.movie, args.out_dir)
    print(f"Analyzed {args.movie}")
    for table in result["tables"]:
        print(f"  table: {table}")
    if result["sheets"]:
        print(f"  decode sheet:  {result['sheets']['decode_sheet']}")
        print(f"  display sheet: {result['sheets']['display_sheet']}")
        print(f"  thumbnails:    {result['sheets']['thumb_dir']}")


if __name__ == "__main__":
    main()
