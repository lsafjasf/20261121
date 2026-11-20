from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from flashrgb.imageio import GrayImage, read_gray
from flashrgb.render import render_sequence
from flashrgb.sync import (
    FrameRecord,
    decode_flash,
    group_exposures,
    parse_roi,
)
from flashrgb.synth import generate_dataset


def _record(order: int, channel: str) -> FrameRecord:
    return FrameRecord(Path(f"frame_{order:04d}.pgm"), order, channel, 0.9, (0, 0, 0))


class SyncTests(unittest.TestCase):
    def test_spatial_flash_code(self) -> None:
        width = 96
        height = 32
        for cell, expected in enumerate(("R", "G", "B")):
            pixels = bytearray(width * height)
            for y in range(height):
                for x in range(cell * 32 + 2, (cell + 1) * 32 - 2):
                    pixels[y * width + x] = 240
            channel, confidence, _ = decode_flash(
                GrayImage(width, height, pixels), (0, 0, width, height)
            )
            self.assertEqual(channel, expected)
            self.assertGreater(confidence, 0.9)

    def test_grouping_marks_missing_exposures(self) -> None:
        labels = ["R", "G", "B", "R", "B", "R", "R", "G", "B", "G", "B"]
        groups = group_exposures([_record(index, item) for index, item in enumerate(labels)])
        self.assertEqual([group.status for group in groups], [
            "complete",
            "missing_1:G",
            "missing_2:GB",
            "complete",
            "missing_1:R",
        ])
        self.assertEqual([group.group_id for group in groups], [0, 1, 2, 3, 4])

    def test_auto_roi(self) -> None:
        self.assertEqual(parse_roi("auto", 320, 240), (0, 0, 96, 32))


class EndToEndTests(unittest.TestCase):
    def test_render_and_edge_metric(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            output = root / "output"
            generate_dataset(data, group_count=7, width=320, height=240)
            report = render_sequence(
                input_path=str(data / "frames"),
                output_dir=output,
                search_radius=16,
                fps=4,
                truth_path=str(data / "truth.json"),
                compensate=True,
            )
            self.assertEqual(report["group_count"], 7)
            self.assertEqual(report["complete_groups"], 5)
            self.assertEqual(report["groups_missing_two_exposures"], 1)
            self.assertGreater(report["edge_metrics"]["improvement_percent"], 40.0)
            for name in (
                "color_before.mp4",
                "color_after.mp4",
                "comparison.mp4",
                "pair_index.csv",
                "pair_index.json",
                "edge_metrics.json",
                "sync_report.json",
                "run_report.json",
            ):
                self.assertTrue((output / name).is_file(), name)
            rows = (output / "pair_index.csv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 8)
            self.assertTrue(any("missing_2:" in row for row in rows))
            sync = json.loads((output / "sync_report.json").read_text(encoding="utf-8"))
            self.assertEqual(sync["decoded_frames"], sync["total_frames"])
            first = read_gray(sorted((data / "frames").glob("*.pgm"))[0])
            self.assertEqual((first.width, first.height), (320, 240))


if __name__ == "__main__":
    unittest.main()
