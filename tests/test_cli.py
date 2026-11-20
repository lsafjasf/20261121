from __future__ import annotations

import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path

import cv2

from film_stabilizer.cli import main
from film_stabilizer.io import open_source
from film_stabilizer.synthetic import SyntheticConfig, SyntheticSequence


class CliTests(unittest.TestCase):
    def test_outputs_keep_original_frame_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames_dir = root / "scans"
            frames_dir.mkdir()
            sequence = SyntheticSequence(
                SyntheticConfig(
                    width=320,
                    height=240,
                    frames=8,
                    splice_frame=None,
                )
            )
            for index, _, _, frame in sequence.iter_frames():
                cv2.imwrite(str(frames_dir / f"scan_{100 + index:06d}.png"), frame)

            source = open_source(frames_dir)
            self.assertEqual([item[2] for item in source.iter_frames()], list(range(100, 108)))
            out_dir = root / "result"
            with contextlib.redirect_stdout(io.StringIO()):
                main(
                    [
                        str(frames_dir),
                        "--out",
                        str(out_dir),
                        "--write-frames",
                        "--burn-frame-number",
                    ]
                )

            with (out_dir / "crop_ranges.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["original_frame"]) for row in rows], list(range(100, 108)))
            self.assertTrue(
                all(int(row["crop_width"]) > 0 for row in rows)
            )
            self.assertTrue((out_dir / "frames_stabilized" / "scan_000100.png").exists())
            self.assertTrue((out_dir / "stabilized.mp4").exists())
            self.assertTrue((out_dir / "stabilized_cropped.mp4").exists())
            self.assertTrue((out_dir / "manifest.json").exists())

            capture = cv2.VideoCapture(str(out_dir / "stabilized.mp4"))
            count = 0
            while True:
                ok, _ = capture.read()
                if not ok:
                    break
                count += 1
            capture.release()
            self.assertEqual(count, 8)


if __name__ == "__main__":
    unittest.main()
