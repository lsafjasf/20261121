from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from venue_wiring.cli import main


class CliTests(unittest.TestCase):
    def test_generate_and_analyze_commands_write_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenario = root / "scenario"
            report = root / "report"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = main(
                    [
                        "generate",
                        "--output",
                        str(scenario),
                        "--sample-rate",
                        "2500",
                        "--duration",
                        "1.3",
                        "--reverb",
                        "0.25",
                        "--start-offsets-ms",
                        "0,15,35,8,50,25,4,30",
                    ]
                )
            self.assertEqual(status, 0)
            self.assertTrue((scenario / "reference.wav").exists())
            self.assertTrue((scenario / "ground_truth.json").exists())

            with contextlib.redirect_stdout(output):
                status = main(
                    [
                        "analyze",
                        "--reference",
                        str(scenario / "reference.wav"),
                        "--recording",
                        str(scenario / "recording.wav"),
                        "--reference-labels",
                        "S1,S2,S3,S4,S5,S6,S7,S8",
                        "--return-labels",
                        "A,B,C,D,E,F,G,H",
                        "--max-lag-seconds",
                        "0.8",
                        "--output",
                        str(report),
                    ]
                )
            self.assertEqual(status, 0)
            self.assertTrue((report / "report.json").exists())
            self.assertTrue((report / "report.md").exists())
            self.assertTrue((report / "evidence.csv").exists())
            self.assertIn("分析完成", output.getvalue())


if __name__ == "__main__":
    unittest.main()
