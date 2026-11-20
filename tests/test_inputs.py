from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from oscilloscope_video.inputs import load_input


class InputTests(unittest.TestCase):
    def test_csv_trigger_edges_and_unit_preservation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            path.write_text(
                "time_s,voltage,unit,trigger\n"
                "0.000,0.1,mV,0\n"
                "0.001,0.2,mV,1\n"
                "0.002,,mV,1\n"
                "0.003,0.3,mV,0\n",
                encoding="utf-8",
            )
            data = load_input(path)
        self.assertEqual(data.unit, "mV")
        self.assertEqual([event.time_s for event in data.events], [0.001])
        self.assertIsNone(data.voltages[2])

    def test_unit_conflict_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(
                json.dumps(
                    {
                        "unit": "V",
                        "samples": [[0.0, 1.0], [0.1, 2.0]],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_input(path, unit="mV")


if __name__ == "__main__":
    unittest.main()
