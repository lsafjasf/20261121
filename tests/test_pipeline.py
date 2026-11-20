import tempfile
import unittest
from pathlib import Path

from assettone.detector import analyze_file
from assettone.report import attach_ground_truth
from assettone.synth import build_sample


class PipelineTests(unittest.TestCase):
    def test_two_assets_survive_damage_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.wav"
            manifest = build_sample(path)
            report = attach_ground_truth(analyze_file(str(path)), manifest)

        summary = report["summary"]
        self.assertEqual(summary["hits"], 6)
        self.assertEqual(summary["candidates"], 1)
        self.assertEqual(summary["false_positives"], 0)
        self.assertEqual(summary["missed"], 0)
        self.assertEqual(summary["damaged_candidates"], 1)
        self.assertLess(summary["timing_mae_ms"], 5.0)

        candidate = next(item for item in report["detections"] if item["kind"] == "candidate")
        self.assertFalse(candidate["data"]["crc_ok"])
        self.assertTrue(candidate["sync"]["locked"])


if __name__ == "__main__":
    unittest.main()
