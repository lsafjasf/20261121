from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from venue_wiring.analyze import AnalysisConfig, analyze
from venue_wiring.report import write_report
from venue_wiring.synth import GenerationConfig, generate_scenario
from venue_wiring.wavio import WavData


class GeneratedScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.generation = GenerationConfig(
            sample_rate=4000,
            duration_seconds=1.8,
            reverb_seconds=0.35,
            start_offsets_ms=[0.0, 20.0, 50.0, 10.0, 70.0, 35.0, 5.0, 45.0],
        )
        cls.truth = generate_scenario(cls.root / "scenario", cls.generation)
        cls.report = analyze(
            cls.root / "scenario" / "reference.wav",
            cls.root / "scenario" / "recording.wav",
            AnalysisConfig(max_lag_seconds=1.0),
            reference_labels=cls.truth["source_labels"],
            return_labels=cls.truth["return_labels"],
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def test_mixed_faults_are_classified_without_manual_review(self) -> None:
        summary = self.report["summary"]
        self.assertEqual(summary["ok"], 5)
        self.assertEqual(summary["inverted"], 1)
        self.assertEqual(summary["duplicate"], 1)
        self.assertEqual(summary["open"], 1)
        self.assertEqual(summary["review"], 0)
        self.assertFalse(summary["manual_review_required"])

    def test_mapping_polarity_and_open_match_ground_truth(self) -> None:
        truth_by_channel = {
            item["return_channel"]: item for item in self.truth["channels"]
        }
        report_by_channel = {
            item["return_channel"]: item for item in self.report["channels"]
        }
        for label, expected in truth_by_channel.items():
            actual = report_by_channel[label]
            if expected["status"] == "open":
                self.assertEqual(actual["status"], "open")
                self.assertIsNone(actual["mapped_source"])
                continue
            self.assertEqual(actual["mapped_source"], expected["source"])
            self.assertEqual(actual["polarity"], expected["polarity"])

    def test_duplicate_group_and_unobserved_sources(self) -> None:
        self.assertEqual(len(self.report["duplicate_groups"]), 1)
        group = self.report["duplicate_groups"][0]
        self.assertEqual(group["source"], "SRC-2")
        self.assertEqual(set(group["members"]), {"SPK-B", "SPK-G"})
        self.assertEqual(
            set(self.report["summary"]["unobserved_sources"]), {"SRC-7", "SRC-8"}
        )

    def test_every_channel_publishes_all_eight_correlation_peaks(self) -> None:
        for channel in self.report["channels"]:
            peaks = channel["evidence"]["all_source_peaks"]
            self.assertEqual(len(peaks), 8)
            self.assertEqual(len({peak["source"] for peak in peaks}), 8)
            self.assertIn("confidence", channel)
            self.assertTrue(channel["confidence"]["basis"])

    def test_report_files_include_json_markdown_and_csv(self) -> None:
        paths = write_report(self.report, self.root / "report")
        payload = json.loads(paths["json"].read_text(encoding="utf-8"))
        self.assertEqual(payload["summary"]["duplicate"], 1)
        markdown = paths["markdown"].read_text(encoding="utf-8")
        self.assertIn("八路回录接线分析报告", markdown)
        self.assertIn("SPK-H", markdown)
        csv_lines = paths["evidence_csv"].read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(csv_lines), 65)


class AmbiguousMappingTests(unittest.TestCase):
    def test_indistinguishable_sources_are_left_for_review(self) -> None:
        sample_rate = 2000
        rng = random.Random(41)
        base = [rng.uniform(-1.0, 1.0) for _ in range(2400)]
        references = [list(base) for _ in range(8)]
        recordings = []
        for delay in range(8):
            shift = 12 * delay
            recordings.append([0.0] * shift + base[: len(base) - shift])
        report = analyze(
            WavData(sample_rate, references, "test"),
            WavData(sample_rate, recordings, "test"),
            AnalysisConfig(max_lag_seconds=0.2),
        )
        self.assertEqual(report["summary"]["review"], 8)
        self.assertTrue(report["summary"]["manual_review_required"])
        self.assertEqual(report["mapping"], [])
        self.assertEqual(report["duplicate_groups"], [])


if __name__ == "__main__":
    unittest.main()
