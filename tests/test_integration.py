from __future__ import annotations

import json
import shutil
import tempfile
import unittest
import wave
from pathlib import Path

from audio_lab.codecs import CODEC_CONFIGS, verify_codecs
from audio_lab.dsp import estimate_delay
from audio_lab.fixtures import build_fixtures
from audio_lab.runner import run_lab
from audio_lab.wavio import read_mono_wav


HAS_FFMPEG = shutil.which("ffmpeg") is not None


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg is required for integration tests")
class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            verify_codecs()
        except RuntimeError as error:
            raise unittest.SkipTest(str(error)) from error
        cls.temp_dir = tempfile.TemporaryDirectory()
        root = Path(cls.temp_dir.name)
        cls.fixtures = root / "fixtures"
        cls.output = root / "output"
        build_fixtures(cls.fixtures)
        cls.result = run_lab(cls.fixtures, cls.output)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_three_codecs_and_all_cases_execute(self) -> None:
        self.assertEqual(len(self.result.cases), 2)
        for case in self.result.cases:
            self.assertEqual(
                [codec.config.key for codec in case.codecs],
                [config.key for config in CODEC_CONFIGS],
            )
        self.assertTrue(self.result.acceptance["passed"])

    def test_source_energy_and_codec_excess_are_distinguished(self) -> None:
        strong, weak = self.result.cases
        self.assertEqual(strong.expected, "codec-added")
        self.assertEqual(weak.expected, "source-prelude")
        for codec in strong.codecs:
            for onset in codec.metrics.onsets:
                self.assertLessEqual(onset.reference_pre_dbfs, -90.0)
                self.assertIsNotNone(onset.codec_added_dbfs)
                self.assertEqual(onset.classification, "codec-added")
        for codec in weak.codecs:
            for onset in codec.metrics.onsets:
                self.assertGreaterEqual(onset.reference_pre_dbfs, -70.0)
                self.assertNotEqual(onset.classification, "codec-added")
                if onset.source_dominance_db is not None:
                    self.assertGreaterEqual(onset.source_dominance_db, 6.0)

    def test_delay_calibration_has_confidence_and_zero_residual(self) -> None:
        for case in self.result.cases:
            for codec in case.codecs:
                self.assertGreater(codec.delay.confidence, 0.8)
                residual = estimate_delay(
                    case.reference.samples,
                    codec.aligned.samples,
                    case.reference.sample_rate,
                )
                self.assertLessEqual(abs(residual.delay_ms), 1.0)

    def test_aligned_and_ab_audio_have_matching_frames(self) -> None:
        sample_rate = self.result.sample_rate
        for case in self.result.cases:
            reference_frames = len(case.reference.samples)
            for codec in case.codecs:
                aligned = read_mono_wav(self.output / codec.aligned_path)
                self.assertEqual(aligned.sample_rate, sample_rate)
                self.assertEqual(len(aligned.samples), reference_frames)
                with wave.open(str(self.output / codec.ab_path), "rb") as handle:
                    self.assertEqual(handle.getnchannels(), 2)
                    self.assertEqual(handle.getframerate(), sample_rate)
                    self.assertEqual(handle.getnframes(), reference_frames)

    def test_report_and_machine_readable_outputs_exist(self) -> None:
        report = (self.output / "report.html").read_text(encoding="utf-8")
        self.assertEqual(report.count('class="ab-player"'), 6)
        self.assertEqual(report.count('class="wave-svg"'), 6)
        self.assertIn("编码新增", report)
        self.assertIn("源前奏占优", report)
        self.assertIn("下载 A/B WAV", report)
        summary = json.loads((self.output / "summary.json").read_text(encoding="utf-8"))
        self.assertTrue(summary["acceptance"]["passed"])
        rows = (self.output / "metrics.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 31)
        for case in self.result.cases:
            for codec in case.codecs:
                for relative in (
                    codec.encoded_path,
                    codec.aligned_path,
                    codec.ab_path,
                    codec.unaligned_path,
                ):
                    self.assertTrue((self.output / relative).exists(), relative)


if __name__ == "__main__":
    unittest.main()
