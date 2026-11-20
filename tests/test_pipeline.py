"""End-to-end and component tests using the synthetic stereo demo."""

import os
import tempfile
import unittest

import numpy as np

from windtool.config import AnalysisConfig
from windtool.dsp import extract_features
from windtool.drum_phase import circular_concentration, circular_mean
from windtool.io import load_stereo
from windtool.make_demo import build_demo
from windtool.pipeline import analyze_file
from windtool.reduction import _frequency_gain_curve
from windtool.segmentation import build_segments, classify_segments


class DemoPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.wav = os.path.join(cls.tmp.name, "demo_stereo.wav")
        build_demo(cls.wav, seed=7)
        cls.audio, cls.sr = load_stereo(cls.wav)
        cls.cfg = AnalysisConfig()
        cls.features, *_ = extract_features(cls.audio, cls.sr, cls.cfg)
        cls.segments = classify_segments(
            build_segments(cls.features, cls.cfg), cls.features, cls.cfg
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_labels_cover_all_three_phenomena(self):
        labels = {s.label for s in self.segments}
        self.assertIn("drum", labels)
        self.assertIn("wind", labels)
        self.assertIn("bump", labels)

    def test_drum_segments_are_coherent_with_low_ild(self):
        for seg in self.segments:
            if seg.label != "drum":
                continue
            self.assertGreaterEqual(
                seg.metrics["peak_coherence_bass_p90"],
                self.cfg.peak_coherent_threshold,
            )
            self.assertLess(seg.metrics["ild_p80_db"], self.cfg.bump_ild_db)
            self.assertEqual(seg.attenuation_db, 0.0)

    def test_wind_segments_are_incoherent_and_sustained(self):
        wind = [s for s in self.segments if s.label == "wind"]
        self.assertTrue(wind)
        for seg in wind:
            self.assertLess(
                seg.metrics["coherence_wind_band"],
                self.cfg.coherent_threshold,
            )
            self.assertGreaterEqual(
                seg.metrics["low_band_fraction"], self.cfg.low_band_fraction
            )

    def test_bump_segments_have_extreme_ild(self):
        bumps = [s for s in self.segments if s.label == "bump"]
        self.assertTrue(bumps)
        for seg in bumps:
            self.assertGreaterEqual(
                seg.metrics["ild_p80_db"], self.cfg.bump_ild_db
            )

    def test_gain_curve_is_monotonic_and_band_limited(self):
        freqs = np.linspace(10.0, 2000.0, 500)
        gain = _frequency_gain_curve(freqs, 15.0, self.cfg)
        gain_db = 20.0 * np.log10(gain)
        self.assertAlmostEqual(gain_db[0], -15.0, places=1)
        # No discontinuities: adjacent bins never jump more than 0.5 dB.
        self.assertLess(np.max(np.abs(np.diff(gain_db))), 0.5)
        self.assertTrue(np.all(gain_db[freqs >= 500.0] == 0.0))

    def test_circular_stats(self):
        angles = np.full(20, 0.3)
        self.assertAlmostEqual(circular_mean(angles), 0.3)
        self.assertAlmostEqual(circular_concentration(angles), 1.0)
        uniform = np.linspace(-np.pi, np.pi, 100, endpoint=False)
        self.assertLess(circular_concentration(uniform), 0.1)

    def test_end_to_end_outputs_and_preservation(self):
        out_dir = os.path.join(self.tmp.name, "out")
        payload = analyze_file(self.wav, out_dir)
        for key in ("processed_wav", "overview_png", "report_json", "report_html"):
            self.assertTrue(os.path.exists(payload["outputs"][key]))
        processed, sr = load_stereo(payload["outputs"]["processed_wav"])
        self.assertEqual(processed.shape, self.audio.shape)
        # Drum hits must survive untouched: find a drum segment time range.
        drum = next(s for s in self.segments if s.label == "drum")
        a, b = int(drum.start_sec * sr), int(drum.end_sec * sr)
        self.assertTrue(
            np.allclose(processed[:, a:b], self.audio[:, a:b], atol=1e-9)
        )
        # Every rendered wind segment shows measurable low-band reduction.
        for row in payload["wind_reduction_check"]:
            if row["label"] == "wind":
                self.assertGreater(row["measured_low_band_reduction_db"], 5.0)
        # Drum phase check must accept the synthetic hits.
        self.assertTrue(payload["drum_phase_check"])
        for row in payload["drum_phase_check"]:
            self.assertEqual(row["verdict"], "stable shared-source phase")

    def test_rejects_non_stereo(self):
        mono_path = os.path.join(self.tmp.name, "mono.wav")
        import soundfile as sf

        sf.write(mono_path, np.zeros(1000), self.sr)
        with self.assertRaises(ValueError):
            load_stereo(mono_path)


if __name__ == "__main__":
    unittest.main()
