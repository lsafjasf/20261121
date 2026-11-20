from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

from miccal.ambiguity import audit_scene, catalog_layouts
from miccal.beam import compute_beam_pattern, write_pattern_svg
from miccal.config import Scene
from miccal.demo import demo_scene, demo_scene_dict
from miccal.dsp import fft, ifft
from miccal.estimate import analyze_recordings
from miccal.simulate import simulate_scene


class DemoPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.scene = demo_scene()
        cls.recordings = Path(cls.temp.name) / "recordings"
        cls.truth = simulate_scene(cls.scene, cls.recordings)
        cls.result = analyze_recordings(cls.scene, cls.recordings)
        cls.pattern = compute_beam_pattern(cls.scene, cls.result)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_geometry_selects_direct_through_strong_reflection(self) -> None:
        self.assertEqual(self.result.best_map, (0, 1, 3, 2))
        self.assertFalse(self.result.map_ambiguous)
        for estimate in self.result.source_estimates:
            chosen = estimate.chosen
            for channel, peaks in enumerate(estimate.candidates):
                later = [peak for peak in peaks if peak.delay_s > chosen[channel].delay_s]
                self.assertLess(chosen[channel].delay_s, min(p.delay_s for p in later))
        strong = [
            finding
            for finding in self.result.findings
            if finding["code"] == "strong_reflection"
        ]
        self.assertTrue(strong)

    def test_estimates_gain_polarity_and_position_residual(self) -> None:
        self.assertLess(self.result.polarity[2], 0.0)
        self.assertAlmostEqual(self.result.gain_error_db[1], 1.20, delta=0.25)
        self.assertAlmostEqual(self.result.gain_error_db[2], -1.30, delta=0.25)
        self.assertLess(self.result.delay_residual_mm[2], -8.0)
        self.assertLess(abs(self.result.delay_residual_mm[1]), 1.0)

    def test_findings_report_mapping_polarity_and_reflection(self) -> None:
        codes = {finding["code"] for finding in self.result.findings}
        self.assertIn("channel_map", codes)
        self.assertIn("polarity", codes)
        self.assertIn("position_delay", codes)
        self.assertIn("strong_reflection", codes)

    def test_beam_returns_to_source_after_calibration(self) -> None:
        self.assertAlmostEqual(self.pattern.after_peak_deg, 0.0, delta=2.0)
        center = min(
            range(len(self.pattern.angles_deg)),
            key=lambda index: abs(self.pattern.angles_deg[index]),
        )
        self.assertGreater(self.pattern.after_db[center], -0.5)
        self.assertNotEqual(self.pattern.before_peak_deg, 0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "beam.svg"
            write_pattern_svg(self.pattern, path, "test", "test")
            self.assertIn("<svg", path.read_text(encoding="utf-8"))

    def test_current_layout_is_unique_at_two_sources(self) -> None:
        audit = audit_scene(
            self.scene, result_map=self.result.best_map, name=self.scene.name
        )
        self.assertFalse(audit.mapping_ambiguous)
        self.assertEqual(len(audit.equivalent_orders), 1)


class CleanSceneTest(unittest.TestCase):
    def test_clean_installation_passes(self) -> None:
        data = demo_scene_dict()
        data["name"] = "clean"
        data["installation"] = {
            "channel_order": [0, 1, 2, 3],
            "gain": [1.0, 1.0, 1.0, 1.0],
            "polarity": [1.0, 1.0, 1.0, 1.0],
        }
        scene = Scene.from_dict(data)
        with tempfile.TemporaryDirectory() as directory:
            recordings = Path(directory)
            simulate_scene(scene, recordings)
            result = analyze_recordings(scene, recordings)
        self.assertEqual(result.best_map, (0, 1, 2, 3))
        self.assertFalse(result.map_ambiguous)
        self.assertEqual(result.status, "ok")
        self.assertTrue(all(polarity > 0 for polarity in result.polarity))
        self.assertTrue(all(abs(value) < 0.25 for value in result.gain_error_db[1:]))
        self.assertTrue(
            all(abs(value) < 2.0 for value in result.delay_residual_mm[1:])
        )


class AmbiguityCatalogTest(unittest.TestCase):
    def test_catalog_reports_symmetric_blind_spots(self) -> None:
        items = {item.name: item for item in catalog_layouts()}
        self.assertTrue(items["square-centered-source"].mapping_ambiguous)
        self.assertGreater(
            len(items["square-centered-source"].equivalent_orders), 1
        )
        self.assertFalse(items["asymmetric-two-source"].mapping_ambiguous)
        self.assertTrue(items["collinear-broadside"].mapping_ambiguous)


class DspTest(unittest.TestCase):
    def test_fft_round_trip(self) -> None:
        values = [complex(index * 0.25, -index * 0.1) for index in range(64)]
        recovered = ifft(fft(values))
        for expected, actual in zip(values, recovered):
            self.assertAlmostEqual(expected.real, actual.real, places=9)
            self.assertAlmostEqual(expected.imag, actual.imag, places=9)


if __name__ == "__main__":
    unittest.main()
