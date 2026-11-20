from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from lcdlab.calibration import calibrate, save_calibration
from lcdlab.config import LabConfig
from lcdlab.metrics import compute_metrics, save_metrics
from lcdlab.model import ResponseParams, simulate_response
from lcdlab.restoration import restore_sequence
from lcdlab.synth import build_calibration_clip, build_demo_clip


class ResponseModelTests(unittest.TestCase):
    def test_noiseless_inverse_recovers_source(self) -> None:
        rng = np.random.default_rng(7)
        source = rng.uniform(0.0, 1.0, size=(20, 12, 16, 3)).astype(np.float32)
        params = ResponseParams(
            alpha_rise=np.array([0.55, 0.31, 0.14]),
            alpha_fall=np.array([0.69, 0.42, 0.22]),
        )
        captured = simulate_response(
            source, params.alpha_rise, params.alpha_fall
        )
        restored = restore_sequence(captured, params, LabConfig())
        np.testing.assert_allclose(restored.frames[1:], source[1:], atol=2e-5)

    def test_calibration_recovers_known_time_constants(self) -> None:
        config = LabConfig(
            width=96,
            height=72,
            noise_sigma=0.0,
            output_dir=Path("/tmp/lcdlab-test"),
        )
        clip = build_calibration_clip(config)
        result = calibrate(clip, config)
        np.testing.assert_allclose(result.params.tau_rise, config.tau_rise, atol=0.08)
        np.testing.assert_allclose(result.params.tau_fall, config.tau_fall, atol=0.08)

    def test_full_report_measurements_are_finite(self) -> None:
        config = LabConfig(
            width=96,
            height=72,
            noise_sigma=0.002,
            output_dir=Path("/tmp/lcdlab-test"),
        )
        calibration = calibrate(build_calibration_clip(config), config)
        clip = build_demo_clip(config, frame_count=48)
        restoration = restore_sequence(clip.captured, calibration.params, config)
        metrics = compute_metrics(clip, restoration, config)
        self.assertTrue(np.isfinite(metrics["ghost"]["restored"]["rms_8bit"]))
        self.assertGreater(metrics["ghost"]["improvement_db"], 10.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            save_calibration(calibration, config, out_dir)
            save_metrics(metrics, out_dir)
            self.assertTrue((out_dir / "calibration.json").exists())
            self.assertTrue((out_dir / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
