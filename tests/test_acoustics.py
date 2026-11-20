from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from acoustics import (
    MONITOR_RATE,
    SourceBank,
    ensure_builtin_sources,
    export_bundle,
    monitor_seconds_to_source,
    process_selection,
    run_selftest,
    source_frame_to_monitor,
    spectrogram_image,
    time_expand,
    waveform_overview,
)


class AcousticCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.bank = SourceBank(Path(cls.temp_dir.name) / "data")
        cls.demo, cls.calibration = ensure_builtin_sources(cls.bank)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_builtin_sources_are_192k(self) -> None:
        self.assertEqual(self.demo.rate, 192_000)
        self.assertEqual(self.calibration.rate, 192_000)
        self.assertGreater(self.demo.frames, 500_000)

    def test_time_expansion_preserves_proportional_mapping(self) -> None:
        start = 12_000
        end = start + 96_000
        factor = 10.0
        result = process_selection(
            self.bank,
            self.demo,
            start,
            end,
            {"mode": "time_expansion", "factor": factor},
        )
        expected_seconds = (end - start) / self.demo.rate * factor
        self.assertAlmostEqual(result["monitor_duration"], expected_seconds, places=4)
        self.assertAlmostEqual(
            monitor_seconds_to_source(
                result["monitor_duration"], start, self.demo.rate, factor
            ),
            end,
            places=6,
        )
        self.assertAlmostEqual(
            source_frame_to_monitor(end, start, self.demo.rate, factor),
            result["monitor_duration"],
            places=6,
        )

    def test_short_clip_has_expected_duration(self) -> None:
        clip = np.zeros(1_920, dtype=np.float32)
        clip[400:500] = 0.8
        monitor = time_expand(clip, 192_000, 5.0)
        self.assertEqual(len(monitor), 2_400)
        self.assertEqual(MONITOR_RATE, 48_000)

    def test_selftest_passes_sideband_and_endpoint_checks(self) -> None:
        result = run_selftest(self.bank, self.calibration)
        self.assertTrue(result["passed"], result)
        self.assertEqual(len(result["checks"]), 7)
        self.assertTrue(all(check["pass"] for check in result["checks"]))

    def test_spectrogram_and_waveform_dimensions(self) -> None:
        image = spectrogram_image(
            self.bank, self.demo, 0, self.demo.frames, 10_000, 96_000, 320, 180
        )
        self.assertEqual(image["width"], 320)
        self.assertEqual(image["height"], 180)
        self.assertEqual(len(image["bytes"]), 320 * 180)
        waveform = waveform_overview(self.bank, self.demo, 0, self.demo.frames, 500)
        self.assertEqual(waveform["width"], 500)
        self.assertEqual(len(waveform["min"]), 500)
        self.assertEqual(len(waveform["max"]), 500)

    def test_export_bundle_contains_two_audio_files_and_mapping(self) -> None:
        start = 19_200
        end = start + 48_000
        rendered = process_selection(
            self.bank,
            self.demo,
            start,
            end,
            {
                "mode": "heterodyne",
                "center_hz": 70_000,
                "bandwidth_hz": 16_000,
                "sideband": "USB",
            },
        )
        bundle = export_bundle(
            self.bank,
            self.demo,
            start,
            end,
            rendered,
            Path(self.temp_dir.name) / "exports",
        )
        manifest = bundle["manifest"]
        raw_info = sf.info(str(bundle["raw_path"]))
        monitor_info = sf.info(str(bundle["monitor_path"]))
        self.assertEqual(raw_info.frames, end - start)
        self.assertEqual(raw_info.samplerate, 192_000)
        self.assertEqual(monitor_info.frames, rendered["monitor_frames"])
        self.assertEqual(monitor_info.samplerate, 48_000)
        self.assertEqual(manifest["mapping"]["start"]["source_frame"], start)
        self.assertEqual(manifest["mapping"]["end_exclusive"]["source_frame"], end)
        self.assertTrue(bundle["mapping_path"].is_file())
        self.assertTrue(bundle["manifest_path"].is_file())
        csv_text = bundle["mapping_path"].read_text(encoding="utf-8")
        self.assertIn("source_frame,source_seconds,monitor_frame,monitor_seconds", csv_text)
        self.assertIn(f"\n{end},", csv_text)
        saved_manifest = json.loads(bundle["manifest_path"].read_text(encoding="utf-8"))
        self.assertEqual(saved_manifest["schema"], "echolab.mapping.v1")
        self.assertIn("sha256", saved_manifest["files"]["raw_clip"])


if __name__ == "__main__":
    unittest.main()

