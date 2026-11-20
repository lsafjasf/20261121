from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from color_audit.audit import AuditError, inspect_media, verify
from color_audit.ffmpeg import discover_tools
from color_audit.fixtures import generate_fixtures
from color_audit.report import write_artifacts


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tools = discover_tools()
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.fixture_dir = cls.root / "fixtures"
        cls.manifest_data = generate_fixtures(cls.tools, cls.fixture_dir)
        cls.mistagged = cls.fixture_dir / cls.manifest_data["assets"]["mistagged_container"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_layers_are_independent_and_conflicts_are_reported(self) -> None:
        inspection = inspect_media(self.tools, self.mistagged)
        self.assertEqual(inspection["container"].values["primaries"].name, "bt2020")
        self.assertEqual(inspection["codec"].values["primaries"].name, "bt709")
        self.assertEqual(inspection["container"].values["range"].name, "full")
        self.assertEqual(inspection["codec"].values["range"].name, "limited")
        self.assertEqual(len(inspection["conflicts"]), 4)

    def test_verified_codec_interpretation_passes(self) -> None:
        output = self.root / "correct"
        audit = verify(
            self.tools,
            self.mistagged,
            output,
            explicit_decisions={},
            assume_layer="codec",
            target=None,
            manifest_path=self.fixture_dir / "manifest.json",
            tolerance=2,
            interactive=False,
        )
        self.assertEqual(audit["result"]["status"], "pass")
        self.assertEqual(len(audit["samples"]), 17)
        self.assertEqual(audit["samples"][0]["output_yuv"], {"y": 16, "u": 128, "v": 128})
        red = next(row for row in audit["samples"] if row["id"] == "red")
        self.assertEqual(red["output_yuv"], {"y": 63, "u": 102, "v": 240})
        self.assertIn("zscale=primariesin=1", audit["conversion"]["filter_graph"])
        self.assertIn("-vf", audit["conversion"]["ffmpeg_command"])
        artifacts = write_artifacts(audit, output)
        report = artifacts["report_md"].read_text(encoding="utf-8")
        self.assertIn("**Result:** `PASS`", report)
        self.assertIn("output_samples_match_manifest", report)
        with artifacts["samples_csv"].open(encoding="utf-8") as handle:
            self.assertIn("gray_16,grayscale,35,90", handle.read())
        saved = json.loads(artifacts["audit_json"].read_text(encoding="utf-8"))
        self.assertEqual(saved["result"]["status"], "pass")

    def test_wrong_container_interpretation_fails(self) -> None:
        output = self.root / "wrong"
        audit = verify(
            self.tools,
            self.mistagged,
            output,
            explicit_decisions={},
            assume_layer="container",
            target=None,
            manifest_path=self.fixture_dir / "manifest.json",
            tolerance=2,
            interactive=False,
        )
        self.assertEqual(audit["result"]["status"], "fail")
        self.assertIn("output_samples_match_manifest", audit["result"]["failures"])
        self.assertIn("black_level", audit["result"]["failures"])
        self.assertIn("white_level", audit["result"]["failures"])
        red = next(row for row in audit["samples"] if row["id"] == "red")
        self.assertNotEqual(red["output_yuv"], red["expected_yuv"])
        self.assertGreater(max(abs(value) for value in red["output_delta"].values()), 2)

    def test_unresolved_conflict_is_not_silently_accepted(self) -> None:
        with self.assertRaises(AuditError):
            verify(
                self.tools,
                self.mistagged,
                self.root / "unresolved",
                explicit_decisions={},
                assume_layer=None,
                target=None,
                manifest_path=None,
                tolerance=2,
                interactive=False,
            )


if __name__ == "__main__":
    unittest.main()
