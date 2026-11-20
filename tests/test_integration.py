from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from roomtone_handoff.cli import main
from roomtone_handoff.errors import ManifestError
from roomtone_handoff.model import load_config
from roomtone_handoff.service import render_all
from roomtone_handoff.verify import verify_manifest

from .fixtures import create_fixture_set


def rules_in_order() -> list[dict[str, object]]:
    return [
        {
            "id": "left-bed",
            "channel": "left",
            "start_frame": 3,
            "end_frame": 39,
            "priority": 1,
            "noise_offset_frames": 2,
        },
        {
            "id": "left-fix",
            "channel": "left",
            "start_frame": 17,
            "end_frame": 53,
            "priority": 9,
            "noise_offset_frames": 0,
        },
        {
            "id": "right-bed",
            "channel": "right",
            "start_frame": 8,
            "end_frame": 44,
            "priority": 2,
            "noise_offset_frames": 5,
        },
        {
            "id": "right-fix",
            "channel": "right",
            "start_frame": 31,
            "end_frame": 62,
            "priority": 8,
            "noise_offset_frames": 7,
        },
        {
            "id": "right-tie",
            "channel": "right",
            "start_frame": 37,
            "end_frame": 48,
            "priority": 8,
            "noise_offset_frames": 3,
        },
    ]


class IntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source_path, self.room_tone_path = create_fixture_set(self.root)

    def write_config(
        self, name: str, rules: list[dict[str, object]], output_dir: str
    ) -> Path:
        payload = {
            "schema": "roomtone-handoff.config.v1",
            "source": self.source_path.name,
            "room_tone": self.room_tone_path.name,
            "output_dir": output_dir,
            "versions": [
                {
                    "name": "handoff-a",
                    "regions": rules,
                },
                {
                    "name": "handoff-b",
                    "regions": [
                        rule
                        for rule in rules
                        if rule["id"] in {"left-fix", "right-bed"}
                    ],
                },
            ],
        }
        path = self.root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_render_manifests_and_independent_verification(self) -> None:
        config_path = self.write_config("config.json", rules_in_order(), "renders")
        config = load_config(config_path)
        original_bytes = self.source_path.read_bytes()
        outputs = render_all(config)
        self.assertEqual(len(outputs), 2)
        for output in outputs:
            self.assertTrue(output.wav_path.is_file())
            self.assertTrue(output.manifest_path.is_file())
            self.assertTrue(output.csv_path.is_file())
            report = verify_manifest(output.manifest_path)
            self.assertEqual(report.version, output.version)
            self.assertGreater(report.outside_pcm_bytes_checked, 0)
            document = json.loads(output.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                len(output.csv_path.read_text(encoding="utf-8").splitlines()),
                len(document["effective_replacements"]) + 1,
            )
            self.assertGreater(
                len(document["effective_replacements"]),
                len(document["rules"]) - 1,
            )
        self.assertEqual(self.source_path.read_bytes(), original_bytes)

    def test_manifest_and_output_do_not_depend_on_config_entry_order(self) -> None:
        first_config = load_config(
            self.write_config("first.json", rules_in_order(), "first")
        )
        reversed_rules = list(reversed(rules_in_order()))
        second_config = load_config(
            self.write_config("second.json", reversed_rules, "second")
        )
        first = {item.version: item for item in render_all(first_config)}
        second = {item.version: item for item in render_all(second_config)}
        for name in first:
            self.assertEqual(first[name].wav_path.read_bytes(), second[name].wav_path.read_bytes())
            self.assertEqual(
                first[name].manifest_path.read_text(encoding="utf-8"),
                second[name].manifest_path.read_text(encoding="utf-8"),
            )
            self.assertEqual(
                first[name].csv_path.read_text(encoding="utf-8"),
                second[name].csv_path.read_text(encoding="utf-8"),
            )

    def test_tampered_output_is_rejected(self) -> None:
        config = load_config(
            self.write_config("tamper.json", rules_in_order(), "tamper-out")
        )
        output = render_all(config)[0]
        payload = bytearray(output.wav_path.read_bytes())
        payload[-1] ^= 0x01
        output.wav_path.write_bytes(payload)
        with self.assertRaisesRegex(ManifestError, "hash"):
            verify_manifest(output.manifest_path)

    def test_cli_render_and_verify(self) -> None:
        config_path = self.write_config("cli.json", rules_in_order(), "cli-out")
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(["render", str(config_path), "--json"])
        self.assertEqual(code, 0, stderr.getvalue())
        summary = json.loads(stdout.getvalue())
        self.assertEqual(len(summary["versions"]), 2)
        manifests = [item["manifest"] for item in summary["versions"]]
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(["verify", *manifests])
        self.assertEqual(code, 0, stderr.getvalue())
        self.assertIn("outside PCM bytes checked", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
