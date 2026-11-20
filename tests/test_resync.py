from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from piano_resync import Recording, SyncConfig, resynchronize


def make_notes(
    count: int = 14,
    step: float = 0.27,
    start: float = 0.15,
) -> list[dict[str, object]]:
    return [
        {
            "note": 60 + index % 8,
            "start": start + index * step,
            "duration": 0.20,
            "velocity": 30 + (index * 17) % 98,
        }
        for index in range(count)
    ]


def make_taps(
    notes: list[dict[str, object]],
    rate: float,
    offset: float,
    jitter_ms: tuple[float, ...] = (0.8, -1.4, 1.2, -0.7),
) -> list[dict[str, object]]:
    return [
        {
            "time": rate * float(note["start"])
            + offset
            + jitter_ms[index % len(jitter_ms)] / 1000.0,
            "strength": float(note["velocity"]) / 127.0,
        }
        for index, note in enumerate(notes)
    ]


class ResyncTests(unittest.TestCase):
    def test_running_speed_and_pedal_rewrite(self) -> None:
        notes = make_notes()
        rate = 1.0017
        offset = 0.041
        payload = {
            "notes": notes,
            "pedals": [
                {"kind": "sustain", "start": 0.4, "end": 2.6},
                {"controller": 67, "start": 2.8, "end": 3.4, "value": 0.5},
            ],
            "taps": make_taps(notes, rate, offset),
        }
        result = resynchronize(payload, SyncConfig())
        self.assertAlmostEqual(result.transform.rate, rate, delta=0.0002)
        self.assertAlmostEqual(result.transform.offset, offset, delta=0.004)
        self.assertEqual(result.statistics["outlier_count"], 0)
        self.assertLess(result.statistics["median_abs_error_ms"], 3.0)
        mapped_pedal = result.pedals[0]
        self.assertAlmostEqual(
            mapped_pedal.start, result.transform.apply(0.4), places=9
        )
        self.assertAlmostEqual(
            mapped_pedal.end, result.transform.apply(2.6), places=9
        )
        self.assertEqual(mapped_pedal.kind, "sustain")
        self.assertEqual(result.pedals[1].kind, "soft")

    def test_edge_mismatch_drops_extra_transients(self) -> None:
        notes = make_notes(count=11, step=0.30)
        rate = 0.9988
        offset = 0.053
        taps = make_taps(notes, rate, offset, (11.0, -1.2, 0.9, -0.5))
        taps[0] = {"time": float(taps[0]["time"]) + 0.23, "strength": 0.15}
        taps.insert(0, {"time": 0.010, "strength": 0.2})
        taps.append(
            {"time": float(taps[-1]["time"]) + 0.26, "strength": 0.25}
        )
        result = resynchronize(
            {"notes": notes, "taps": taps},
            SyncConfig(),
        )
        self.assertAlmostEqual(result.transform.rate, rate, delta=0.0008)
        self.assertEqual(result.statistics["unmatched_note_count"], 1)
        self.assertEqual(result.statistics["unmatched_tap_count"], 3)
        self.assertEqual(result.unmatched_notes[0]["note"], notes[0]["note"])
        self.assertTrue(
            any("outside the mapped note span" in text for text in result.warnings)
        )

    def test_brief_dropouts_do_not_shift_later_anchors(self) -> None:
        notes = make_notes(count=18, step=0.235)
        rate = 1.0011
        offset = 0.026
        taps = make_taps(notes, rate, offset)
        for index in sorted((5, 6, 12), reverse=True):
            del taps[index]
        result = resynchronize({"notes": notes, "taps": taps}, SyncConfig())
        self.assertAlmostEqual(result.transform.rate, rate, delta=0.0002)
        self.assertEqual(result.statistics["unmatched_note_count"], 3)
        self.assertEqual(result.statistics["unmatched_tap_count"], 0)
        late_errors = [
            abs(float(row["error_ms"]))
            for row in result.comparisons
            if float(row["source_onset"]) > 3.0
        ]
        self.assertTrue(late_errors)
        self.assertLess(max(late_errors), 4.0)

    def test_short_input_warns_instead_of_crashing(self) -> None:
        payload = {
            "notes": [
                {"note": 60, "start": 0.10, "duration": 0.2, "velocity": 40},
                {"note": 64, "start": 0.40, "duration": 0.2, "velocity": 100},
            ],
            "taps": [
                {"time": 0.111, "strength": 0.3},
                {"time": 0.414, "strength": 0.8},
            ],
        }
        result = resynchronize(payload, SyncConfig())
        self.assertTrue(
            any("only 2 reliable anchors" in warning for warning in result.warnings)
        )

    def test_aliases_and_nested_input(self) -> None:
        recording = Recording.from_dict(
            {
                "midi": {
                    "notes": [
                        {
                            "key": 48,
                            "onset": 1.0,
                            "offset": 1.5,
                            "vel": 0.5,
                        }
                    ],
                    "pedals": [{"cc": 64, "on": 0.5, "off": 2.0}],
                },
                "audio": {
                    "onsets": [{"onset": 1.01, "amplitude": 0.51}]
                },
            }
        )
        self.assertEqual(recording.notes[0].note, 48)
        self.assertEqual(recording.notes[0].velocity, 64)
        self.assertEqual(recording.notes[0].end, 1.5)
        self.assertEqual(recording.pedals[0].kind, "sustain")
        self.assertEqual(recording.taps[0].strength, 0.51)

    def test_missing_taps_is_an_input_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one audio tap"):
            resynchronize(
                {
                    "notes": [
                        {"note": 60, "start": 0.0, "duration": 0.2}
                    ],
                    "taps": [],
                }
            )

    def test_cli_json_round_trip(self) -> None:
        notes = make_notes(count=8)
        payload = {
            "notes": notes,
            "pedals": [{"kind": "sustain", "start": 0.2, "end": 1.0}],
            "taps": make_taps(notes, 1.0008, 0.019),
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.json"
            output = Path(directory) / "output.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "piano_resync",
                    str(source),
                    "-o",
                    str(output),
                    "--pretty",
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved["timeline"], "audio")
            self.assertEqual(len(saved["notes"]), 8)
            self.assertEqual(saved["pedals"][0]["kind"], "sustain")
            self.assertIn("comparisons", saved["diagnostics"])


if __name__ == "__main__":
    unittest.main()
