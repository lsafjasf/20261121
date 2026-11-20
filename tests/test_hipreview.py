"""hipreview 核心行为测试：纯标准库 unittest，无需额外依赖。"""

import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hipreview.analyzer import (  # noqa: E402
    DELAY_TOLERANCE_MS, analyze_reel, detect_language, find_duplicate_segments,
    normalized_residual, xcorr_delay,
)
from hipreview.dataset import generate  # noqa: E402
from hipreview.report import run_audit  # noqa: E402
from hipreview.synth import SAMPLE_RATE, build_track  # noqa: E402


def shift(track: np.ndarray, ms: float) -> np.ndarray:
    shift_n = int(round(ms / 1000.0 * SAMPLE_RATE))
    out = np.zeros_like(track)
    if shift_n >= 0:
        out[shift_n:] = track[:-shift_n]
    else:
        out[:shift_n] = track[-shift_n:]
    return out


class SynthAndLanguageTests(unittest.TestCase):
    def test_language_roundtrip(self):
        for lang in ("CN", "EN", "JA", "FR", "DE", "ES"):
            with self.subTest(lang=lang):
                detected, _ = detect_language(build_track(6, lang, seed=2))
                self.assertEqual(detected, lang)

    def test_independent_tracks_not_duplicate(self):
        main = build_track(10, "CN", seed=5)
        hi = build_track(10, "CN",
                         word_pattern=[7, 2, 9, 0, 5, 3, 8, 1, 6, 4],
                         seed=6)
        self.assertGreater(normalized_residual(hi, main), 0.05)
        result = analyze_reel("OK", main, hi, "CN")
        self.assertEqual(result.issues, [])


class DelayTests(unittest.TestCase):
    def test_positive_and_negative_delay(self):
        main = build_track(8, "CN", seed=7)
        for ms in (-250.0, 320.0):
            with self.subTest(ms=ms):
                lag, _ = xcorr_delay(main, shift(main, ms))
                self.assertAlmostEqual(lag / SAMPLE_RATE * 1000.0,
                                       ms, delta=20.0)

    def test_delay_issue_threshold(self):
        main = build_track(8, "CN", seed=8)
        result = analyze_reel("D", main, shift(main, 320.0), "CN")
        cats = {i.category for i in result.issues}
        self.assertIn("DELAY", cats)

    def test_in_sync_no_delay_issue(self):
        main = build_track(8, "CN", seed=9)
        lag, _ = xcorr_delay(main, main)
        self.assertLess(abs(lag / SAMPLE_RATE * 1000.0),
                        DELAY_TOLERANCE_MS)


class DuplicateTests(unittest.TestCase):
    def test_exact_copy_flagged_dup_track(self):
        main = build_track(12, "CN", seed=10)
        result = analyze_reel("C", main, main.copy(), "CN")
        self.assertIn("DUP_TRACK", {i.category for i in result.issues})

    def test_in_segment_duplicate(self):
        hi = build_track(16, "CN",
                         word_pattern=[3, 7, 1, 8, 4, 0, 6, 2, 9, 5,
                                       3, 7, 4, 0, 6, 2], seed=11)
        hi[12 * SAMPLE_RATE:16 * SAMPLE_RATE] = \
            hi[4 * SAMPLE_RATE:8 * SAMPLE_RATE]
        hits = find_duplicate_segments(hi)
        self.assertTrue(hits)
        again = hits[0][1]
        self.assertAlmostEqual(again, 12.0, delta=2.0)


class LanguageMismatchTests(unittest.TestCase):
    def test_wrong_language_flagged(self):
        main = build_track(8, "CN", seed=12)
        hi = build_track(8, "EN", seed=13)
        result = analyze_reel("L", main, hi, "CN")
        lang_issues = [i for i in result.issues
                       if i.category == "LANG_MISMATCH"]
        self.assertEqual(len(lang_issues), 1)
        self.assertEqual(lang_issues[0].evidence["detected"], "EN")


class EndToEndTests(unittest.TestCase):
    def test_demo_pipeline_report_and_clips(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            audit_dir = os.path.join(tmp, "audit")
            generate(data_dir)
            playlist_path = os.path.join(data_dir, "playlist.json")
            payload = run_audit(playlist_path, audit_dir)

            with open(os.path.join(audit_dir, "report.json"),
                      encoding="utf-8") as fh:
                on_disk = json.load(fh)
            self.assertEqual(len(on_disk["results"]), 5)

            by_id = {r["reel_id"]: r for r in payload["results"]}
            self.assertEqual(by_id["R1"]["issues"], [])
            expected = {
                "R2": "DELAY",
                "R3": "DUP_TRACK",
                "R4": "DUP_SEGMENT",
                "R5": "LANG_MISMATCH",
            }
            for reel_id, category in expected.items():
                cats = [i["category"] for i in by_id[reel_id]["issues"]]
                self.assertIn(category, cats)
                issue = next(i for i in by_id[reel_id]["issues"]
                             if i["category"] == category)
                self.assertTrue(os.path.isfile(os.path.join(
                    audit_dir, issue["clip_hi"])))
                self.assertTrue(os.path.isfile(os.path.join(
                    audit_dir, issue["clip_main"])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
