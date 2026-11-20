"""End-to-end and unit validation for the relisten package builder."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relisten.audio import read_wav
from relisten.builder import build_package
from relisten.cutting import rms
from relisten.merge import find_overlaps, merge_utterances
from relisten.model import BuildConfig, Utterance

from tests.fixtures import (
    EXPECTED_GROUPS,
    LABELS,
    SPEAKER_RATES,
    WEAK_TAIL_AMPLITUDE,
    WEAK_TAIL_START,
    build_meeting,
)


class MergeTests(unittest.TestCase):
    def test_nearby_short_utterances_merge_long_gap_stays_separate(self):
        config = BuildConfig()
        meeting = build_meeting(tempfile.mkdtemp())
        grouped = merge_utterances(meeting["utterances"], config)
        counts = {
            speaker_id: len(groups)
            for speaker_id, groups in grouped.items()
        }
        self.assertEqual(counts, EXPECTED_GROUPS)

        merged_a = grouped["A"][0]
        self.assertEqual(merged_a.indices, [0, 1])
        self.assertEqual(
            merged_a.intervals, [(1.0, 2.0), (2.6, 3.4)]
        )

        separate_d = grouped["D"][-1]
        self.assertEqual(separate_d.intervals, [(14.0, 14.4)])

    def test_overlap_evidence_keeps_cross_talk_source(self):
        config = BuildConfig()
        meeting = build_meeting(tempfile.mkdtemp())
        grouped = merge_utterances(meeting["utterances"], config)

        a_group_one = grouped["A"][0]
        overlaps = find_overlaps(a_group_one, meeting["utterances"])
        self.assertEqual(len(overlaps), 1)
        self.assertEqual(overlaps[0]["speaker_id"], "B")
        self.assertAlmostEqual(overlaps[0]["start"], 3.0)
        self.assertAlmostEqual(overlaps[0]["end"], 3.4)

        a_group_two = grouped["A"][1]
        sources = {
            item["speaker_id"]
            for item in find_overlaps(
                a_group_two, meeting["utterances"]
            )
        }
        self.assertEqual(sources, {"C"})


class CuttingTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.mkdtemp()
        self.meeting = build_meeting(self.workspace)
        self.output = Path(self.workspace) / "out"
        self.manifest = build_package(
            self.meeting["tracks"],
            self.meeting["utterances"],
            self.output,
            BuildConfig(),
        )

    def test_sample_exact_mapping_round_trip(self):
        for entry in self.manifest["clips"]:
            track = self.meeting["tracks"][entry["speaker_id"]]
            clip = read_wav(self.output / entry["output_path"])
            source_slice = track.samples[
                entry["source_start_sample"] : entry["source_end_sample"]
            ]
            self.assertEqual(
                len(clip.samples),
                entry["source_end_sample"]
                - entry["source_start_sample"],
            )
            self.assertEqual(
                clip.sample_rate,
                SPEAKER_RATES[entry["speaker_id"]],
            )
            self.assertEqual(clip.sample_rate, entry["sample_rate"])
            self.assertEqual(clip.samples, source_slice)
            self.assertEqual(
                entry["mapping"]["source_start_sample"],
                entry["source_start_sample"],
            )
            self.assertTrue(entry["mapping"]["sample_exact"])

    def test_no_labelled_speech_is_clipped(self):
        for entry in self.manifest["clips"]:
            rate = entry["sample_rate"]
            for interval in entry["merged_intervals"]:
                labelled_start = round(interval["start"] * rate)
                labelled_end = round(interval["end"] * rate)
                self.assertLessEqual(
                    entry["source_start_sample"], labelled_start
                )
                self.assertGreaterEqual(
                    entry["source_end_sample"], labelled_end
                )

    def test_weak_final_syllable_survives_without_truncation(self):
        a_clip = next(
            entry
            for entry in self.manifest["clips"]
            if entry["speaker_id"] == "A"
            and any(
                interval["end"] == 9.0
                for interval in entry["merged_intervals"]
            )
        )
        rate = a_clip["sample_rate"]
        track = self.meeting["tracks"]["A"]
        weak_start = round(WEAK_TAIL_START * rate)
        labelled_end = round(9.0 * rate)
        weak_energy = rms(
            track.samples, weak_start, labelled_end
        )
        self.assertGreater(weak_energy, 0.0)
        self.assertLess(weak_energy, WEAK_TAIL_AMPLITUDE * 0.8)

        clip = read_wav(self.output / a_clip["output_path"])
        clip_weak_start = weak_start - a_clip["source_start_sample"]
        clip_weak_end = labelled_end - a_clip["source_start_sample"]
        preserved = rms(
            clip.samples, clip_weak_start, clip_weak_end
        )
        self.assertAlmostEqual(preserved, weak_energy, places=12)
        self.assertGreaterEqual(
            a_clip["source_end_sample"], labelled_end
        )

    def test_cuts_land_in_silence_not_on_speech(self):
        for entry in self.manifest["clips"]:
            track = self.meeting["tracks"][entry["speaker_id"]]
            rate = entry["sample_rate"]
            guard = max(1, round(0.02 * rate))
            for boundary in (
                entry["source_start_sample"],
                entry["source_end_sample"],
            ):
                window = rms(
                    track.samples,
                    max(0, boundary - guard),
                    min(len(track.samples), boundary + guard),
                )
                self.assertEqual(window, 0.0)

    def test_tiny_context_falls_back_without_clipping_speech(self):
        tight = BuildConfig(
            context_pre=0.005,
            context_post=0.005,
            energy_search=0.3,
        )
        output = Path(self.workspace) / "tight"
        manifest = build_package(
            self.meeting["tracks"],
            self.meeting["utterances"],
            output,
            tight,
        )
        for entry in manifest["clips"]:
            rate = entry["sample_rate"]
            for interval in entry["merged_intervals"]:
                self.assertLessEqual(
                    entry["source_start_sample"],
                    round(interval["start"] * rate),
                )
                self.assertGreaterEqual(
                    entry["source_end_sample"],
                    round(interval["end"] * rate),
                )


class OutputTests(unittest.TestCase):
    def test_independent_per_speaker_outputs_and_manifest(self):
        workspace = tempfile.mkdtemp()
        meeting = build_meeting(workspace)
        output = Path(workspace) / "out"
        manifest = build_package(
            meeting["tracks"], meeting["utterances"], output
        )

        speakers = set(manifest["speakers"])
        self.assertEqual(speakers, {"A", "B", "C", "D"})
        for speaker_id in speakers:
            speaker_dir = output / speaker_id
            wav_files = sorted(speaker_dir.glob("*.wav"))
            self.assertEqual(
                len(wav_files), EXPECTED_GROUPS[speaker_id]
            )
            for entry in manifest["clips"]:
                if entry["speaker_id"] == speaker_id:
                    self.assertTrue(
                        (output / entry["output_path"]).exists()
                    )

        for entry in manifest["clips"]:
            path = Path(entry["source_path"])
            self.assertEqual(
                path.name, f"speaker_{entry['speaker_id']}.wav"
            )

        on_disk = json.loads(
            (output / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            len(on_disk["clips"]), len(manifest["clips"])
        )

    def test_two_sample_rate_versions_preserved(self):
        workspace = tempfile.mkdtemp()
        meeting = build_meeting(workspace)
        output = Path(workspace) / "out"
        manifest = build_package(
            meeting["tracks"], meeting["utterances"], output
        )
        rates = {
            entry["speaker_id"]: entry["sample_rate"]
            for entry in manifest["clips"]
        }
        for speaker_id, rate in SPEAKER_RATES.items():
            self.assertEqual(rates[speaker_id], rate)

        rates_present = {entry["sample_rate"] for entry in manifest["clips"]}
        self.assertEqual(rates_present, {16000, 24000})

    def test_annotation_sanity(self):
        for speaker, start, end, _ in LABELS:
            self.assertIn(speaker, SPEAKER_RATES)
            self.assertLess(start, end)


if __name__ == "__main__":
    unittest.main()
