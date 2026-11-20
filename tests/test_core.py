from __future__ import annotations

import random
import unittest

from audio_lab.dsp import align_signal, estimate_delay
from audio_lab.metrics import MetricSettings, measure_onsets


def _transient_signal(length: int, delay: int = 0) -> list[float]:
    rng = random.Random(42)
    samples = [0.0] * length
    for onset in (0.20, 0.55, 0.80):
        start = round(onset * 48_000) + delay
        for offset in range(round(0.030 * 48_000)):
            if start + offset >= length:
                break
            envelope = 0.9 * (0.72 ** (offset / 48.0))
            samples[start + offset] = envelope * (
                0.7 * rng.uniform(-1.0, 1.0)
                + 0.3 * ((offset // 4) % 2 * 2 - 1)
            )
    return samples


class DelayTests(unittest.TestCase):
    def test_estimates_and_removes_positive_delay(self) -> None:
        reference = _transient_signal(48_000)
        delayed = [0.0] * 137 + reference[:-137]
        estimate = estimate_delay(reference, delayed, 48_000)
        self.assertEqual(estimate.delay_samples, 137)
        self.assertGreater(estimate.confidence, 0.90)
        aligned = align_signal(delayed, len(reference), estimate.delay_samples)
        self.assertEqual(aligned, reference)

    def test_align_handles_negative_delay(self) -> None:
        reference = [0.0, 0.25, -0.5, 0.75, 0.0]
        early = [-0.5, 0.75, 0.0]
        aligned = align_signal(early, len(reference), -1)
        self.assertEqual(aligned, [0.0, -0.5, 0.75, 0.0, 0.0])

    def test_estimates_negative_delay_sign(self) -> None:
        reference = _transient_signal(48_000)
        early = reference[137:] + [0.0] * 137
        estimate = estimate_delay(reference, early, 48_000)
        self.assertEqual(estimate.delay_samples, -137)
        self.assertEqual(
            align_signal(early, len(reference), estimate.delay_samples), reference
        )

    def test_alignment_is_exactly_reference_length(self) -> None:
        aligned = align_signal([1.0, 2.0, 3.0], 6, 1)
        self.assertEqual(len(aligned), 6)
        self.assertEqual(aligned, [2.0, 3.0, 0.0, 0.0, 0.0, 0.0])


class LeakageSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = MetricSettings()
        self.sample_rate = 1000
        self.onsets = [0.20]

    def test_silent_reference_attributes_energy_to_codec(self) -> None:
        reference = [0.0] * 500
        decoded = [0.0] * 500
        for index in range(170, 198):
            decoded[index] = 0.01
        summary = measure_onsets(
            reference, decoded, self.sample_rate, self.onsets, self.settings
        )
        onset = summary.onsets[0]
        self.assertEqual(onset.classification, "codec-added")
        self.assertAlmostEqual(onset.codec_added_dbfs, -40.0, places=2)
        self.assertIsNone(onset.source_dominance_db)

    def test_real_prelude_dominates_small_codec_excess(self) -> None:
        reference = [0.0] * 500
        decoded = [0.0] * 500
        for index in range(170, 198):
            reference[index] = 0.02
            decoded[index] = 0.0205
        summary = measure_onsets(
            reference, decoded, self.sample_rate, self.onsets, self.settings
        )
        onset = summary.onsets[0]
        self.assertEqual(onset.classification, "source-prelude")
        self.assertGreater(onset.source_dominance_db or 0.0, 10.0)
        self.assertLess(onset.codec_added_dbfs or -999.0, -35.0)
        self.assertEqual(summary.source_prelude_count, 1)
        self.assertEqual(summary.codec_added_count, 0)

    def test_visible_source_without_excess_is_source_prelude(self) -> None:
        reference = [0.0] * 500
        decoded = [0.0] * 500
        for index in range(170, 198):
            reference[index] = 0.02
            decoded[index] = 0.0202
        summary = measure_onsets(
            reference, decoded, self.sample_rate, self.onsets, self.settings
        )
        self.assertEqual(summary.onsets[0].classification, "source-prelude")


if __name__ == "__main__":
    unittest.main()
