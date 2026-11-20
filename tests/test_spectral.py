from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from venue_wiring.spectral import fft_convolve, normalized_xcorr_peak
from venue_wiring.wavio import read_wav, write_wav


class SpectralTests(unittest.TestCase):
    def test_fft_convolution_matches_direct_convolution(self) -> None:
        signal = [0.0, 1.0, -0.5, 0.25, 0.0, 0.0]
        response = [1.0, 0.0, 0.5]
        expected = [0.0, 1.0, -0.5, 0.75, -0.25, 0.125, 0.0, 0.0]
        actual = fft_convolve(signal, response)
        self.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected):
            self.assertAlmostEqual(left, right, places=9)

    def test_correlation_finds_positive_and_negative_lag(self) -> None:
        reference = [0.0, 0.0, 1.0, -0.5, 0.25, 0.0, 0.0, 0.0]
        delayed = [0.0, 0.0, 0.0, 0.0, 1.0, -0.5, 0.25, 0.0, 0.0]
        positive = normalized_xcorr_peak(reference, delayed, 1000, min_overlap=0.2)
        self.assertGreater(positive.signed_score, 0.95)
        self.assertEqual(positive.lag_samples, 2)
        self.assertEqual(positive.polarity, 1)

        inverted = [-value for value in delayed]
        negative = normalized_xcorr_peak(reference, inverted, 1000, min_overlap=0.2)
        self.assertLess(negative.signed_score, -0.95)
        self.assertEqual(negative.polarity, -1)

    def test_correlation_finds_early_recording_start(self) -> None:
        reference = [0.0, 0.0, 0.0, 1.0, -0.5, 0.25, 0.0, 0.0]
        early = [1.0, -0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0]
        peak = normalized_xcorr_peak(reference, early, 1000, min_overlap=0.2)
        self.assertLess(peak.lag_samples, 0)
        self.assertGreater(peak.signed_score, 0.90)

    def test_wav_pcm16_and_float32_round_trip(self) -> None:
        channels = [[0.0, 0.5, -0.5], [0.25, -0.25, 0.0]]
        with tempfile.TemporaryDirectory() as directory:
            for subtype in ("pcm16", "float32"):
                path = Path(directory) / f"{subtype}.wav"
                write_wav(path, channels, 48000, subtype=subtype)
                loaded = read_wav(path)
                self.assertEqual(loaded.sample_rate, 48000)
                self.assertEqual(loaded.channel_count, 2)
                for expected_channel, actual_channel in zip(channels, loaded.channels):
                    for expected, actual in zip(expected_channel, actual_channel):
                        self.assertAlmostEqual(expected, actual, places=4)


if __name__ == "__main__":
    unittest.main()
