"""Tests for pilot tracking, bounded warp estimation and banded resampling."""

import unittest

import numpy as np

from wow_flutter.pilot import track_pilot
from wow_flutter.warp import estimate_warp, _extrapolate_bounded
from wow_flutter.resample import banded_resample, _sinc_table, _interpolate
from wow_flutter.synthesis import generate_tape
from wow_flutter.measure import note_frequency_track, trim_edges, vibrato_metrics


def make_warped_tone(fs=8000.0, duration=3.0, f0=1000.0, note=440.0):
    n = int(fs * duration)
    idx = np.arange(n)
    t = idx / fs
    speed = 0.005 * np.sin(2 * np.pi * 2 * t)
    position = idx + np.cumsum(speed)
    table = _sinc_table(16, 64)
    pilot = _interpolate(np.sin(2 * np.pi * f0 * t), position, table)
    audio = _interpolate(np.sin(2 * np.pi * note * t), position, table)
    return t, speed, position, pilot, audio


class PilotTrackingTests(unittest.TestCase):
    def test_tracks_modulated_frequency(self):
        fs = 8000.0
        t, speed, _, pilot, _ = make_warped_tone(fs=fs)
        track = track_pilot(pilot, fs, 1000.0)
        true_if = 1000.0 * (1.0 + np.interp(track.frame_times, t, speed))
        mid = slice(20, -20)
        corr = np.corrcoef(track.freq[mid], true_if[mid])[0, 1]
        self.assertGreater(corr, 0.95)
        error = np.sqrt(np.mean((track.freq[mid] - true_if[mid]) ** 2))
        self.assertLess(error, 1.0)

    def test_detects_dropout(self):
        tape = generate_tape()
        track = track_pilot(tape.pilot, tape.sample_rate, 1000.0)
        detected = ~track.reliable
        self.assertGreater(detected.mean(), 0.05)
        self.assertTrue(np.all((track.confidence >= 0) & (track.confidence <= 1)))

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            track_pilot(np.zeros(100), 8000.0, 5000.0)
        with self.assertRaises(ValueError):
            track_pilot(np.zeros((2, 2)), 8000.0, 1000.0)


class WarpEstimationTests(unittest.TestCase):
    def test_extrapolation_is_bounded(self):
        values = np.array([0.01, 0.011, 0.009, 0.012, 0.01])
        mask = np.ones(5, dtype=bool)
        mask[2] = False
        filled, extrapolated = _extrapolate_bounded(values, mask, 0.04)
        self.assertTrue(extrapolated[2])
        self.assertTrue(np.all(np.abs(filled) <= 0.04))

    def test_all_missing_returns_zero(self):
        values = np.ones(6) * 0.03
        mask = np.zeros(6, dtype=bool)
        filled, extrapolated = _extrapolate_bounded(values, mask, 0.04)
        np.testing.assert_allclose(filled, 0.0)
        self.assertTrue(np.all(extrapolated))

    def test_long_gap_extrapolation_stays_bounded(self):
        values = np.full(50, 0.02)
        mask = np.ones(50, dtype=bool)
        mask[10:40] = False
        filled, extrapolated = _extrapolate_bounded(values, mask, 0.04)
        self.assertTrue(np.all(extrapolated[10:40]))
        self.assertTrue(np.all(np.abs(filled) <= 0.04))
        # Interior gap must be bridged (finite, non-zero, smooth at edges).
        self.assertTrue(np.all(np.isfinite(filled)))
        np.testing.assert_allclose(filled[9], filled[10], atol=0.01)

    def test_map_monotonic_and_covered(self):
        tape = generate_tape()
        track = track_pilot(tape.pilot, tape.sample_rate, 1000.0)
        warp = estimate_warp(track, tape.sample_rate)
        self.assertTrue(np.all(np.diff(warp.input_map) > 0))
        self.assertTrue(np.all(np.isfinite(warp.input_map)))
        self.assertTrue(np.all(np.abs(warp.speed_error) <= 0.04 + 1e-12))


class ResamplingTests(unittest.TestCase):
    def test_identity_map_preserves_tone(self):
        fs = 8000.0
        t, _, _, _, audio = make_warped_tone(fs=fs)
        identity = np.arange(len(audio), dtype=float)
        out = banded_resample(audio, identity, sample_rate=fs)
        self.assertEqual(len(out), len(audio))
        # Interior samples stay close to the input under an identity map.
        np.testing.assert_allclose(out[500:-500], audio[500:-500], atol=0.02)

    def test_constant_speed_error_restores_pitch(self):
        fs = 8000.0
        n = int(fs * 2.0)
        idx = np.arange(n)
        t = idx / fs
        r = 0.01
        recorded = np.sin(2 * np.pi * 440.0 * (1 + r) * t)
        input_map = idx / (1.0 + r)
        out = banded_resample(recorded, input_map, sample_rate=fs)
        tt, ff = note_frequency_track(out, fs, 440.0)
        ff = trim_edges(tt, ff)
        self.assertAlmostEqual(float(np.mean(ff)), 440.0, delta=0.5)


class EndToEndTests(unittest.TestCase):
    def test_stable_variance_reduced_and_vibrato_preserved(self):
        tape = generate_tape()
        fs = tape.sample_rate
        track = track_pilot(tape.pilot, fs, 1000.0)
        warp = estimate_warp(track, fs)
        restored = banded_resample(tape.program, warp.input_map, sample_rate=fs)

        def stable_std(sig):
            tt, ff = note_frequency_track(sig, fs, 392.0)
            return float(np.std(trim_edges(tt, ff)))

        warped_std = stable_std(tape.program)
        restored_std = stable_std(restored)
        self.assertLess(restored_std, 0.7 * warped_std)

        ct, clean_f = note_frequency_track(tape.clean_program, fs, 523.25)
        rt, rest_f = note_frequency_track(restored, fs, 523.25)
        clean = vibrato_metrics(trim_edges(ct, clean_f), 523.25)
        restored_metrics = vibrato_metrics(trim_edges(rt, rest_f), 523.25)
        self.assertAlmostEqual(
            restored_metrics["vibrato_rate_hz"],
            clean["vibrato_rate_hz"],
            delta=0.3,
        )
        self.assertGreater(restored_metrics["vibrato_depth_cents"], 60.0)


if __name__ == "__main__":
    unittest.main()
