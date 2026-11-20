from __future__ import annotations

import unittest

import numpy as np

from film_stabilizer.detector import PerforationDetector
from film_stabilizer.stabilizer import StabilizerConfig, analyze_records
from film_stabilizer.synthetic import SyntheticConfig, SyntheticSequence
from film_stabilizer.types import FrameRecord
from film_stabilizer.verify import verify_against_truth


def analyze(sequence: SyntheticSequence):
    detector = PerforationDetector()
    records = []
    for index, name, number, frame in sequence.iter_frames():
        records.append(
            FrameRecord(index, name, number, detector.detect(frame))
        )
    return analyze_records(sequence.info, records, StabilizerConfig())


class TrackingTests(unittest.TestCase):
    def test_tracks_static_gate_while_actor_moves(self) -> None:
        sequence = SyntheticSequence(
            SyntheticConfig(
                width=480,
                height=360,
                frames=32,
                splice_frame=None,
                gate_jitter_px=0.0,
                gate_rotation_deg=0.0,
            )
        )
        result = analyze(sequence)
        for record in result.records:
            tx, ty = record.matrix[:2, 2]
            self.assertLess(np.hypot(tx, ty), 0.45)
            self.assertLess(abs(record.matrix[1, 0]), 0.0025)
        actor = np.asarray([item.actor for item in sequence.truth])
        self.assertGreater(np.ptp(actor[:, 0]), 100.0)

    def test_actor_does_not_change_perforation_measurement(self) -> None:
        sequence = SyntheticSequence(
            SyntheticConfig(
                width=480,
                height=360,
                frames=16,
                splice_frame=None,
            )
        )
        detector = PerforationDetector()
        height, width = sequence.info.height, sequence.info.width
        x0, x1 = int(width * 0.145), int(width * 0.855)
        y0, y1 = int(height * 0.060), int(height * 0.940)
        for _, _, _, frame in sequence.iter_frames():
            with_actor = detector.detect(frame)
            scrambled = frame.copy()
            scrambled[y0:y1, x0:x1] = (17, 17, 17)
            without_actor = detector.detect(scrambled)
            self.assertEqual(len(with_actor), len(without_actor))
            for first, second in zip(with_actor, without_actor):
                self.assertEqual(first.side, second.side)
                self.assertAlmostEqual(first.x, second.x, places=5)
                self.assertAlmostEqual(first.y, second.y, places=5)

    def test_missing_holes_and_splice_jump(self) -> None:
        sequence = SyntheticSequence(
            SyntheticConfig(
                width=640,
                height=480,
                frames=40,
                splice_frame=24,
                missing_left={7, 8, 9, 15},
                missing_right={7, 8, 9, 16},
            )
        )
        result = analyze(sequence)
        statuses = [record.status for record in result.records]
        self.assertEqual(statuses[7], "interpolated")
        self.assertEqual(statuses[8], "interpolated")
        self.assertIn("measured_single", statuses)
        report = verify_against_truth(result, sequence.truth)
        self.assertTrue(report.passed, report)
        self.assertIn(24, result.splice_frames)
        self.assertLess(report.gate_alignment_rms_px, 1.0)
        self.assertGreater(report.actor_motion_range_px, 3.0 * report.gate_motion_range_px)


if __name__ == "__main__":
    unittest.main()
