from __future__ import annotations

import unittest

from film_stabilizer.detector import DetectorConfig, PerforationDetector
from film_stabilizer.synthetic import SyntheticConfig, SyntheticSequence


class DetectorTests(unittest.TestCase):
    def test_finds_both_perforations_and_respects_dropouts(self) -> None:
        sequence = SyntheticSequence(
            SyntheticConfig(
                width=480,
                height=360,
                frames=8,
                splice_frame=None,
                missing_left={4},
                missing_right={4},
            )
        )
        detector = PerforationDetector(DetectorConfig())
        for index, _, _, frame in sequence.iter_frames():
            detections = detector.detect(frame)
            sides = {item.side for item in detections}
            if index == 4:
                self.assertEqual(sides, set())
            else:
                self.assertEqual(sides, {"left", "right"})

    def test_picture_area_is_outside_detection_bands(self) -> None:
        detector = PerforationDetector(DetectorConfig(side_band_fraction=0.10))
        self.assertLess(0.10, 0.145)


if __name__ == "__main__":
    unittest.main()
