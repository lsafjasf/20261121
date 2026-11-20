"""End-to-end and unit tests for the offline spatial exporter."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatial_export.audio import RenderConfig, read_wav  # noqa: E402
from spatial_export.exporter import export_layout  # noqa: E402
from spatial_export.geometry import (  # noqa: E402
    angle_between,
    det3,
    normalize,
    unit_from_az_el,
)
from spatial_export.layouts import LAYOUT_ASYM, LAYOUT_CUBE  # noqa: E402
from spatial_export.panner import VbapPanner  # noqa: E402
from spatial_export.trajectory import demo_trajectory  # noqa: E402
from spatial_export.verifier import verify_all  # noqa: E402


class GeometryTests(unittest.TestCase):
    def test_unit_sphere_and_angles(self) -> None:
        u = unit_from_az_el(123.0, -27.0)
        self.assertAlmostEqual(sum(v * v for v in u), 1.0, places=12)
        self.assertAlmostEqual(
            angle_between(unit_from_az_el(0, 0), unit_from_az_el(90, 0)),
            math.pi / 2,
            places=12,
        )

    def test_det3_sign(self) -> None:
        self.assertAlmostEqual(
            det3(((1, 0, 0), (0, 1, 0), (0, 0, 1))), 1.0, places=12
        )


class PannerTests(unittest.TestCase):
    def test_symmetric_layout_covers_every_direction(self) -> None:
        panner = VbapPanner(LAYOUT_CUBE)
        self.assertGreater(len(panner.triplets), 0)
        for az in range(0, 360, 17):
            for el in (-80, -40, 0, 40, 80):
                result = panner.pan(unit_from_az_el(az, el))
                self.assertFalse(result.degraded, (az, el))
                self.assertLess(result.angular_error_deg, 1.0)

    def test_asymmetric_layout_reports_degradation(self) -> None:
        panner = VbapPanner(LAYOUT_ASYM)
        rear = panner.pan(unit_from_az_el(180.0, 0.0))
        self.assertTrue(rear.degraded)
        self.assertEqual(rear.degradation_kind, "uncovered_nearest_speaker")
        self.assertEqual(len(rear.active), 1)
        front = panner.pan(unit_from_az_el(0.0, 10.0))
        self.assertFalse(front.degraded)
        self.assertLess(front.angular_error_deg, 1.0)

    def test_gains_are_non_negative_and_normalized(self) -> None:
        panner = VbapPanner(LAYOUT_CUBE)
        result = panner.pan(unit_from_az_el(37.0, 21.0))
        self.assertTrue(all(g >= -1e-12 for g in result.gains))
        self.assertAlmostEqual(sum(g * g for g in result.gains), 1.0, places=9)


class ExportVerificationTests(unittest.TestCase):
    def test_both_layouts_pass_full_pipeline(self) -> None:
        config = RenderConfig(sample_rate=24000, block_size=256, bit_depth=24)
        trajectory = demo_trajectory(duration_s=4.0)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            for layout in (LAYOUT_CUBE, LAYOUT_ASYM):
                result = export_layout(layout, trajectory, config, out_dir)
                report = verify_all(layout, trajectory, config, result)
                self.assertTrue(report.passed, report.checks)

                channels, rate, depth = read_wav(result.multichannel_wav)
                self.assertEqual(rate, 24000)
                self.assertEqual(depth, 24)
                self.assertEqual(len(channels), len(layout.speakers))

            # Asymmetric layout must have degraded blocks on the rear sweep;
            # the symmetric cube must have none.
            cube = export_layout(
                LAYOUT_CUBE, trajectory, config, out_dir / "cube"
            )
            asym = export_layout(
                LAYOUT_ASYM, trajectory, config, out_dir / "asym"
            )
            self.assertEqual(cube.degradation_blocks, 0)
            self.assertGreater(asym.degradation_blocks, 0)
            self.assertGreaterEqual(len(asym.degradation_events), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
