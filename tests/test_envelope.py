import unittest

from helpers import SAMPLE_RATE  # noqa: F401  (ensures path setup)

from bwf_batch.envelope import (
    MERGE_LAST_WINS,
    MERGE_MAX,
    MERGE_MULTIPLY,
    ControlPoint,
    Envelope,
    Segment,
)


def seg(name, *points):
    return Segment(name, [ControlPoint(t, db) for t, db in points])


class TestSegmentEval(unittest.TestCase):
    def test_linear_interpolation_in_db(self):
        s = seg("a", (0.0, 0.0), (1.0, -12.0))
        self.assertAlmostEqual(s.eval_db(0.5), -6.0)

    def test_holds_edge_values_outside_span(self):
        s = seg("a", (1.0, -3.0), (2.0, -9.0))
        self.assertEqual(s.eval_db(0.0), -3.0)
        self.assertEqual(s.eval_db(5.0), -9.0)

    def test_pre_head_point_shapes_file_start(self):
        # Point before the head pulls the gain at t=0 towards its value.
        s = seg("a", (-1.0, -60.0), (1.0, 0.0))
        self.assertAlmostEqual(s.eval_db(0.0), -30.0)

    def test_post_tail_point_shapes_file_end(self):
        s = seg("a", (0.5, 0.0), (2.0, -30.0))
        # File ends at 1.0s; the post-tail point still shapes t<1.0.
        self.assertAlmostEqual(s.eval_db(1.0), -10.0)


class TestMergeSemantics(unittest.TestCase):
    def test_last_wins_default(self):
        env = Envelope(
            [seg("first", (0.0, -20.0), (1.0, -20.0)),
             seg("second", (0.5, -2.0), (1.5, -2.0))],
            merge_mode=MERGE_LAST_WINS,
        )
        self.assertEqual(env.eval_db(0.25), -20.0)   # only "first" covers
        self.assertEqual(env.eval_db(0.75), -2.0)    # overlap: later wins
        self.assertEqual(env.eval_db(1.25), -2.0)    # only "second" covers

    def test_multiply_sums_db(self):
        env = Envelope(
            [seg("a", (0.0, -6.0), (1.0, -6.0)),
             seg("b", (0.0, -6.0), (1.0, -6.0))],
            merge_mode=MERGE_MULTIPLY,
        )
        self.assertAlmostEqual(env.eval_db(0.5), -12.0)

    def test_max_picks_highest(self):
        env = Envelope(
            [seg("a", (0.0, -6.0), (1.0, -6.0)),
             seg("b", (0.0, -12.0), (1.0, -12.0))],
            merge_mode=MERGE_MAX,
        )
        self.assertEqual(env.eval_db(0.5), -6.0)

    def test_max_crossing_breakpoint(self):
        # The merged max curve switches at the crossing of two lines, which
        # is not a control point; breakpoints() must include it.
        env = Envelope(
            [seg("up", (0.0, -10.0), (1.0, 0.0)),
             seg("down", (0.0, 0.0), (1.0, -10.0))],
            merge_mode=MERGE_MAX,
        )
        crossings = [t for t in env.breakpoints() if abs(t - 0.5) < 1e-9]
        self.assertTrue(crossings)
        self.assertAlmostEqual(env.eval_db(0.5), -5.0)

    def test_gap_holds_nearest_edge(self):
        env = Envelope(
            [seg("a", (0.0, -4.0), (1.0, -8.0)),
             seg("b", (3.0, -2.0), (4.0, -2.0))],
        )
        self.assertEqual(env.eval_db(1.4), -8.0)  # nearer to a's tail
        self.assertEqual(env.eval_db(2.6), -2.0)  # nearer to b's head

    def test_empty_envelope_is_unity(self):
        self.assertEqual(Envelope().eval_db(0.3), 0.0)

    def test_outside_all_segments_holds_global_edges(self):
        env = Envelope([seg("a", (1.0, -6.0), (2.0, -12.0))])
        self.assertEqual(env.eval_db(0.0), -6.0)
        self.assertEqual(env.eval_db(99.0), -12.0)


class TestAffectedRange(unittest.TestCase):
    def test_no_change_returns_none(self):
        env = Envelope([seg("a", (0.0, -6.0), (1.0, -6.0))])
        self.assertIsNone(env.affected_range(env.copy(), 1.0))

    def test_change_outside_material_returns_none(self):
        # The post-tail segment is shadowed inside [0, 1] by "main"
        # (last_wins), so editing it cannot affect the material.
        before = Envelope(
            [seg("main", (0.0, 0.0), (1.0, 0.0)),
             seg("tail", (2.0, -6.0), (5.0, -6.0))],
        )
        after = Envelope(
            [seg("main", (0.0, 0.0), (1.0, 0.0)),
             seg("tail", (2.0, -9.0), (5.0, -9.0))],
        )
        self.assertIsNone(before.affected_range(after, 1.0))

    def test_range_expands_to_neighbouring_breakpoints(self):
        before = Envelope([seg("a", (0.0, 0.0), (0.5, 0.0), (1.0, 0.0))])
        after = Envelope([seg("a", (0.0, 0.0), (0.5, -6.0), (1.0, 0.0))])
        lo, hi = before.affected_range(after, 1.0)
        self.assertLessEqual(lo, 0.5)
        self.assertGreaterEqual(hi, 0.5)
        self.assertAlmostEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 1.0)

    def test_range_clamped_at_tail(self):
        before = Envelope([seg("a", (0.0, -3.0), (5.0, -3.0))])
        after = Envelope([seg("a", (0.0, -3.0), (0.75, -3.0), (5.0, -9.0))])
        lo, hi = before.affected_range(after, 1.0)
        self.assertAlmostEqual(lo, 0.75)
        self.assertEqual(hi, 1.0)  # clamped from the post-tail point at 5.0

    def test_range_clamped_at_head(self):
        before = Envelope()
        after = Envelope([seg("a", (-5.0, -3.0), (0.25, -3.0))])
        lo, hi = before.affected_range(after, 1.0)
        self.assertEqual(lo, 0.0)  # clamped from the pre-head point at -5.0
        self.assertEqual(hi, 1.0)  # edge hold extends the gain to the tail

    def test_range_matches_dense_sampling(self):
        # Cross-check the analytic range against dense per-sample evaluation.
        before = Envelope(
            [seg("a", (0.0, -6.0), (0.4, -6.0), (1.0, -6.0)),
             seg("b", (0.3, -3.0), (0.6, -3.0))],
            merge_mode=MERGE_MAX,
        )
        after = before.copy()
        after.move_point("b", 0, 0.35, -1.0)
        duration = 1.0
        lo, hi = before.affected_range(after, duration)
        n = SAMPLE_RATE
        for f in range(n):
            t = f / n
            differs = abs(before.eval_db(t) - after.eval_db(t)) > 1e-9
            in_range = lo <= t <= hi
            if differs:
                self.assertTrue(in_range, "t=%f differs but outside range" % t)


class TestSerialization(unittest.TestCase):
    def test_roundtrip(self):
        env = Envelope(
            [seg("a", (-0.5, -60.0), (1.0, 0.0))], merge_mode=MERGE_MULTIPLY
        )
        clone = Envelope.from_dict(env.to_dict())
        self.assertEqual(clone.to_dict(), env.to_dict())
        self.assertEqual(clone.merge_mode, MERGE_MULTIPLY)


if __name__ == "__main__":
    unittest.main()
