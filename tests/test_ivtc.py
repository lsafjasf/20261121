"""Ground-truth tests for cadence detection, IVTC reconstruction and combing.

Run with:  python3 -m unittest tests.test_ivtc -v
"""

import unittest

from ivtc.detector import analyze
from ivtc.imageio import split_fields, weave
from ivtc.metrics import comb_energy, same_parity_residues
from ivtc.reconstruct import manifest, reconstruct
from ivtc.synth import build_mixed_clip, timeline_truth
from ivtc.verify import verify_reconstruction


class CadenceTests(unittest.TestCase):
    def setUp(self):
        self.frames, self.truth = build_mixed_clip()
        self.analysis = analyze(self.frames)
        self.gt = timeline_truth()

    def test_segment_boundaries_match_ground_truth(self):
        kinds = [(s.kind, s.field_start, s.field_end)
                 for s in self.analysis.segments]
        expected = [(s["kind"], s["field_start"], s["field_end"])
                    for s in self.gt["segments"]]
        self.assertEqual(kinds, expected)

    def test_pulldown_anchors_are_exact_five_field_spaced(self):
        for seg in self.analysis.segments:
            if seg.kind != "pulldown":
                continue
            gaps = {b - a for a, b in
                    zip(seg.anchors, seg.anchors[1:])}
            self.assertEqual(gaps, {5})
            self.assertGreaterEqual(len(seg.anchors), 2)

    def test_cuts_at_ground_truth_frames(self):
        cut_frames = sorted(c.frame_index for c in self.analysis.cuts)
        self.assertEqual(cut_frames, [c["frame_index"] for c in self.gt["cuts"]])

    def test_native_segment_has_no_low_residues(self):
        residues = same_parity_residues(self.analysis.fields)
        native = next(s for s in self.analysis.segments if s.kind == "native")
        vals = [r for r in residues[native.field_start:native.field_end]
                if r is not None]
        self.assertTrue(all(r > self.analysis.threshold for r in vals))


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.frames, self.truth = build_mixed_clip()
        self.analysis = analyze(self.frames)
        self.rebuilt = reconstruct(self.analysis)

    def test_only_confident_pulldown_segments_reconstructed(self):
        confident = [s for s in self.analysis.segments
                     if s.kind == "pulldown" and s.confidence >= 0.9]
        # anchors whose weave window stays inside the segment are usable;
        # the first anchor of the post-cut run reaches into native fields
        usable = 0
        for seg in confident:
            for anchor in seg.anchors:
                if anchor % 2 == 0:
                    top_i, bottom_i = anchor - 4, anchor - 3
                else:
                    top_i, bottom_i = anchor - 1, anchor - 2
                if top_i >= seg.field_start and bottom_i < seg.field_end:
                    usable += 1
        expected = usable
        self.assertEqual(len(self.rebuilt), expected)

    def test_each_frame_cites_two_opposite_parity_input_fields(self):
        fields = self.analysis.fields
        for item in self.rebuilt:
            top = fields[item.top_field]
            bottom = fields[item.bottom_field]
            self.assertEqual(top.parity, "top")
            self.assertEqual(bottom.parity, "bottom")
            self.assertNotEqual(top.index, bottom.index)

    def test_recovered_frames_match_ground_truth_film_frames(self):
        for item in self.rebuilt:
            tt = self.truth[item.top_field]
            tb = self.truth[item.bottom_field]
            self.assertEqual(tt.kind, "pulldown")
            self.assertEqual(tb.kind, "pulldown")
            self.assertEqual(tt.scene, tb.scene)
            self.assertEqual(tt.film_frame, tb.film_frame)

    def test_no_reconstruction_crosses_a_cut(self):
        cut_fields = {c.field_index for c in self.analysis.cuts}
        for item in self.rebuilt:
            lo = min(item.top_field, item.bottom_field)
            hi = max(item.top_field, item.bottom_field)
            for cut in cut_fields:
                self.assertFalse(lo < cut <= hi,
                                 f"frame {item.frame.index} crosses cut {cut}")

    def test_manifest_is_auditable_and_pixels_match(self):
        man = manifest(self.rebuilt)
        self.assertEqual(man["frame_count"], len(self.rebuilt))
        fields = self.analysis.fields
        for entry, item in zip(man["frames"], self.rebuilt):
            top = fields[entry["top_field_index"]]
            bottom = fields[entry["bottom_field_index"]]
            rewoven = weave(top, bottom, item.frame.index, {})
            self.assertEqual(rewoven.buf, item.frame.buf)


class CombingTests(unittest.TestCase):
    def setUp(self):
        frames, _ = build_mixed_clip()
        self.analysis = analyze(frames)
        self.rebuilt = reconstruct(self.analysis)

    def test_reconstructed_frames_have_no_motion_edge_combing(self):
        for item in self.rebuilt:
            stats = comb_energy(item.frame)
            self.assertLess(stats["comb_ratio"], 2.0)
            self.assertLess(stats["boundary_mismatch"], 2.0)

    def test_deliberate_mismatch_is_flagged(self):
        result = verify_reconstruction(self.analysis, self.rebuilt)
        self.assertTrue(result["all_passed"])
        ctrl = result["negative_control"]
        self.assertTrue(ctrl["mismatch_detected"])
        self.assertGreater(ctrl["mismatched_weave"]["comb_ratio"], 2.0)
        self.assertLess(ctrl["matched_weave"]["comb_ratio"], 2.0)

    def test_split_and_weave_roundtrip(self):
        frames, _ = build_mixed_clip()
        top, bottom = split_fields(frames[0])
        rewoven = weave(top, bottom, 0, {})
        self.assertEqual(rewoven.buf, frames[0].buf)


class NoiseRobustnessTests(unittest.TestCase):
    def test_detection_survives_sensor_noise(self):
        import random

        from ivtc.imageio import VideoFrame
        rng = random.Random(20261121)
        frames, _ = build_mixed_clip()
        noisy = []
        for frame in frames:
            buf = bytearray(frame.buf)
            for i in range(0, len(buf), 3):  # sparse +-6 luma noise
                buf[i] = max(0, min(255, buf[i] + rng.randint(-6, 6)))
            noisy.append(VideoFrame(frame.index, frame.width, frame.height,
                                    bytes(buf), dict(frame.source)))
        analysis = analyze(noisy)
        kinds = [(s.kind, s.field_start, s.field_end)
                 for s in analysis.segments]
        self.assertEqual(kinds, [
            ("pulldown", 0, 30), ("native", 30, 46),
            ("pulldown", 46, 60)])
        self.assertEqual(sorted(c.frame_index for c in analysis.cuts),
                         [15, 23])
        rebuilt = reconstruct(analysis)
        result = verify_reconstruction(analysis, rebuilt)
        self.assertTrue(result["all_passed"])


class CadencePhaseTests(unittest.TestCase):
    def test_all_entry_phases_reconstruct_without_combing(self):
        from ivtc.synth import _weave_record_pair, telecine_fields

        for phase in range(5):
            records, _ = telecine_fields(24, "A", phase_shift=phase)
            records = records[:(len(records) // 2) * 2]
            frames = [_weave_record_pair(records[2 * fi],
                                         records[2 * fi + 1], fi)
                      for fi in range(len(records) // 2)]
            analysis = analyze(frames)
            lows = [n for n, r in enumerate(analysis.residues)
                    if r is not None and r <= analysis.threshold]
            gaps = {b - a for a, b in zip(lows, lows[1:])}
            self.assertEqual(gaps, {5}, f"phase {phase}: cadence broken")
            rebuilt = reconstruct(analysis)
            self.assertGreaterEqual(len(rebuilt), 10)
            for item in rebuilt:
                self.assertLess(comb_energy(item.frame)["comb_ratio"], 2.0)


if __name__ == "__main__":
    unittest.main()
