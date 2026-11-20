import json
import os
import tempfile
import unittest

from helpers import SAMPLE_RATE, write_source

from bwf_batch.batch import BatchProcessor
from bwf_batch.bwf import read_bwf
from bwf_batch.render import render_full
from bwf_batch.store import ProjectStore


class TestBatchRender(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.src_path = write_source(self.tmp.name, frames=2000)
        self.outdir = os.path.join(self.tmp.name, "out")

    def run_edits(self, edits):
        processor = BatchProcessor(self.outdir)
        return processor.process_file(self.src_path, edits)

    def master(self):
        return read_bwf(os.path.join(self.outdir, "src", "master.wav"))

    def test_first_edit_renders_full(self):
        versions = self.run_edits(
            [{"op": "add_segment", "name": "dip",
              "points": [[0.1, 0.0], [0.15, -12.0], [0.2, 0.0]]}]
        )
        self.assertEqual(versions, ["v0000"])
        meta = self._meta("v0000")
        self.assertEqual(meta["render_mode"], "full")
        self.assertEqual(meta["affected_frames"], {"start": 0, "end": 2000})

    def test_incremental_matches_full_render(self):
        edits = [
            {"op": "add_segment", "name": "fade",
             "points": [[-0.05, -40.0], [0.05, 0.0]]},          # pre-head point
            {"op": "add_segment", "name": "dip",
             "points": [[0.1, 0.0], [0.125, -18.0], [0.15, 0.0]]},
            {"op": "add_point", "segment": "fade", "time": 0.3, "gain_db": -6.0},
            {"op": "move_point", "segment": "dip", "index": 1,
             "time": 0.13, "gain_db": -24.0},
            {"op": "add_segment", "name": "tail",
             "points": [[0.2, 0.0], [0.35, -60.0]]},            # post-tail point
        ]
        self.run_edits(edits)
        store = ProjectStore(self.outdir)
        incremental = store.load_master("src")
        full = render_full(read_bwf(self.src_path), store.load_envelope("src"))
        self.assertEqual(incremental.samples, full.samples)

    def test_incremental_matches_full_overlapping_segments(self):
        edits = [
            {"op": "add_segment", "name": "a",
             "points": [[0.05, -6.0], [0.15, -6.0]]},
            {"op": "add_segment", "name": "b",
             "points": [[0.10, -12.0], [0.20, -12.0]]},          # overlaps a
            {"op": "set_merge_mode", "mode": "multiply"},
            {"op": "set_merge_mode", "mode": "max"},
            {"op": "set_merge_mode", "mode": "last_wins"},
            {"op": "remove_segment", "name": "a"},
        ]
        self.run_edits(edits)
        store = ProjectStore(self.outdir)
        incremental = store.load_master("src")
        full = render_full(read_bwf(self.src_path), store.load_envelope("src"))
        self.assertEqual(incremental.samples, full.samples)

    def test_noop_edit_skips_render(self):
        self.run_edits(
            [{"op": "add_segment", "name": "far",
              "points": [[5.0, -6.0], [9.0, -6.0]]},             # post-tail only…
             {"op": "add_segment", "name": "cover",
              "points": [[0.0, 0.0], [0.25, 0.0]]},              # …and shadowed
             {"op": "move_point", "segment": "far", "index": 0,
              "time": 6.0, "gain_db": -30.0}]
        )
        meta = self._meta("v0002")
        self.assertEqual(meta["render_mode"], "none")
        self.assertIsNone(meta["affected_frames"])

    def test_versions_store_source_samples_independently(self):
        self.run_edits(
            [{"op": "add_segment", "name": "dip",
              "points": [[0.1, 0.0], [0.125, -18.0], [0.15, 0.0]]},
             {"op": "move_point", "segment": "dip", "index": 1,
              "time": 0.13, "gain_db": -24.0}]
        )
        vdir = os.path.join(self.outdir, "src", "versions")
        # Baseline version references the whole source.
        ref0 = read_bwf(os.path.join(vdir, "v0000", "source_ref.wav"))
        src = read_bwf(self.src_path)
        self.assertEqual(ref0.samples, src.samples)
        # The adjustment references exactly its affected source range.
        meta1 = self._meta("v0001")
        ref1 = read_bwf(os.path.join(vdir, "v0001", "source_ref.wav"))
        start = meta1["affected_frames"]["start"]
        end = meta1["affected_frames"]["end"]
        self.assertEqual(ref1.frame_count, end - start)
        ch = src.channels
        self.assertEqual(ref1.samples, src.samples[start * ch : end * ch])
        # Envelope snapshot and hashes are recorded per version.
        self.assertIn("envelope_sha256_before", meta1)
        with open(os.path.join(vdir, "v0001", "envelope.json")) as fh:
            snap = json.load(fh)
        self.assertEqual(snap["segments"][0]["name"], "dip")

    def test_source_change_detected(self):
        self.run_edits(
            [{"op": "add_segment", "name": "a",
              "points": [[0.0, -3.0], [0.1, -3.0]]}]
        )
        write_source(self.tmp.name, frames=1500)  # same path, new content
        with self.assertRaises(ValueError):
            self.run_edits(
                [{"op": "add_segment", "name": "b",
                  "points": [[0.0, -3.0], [0.1, -3.0]]}]
            )

    def test_bext_survives_to_master(self):
        self.run_edits(
            [{"op": "add_segment", "name": "a",
              "points": [[0.0, -3.0], [0.1, -3.0]]}]
        )
        master = self.master()
        self.assertIsNotNone(master.bext)
        self.assertEqual(master.bext.originator_reference, "TESTREF0001")

    def _meta(self, version):
        path = os.path.join(self.outdir, "src", "versions", version, "meta.json")
        with open(path) as fh:
            return json.load(fh)


if __name__ == "__main__":
    unittest.main()
