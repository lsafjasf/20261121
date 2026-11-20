import json
import os
import tempfile
import unittest

from helpers import write_source

from bwf_batch.batch import BatchProcessor
from bwf_batch.bwf import read_bwf, write_bwf
from bwf_batch.store import ProjectStore
from bwf_batch.verify import (
    BOUNDARY_HALF_WINDOW,
    boundary_probes,
    compare,
    fixed_probes,
    verify_all,
    verify_file,
)

EDITS = [
    {"op": "add_segment", "name": "fade",
     "points": [[-0.02, -30.0], [0.04, 0.0]]},
    {"op": "add_segment", "name": "dip",
     "points": [[0.1, 0.0], [0.125, -18.0], [0.15, 0.0]]},
    {"op": "move_point", "segment": "dip", "index": 1,
     "time": 0.13, "gain_db": -24.0},
]


class TestProbes(unittest.TestCase):
    def test_fixed_probes_cover_head_and_tail(self):
        probes = fixed_probes(10000, count=4, length=100)
        self.assertEqual(probes[0], (0, 100))
        self.assertEqual(probes[-1], (9900, 10000))
        for start, end in probes:
            self.assertEqual(end - start, 100)

    def test_fixed_probes_short_file(self):
        self.assertEqual(fixed_probes(10, length=64), [(0, 10)])

    def test_boundary_probes_straddle_boundary(self):
        probes = boundary_probes([500], 1000, half=8)
        self.assertEqual(probes, [(500, 492, 508)])

    def test_boundary_probes_skip_file_edges(self):
        self.assertEqual(boundary_probes([0, 1000], 1000), [])


class TestVerify(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.src_path = write_source(self.tmp.name, frames=2000)
        self.outdir = os.path.join(self.tmp.name, "out")
        BatchProcessor(self.outdir).process_file(self.src_path, EDITS)

    def test_clean_history_reports_equal(self):
        store = ProjectStore(self.outdir)
        report = verify_file(store, "src")
        self.assertTrue(report["equal"])
        self.assertEqual(report["total_mismatched_samples"], 0)
        kinds = {p["kind"] for p in report["probes"]}
        self.assertEqual(kinds, {"fixed", "boundary"})
        # Report is persisted next to the master.
        self.assertTrue(
            os.path.exists(os.path.join(self.outdir, "src", "verify_report.json"))
        )

    def test_corrupted_master_yields_boundary_sample_diffs(self):
        store = ProjectStore(self.outdir)
        master = store.load_master("src")
        # Corrupt samples around a known splice boundary from the history.
        meta = list(store.iter_version_meta("src"))[-1]
        boundary = meta["affected_frames"]["start"]
        ch = master.channels
        master.samples[boundary * ch] += 1000
        master.samples[(boundary + 2) * ch + 1] -= 500
        write_bwf(store.master_path("src"), master)

        report = verify_file(ProjectStore(self.outdir), "src")
        self.assertFalse(report["equal"])
        boundary_probe = [
            p for p in report["probes"]
            if p["kind"] == "boundary" and p["boundary_frame"] == boundary
        ][0]
        self.assertEqual(boundary_probe["mismatched_samples"], 2)
        self.assertEqual(boundary_probe["max_abs_delta"], 1000)
        diffs = boundary_probe["sample_diffs"]
        self.assertEqual(
            {(d["frame"], d["channel"]) for d in diffs},
            {(boundary, 0), (boundary + 2, 1)},
        )
        deltas = {d["frame"]: d["delta"] for d in diffs}
        self.assertEqual(deltas[boundary], 1000)
        self.assertEqual(deltas[boundary + 2], -500)

    def test_verify_all_aggregates(self):
        result = verify_all(ProjectStore(self.outdir))
        self.assertTrue(result["equal"])
        self.assertIn("src", result["files"])

    def test_compare_lists_only_bounded_diffs(self):
        # sample_diffs listing is capped; counts stay exact.
        from helpers import make_bwf
        full = make_bwf(frames=100)
        incr = make_bwf(frames=100)
        for i in range(len(incr.samples)):
            incr.samples[i] += 1
        report = compare(full, incr, boundaries=[50])
        self.assertFalse(report["equal"])
        listed = sum(len(p.get("sample_diffs", [])) for p in report["probes"])
        self.assertLessEqual(listed, 32)


if __name__ == "__main__":
    unittest.main()
