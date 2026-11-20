import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from helpers import write_source

from bwf_batch.cli import main


class TestCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.src = write_source(self.tmp.name, name="take1.wav", frames=2000)
        self.job = {
            "outdir": "out",
            "files": [
                {
                    "path": "take1.wav",
                    "edits": [
                        {"op": "add_segment", "name": "fadein",
                         "points": [[-0.02, -40.0], [0.05, 0.0]]},
                        {"op": "add_segment", "name": "dip",
                         "points": [[0.1, 0.0], [0.125, -12.0], [0.15, 0.0]]},
                        {"op": "set_merge_mode", "mode": "last_wins"},
                        {"op": "move_point", "segment": "dip", "index": 1,
                         "time": 0.13, "gain_db": -20.0},
                    ],
                }
            ],
        }
        self.job_path = os.path.join(self.tmp.name, "job.json")
        with open(self.job_path, "w") as fh:
            json.dump(self.job, fh)
        self.outdir = os.path.join(self.tmp.name, "out")

    def test_run_then_verify_then_show(self):
        out = io.StringIO()
        with redirect_stdout(out):
            rc = main(["run", self.job_path])
        self.assertEqual(rc, 0)
        self.assertIn("take1: 4 adjustment(s) -> v0003", out.getvalue())
        self.assertTrue(
            os.path.exists(os.path.join(self.outdir, "take1", "master.wav"))
        )

        out = io.StringIO()
        with redirect_stdout(out):
            rc = main(["verify", self.outdir])
        self.assertEqual(rc, 0)
        self.assertIn("take1 [OK]", out.getvalue())

        out = io.StringIO()
        with redirect_stdout(out):
            rc = main(["show", self.outdir, "take1"])
        self.assertEqual(rc, 0)
        text = out.getvalue()
        self.assertIn('"fadein"', text)
        self.assertIn("v0001 add_segment render=incremental", text)

    def test_verify_fails_on_mismatch(self):
        main(["run", self.job_path])
        # Corrupt the master at a recorded splice boundary.
        from bwf_batch.bwf import read_bwf, write_bwf
        meta_path = os.path.join(
            self.outdir, "take1", "versions", "v0003", "meta.json"
        )
        with open(meta_path) as fh:
            boundary = json.load(fh)["affected_frames"]["start"]
        master_path = os.path.join(self.outdir, "take1", "master.wav")
        master = read_bwf(master_path)
        master.samples[boundary * master.channels] += 3000
        write_bwf(master_path, master)
        out = io.StringIO()
        with redirect_stdout(out):
            rc = main(["verify", self.outdir])
        self.assertEqual(rc, 1)
        self.assertIn("MISMATCH", out.getvalue())
        self.assertIn("delta=3000", out.getvalue())


if __name__ == "__main__":
    unittest.main()
