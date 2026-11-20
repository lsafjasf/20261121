"""Randomized invariant: after every edit, master == full re-render."""
import os
import random
import tempfile
import unittest

from helpers import write_source

from bwf_batch.batch import BatchProcessor
from bwf_batch.bwf import read_bwf
from bwf_batch.render import render_full
from bwf_batch.store import ProjectStore


class TestIncrementalFuzz(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_random_edit_sequences(self):
        rng = random.Random(20261121)
        modes = ["last_wins", "multiply", "max"]
        for trial in range(8):
            frames = 1200
            duration = frames / 8000.0
            src_path = write_source(self.tmp.name, name="fuzz.wav", frames=frames)
            outdir = os.path.join(self.tmp.name, "out%d" % trial)
            processor = BatchProcessor(outdir)
            created = []
            for step in range(12):
                op = self._random_op(rng, step, duration, modes, created)
                processor.process_file(src_path, [op])
                store = ProjectStore(outdir)
                master = store.load_master("fuzz")
                env = store.load_envelope("fuzz")
                full = render_full(read_bwf(src_path), env)
                self.assertEqual(
                    master.samples,
                    full.samples,
                    "trial %d step %d op %s" % (trial, step, op),
                )

    def _random_op(self, rng, step, duration, modes, created):
        choices = ["add_segment", "set_merge_mode"]
        if created:
            choices += ["add_point", "move_point", "add_segment"]
        kind = rng.choice(choices)
        if kind == "set_merge_mode":
            return {"op": "set_merge_mode", "mode": rng.choice(modes)}
        if kind == "add_segment":
            name = "s%d" % step
            created.append(name)
            points = sorted(
                (round(rng.uniform(-0.05, duration + 0.05), 4),
                 round(rng.uniform(-30.0, 0.0), 2))
                for _ in range(rng.randint(1, 4))
            )
            return {"op": "add_segment", "name": name, "points": points}
        name = rng.choice(created)
        if kind == "add_point":
            return {"op": "add_point", "segment": name,
                    "time": round(rng.uniform(-0.05, duration + 0.05), 4),
                    "gain_db": round(rng.uniform(-30.0, 0.0), 2)}
        return {"op": "move_point", "segment": name, "index": 0,
                "time": round(rng.uniform(-0.05, duration + 0.05), 4),
                "gain_db": round(rng.uniform(-30.0, 0.0), 2)}


if __name__ == "__main__":
    unittest.main()
