from __future__ import annotations

import math
import random
import shutil
import tempfile
import unittest
from pathlib import Path

from oscilloscope_video.model import Event, TraceData
from oscilloscope_video.pipeline import render_trace_video
from oscilloscope_video.verify import verify_mp4


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg")
class IntegrationTests(unittest.TestCase):
    def test_mp4_decode_matches_cue_samples_and_event_first_frame(self) -> None:
        rng = random.Random(7)
        times = []
        voltages = []
        time_s = 0.0
        for _ in range(30000):
            time_s += (0.7 + 0.6 * rng.random()) / 30000.0
            if 0.63 < time_s < 0.68:
                continue
            if time_s > 1.0:
                break
            carrier = 0.35 * math.sin(2.0 * math.pi * 4200.0 * time_s)
            transient = 0.9 * math.exp(-((time_s - 0.4173) / 0.0005) ** 2)
            times.append(time_s)
            voltages.append(0.1 + carrier + transient)
        data = TraceData(
            times,
            voltages,
            "V",
            [Event(0.4173, "A", cue_hz=1320.0)],
            gap_factor=8.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trace.mp4"
            render_trace_video(
                data,
                output,
                fps=10,
                window_seconds=0.3,
                duration_seconds=1.0,
                width=320,
                height=180,
                supersample=1,
            )
            report = verify_mp4(
                output,
                output.with_suffix(".manifest.json"),
                audio_tolerance_ms=10.0,
            )
        self.assertTrue(report["passed"], report["issues"])
        event = report["events"][0]
        self.assertLessEqual(abs(event["sample_error"]), 240)
        self.assertGreaterEqual(event["waveform_correlation"], 0.62)
        self.assertEqual(event["decoded_first_frame"], event["expected_first_frame"])


if __name__ == "__main__":
    unittest.main()
