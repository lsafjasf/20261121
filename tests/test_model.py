from __future__ import annotations

import unittest

from oscilloscope_video.model import Event, Timeline, TraceData, merge_intervals


class TimelineTests(unittest.TestCase):
    def test_event_frame_is_not_quantized_for_audio(self) -> None:
        timeline = Timeline.fit(0.0, 2.0, 0.5, 30)
        event = Event(1.1373, "A")
        self.assertEqual(timeline.first_frame(event.time_s), 35)
        self.assertAlmostEqual(timeline.video_time(event.time_s), 1.1373)
        self.assertAlmostEqual(timeline.window_center(35), 35 / 30)

    def test_compressed_playback_changes_frame_mapping(self) -> None:
        timeline = Timeline.fit(0.0, 2.0, 0.5, 30, duration_s=1.0)
        self.assertAlmostEqual(timeline.time_scale, 2.0)
        self.assertEqual(timeline.first_frame(1.1373), 18)

    def test_capture_end_has_a_frame_at_or_after_event(self) -> None:
        timeline = Timeline.fit(0.0, 7.199990793, 0.75, 30)
        self.assertGreater(timeline.frame_count, timeline.duration_s * 30)
        end_frame = timeline.first_frame(timeline.capture_end_s)
        self.assertGreaterEqual(
            end_frame / timeline.fps,
            timeline.video_time(timeline.capture_end_s) - 1e-12,
        )
        self.assertLess(end_frame, timeline.frame_count)


class GapTests(unittest.TestCase):
    def test_merge_intervals(self) -> None:
        self.assertEqual(
            merge_intervals([(2.0, 3.0), (1.0, 2.5), (5.0, 5.1)]),
            [(1.0, 3.0), (5.0, 5.1)],
        )

    def test_sparse_rows_become_unknown_interval(self) -> None:
        data = TraceData(
            [0.0, 0.001, 0.002, 0.050, 0.051],
            [0.0, 0.1, 0.2, 0.3, 0.4],
            "V",
            gap_factor=8.0,
        )
        self.assertEqual(data.effective_unknown_intervals(), [(0.002, 0.05)])
        self.assertGreaterEqual(len(set(data.sample_runs())), 2)

    def test_invalid_voltage_splits_trace(self) -> None:
        data = TraceData(
            [0.0, 0.001, 0.002, 0.003],
            [0.0, None, 0.2, 0.3],
            "mV",
        )
        self.assertEqual(len(set(data.sample_runs())), 2)


if __name__ == "__main__":
    unittest.main()
