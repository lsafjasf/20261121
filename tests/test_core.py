from __future__ import annotations

import itertools
from pathlib import Path
import tempfile
import unittest

from roomtone_handoff.model import LEFT, RIGHT, Region, Version
from roomtone_handoff.render import render_version
from roomtone_handoff.wav import read_wav

from .fixtures import create_fixture_set, desynchronized_stereo_frames


class RenderingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        directory = Path(self.temporary.name)
        self.source_path, self.room_tone_path = create_fixture_set(directory)
        self.source = read_wav(self.source_path)
        self.room_tone = read_wav(self.room_tone_path)
        self.frames = desynchronized_stereo_frames()

    def render(self, rules: tuple[Region, ...]):
        return render_version(self.source, self.room_tone, Version("check", rules))

    def test_pcm_bytes_outside_requested_intervals_are_identical(self) -> None:
        rules = (
            Region("left-base", LEFT, 4, 44, 1, 2),
            Region("left-focus", LEFT, 20, 52, 9, 0),
            Region("right-base", RIGHT, 7, 38, 2, 5),
            Region("right-focus", RIGHT, 28, 58, 8, 1),
            Region("right-tie", RIGHT, 35, 49, 8, 3),
        )
        output = self.render(rules)
        rendered = read_wav_from_bytes(output.output_bytes)
        self.assertEqual(
            rendered.raw[: rendered.data_offset],
            self.source.raw[: self.source.data_offset],
        )
        source_end = self.source.data_offset + self.source.data_size
        rendered_end = rendered.data_offset + rendered.data_size
        self.assertEqual(rendered.raw[rendered_end:], self.source.raw[source_end:])

        checked = 0
        changed = 0
        for frame_index, (original_left, original_right) in enumerate(self.frames):
            for channel, channel_index, original in (
                (LEFT, 0, original_left),
                (RIGHT, 1, original_right),
            ):
                inside = any(
                    rule.channel == channel
                    and rule.start_frame <= frame_index < rule.end_frame
                    for rule in rules
                )
                rendered_value = sample_at(rendered, frame_index, channel_index)
                if inside:
                    changed += rendered_value != original
                else:
                    width = self.source.format.sample_width
                    byte_offset = (
                        self.source.data_offset
                        + frame_index * self.source.format.block_align
                        + channel_index * width
                    )
                    self.assertEqual(
                        rendered.raw[byte_offset : byte_offset + width],
                        self.source.raw[byte_offset : byte_offset + width],
                        f"{channel} PCM bytes {frame_index} changed outside intervals",
                    )
                    self.assertEqual(
                        rendered_value,
                        original,
                        f"{channel} sample {frame_index} changed outside requested intervals",
                    )
                checked += 1
        self.assertGreater(changed, 0)
        self.assertGreater(checked, 0)

    def test_overlap_result_is_independent_of_rule_entry_order(self) -> None:
        rules = (
            Region("left-base", LEFT, 4, 44, 1, 2),
            Region("left-focus", LEFT, 20, 52, 9, 0),
            Region("right-base", RIGHT, 7, 38, 2, 5),
            Region("right-focus", RIGHT, 28, 58, 8, 1),
            Region("right-tie", RIGHT, 35, 49, 8, 3),
        )
        baseline = self.render(rules)
        baseline_spans = [
            (
                item.span.channel,
                item.span.start_frame,
                item.span.end_frame,
                item.span.region.rule_id,
            )
            for item in baseline.replacements
        ]
        self.assertIn(
            (RIGHT, 35, 49, "right-tie"),
            baseline_spans,
            "equal-priority overlap must use the documented stable tie-break",
        )
        for permutation in itertools.permutations(rules):
            candidate = self.render(permutation)
            self.assertEqual(candidate.output_bytes, baseline.output_bytes)
            self.assertEqual(
                [
                    (
                        item.span.channel,
                        item.span.start_frame,
                        item.span.end_frame,
                        item.span.region.rule_id,
                    )
                    for item in candidate.replacements
                ],
                baseline_spans,
            )

    def test_versions_render_only_from_the_original_source(self) -> None:
        first = self.render((Region("left-only", LEFT, 10, 30, 5, 0),))
        second = self.render((Region("right-only", RIGHT, 40, 60, 5, 0),))
        source_left_20 = sample_at(self.source, 20, 0)
        self.assertNotEqual(
            sample_at(read_wav_from_bytes(first.output_bytes), 20, 0),
            source_left_20,
        )
        self.assertEqual(sample_at(read_wav_from_bytes(second.output_bytes), 20, 0), source_left_20)
        self.assertEqual(
            sample_at(read_wav_from_bytes(second.output_bytes), 40, 0),
            sample_at(self.source, 40, 0),
        )


def read_wav_from_bytes(payload: bytes):
    handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        handle.write(payload)
        handle.close()
        return read_wav(Path(handle.name))
    finally:
        Path(handle.name).unlink(missing_ok=True)


def sample_at(audio, frame_index: int, channel_index: int) -> int:
    offset = (
        audio.data_offset
        + frame_index * audio.format.block_align
        + channel_index * audio.format.sample_width
    )
    return int.from_bytes(
        audio.raw[offset : offset + audio.format.sample_width], "little", signed=True
    )


if __name__ == "__main__":
    unittest.main()
