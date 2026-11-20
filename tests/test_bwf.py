import os
import struct
import tempfile
import unittest

from helpers import make_bwf, write_source

from bwf_batch.bwf import (
    WAVE_FORMAT_IEEE_FLOAT,
    read_bwf,
    write_bwf,
)


class TestBwfRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def roundtrip(self, **kwargs):
        path = write_source(self.tmp.name, **kwargs)
        return read_bwf(path)

    def test_pcm16_roundtrip_bitexact(self):
        bwf = self.roundtrip(bit_depth=16)
        self.assertEqual(bwf.samples, make_bwf(bit_depth=16).samples)
        self.assertEqual(bwf.sample_rate, 8000)
        self.assertEqual(bwf.channels, 2)
        self.assertEqual(bwf.frame_count, 1000)

    def test_pcm24_roundtrip_bitexact(self):
        bwf = self.roundtrip(bit_depth=24)
        self.assertEqual(bwf.samples, make_bwf(bit_depth=24).samples)

    def test_pcm32_roundtrip_bitexact(self):
        bwf = self.roundtrip(bit_depth=32)
        self.assertEqual(bwf.samples, make_bwf(bit_depth=32).samples)

    def test_float32_roundtrip(self):
        bwf = self.roundtrip(audio_format=WAVE_FORMAT_IEEE_FLOAT, bit_depth=32)
        expected = make_bwf(audio_format=WAVE_FORMAT_IEEE_FLOAT, bit_depth=32)
        as_f32 = [
            struct.unpack("<f", struct.pack("<f", v))[0] for v in expected.samples
        ]
        self.assertEqual(bwf.samples, as_f32)

    def test_bext_preserved(self):
        bwf = self.roundtrip()
        self.assertIsNotNone(bwf.bext)
        self.assertEqual(bwf.bext.description, "test material")
        self.assertEqual(bwf.bext.originator_reference, "TESTREF0001")
        self.assertEqual(bwf.bext.origination_date, "2026-11-21")

    def test_extra_chunks_preserved(self):
        bwf = make_bwf()
        bwf.extra_chunks.append((b"test", b"abc"))  # odd size -> padding
        path = os.path.join(self.tmp.name, "extra.wav")
        write_bwf(path, bwf)
        back = read_bwf(path)
        self.assertIn((b"test", b"abc"), back.extra_chunks)
        self.assertEqual(back.samples, bwf.samples)

    def test_slice_frames_bitexact(self):
        bwf = make_bwf()
        part = bwf.slice_frames(100, 250)
        self.assertEqual(part.frame_count, 150)
        self.assertEqual(part.samples, bwf.samples[200:500])

    def test_slice_clamped(self):
        bwf = make_bwf()
        part = bwf.slice_frames(-10, 100000)
        self.assertEqual(part.frame_count, bwf.frame_count)


if __name__ == "__main__":
    unittest.main()
