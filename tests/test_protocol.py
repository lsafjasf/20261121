import unittest

from assettone.protocol import (
    PROTOCOL,
    build_frame_bits,
    crc16_ccitt,
    decode_body_bits,
)


class ProtocolTests(unittest.TestCase):
    def test_crc16_check_vector(self):
        self.assertEqual(crc16_ccitt(b"123456789"), 0x29B1)

    def test_frame_round_trip(self):
        frame, checksum = build_frame_bits(0x7781)
        self.assertEqual(len(frame), PROTOCOL.frame_bits)
        decoded = decode_body_bits(frame[PROTOCOL.sync_bits :])
        self.assertEqual(decoded["version"], 1)
        self.assertEqual(decoded["asset_id"], 0x7781)
        self.assertEqual(decoded["meta"], 3)
        self.assertEqual(decoded["crc_expected"], checksum)
        self.assertEqual(decoded["crc_received"], checksum)


if __name__ == "__main__":
    unittest.main()

