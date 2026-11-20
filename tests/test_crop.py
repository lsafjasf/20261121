from __future__ import annotations

import unittest

import numpy as np

from film_stabilizer.crop import largest_rectangle
from film_stabilizer.types import rigid_matrix


class CropTests(unittest.TestCase):
    def test_identity_returns_everything(self) -> None:
        mask = np.ones((40, 60), dtype=np.uint8) * 255
        self.assertEqual(largest_rectangle(mask), (0, 0, 60, 40))

    def test_translation_leaves_expected_safe_rectangle(self) -> None:
        matrix = rigid_matrix(0.0, 5.0, 7.0)
        mask = np.zeros((40, 60), dtype=np.uint8)
        mask[7:, 5:] = 255
        self.assertEqual(largest_rectangle(mask), (5, 7, 55, 33))

    def test_rotation_reduces_safe_area(self) -> None:
        mask = np.ones((200, 200), dtype=np.uint8) * 255
        center = (100.0, 100.0)
        angle = np.radians(3.0)
        c, s = np.cos(angle), np.sin(angle)
        tx = center[0] - c * center[0] + s * center[1]
        ty = center[1] - s * center[0] - c * center[1]
        matrix = rigid_matrix(angle, tx, ty)
        warped = np.zeros_like(mask)
        for y in range(200):
            for x in range(200):
                source = matrix[:2, :2].T @ (np.array([x, y]) - matrix[:2, 2])
                if 0 <= source[0] < 200 and 0 <= source[1] < 200:
                    warped[y, x] = 255
        x, y, width, height = largest_rectangle(warped)
        self.assertGreater(width * height, 30000)
        self.assertLess(width, 200)
        self.assertLess(height, 200)
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)


if __name__ == "__main__":
    unittest.main()
