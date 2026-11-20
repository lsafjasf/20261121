from __future__ import annotations

import cv2
import numpy as np

Rect = tuple[int, int, int, int]


def valid_mask(
    matrix: np.ndarray, size: tuple[int, int]
) -> np.ndarray:
    width, height = size
    source = np.ones((height, width), dtype=np.uint8) * 255
    return cv2.warpAffine(
        source,
        matrix[:2, :],
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def largest_rectangle(mask: np.ndarray) -> Rect:
    """Return the largest all-valid axis-aligned rectangle in ``mask``."""
    if mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    height, width = mask.shape
    if height == 0 or width == 0 or not np.any(mask):
        return 0, 0, 0, 0

    valid = mask > 0
    heights = np.zeros(width, dtype=np.int32)
    best_area = 0
    best: Rect = (0, 0, 0, 0)

    for row in range(height):
        heights = np.where(valid[row], heights + 1, 0)
        stack: list[tuple[int, int]] = []
        for column in range(width + 1):
            current = int(heights[column]) if column < width else 0
            start = column
            while stack and stack[-1][1] > current:
                start, bar_height = stack.pop()
                area = bar_height * (column - start)
                if area > best_area:
                    best_area = area
                    best = (
                        start,
                        row - bar_height + 1,
                        column - start,
                        bar_height,
                    )
            if not stack or stack[-1][1] < current:
                stack.append((start, current))
    return best


def common_valid_mask(
    matrices: list[np.ndarray], size: tuple[int, int]
) -> np.ndarray:
    width, height = size
    mask = np.ones((height, width), dtype=np.uint8) * 255
    for matrix in matrices:
        mask = cv2.bitwise_and(mask, valid_mask(matrix, size))
    return mask


def apply_crop(frame: np.ndarray, crop: Rect) -> np.ndarray:
    x, y, width, height = crop
    if width <= 0 or height <= 0:
        raise ValueError("empty crop rectangle")
    return frame[y : y + height, x : x + width]
