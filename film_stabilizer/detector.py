from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .types import Perforation


@dataclass
class DetectorConfig:
    """Geometry and shape limits for the two perforation bands."""

    side_band_fraction: float = 0.10
    top_margin_fraction: float = 0.03
    bottom_margin_fraction: float = 0.03
    polarity: str = "bright"
    min_area_fraction: float = 1.2e-4
    max_area_fraction: float = 8e-3
    min_aspect: float = 0.30
    max_aspect: float = 3.2
    min_extent: float = 0.42
    max_candidates_per_side: int = 4
    otsu_scale: float = 1.0

    def validate(self) -> None:
        if not 0.02 <= self.side_band_fraction <= 0.45:
            raise ValueError("side_band_fraction must be in [0.02, 0.45]")
        if self.polarity not in {"bright", "dark"}:
            raise ValueError("polarity must be 'bright' or 'dark'")
        if not 0 < self.min_area_fraction < self.max_area_fraction:
            raise ValueError("area fractions must satisfy 0 < min < max")


class PerforationDetector:
    """Detect high-contrast sprocket holes in the outer film bands.

    The picture area is never inspected, so independent actor motion cannot
    change the estimated registration.
    """

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self.config.validate()

    def detect(self, frame: np.ndarray) -> list[Perforation]:
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        height, width = gray.shape[:2]
        band = max(4, int(round(width * self.config.side_band_fraction)))
        top = int(round(height * self.config.top_margin_fraction))
        bottom = height - int(round(height * self.config.bottom_margin_fraction))

        left = self._detect_band(gray, "left", 0, band, top, bottom)
        right = self._detect_band(gray, "right", width - band, width, top, bottom)
        left.sort(key=lambda item: item.score, reverse=True)
        right.sort(key=lambda item: item.score, reverse=True)
        cap = self.config.max_candidates_per_side
        return left[:cap] + right[:cap]

    def _detect_band(
        self,
        gray: np.ndarray,
        side: str,
        x0: int,
        x1: int,
        y0: int,
        y1: int,
    ) -> list[Perforation]:
        if x1 - x0 < 3 or y1 - y0 < 3:
            return []
        roi = gray[y0:y1, x0:x1]
        blur = cv2.GaussianBlur(roi, (3, 3), 0)
        otsu, _ = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        if self.config.polarity == "bright":
            highlight = float(np.percentile(blur, 99.5))
            middle = float(np.percentile(blur, 50.0))
            if highlight - middle > 35.0:
                contrast_threshold = middle + 0.55 * (highlight - middle)
            else:
                contrast_threshold = highlight + 3.0
            threshold = float(
                np.clip(
                    max(otsu * self.config.otsu_scale, contrast_threshold),
                    1,
                    254,
                )
            )
            binary = (blur >= threshold).astype(np.uint8)
        else:
            shadow = float(np.percentile(blur, 0.5))
            middle = float(np.percentile(blur, 50.0))
            if middle - shadow > 35.0:
                contrast_threshold = middle - 0.55 * (middle - shadow)
            else:
                contrast_threshold = shadow - 3.0
            threshold = float(
                np.clip(
                    min(otsu * self.config.otsu_scale, contrast_threshold),
                    1,
                    254,
                )
            )
            binary = (blur <= threshold).astype(np.uint8)
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8)
        )

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        frame_area = float(gray.shape[0] * gray.shape[1])
        min_area = self.config.min_area_fraction * frame_area
        max_area = self.config.max_area_fraction * frame_area
        found: list[Perforation] = []
        roi_height, roi_width = roi.shape[:2]

        for label in range(1, count):
            x, y, width, height, area = stats[label]
            if area < min_area or area > max_area:
                continue
            if height <= 1 or width <= 1:
                continue
            aspect = width / float(height)
            if not self.config.min_aspect <= aspect <= self.config.max_aspect:
                continue
            extent = area / float(width * height)
            if extent < self.config.min_extent:
                continue
            if y <= 0 or y + height >= roi_height:
                continue
            touches_outer_edge = (
                x <= 0 if side == "left" else x + width >= roi_width
            )
            if touches_outer_edge:
                continue

            center_x = float(centroids[label][0])
            center_y = float(centroids[label][1])
            shape_penalty = abs(np.log(max(aspect, 1e-6) / 0.75))
            edge_bonus = 1.0 - 0.25 * abs(center_x / max(roi_width, 1) - 0.5)
            score = (
                np.sqrt(area)
                * extent
                * edge_bonus
                * np.exp(-0.35 * shape_penalty)
            )
            found.append(
                Perforation(
                    side=side,
                    x=center_x + x0,
                    y=center_y + y0,
                    width=int(width),
                    height=int(height),
                    area=float(area),
                    score=float(score),
                )
            )
        return found
