from __future__ import annotations

import cv2
import numpy as np

from .config import DiagnosticConfig
from .models import FrameFeatures


class FeatureExtractor:
    def __init__(self, config: DiagnosticConfig):
        self.config = config
        self._previous: np.ndarray | None = None

    def process(self, frame: np.ndarray, index: int, pts_ms: float | None) -> FrameFeatures:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = self._crop_roi(gray)
        luma_mean = float(np.mean(roi))
        p05, p95 = np.percentile(roi, (5, 95))
        contrast = float(p95 - p05)

        gray_f = roi.astype(np.float32)
        smooth = cv2.GaussianBlur(gray_f, (0, 0), self.config.noise_sigma)
        spatial_hp = np.abs(gray_f - smooth)
        detail = float(np.percentile(spatial_hp, 90))
        black = (
            luma_mean < self.config.black_luma
            and contrast < self.config.black_contrast
            and detail < self.config.black_detail
        )

        temporal_mad = None
        temporal_p99 = None
        local_noise = None
        duplicate = False
        if self._previous is not None:
            diff = gray_f - self._previous
            diff_abs = np.abs(diff)
            temporal_mad = float(np.mean(diff_abs))
            temporal_p99 = float(np.percentile(diff_abs, 99))
            local_noise = self._local_temporal_noise(diff)
            duplicate = (
                temporal_mad <= self.config.duplicate_mad
                and temporal_p99 <= self.config.duplicate_p99
            )
        self._previous = gray_f

        return FrameFeatures(
            index=index,
            pts_ms=pts_ms,
            luma_mean=luma_mean,
            contrast=contrast,
            detail=detail,
            temporal_mad=temporal_mad,
            temporal_p99=temporal_p99,
            local_noise=local_noise,
            duplicate=duplicate,
            black=black,
        )

    def _crop_roi(self, gray: np.ndarray) -> np.ndarray:
        height, width = gray.shape[:2]
        margin_y = int(round(height * self.config.roi_margin))
        margin_x = int(round(width * self.config.roi_margin))
        if margin_y or margin_x:
            return gray[margin_y : height - margin_y, margin_x : width - margin_x]
        return gray

    def _local_temporal_noise(self, diff: np.ndarray) -> float:
        """Median per-tile residual, robust to isolated edges and moving objects."""

        residual = np.abs(diff)
        block = self.config.block_size
        height = (residual.shape[0] // block) * block
        width = (residual.shape[1] // block) * block
        if height < block or width < block:
            return float(np.median(residual))
        tiles = residual[:height, :width].reshape(
            height // block, block, width // block, block
        )
        tile_scores = np.mean(tiles, axis=(1, 3))
        return float(np.median(tile_scores))
