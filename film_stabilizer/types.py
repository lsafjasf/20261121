from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Perforation:
    side: str
    x: float
    y: float
    width: int
    height: int
    area: float
    score: float


@dataclass
class FrameRecord:
    index: int
    source_name: str
    original_frame: int
    detections: list[Perforation] = field(default_factory=list)
    matrix: np.ndarray | None = None
    confidence: int = 0
    status: str = "unprocessed"
    left_observed: bool = False
    right_observed: bool = False
    left_anchor: tuple[float, float] | None = None
    right_anchor: tuple[float, float] | None = None
    is_splice: bool = False
    crop: tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass
class SequenceInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    source: str


def rigid_matrix(angle_rad: float, tx: float, ty: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array(
        [[c, -s, tx], [s, c, ty], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def apply_matrix(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    return (matrix @ hom.T).T[:, :2]


def matrix_translation(matrix: np.ndarray) -> tuple[float, float]:
    return float(matrix[0, 2]), float(matrix[1, 2])


def matrix_angle_deg(matrix: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))


