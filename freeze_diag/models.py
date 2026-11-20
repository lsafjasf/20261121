from dataclasses import dataclass, field
from typing import Any


@dataclass
class FrameFeatures:
    index: int
    pts_ms: float | None
    luma_mean: float
    contrast: float
    detail: float
    temporal_mad: float | None
    temporal_p99: float | None
    local_noise: float | None
    duplicate: bool
    black: bool


@dataclass
class TimestampSummary:
    frame_count: int
    available_count: int
    missing_count: int
    nominal_delta_ms: float | None
    continuity: str
    regular_ratio: float
    gap_count: int
    backstep_count: int
    duplicate_pts_count: int
    first_ms: float | None
    last_ms: float | None
    median_delta_ms: float | None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class Interval:
    status: str
    start_frame: int
    end_frame: int
    start_ms: float | None
    end_ms: float | None
    duration_ms: float
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": round(self.duration_ms, 3),
            "evidence": self.evidence,
        }
