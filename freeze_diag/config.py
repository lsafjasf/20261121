from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DiagnosticConfig:
    """Thresholds for the image and timestamp evidence."""

    roi_margin: float = 0.02
    block_size: int = 16
    noise_sigma: float = 1.2
    duplicate_mad: float = 0.15
    duplicate_p99: float = 2.5
    max_duplicate_gap: int = 1
    noise_floor: float = 0.03
    black_luma: float = 14.0
    black_contrast: float = 6.0
    black_detail: float = 2.0
    min_freeze_ms: float = 1000.0
    timestamp_tolerance_ms: float = 3.0
    timestamp_tolerance_ratio: float = 0.20

    def to_dict(self) -> dict:
        return asdict(self)
