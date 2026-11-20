"""Perforation-based stabilization for frame-by-frame scanned film."""

from .detector import DetectorConfig, Perforation, PerforationDetector
from .stabilizer import StabilizerConfig, StabilizationResult, stabilize

__all__ = [
    "DetectorConfig",
    "Perforation",
    "PerforationDetector",
    "StabilizerConfig",
    "StabilizationResult",
    "stabilize",
]

__version__ = "0.1.0"
