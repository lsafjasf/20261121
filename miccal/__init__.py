"""Four-microphone swept-sine calibration toolkit."""

from .config import Scene, SceneError
from .estimate import CalibrationResult, analyze_recordings
from .simulate import simulate_scene

__all__ = [
    "CalibrationResult",
    "Scene",
    "SceneError",
    "analyze_recordings",
    "simulate_scene",
]

__version__ = "0.1.0"
