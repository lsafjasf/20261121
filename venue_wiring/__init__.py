"""Offline analysis of source-to-loudspeaker wiring from eight returns."""

from .analyze import AnalysisConfig, analyze
from .synth import GenerationConfig, generate_scenario

__all__ = ["AnalysisConfig", "GenerationConfig", "analyze", "generate_scenario"]
__version__ = "0.1.0"

