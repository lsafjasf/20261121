"""Freeze diagnostics for fixed-camera video."""

from .analyzer import analyze_video
from .config import DiagnosticConfig

__all__ = ["DiagnosticConfig", "analyze_video"]
