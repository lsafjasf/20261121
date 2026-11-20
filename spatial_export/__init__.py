"""Fully offline multi-channel spatial audio exporter (pure stdlib)."""

from .audio import RenderConfig
from .exporter import ExportResult, export_layout
from .layouts import LAYOUTS, LAYOUT_ASYM, LAYOUT_CUBE, Layout
from .panner import VbapPanner
from .trajectory import Trajectory, demo_trajectory
from .verifier import VerificationReport, verify_all

__all__ = [
    "RenderConfig",
    "ExportResult",
    "export_layout",
    "LAYOUTS",
    "LAYOUT_ASYM",
    "LAYOUT_CUBE",
    "Layout",
    "VbapPanner",
    "Trajectory",
    "demo_trajectory",
    "VerificationReport",
    "verify_all",
]
