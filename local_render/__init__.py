"""Local OTIO/FFmpeg rendering entry point.

Two video tracks plus their audio tracks are supported. Clips may start at a
negative source time (an explicit black/silence prefix), and SMPTE dissolve
transitions require real media handles. Missing handles raise
:class:`~local_render.errors.MissingHandleError` instead of being silently
replaced by freeze frames.
"""

from local_render.errors import LocalRenderError, MissingHandleError
from local_render.planner import RenderPlan, build_plan

__all__ = [
    "LocalRenderError",
    "MissingHandleError",
    "RenderPlan",
    "build_plan",
]
