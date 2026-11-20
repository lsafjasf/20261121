"""Traceable BWF batch processing with editable gain envelopes.

Pipeline overview:
  - bwf:       Broadcast Wave Format (RIFF + bext) reading/writing.
  - envelope:  Editable gain envelope (segments of control points) with
               explicit overlap-merge semantics and head/tail handling.
  - store:     Version store; every adjustment persists the source samples
               it referenced plus version metadata, independently.
  - render:    Full renders and incremental (affected-range-only) renders.
  - verify:    Fixed-probe comparison of full vs. incremental renders with
               per-sample diffs around splice boundaries.
"""

__version__ = "0.1.0"
