from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SyncConfig:
    """Tuning parameters for anchor matching and drift estimation.

    All times are seconds. ``timing_tolerance`` controls how strongly a small
    residual is preferred; ``match_window`` is the hard search radius for a
    MIDI-note/audio-onset pair.
    """

    timing_tolerance: float = 0.025
    match_window: float = 0.150
    max_rate_error: float = 0.020
    rate_steps: int = 41
    initial_offset_window: float = 2.0
    skip_cost: float = 0.75
    velocity_weight: float = 0.40
    outlier_sigma: float = 4.0
    min_anchors: int = 3
    max_iterations: int = 5
    max_alignment_cells: int = 30_000_000

    def __post_init__(self) -> None:
        if self.timing_tolerance <= 0:
            raise ValueError("timing_tolerance must be positive")
        if self.match_window < self.timing_tolerance:
            raise ValueError("match_window must be >= timing_tolerance")
        if self.max_rate_error <= 0 or self.max_rate_error >= 0.5:
            raise ValueError("max_rate_error must be between 0 and 0.5")
        if self.rate_steps < 3:
            raise ValueError("rate_steps must be at least 3")
        if self.initial_offset_window < self.match_window:
            raise ValueError("initial_offset_window must be >= match_window")
        if self.skip_cost <= 0:
            raise ValueError("skip_cost must be positive")
        if not 0 <= self.velocity_weight <= 4:
            raise ValueError("velocity_weight must be between 0 and 4")
        if self.outlier_sigma <= 0:
            raise ValueError("outlier_sigma must be positive")
        if self.min_anchors < 2:
            raise ValueError("min_anchors must be at least 2")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.max_alignment_cells < 1:
            raise ValueError("max_alignment_cells must be positive")

