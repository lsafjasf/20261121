"""Core data structures and tunable configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class Utterance:
    """One manually labelled speech interval on a speaker's own track."""

    speaker_id: str
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(
                f"utterance end {self.end} precedes start {self.start}"
            )

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class BuildConfig:
    """Parameters controlling merging, context padding and cut placement."""

    merge_gap: float = 1.2
    short_utterance: float = 2.0
    context_pre: float = 0.4
    context_post: float = 0.4
    energy_window: float = 0.08
    energy_search: float = 0.3

    def validate(self) -> None:
        for name in (
            "merge_gap",
            "short_utterance",
            "context_pre",
            "context_post",
            "energy_window",
            "energy_search",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass
class Clip:
    """A produced relisten clip and its exact mapping back to the source."""

    clip_id: str
    speaker_id: str
    group_index: int
    source_path: str
    sample_rate: int
    source_start_sample: int
    source_end_sample: int
    clip_start_sample: int = 0
    clip_end_sample: int = 0
    start: float = 0.0
    end: float = 0.0
    utterance_indices: list[int] = field(default_factory=list)
    merged_intervals: list[dict[str, float]] = field(default_factory=list)
    overlaps: list[dict[str, Any]] = field(default_factory=list)
    output_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["clip_duration"] = (
            self.clip_end_sample - self.clip_start_sample
        ) / self.sample_rate
        data["mapping"] = {
            "source_path": self.source_path,
            "sample_rate": self.sample_rate,
            "source_start_sample": self.source_start_sample,
            "source_end_sample": self.source_end_sample,
            "clip_start_sample": self.clip_start_sample,
            "clip_end_sample": self.clip_end_sample,
            "start": self.start,
            "end": self.end,
            "sample_exact": True,
        }
        return data
