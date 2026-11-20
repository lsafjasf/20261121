from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Mapping

from .alignment import Match, align_notes, estimate_initial_transform
from .config import SyncConfig
from .drift import AffineTransform, Fit, robust_fit
from .model import Note, Recording, Tap


@dataclass(frozen=True)
class MappedNote:
    note: int
    start: float
    end: float
    velocity: int
    source_start: float
    source_end: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "note": self.note,
            "start": self.start,
            "end": self.end,
            "velocity": self.velocity,
            "source_start": self.source_start,
            "source_end": self.source_end,
        }


@dataclass(frozen=True)
class MappedPedal:
    kind: str
    start: float
    end: float
    value: float
    source_start: float
    source_end: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "kind": self.kind,
            "start": self.start,
            "end": self.end,
            "value": self.value,
            "source_start": self.source_start,
            "source_end": self.source_end,
        }


@dataclass(frozen=True)
class Anchor:
    source_time: float
    target_time: float
    mapped_time: float
    note: int
    velocity: int
    strength: float
    residual_ms: float
    inlier: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "source_time": self.source_time,
            "target_time": self.target_time,
            "mapped_time": self.mapped_time,
            "note": self.note,
            "velocity": self.velocity,
            "strength": self.strength,
            "residual_ms": self.residual_ms,
            "inlier": self.inlier,
        }


@dataclass(frozen=True)
class SyncResult:
    transform: AffineTransform
    notes: tuple[MappedNote, ...]
    pedals: tuple[MappedPedal, ...]
    anchors: tuple[Anchor, ...]
    comparisons: tuple[dict[str, float | int], ...]
    unmatched_notes: tuple[dict[str, float | int], ...]
    unmatched_taps: tuple[dict[str, float], ...]
    statistics: dict[str, float | int]
    warnings: tuple[str, ...]
    config: SyncConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeline": "audio",
            "transform": self.transform.to_dict(),
            "notes": [note.to_dict() for note in self.notes],
            "pedals": [pedal.to_dict() for pedal in self.pedals],
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "diagnostics": {
                **self.statistics,
                "resynthesized_onsets": [
                    {
                        "note": note.note,
                        "time": note.start,
                        "velocity": note.velocity,
                    }
                    for note in self.notes
                ],
                "comparisons": list(self.comparisons),
                "unmatched_notes": list(self.unmatched_notes),
                "unmatched_taps": list(self.unmatched_taps),
                "warnings": list(self.warnings),
            },
        }


def _fit_matches(
    notes: tuple[Note, ...],
    taps: tuple[Tap, ...],
    matches: tuple[Match, ...],
    transform: AffineTransform,
    config: SyncConfig,
) -> Fit:
    source_times = [notes[match.note_index].start for match in matches]
    target_times = [taps[match.tap_index].time for match in matches]
    return robust_fit(source_times, target_times, transform, config)


def _converged(left: AffineTransform, right: AffineTransform) -> bool:
    return (
        abs(left.rate - right.rate) < 1e-9
        and abs(left.offset - right.offset) < 1e-7
    )


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or len(left) != len(right):
        return 0.0
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    left_energy = sum((a - left_mean) ** 2 for a in left)
    right_energy = sum((b - right_mean) ** 2 for b in right)
    denominator = (left_energy * right_energy) ** 0.5
    return numerator / denominator if denominator else 0.0


def resynchronize(
    recording: Recording | Mapping[str, Any],
    config: SyncConfig | None = None,
) -> SyncResult:
    """Estimate MIDI-to-audio drift and rewrite notes/pedals to audio time."""

    config = config or SyncConfig()
    if isinstance(recording, Mapping):
        recording = Recording.from_dict(recording)
    notes = recording.notes
    taps = recording.taps
    if not notes:
        raise ValueError("at least one note onset is required")
    if not taps:
        raise ValueError("at least one audio tap is required")

    transform = estimate_initial_transform(notes, taps, config)
    alignment = align_notes(notes, taps, transform, config)
    fit = _fit_matches(notes, taps, alignment.matches, transform, config)
    for _ in range(config.max_iterations - 1):
        next_transform = fit.transform
        if _converged(transform, next_transform):
            break
        transform = next_transform
        alignment = align_notes(notes, taps, transform, config)
        fit = _fit_matches(notes, taps, alignment.matches, transform, config)
    transform = fit.transform

    anchors: list[Anchor] = []
    for index, match in enumerate(alignment.matches):
        note = notes[match.note_index]
        tap = taps[match.tap_index]
        mapped = transform.apply(note.start)
        residual = mapped - tap.time
        anchors.append(
            Anchor(
                source_time=note.start,
                target_time=tap.time,
                mapped_time=mapped,
                note=note.note,
                velocity=note.velocity,
                strength=tap.strength,
                residual_ms=residual * 1000.0,
                inlier=fit.inliers[index] if index < len(fit.inliers) else False,
            )
        )

    mapped_notes = tuple(
        sorted(
            (
                MappedNote(
                    note=note.note,
                    start=transform.apply(note.start),
                    end=transform.apply(note.end),
                    velocity=note.velocity,
                    source_start=note.start,
                    source_end=note.end,
                )
                for note in notes
            ),
            key=lambda item: (item.start, item.note),
        )
    )
    mapped_pedals = tuple(
        MappedPedal(
            kind=pedal.kind,
            start=transform.apply(pedal.start),
            end=transform.apply(pedal.end),
            value=pedal.value,
            source_start=pedal.start,
            source_end=pedal.end,
        )
        for pedal in recording.pedals
    )

    comparisons = tuple(
        {
            "note": anchor.note,
            "velocity": anchor.velocity,
            "strength": anchor.strength,
            "source_onset": anchor.source_time,
            "resynthesized_onset": anchor.mapped_time,
            "true_tap": anchor.target_time,
            "error_ms": anchor.residual_ms,
            "inlier": anchor.inlier,
        }
        for anchor in anchors
    )
    unmatched_notes = tuple(
        {
            "note": notes[index].note,
            "source_onset": notes[index].start,
            "mapped_onset": transform.apply(notes[index].start),
            "velocity": notes[index].velocity,
        }
        for index in alignment.note_indices
    )
    unmatched_taps = tuple(
        {"time": taps[index].time, "strength": taps[index].strength}
        for index in alignment.tap_indices
    )

    inlier_anchors = [anchor for anchor in anchors if anchor.inlier]
    abs_errors = [abs(anchor.residual_ms) for anchor in inlier_anchors]
    all_abs_errors = [abs(anchor.residual_ms) for anchor in anchors]
    matched_velocities = [float(anchor.velocity) for anchor in inlier_anchors]
    matched_strengths = [anchor.strength for anchor in inlier_anchors]
    anchor_span = (
        max(anchor.source_time for anchor in inlier_anchors)
        - min(anchor.source_time for anchor in inlier_anchors)
        if inlier_anchors
        else 0.0
    )
    statistics: dict[str, float | int] = {
        "anchor_count": len(anchors),
        "inlier_count": len(inlier_anchors),
        "outlier_count": len(anchors) - len(inlier_anchors),
        "unmatched_note_count": len(unmatched_notes),
        "unmatched_tap_count": len(unmatched_taps),
        "anchor_source_span_seconds": anchor_span,
        "mean_abs_error_ms": mean(abs_errors) if abs_errors else 0.0,
        "median_abs_error_ms": _percentile(abs_errors, 0.50),
        "p95_abs_error_ms": _percentile(abs_errors, 0.95),
        "max_abs_error_ms": max(abs_errors, default=0.0),
        "mean_abs_error_all_anchors_ms": (
            mean(all_abs_errors) if all_abs_errors else 0.0
        ),
        "signed_bias_ms": (
            mean(anchor.residual_ms for anchor in inlier_anchors)
            if inlier_anchors
            else 0.0
        ),
        "velocity_strength_correlation": _correlation(
            matched_velocities, matched_strengths
        ),
        "anchor_velocity_span": (
            max(matched_velocities) - min(matched_velocities)
            if matched_velocities
            else 0.0
        ),
    }
    warnings: list[str] = []
    if len(inlier_anchors) < config.min_anchors:
        warnings.append(
            f"only {len(inlier_anchors)} reliable anchors; at least "
            f"{config.min_anchors} are recommended for stable drift estimation"
        )
    if 0 < anchor_span < 0.5:
        warnings.append(
            "anchor span is shorter than 0.5 s; the clock-rate estimate is weakly constrained"
        )
    if statistics["anchor_velocity_span"] < 8:
        warnings.append(
            "low velocity variation across anchors; matching relied mainly on timing"
        )
    if abs(transform.rate - 1.0) > config.max_rate_error * 1.25:
        warnings.append(
            f"estimated rate {transform.rate:.8f} exceeds the configured search range"
        )
    if unmatched_notes:
        warnings.append(
            f"{len(unmatched_notes)} note onset(s) had no tap within "
            f"{config.match_window:.3f} s; they were still rewritten with the fitted drift"
        )
    if unmatched_taps:
        warnings.append(
            f"{len(unmatched_taps)} audio tap(s) were not paired; they may be extra "
            "transients or MIDI note-on dropouts"
        )
    first_source = min(notes, key=lambda note: note.start).start
    last_source = max(note.end for note in notes)
    boundary_taps = [
        tap
        for tap in taps
        if tap.time < transform.apply(first_source) - config.match_window
        or tap.time > transform.apply(last_source) + config.match_window
    ]
    if boundary_taps:
        warnings.append(
            f"{len(boundary_taps)} tap(s) lie outside the mapped note span; "
            "check recording head/tail edge alignment"
        )
    last_onset = max(note.start for note in notes)
    edge_anchors = [
        anchor
        for anchor in inlier_anchors
        if anchor.source_time <= first_source + config.timing_tolerance
        or anchor.source_time >= last_onset - config.timing_tolerance
    ]
    if any(
        abs(anchor.residual_ms) > config.timing_tolerance * 2000.0
        for anchor in edge_anchors
    ):
        warnings.append("edge anchors disagree with the fitted drift line")

    return SyncResult(
        transform=transform,
        notes=mapped_notes,
        pedals=mapped_pedals,
        anchors=tuple(anchors),
        comparisons=comparisons,
        unmatched_notes=unmatched_notes,
        unmatched_taps=unmatched_taps,
        statistics=statistics,
        warnings=tuple(warnings),
        config=config,
    )
