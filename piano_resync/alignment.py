from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from math import exp
from typing import Sequence

from .config import SyncConfig
from .drift import AffineTransform
from .model import Note, Tap
from .normalize import index_window, robust_unit


@dataclass(frozen=True)
class Match:
    note_index: int
    tap_index: int
    cost: float


@dataclass(frozen=True)
class Alignment:
    matches: tuple[Match, ...]
    note_indices: tuple[int, ...]
    tap_indices: tuple[int, ...]


def _pair_cost(
    residual: float,
    velocity_difference: float,
    config: SyncConfig,
    dynamics_confidence: float,
) -> float:
    scaled_time = residual / config.timing_tolerance
    time_cost = min(scaled_time * scaled_time, 64.0)
    dynamic_cost = (
        config.velocity_weight
        * dynamics_confidence
        * velocity_difference
        * velocity_difference
    )
    return time_cost + dynamic_cost


def align_notes(
    notes: Sequence[Note],
    taps: Sequence[Tap],
    transform: AffineTransform,
    config: SyncConfig,
    *,
    use_dynamics: bool = True,
) -> Alignment:
    """Monotonically align note onsets to taps, permitting skipped events."""

    note_count = len(notes)
    tap_count = len(taps)
    if not note_count or not tap_count:
        return Alignment((), tuple(range(note_count)), tuple(range(tap_count)))

    cells = (note_count + 1) * (tap_count + 1)
    if cells > config.max_alignment_cells:
        raise ValueError(
            "alignment matrix is too large "
            f"({note_count} notes x {tap_count} taps); raise max_alignment_cells "
            "or split the recording"
        )

    tap_times = [tap.time for tap in taps]
    note_velocity, note_span = robust_unit([note.velocity for note in notes])
    tap_strength, tap_span = robust_unit([tap.strength for tap in taps])
    dynamics_confidence = min(1.0, note_span / 16.0, tap_span / 0.20)
    if not use_dynamics:
        dynamics_confidence = 0.0

    skip_note = config.skip_cost
    skip_tap = config.skip_cost
    previous = [index * skip_tap for index in range(tap_count + 1)]
    backtrack = [bytearray(tap_count + 1) for _ in range(note_count + 1)]
    for tap_index in range(1, tap_count + 1):
        backtrack[0][tap_index] = 1

    for note_index, note in enumerate(notes, start=1):
        predicted = transform.apply(note.start)
        low_index, high_index = index_window(
            tap_times,
            predicted - config.match_window,
            predicted + config.match_window + 1e-12,
        )
        pair_costs: dict[int, float] = {}
        for tap_index in range(low_index, high_index):
            residual = abs(predicted - taps[tap_index].time)
            velocity_difference = abs(
                note_velocity[note_index - 1] - tap_strength[tap_index]
            )
            pair_costs[tap_index] = _pair_cost(
                residual, velocity_difference, config, dynamics_confidence
            )

        current = [note_index * skip_note] + [0.0] * tap_count
        row = backtrack[note_index]
        row[0] = 1
        for tap_index in range(1, tap_count + 1):
            best_cost = previous[tap_index] + skip_note
            direction = 0
            skip_tap_cost = current[tap_index - 1] + skip_tap
            if skip_tap_cost < best_cost:
                best_cost = skip_tap_cost
                direction = 1
            pair_cost = pair_costs.get(tap_index - 1)
            if pair_cost is not None:
                match_cost = previous[tap_index - 1] + pair_cost
                if match_cost < best_cost:
                    best_cost = match_cost
                    direction = 2
            current[tap_index] = best_cost
            row[tap_index] = direction
        previous = current

    matches: list[Match] = []
    note_index = note_count
    tap_index = tap_count
    while note_index > 0 or tap_index > 0:
        direction = backtrack[note_index][tap_index]
        if note_index > 0 and tap_index > 0 and direction == 2:
            predicted = transform.apply(notes[note_index - 1].start)
            residual = abs(predicted - taps[tap_index - 1].time)
            velocity_difference = abs(
                note_velocity[note_index - 1] - tap_strength[tap_index - 1]
            )
            matches.append(
                Match(
                    note_index=note_index - 1,
                    tap_index=tap_index - 1,
                    cost=_pair_cost(
                        residual,
                        velocity_difference,
                        config,
                        dynamics_confidence,
                    ),
                )
            )
            note_index -= 1
            tap_index -= 1
        elif note_index > 0 and (tap_index == 0 or direction == 0):
            note_index -= 1
        else:
            tap_index -= 1
    matches.reverse()
    matched_notes = {match.note_index for match in matches}
    matched_taps = {match.tap_index for match in matches}
    return Alignment(
        matches=tuple(matches),
        note_indices=tuple(
            index for index in range(note_count) if index not in matched_notes
        ),
        tap_indices=tuple(
            index for index in range(tap_count) if index not in matched_taps
        ),
    )


def estimate_initial_transform(
    notes: Sequence[Note], taps: Sequence[Tap], config: SyncConfig
) -> AffineTransform:
    """Coarse (rate, offset) search before dynamic-programming alignment."""

    if not notes or not taps:
        raise ValueError("at least one note and one tap are required")

    note_velocity, _ = robust_unit([note.velocity for note in notes])
    tap_strength, _ = robust_unit([tap.strength for tap in taps])
    tap_times = [tap.time for tap in taps]
    bin_width = max(0.004, config.timing_tolerance / 4.0)
    half_steps = config.rate_steps // 2
    rate_candidates = [
        1.0 + (step / half_steps) * config.max_rate_error
        for step in range(-half_steps, half_steps + 1)
    ]

    best_transform = AffineTransform(1.0, 0.0)
    best_score = float("-inf")
    for rate in rate_candidates:
        votes: dict[int, float] = defaultdict(float)
        for note_index, note in enumerate(notes):
            center = rate * note.start
            low = center - config.initial_offset_window
            high = center + config.initial_offset_window
            low_index, high_index = index_window(tap_times, low, high)
            for tap_index in range(low_index, high_index):
                delta = tap_times[tap_index] - center
                velocity_similarity = 1.0 - abs(
                    note_velocity[note_index] - tap_strength[tap_index]
                )
                weight = 0.20 + velocity_similarity
                key = round(delta / bin_width)
                votes[key - 1] += weight * 0.25
                votes[key] += weight
                votes[key + 1] += weight * 0.25

        if not votes:
            continue
        candidate_bins = sorted(votes, key=votes.get, reverse=True)[:8]
        for key in candidate_bins:
            offset = key * bin_width
            candidate = AffineTransform(rate, offset)
            score = _coarse_alignment_score(
                notes,
                taps,
                candidate,
                note_velocity,
                tap_strength,
                config,
            )
            if score > best_score:
                best_score = score
                best_transform = candidate

    if best_score == float("-inf") or best_score <= 0:
        offset = taps[0].time - notes[0].start
        return AffineTransform(1.0, offset)
    return best_transform


def _coarse_alignment_score(
    notes: Sequence[Note],
    taps: Sequence[Tap],
    transform: AffineTransform,
    note_velocity: Sequence[float],
    tap_strength: Sequence[float],
    config: SyncConfig,
) -> float:
    tap_times = [tap.time for tap in taps]
    used: set[int] = set()
    score = 0.0
    for note_index, note in enumerate(notes):
        predicted = transform.apply(note.start)
        low_index, high_index = index_window(
            tap_times,
            predicted - config.match_window,
            predicted + config.match_window + 1e-12,
        )
        candidates = [
            tap_index for tap_index in range(low_index, high_index)
            if tap_index not in used
        ]
        if not candidates:
            continue
        tap_index = min(
            candidates,
            key=lambda index: (
                abs(predicted - taps[index].time) / config.timing_tolerance
                + abs(note_velocity[note_index] - tap_strength[index])
            ),
        )
        residual = abs(predicted - taps[tap_index].time)
        velocity_difference = abs(
            note_velocity[note_index] - tap_strength[tap_index]
        )
        quality = exp(-0.5 * (residual / config.timing_tolerance) ** 2)
        quality *= 1.0 - 0.5 * velocity_difference
        score += 0.50 + quality
        used.add(tap_index)
    return score
