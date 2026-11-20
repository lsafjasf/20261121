"""Merge neighbouring short utterances while preserving crosstalk sources.

Merging is performed per speaker.  Two consecutive intervals merge when the
gap between them is no larger than ``merge_gap`` and at least one side is a
short utterance (``<= short_utterance``).  Overlapping/abutting intervals from
the same speaker are always defensively unioned.

Crosstalk is never used to merge audio: intervals of *other* speakers that
intersect a group are recorded as overlap evidence so the origin of an
interruption stays attributable in the manifest.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import BuildConfig, Utterance


@dataclass
class UtteranceGroup:
    speaker_id: str
    indices: list[int]
    intervals: list[tuple[float, float]]

    @property
    def start(self) -> float:
        return self.intervals[0][0]

    @property
    def end(self) -> float:
        return self.intervals[-1][1]


def _should_merge(
    previous: tuple[float, float],
    current: tuple[float, float],
    config: BuildConfig,
) -> bool:
    gap = current[0] - previous[1]
    if gap <= 0:
        return True
    if gap > config.merge_gap:
        return False
    previous_short = (previous[1] - previous[0]) <= config.short_utterance
    current_short = (current[1] - current[0]) <= config.short_utterance
    return previous_short or current_short


def group_speaker(
    speaker_id: str,
    speaker_utts: list[tuple[int, Utterance]],
    config: BuildConfig,
) -> list[UtteranceGroup]:
    ordered = sorted(speaker_utts, key=lambda pair: pair[1].start)
    groups: list[UtteranceGroup] = []
    for index, utterance in ordered:
        interval = (utterance.start, utterance.end)
        if groups and _should_merge(
            groups[-1].intervals[-1], interval, config
        ):
            groups[-1].indices.append(index)
            groups[-1].intervals.append(interval)
        else:
            groups.append(
                UtteranceGroup(speaker_id, [index], [interval])
            )

    # Normalise any unioned (overlapping same-speaker) group bounds while
    # keeping the original labelled intervals for the mapping output.
    for group in groups:
        group.intervals = sorted(group.intervals)
    return groups


def merge_utterances(
    utterances: list[Utterance], config: BuildConfig
) -> dict[str, list[UtteranceGroup]]:
    config.validate()
    per_speaker: dict[str, list[tuple[int, Utterance]]] = {}
    for index, utterance in enumerate(utterances):
        per_speaker.setdefault(utterance.speaker_id, []).append(
            (index, utterance)
        )
    return {
        speaker_id: group_speaker(speaker_id, items, config)
        for speaker_id, items in sorted(per_speaker.items())
    }


def find_overlaps(
    group: UtteranceGroup, all_utterances: list[Utterance]
) -> list[dict[str, float | str]]:
    """Return labelled intervals of other speakers crossing this group."""

    overlaps: list[dict[str, float | str]] = []
    for index, utterance in enumerate(all_utterances):
        if utterance.speaker_id == group.speaker_id:
            continue
        overlap_start = max(utterance.start, group.start)
        overlap_end = min(utterance.end, group.end)
        if overlap_start < overlap_end:
            overlaps.append(
                {
                    "utterance_index": index,
                    "speaker_id": utterance.speaker_id,
                    "start": overlap_start,
                    "end": overlap_end,
                    "kind": "cross_talk",
                }
            )
    return overlaps
