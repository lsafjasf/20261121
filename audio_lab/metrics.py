"""Pre-onset energy decomposition and acceptance classification."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .dsp import mean_square
from .wavio import dbfs


@dataclass(frozen=True)
class MetricSettings:
    window_ms: float = 20.0
    guard_ms: float = 2.0
    attack_ms: float = 50.0
    source_visible_dbfs: float = -70.0
    silence_dbfs: float = -90.0
    added_visible_dbfs: float = -80.0
    source_dominance_db: float = 6.0
    min_delay_confidence: float = 0.80
    floor_db: float = -180.0


@dataclass(frozen=True)
class OnsetMetrics:
    index: int
    onset_s: float
    window_start_s: float
    window_stop_s: float
    reference_pre_energy: float
    decoded_pre_energy: float
    attack_energy: float
    reference_pre_dbfs: float
    decoded_pre_dbfs: float
    codec_added_dbfs: float | None
    level_change_db: float | None
    source_dominance_db: float | None
    reference_pre_to_attack_db: float
    decoded_pre_to_attack_db: float
    classification: str


@dataclass(frozen=True)
class MeasurementSummary:
    onsets: list[OnsetMetrics]
    worst_added_dbfs: float | None
    worst_added_onset_index: int | None
    codec_added_count: int
    source_prelude_count: int
    mixed_count: int
    no_added_energy_count: int


def _ratio_db(numerator: float, denominator: float, floor_db: float) -> float:
    if numerator <= 0.0:
        return -abs(floor_db)
    if denominator <= 0.0:
        return abs(floor_db)
    return 10.0 * math.log10(numerator / denominator)


def classify_onset(
    reference_energy: float,
    decoded_energy: float,
    settings: MetricSettings,
) -> tuple[str, float | None, float | None]:
    added_energy = max(0.0, decoded_energy - reference_energy)
    reference_db = dbfs(reference_energy, settings.floor_db)
    added_db = dbfs(added_energy, settings.floor_db) if added_energy > 0.0 else None
    dominance_db = (
        reference_db - added_db
        if added_db is not None and reference_db > settings.silence_dbfs
        else None
    )
    if reference_db >= settings.source_visible_dbfs:
        if added_db is None or dominance_db >= settings.source_dominance_db:
            return "source-prelude", added_db, dominance_db
    if (
        reference_db <= settings.silence_dbfs
        and added_db is not None
        and added_db >= settings.added_visible_dbfs
    ):
        return "codec-added", added_db, dominance_db
    if added_db is None or added_db < settings.added_visible_dbfs:
        return "no-added-energy", added_db, dominance_db
    return "mixed", added_db, dominance_db


def measure_onsets(
    reference: list[float],
    decoded: list[float],
    sample_rate: int,
    onsets_s: list[float],
    settings: MetricSettings | None = None,
) -> MeasurementSummary:
    settings = settings or MetricSettings()
    if len(reference) != len(decoded):
        raise ValueError("reference and decoded signals must already be aligned")
    window_frames = max(1, round(settings.window_ms * sample_rate / 1000.0))
    guard_frames = max(0, round(settings.guard_ms * sample_rate / 1000.0))
    attack_frames = max(1, round(settings.attack_ms * sample_rate / 1000.0))
    results: list[OnsetMetrics] = []
    for index, onset_s in enumerate(onsets_s, start=1):
        onset_frame = round(onset_s * sample_rate)
        window_stop = max(0, onset_frame - guard_frames)
        window_start = max(0, window_stop - window_frames)
        reference_energy = mean_square(reference, window_start, window_stop)
        decoded_energy = mean_square(decoded, window_start, window_stop)
        attack_energy = mean_square(
            reference, onset_frame, min(len(reference), onset_frame + attack_frames)
        )
        classification, added_db, dominance_db = classify_onset(
            reference_energy, decoded_energy, settings
        )
        reference_db = dbfs(reference_energy, settings.floor_db)
        decoded_db = dbfs(decoded_energy, settings.floor_db)
        results.append(
            OnsetMetrics(
                index=index,
                onset_s=onset_s,
                window_start_s=window_start / sample_rate,
                window_stop_s=window_stop / sample_rate,
                reference_pre_energy=reference_energy,
                decoded_pre_energy=decoded_energy,
                attack_energy=attack_energy,
                reference_pre_dbfs=reference_db,
                decoded_pre_dbfs=decoded_db,
                codec_added_dbfs=added_db,
                level_change_db=(
                    decoded_db - reference_db
                    if reference_db > settings.floor_db
                    else None
                ),
                source_dominance_db=dominance_db,
                reference_pre_to_attack_db=_ratio_db(
                    reference_energy, attack_energy, settings.floor_db
                ),
                decoded_pre_to_attack_db=_ratio_db(
                    decoded_energy, attack_energy, settings.floor_db
                ),
                classification=classification,
            )
        )

    added = [
        (item.codec_added_dbfs, item.index)
        for item in results
        if item.codec_added_dbfs is not None
    ]
    worst = max(added, default=None, key=lambda item: item[0])
    return MeasurementSummary(
        onsets=results,
        worst_added_dbfs=worst[0] if worst else None,
        worst_added_onset_index=worst[1] if worst else None,
        codec_added_count=sum(item.classification == "codec-added" for item in results),
        source_prelude_count=sum(
            item.classification == "source-prelude" for item in results
        ),
        mixed_count=sum(item.classification == "mixed" for item in results),
        no_added_energy_count=sum(
            item.classification == "no-added-energy" for item in results
        ),
    )


def evaluate_acceptance(
    summary: MeasurementSummary,
    expected: str,
    delay_confidence: float,
    settings: MetricSettings | None = None,
) -> dict[str, object]:
    settings = settings or MetricSettings()
    wrong = [
        item.index
        for item in summary.onsets
        if item.classification != expected
    ]
    confidence_ok = delay_confidence >= settings.min_delay_confidence
    passed = not wrong and confidence_ok
    return {
        "passed": passed,
        "expected": expected,
        "confidence_ok": confidence_ok,
        "unexpected_onset_indices": wrong,
        "message": (
            "分类与预期一致"
            if passed
            else (
                f"起音 {wrong} 分类不符" if wrong else "延迟校准置信度不足"
            )
        ),
    }
