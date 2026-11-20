from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import math
from pathlib import Path
from typing import Any, Sequence

from .config import Scene
from .dsp import (
    Peak,
    deconvolve_sweep,
    detect_peaks,
    exponential_sweep,
    read_wav,
)
from .geometry import all_permutations, direct_distances, predicted_relative_delays


class AnalysisError(ValueError):
    pass


@dataclass(frozen=True)
class SourcePeaks:
    source_name: str
    reference_channel: int
    candidates: tuple[tuple[Peak, ...], ...]
    chosen: tuple[Peak, ...]

    @property
    def relative_delays_s(self) -> tuple[float, ...]:
        base = self.chosen[self.reference_channel].delay_s
        return tuple(peak.delay_s - base for peak in self.chosen)


@dataclass(frozen=True)
class MapCandidate:
    order: tuple[int, int, int, int]
    score_samples: float
    source_choices: tuple[tuple[Peak, ...], ...]


@dataclass
class CalibrationResult:
    scene: Scene
    source_estimates: list[SourcePeaks]
    best_map: tuple[int, int, int, int]
    map_candidates: list[MapCandidate]
    map_ambiguous: bool
    relative_delays_s: tuple[float, ...]
    delay_residual_mm: tuple[float, ...]
    gain_relative: tuple[float, ...]
    polarity: tuple[float, ...]
    findings: list[dict[str, str]] = field(default_factory=list)

    @property
    def status(self) -> str:
        if any(item["severity"] == "error" for item in self.findings):
            return "fail"
        if any(item["severity"] == "warning" for item in self.findings):
            return "attention"
        return "ok"

    @property
    def gain_error_db(self) -> tuple[float, ...]:
        result = []
        for value in self.gain_relative:
            magnitude = max(abs(value), 1e-12)
            result.append(20.0 * math.log10(magnitude))
        return tuple(result)

    @property
    def ambiguous_maps(self) -> list[tuple[int, int, int, int]]:
        return [candidate.order for candidate in self.map_candidates]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene": self.scene.name,
            "status": self.status,
            "map": list(self.best_map),
            "map_ambiguous": self.map_ambiguous,
            "ambiguous_maps": [list(order) for order in self.ambiguous_maps],
            "relative_delays_s": list(self.relative_delays_s),
            "delay_residual_mm": list(self.delay_residual_mm),
            "gain_relative": list(self.gain_relative),
            "gain_error_db": list(self.gain_error_db),
            "polarity": list(self.polarity),
            "findings": self.findings,
            "sources": [
                {
                    "name": estimate.source_name,
                    "relative_delays_s": list(estimate.relative_delays_s),
                    "chosen_peaks": [
                        {
                            "delay_s": peak.delay_s,
                            "signed_amplitude": peak.signed_amplitude,
                            "envelope": peak.envelope,
                            "energy": peak.energy,
                        }
                        for peak in estimate.chosen
                    ],
                    "candidate_count": [len(items) for items in estimate.candidates],
                }
                for estimate in self.source_estimates
            ],
        }


def _load_candidates(
    scene: Scene,
    recordings_dir: Path,
) -> list[list[tuple[Peak, ...]]]:
    sweep = exponential_sweep(scene.sweep, scene.sample_rate)
    all_candidates: list[list[tuple[Peak, ...]]] = []
    for source in scene.sources:
        path = recordings_dir / f"{source.name}.wav"
        if not path.exists():
            raise AnalysisError(f"missing recording: {path}")
        channels, sample_rate = read_wav(path)
        if sample_rate != scene.sample_rate:
            raise AnalysisError(
                f"{path}: sample rate is {sample_rate}, expected {scene.sample_rate}"
            )
        if len(channels) != 4:
            raise AnalysisError(f"{path}: expected four channels, got {len(channels)}")
        per_channel: list[tuple[Peak, ...]] = []
        for channel in channels:
            ir = deconvolve_sweep(
                channel,
                sweep,
                sample_rate,
                scene.sweep.f_start,
                scene.sweep.f_stop,
            )
            peaks = detect_peaks(
                ir,
                sample_rate=sample_rate,
                max_delay_s=scene.analysis.max_delay_s,
                min_separation_s=max(0.0009, 0.75 / scene.sweep.f_start),
                count=scene.analysis.candidate_count,
                relative_threshold=scene.analysis.peak_rel_threshold,
            )
            if not peaks:
                raise AnalysisError(f"{path}: no arrival peaks detected")
            per_channel.append(tuple(peaks))
        all_candidates.append(per_channel)
    return all_candidates


def _source_options(
    scene: Scene,
    source_position: tuple[float, float, float],
    candidates_for_source: Sequence[Sequence[Peak]],
    order: tuple[int, int, int, int],
    limit: int,
) -> list[tuple[float, tuple[Peak, ...], tuple[float, ...]]]:
    mics = tuple(scene.microphones[index] for index in order)
    reference = scene.reference_channel
    predicted = predicted_relative_delays(
        mics,
        [source_position],
        reference,
        scene.sound_speed,
    )[0]
    options: list[tuple[float, tuple[Peak, ...], tuple[float, ...]]] = []
    for combination in itertools.product(*candidates_for_source):
        observed = tuple(
            peak.delay_s - combination[reference].delay_s
            for peak in combination
        )
        residuals = tuple(
            (obs - pred) * scene.sample_rate
            for obs, pred in zip(observed, predicted)
        )
        score = math.sqrt(
            sum(value * value for value in residuals) / len(residuals)
        )
        options.append((score, tuple(combination), residuals))
    options.sort(key=lambda item: item[0])
    return options[:limit]


def _joint_score(
    options: Sequence[tuple[float, tuple[Peak, ...], tuple[float, ...]]],
) -> tuple[float, tuple[tuple[Peak, ...], ...]]:
    count = len(options)
    offsets = [
        sum(option[2][channel] for option in options) / count
        for channel in range(4)
    ]
    squared = 0.0
    for option in options:
        squared += sum(
            (residual - offset) ** 2
            for residual, offset in zip(option[2], offsets)
        )
    residual_rms = math.sqrt(squared / (count * 4))
    offset_rms = math.sqrt(sum(value * value for value in offsets) / 4)
    score = residual_rms + 0.02 * offset_rms
    return score, tuple(option[1] for option in options)


def _map_candidates(
    scene: Scene,
    all_candidates: list[list[tuple[Peak, ...]]],
) -> list[MapCandidate]:
    candidates: list[MapCandidate] = []
    option_limit = 32 if len(scene.sources) <= 2 else 16
    for order in all_permutations():
        source_options = []
        for source, source_candidates in zip(scene.sources, all_candidates):
            options = _source_options(
                scene, source.position, source_candidates, order, option_limit
            )
            if not options:
                break
            source_options.append(options)
        if len(source_options) != len(scene.sources):
            continue
        best_score = math.inf
        best_choices: tuple[tuple[Peak, ...], ...] = ()
        for option_tuple in itertools.product(*source_options):
            score, choices = _joint_score(option_tuple)
            if score < best_score:
                best_score = score
                best_choices = choices
        score = best_score
        source_choices = list(best_choices)
        candidates.append(MapCandidate(order, score, tuple(source_choices)))
    candidates.sort(key=lambda item: item.score_samples)
    return candidates


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _aggregate(
    scene: Scene,
    source_estimates: list[SourcePeaks],
    order: tuple[int, int, int, int],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    gain_samples: list[list[float]] = [[] for _ in range(4)]
    delay_samples: list[list[float]] = [[] for _ in range(4)]
    reference = scene.reference_channel
    mics = tuple(scene.microphones[index] for index in order)
    for estimate, source in zip(source_estimates, scene.sources):
        distances = direct_distances(mics, source.position)
        reference_amplitude = estimate.chosen[reference].energy
        if reference_amplitude < 1e-15:
            reference_amplitude = 1e-15
        reference_polarity = estimate.chosen[reference].polarity
        for channel, peak in enumerate(estimate.chosen):
            gain = (
                peak.energy
                / reference_amplitude
                * peak.polarity
                * reference_polarity
                * distances[channel]
                / distances[reference]
            )
            gain_samples[channel].append(gain)
        observed = estimate.relative_delays_s
        predicted = predicted_relative_delays(
            mics,
            [source.position],
            reference,
            scene.sound_speed,
        )[0]
        for channel in range(4):
            delay_samples[channel].append(observed[channel] - predicted[channel])
    gains = tuple(_median(values) for values in gain_samples)
    delays = tuple(_median(values) for values in delay_samples)
    residuals_mm = tuple(
        value * scene.sound_speed * 1000.0 for value in delays
    )
    return gains, delays, residuals_mm


def _make_findings(
    scene: Scene,
    result: CalibrationResult,
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    reference = scene.reference_channel
    if result.best_map != (0, 1, 2, 3):
        if result.map_ambiguous:
            findings.append(
                {
                    "severity": "warning",
                    "code": "channel_map_ambiguous",
                    "message": (
                        "通道/麦克风映射不唯一；最佳候选为 "
                        f"{list(result.best_map)}"
                    ),
                }
            )
        else:
            findings.append(
                {
                    "severity": "error",
                    "code": "channel_map",
                    "message": (
                        "测得延时对应通道顺序 "
                        f"{list(result.best_map)}，不是标称 [0, 1, 2, 3]"
                    ),
                }
            )
    elif result.map_ambiguous:
        findings.append(
            {
                "severity": "warning",
                "code": "channel_map_ambiguous",
                "message": "多组通道/麦克风映射都满足当前几何约束",
            }
        )
    for channel, polarity in enumerate(result.polarity):
        if channel == reference:
            continue
        if polarity < 0:
            findings.append(
                {
                    "severity": "error",
                    "code": "polarity",
                    "message": f"通道 {channel} 相对参考通道反相",
                }
            )
    for channel, error_db in enumerate(result.gain_error_db):
        if channel == reference:
            continue
        if abs(error_db) > scene.analysis.gain_warn_db:
            findings.append(
                {
                    "severity": "warning",
                    "code": "gain",
                    "message": (
                        f"通道 {channel} 相对增益为 {error_db:+.1f} dB"
                    ),
                }
            )
    for channel, residual_mm in enumerate(result.delay_residual_mm):
        if channel == reference:
            continue
        limit = scene.analysis.delay_warn_mm
        if abs(residual_mm) > limit:
            severity = "error" if abs(residual_mm) > 2.0 * limit else "warning"
            findings.append(
                {
                    "severity": severity,
                    "code": "position_delay",
                    "message": (
                        f"通道 {channel} 等效位置延时残差为 "
                        f"{residual_mm:+.1f} mm"
                    ),
                }
            )
    for estimate in result.source_estimates:
        for channel, (chosen, candidates) in enumerate(
            zip(estimate.chosen, estimate.candidates)
        ):
            stronger = [
                peak
                for peak in candidates
                if peak.delay_s > chosen.delay_s
                and peak.envelope > chosen.envelope * 1.02
            ]
            if stronger:
                reflection = max(stronger, key=lambda peak: peak.envelope)
                ratio = reflection.envelope / max(chosen.envelope, 1e-15)
                findings.append(
                    {
                        "severity": "info",
                        "code": "strong_reflection",
                        "message": (
                            f"{estimate.source_name} 通道 {channel}："
                            f"后到峰是直达声的 {ratio:.2f} 倍；"
                            "几何匹配已选择更早的直达峰"
                        ),
                    }
                )
    if not findings:
        findings.append(
            {
                "severity": "info",
                "code": "ok",
                "message": "未发现超过配置阈值的安装误差",
            }
        )
    return findings


def analyze_recordings(
    scene: Scene,
    recordings_dir: str | Path,
) -> CalibrationResult:
    all_candidates = _load_candidates(scene, Path(recordings_dir))
    candidates = _map_candidates(scene, all_candidates)
    if not candidates or not candidates[0].source_choices:
        raise AnalysisError("could not associate four-channel arrivals")
    best = candidates[0]
    window = scene.analysis.map_tolerance_samples
    ambiguous = [
        candidate
        for candidate in candidates
        if candidate.score_samples <= best.score_samples + window
    ]
    map_ambiguous = len(ambiguous) > 1
    source_estimates = [
        SourcePeaks(
            source_name=source.name,
            reference_channel=scene.reference_channel,
            candidates=tuple(all_candidates[index]),
            chosen=best.source_choices[index],
        )
        for index, source in enumerate(scene.sources)
    ]
    gains, _, residuals_mm = _aggregate(scene, source_estimates, best.order)
    mean_delays = tuple(
        sum(
            estimate.relative_delays_s[channel]
            for estimate in source_estimates
        )
        / len(source_estimates)
        for channel in range(4)
    )
    polarity = tuple(1.0 if value >= 0 else -1.0 for value in gains)
    result = CalibrationResult(
        scene=scene,
        source_estimates=source_estimates,
        best_map=best.order,
        map_candidates=ambiguous,
        map_ambiguous=map_ambiguous,
        relative_delays_s=mean_delays,
        delay_residual_mm=residuals_mm,
        gain_relative=gains,
        polarity=polarity,
    )
    result.findings = _make_findings(scene, result)
    return result
