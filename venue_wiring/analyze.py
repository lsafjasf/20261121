"""Decision logic for source mapping, polarity, duplicate wiring, and opens."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any

from .spectral import CorrelationPeak, normalized_xcorr_peak
from .wavio import WavData, read_wav


@dataclass
class AnalysisConfig:
    expected_channels: int = 8
    accept_threshold: float = 0.28
    open_threshold: float = 0.12
    margin_threshold: float = 0.030
    min_prominence_db: float = 1.5
    min_overlap: float = 0.30
    max_lag_seconds: float = 1.5
    exclusion_seconds: float = 0.002
    duplicate_ratio: float = 0.72
    duplicate_similarity: float = 0.35


@dataclass
class ChannelDecision:
    return_index: int
    return_label: str
    top: CorrelationPeak
    top_source_index: int
    second_source_index: int
    source_margin: float
    all_peaks: list[CorrelationPeak] = field(default_factory=list)
    status: str = "review"
    mapped_source_index: int | None = None
    duplicate_group: str | None = None
    duplicate_reference_index: int | None = None
    duplicate_reference_label: str | None = None
    duplicate_similarity: float | None = None
    return_rms: float = 0.0
    return_peak: float = 0.0
    reasons: list[str] = field(default_factory=list)


def _labels(
    count: int, supplied: list[str] | None, prefix: str
) -> list[str]:
    if supplied is None:
        return [f"{prefix}-{index + 1}" for index in range(count)]
    if len(supplied) != count:
        raise ValueError(f"expected {count} labels, received {len(supplied)}")
    cleaned = [value.strip() for value in supplied]
    if any(not value for value in cleaned):
        raise ValueError("channel labels must not be blank")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("channel labels must be unique")
    return cleaned


def _as_wav(value: str | Path | WavData) -> WavData:
    return value if isinstance(value, WavData) else read_wav(value)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _validate_config(config: AnalysisConfig) -> None:
    if config.expected_channels < 2:
        raise ValueError("expected_channels must be at least 2")
    if not 0.0 <= config.open_threshold < config.accept_threshold <= 1.0:
        raise ValueError(
            "thresholds must satisfy 0 <= open_threshold < accept_threshold <= 1"
        )
    if config.margin_threshold < 0.0:
        raise ValueError("margin_threshold must not be negative")
    if config.min_prominence_db < 0.0:
        raise ValueError("min_prominence_db must not be negative")
    if not 0.0 < config.min_overlap < 1.0:
        raise ValueError("min_overlap must be between 0 and 1")
    if config.max_lag_seconds <= 0.0:
        raise ValueError("max_lag_seconds must be positive")
    if not 0.0 <= config.duplicate_ratio <= 1.0:
        raise ValueError("duplicate_ratio must be between 0 and 1")
    if not 0.0 <= config.duplicate_similarity <= 1.0:
        raise ValueError("duplicate_similarity must be between 0 and 1")


def _confidence_factors(
    decision: ChannelDecision, config: AnalysisConfig
) -> dict[str, float]:
    correlation = abs(decision.top.signed_score)
    return {
        "correlation_strength": _clamp(
            (correlation - config.open_threshold) / max(0.01, 0.95 - config.open_threshold)
        ),
        "source_margin": _clamp(
            decision.source_margin / max(0.01, config.margin_threshold * 4.0)
        ),
        "peak_prominence": _clamp(
            (decision.top.peak_to_secondary_db - config.min_prominence_db + 3.0)
            / 20.0
        ),
        "overlap": _clamp(
            (decision.top.overlap_ratio - config.min_overlap)
            / max(0.01, 0.80 - config.min_overlap)
        ),
    }


def _confidence(
    decision: ChannelDecision, config: AnalysisConfig
) -> dict[str, Any]:
    magnitude = abs(decision.top.signed_score)
    if decision.status == "open":
        openness = _clamp(
            (config.open_threshold - magnitude)
            / max(1e-9, config.open_threshold * 0.75)
        )
        score = 0.65 + 0.35 * openness
        level = "high" if score >= 0.78 else "medium"
        return {
            "level": level,
            "score": _round(score, 4),
            "factors": {
                "open_evidence": openness,
                "correlation_strength": _clamp(1.0 - magnitude / config.open_threshold),
            },
            "basis": [
                f"最强归一化互相关 {magnitude:.3f}",
                f"低于开路阈值 {config.open_threshold:.3f}",
                f"回录 RMS {decision.return_rms:.6f}",
                "未出现任何测试源的可辨识相关峰",
            ],
        }
    factors = _confidence_factors(decision, config)
    score = (
        0.45 * factors["correlation_strength"]
        + 0.25 * factors["source_margin"]
        + 0.20 * factors["peak_prominence"]
        + 0.10 * factors["overlap"]
    )
    if decision.duplicate_similarity is not None:
        factors["duplicate_waveform"] = _clamp(decision.duplicate_similarity)
        score = 0.82 * score + 0.18 * factors["duplicate_waveform"]
    if score >= 0.78:
        level = "high"
    elif score >= 0.55:
        level = "medium"
    else:
        level = "low"
    basis = [
        f"最强归一化互相关 {abs(decision.top.signed_score):.3f}",
        f"与次优源的裕量 {decision.source_margin:.3f}",
        f"峰-次峰 {decision.top.peak_to_secondary_db:.1f} dB",
        f"有效重叠 {decision.top.overlap_ratio:.0%}",
        f"峰-噪声底 {decision.top.peak_to_noise:.1f}×",
    ]
    if decision.duplicate_similarity is not None:
        basis.append(f"重复通道波形相似度 {decision.duplicate_similarity:.3f}")
    return {"level": level, "score": _round(score, 4), "factors": factors, "basis": basis}


def _base_status(decision: ChannelDecision, config: AnalysisConfig) -> None:
    magnitude = abs(decision.top.signed_score)
    if magnitude < config.open_threshold:
        decision.status = "open"
        decision.reasons.append(
            f"最强互相关 {magnitude:.3f} 低于开路阈值 {config.open_threshold:.3f}"
        )
        return
    automatic = (
        magnitude >= config.accept_threshold
        and decision.source_margin >= config.margin_threshold
        and decision.top.overlap_ratio >= config.min_overlap
    )
    if not automatic:
        decision.status = "review"
        if magnitude < config.accept_threshold:
            decision.reasons.append(
                f"互相关 {magnitude:.3f} 未达到自动接受阈值 {config.accept_threshold:.3f}"
            )
        if decision.source_margin < config.margin_threshold:
            decision.reasons.append(
                f"与次优源裕量 {decision.source_margin:.3f} 过小"
            )
        if decision.top.overlap_ratio < config.min_overlap:
            decision.reasons.append(
                f"有效重叠 {decision.top.overlap_ratio:.0%} 不足"
            )
        return
    decision.mapped_source_index = decision.top_source_index
    decision.status = "inverted" if decision.top.polarity < 0 else "ok"
    if decision.top.polarity < 0:
        decision.reasons.append("最强相关峰为负值，极性判断为反相")
    else:
        decision.reasons.append("最强相关峰为正值，极性判断为正常")


def _find_duplicate_groups(
    decisions: list[ChannelDecision],
    recording: WavData,
    config: AnalysisConfig,
) -> list[dict[str, Any]]:
    by_source: dict[int, list[ChannelDecision]] = {}
    for decision in decisions:
        if decision.status == "open":
            continue
        if decision.status not in ("ok", "inverted"):
            continue
        by_source.setdefault(decision.top_source_index, []).append(decision)

    groups: list[dict[str, Any]] = []
    for source_index, members in sorted(by_source.items()):
        if len(members) < 2:
            continue
        canonical = max(members, key=lambda item: abs(item.top.signed_score))
        accepted = [canonical]
        for member in members:
            if member is canonical:
                continue
            ratio = abs(member.top.signed_score) / max(
                abs(canonical.top.signed_score), 1e-12
            )
            similarity = abs(
                normalized_xcorr_peak(
                    recording.channels[member.return_index],
                    recording.channels[canonical.return_index],
                    recording.sample_rate,
                    min_overlap=config.min_overlap,
                    max_lag_seconds=config.max_lag_seconds,
                    exclusion_seconds=config.exclusion_seconds,
                ).signed_score
            )
            member.duplicate_similarity = similarity
            if ratio >= config.duplicate_ratio and (
                similarity >= config.duplicate_similarity or ratio >= 0.90
            ):
                accepted.append(member)
            else:
                member.status = "review"
                member.mapped_source_index = None
                member.reasons.append(
                    f"与 {canonical.return_label} 共享候选源，但重复接线证据不足"
                )

        if len(accepted) < 2:
            if canonical.status in ("ok", "inverted"):
                canonical.status = "review"
                canonical.mapped_source_index = None
                canonical.reasons.append("候选源被多个通道占用，重复证据不足")
            continue

        group_id = f"DUP-{len(groups) + 1}"
        for member in accepted:
            member.duplicate_group = group_id
            member.mapped_source_index = source_index
            if member is canonical:
                member.duplicate_reference_index = None
                member.status = "inverted" if member.top.polarity < 0 else "ok"
                member.reasons.append(
                    f"{group_id} 中相关最强，作为参考通道；同组 {len(accepted) - 1} 路重复"
                )
            else:
                member.duplicate_reference_index = canonical.return_index
                member.duplicate_reference_label = canonical.return_label
                member.status = "duplicate"
                member.reasons.append(
                    f"与 {canonical.return_label} 在同一候选源上重复，"
                    f"相关比 {ratio:.2f}，波形相似度 {similarity:.2f}"
                )
        groups.append(
            {
                "id": group_id,
                "source_index": source_index,
                "reference_channel_index": canonical.return_index,
                "member_indices": [member.return_index for member in accepted],
            }
        )
    return groups


def _peak_payload(peak: CorrelationPeak) -> dict[str, Any]:
    return {
        "signed_correlation": _round(peak.signed_score),
        "absolute_correlation": _round(abs(peak.signed_score)),
        "lag_ms": _round(peak.lag_seconds * 1000.0, 3),
        "overlap_fraction": _round(peak.overlap_ratio, 4),
        "polarity": "normal" if peak.polarity >= 0 else "inverted",
        "peak_to_secondary_db": _round(peak.peak_to_secondary_db, 3),
        "peak_to_noise": _round(peak.peak_to_noise, 3),
    }


def _channel_payload(
    decision: ChannelDecision,
    source_labels: list[str],
    config: AnalysisConfig,
) -> dict[str, Any]:
    automatic = decision.status in ("ok", "inverted", "duplicate", "open")
    mapped_source = (
        source_labels[decision.mapped_source_index]
        if decision.mapped_source_index is not None
        else None
    )
    if decision.status == "open":
        polarity = "unknown"
        polarity_candidate = None
    elif decision.status in ("ok", "inverted", "duplicate"):
        polarity = "inverted" if decision.top.polarity < 0 else "normal"
        polarity_candidate = polarity
    else:
        polarity = "undetermined"
        polarity_candidate = "inverted" if decision.top.polarity < 0 else "normal"

    ranked = sorted(
        enumerate(decision.all_peaks),
        key=lambda item: abs(item[1].signed_score),
        reverse=True,
    )
    source_peaks = [
        {
            "source": source_labels[index],
            **_peak_payload(peak),
        }
        for index, peak in ranked
    ]
    duplicate = None
    if decision.duplicate_group is not None:
        duplicate = {
            "group": decision.duplicate_group,
            "role": "reference"
            if decision.duplicate_reference_index is None
            else "duplicate",
            "reference_channel": decision.duplicate_reference_label,
            "waveform_similarity": None
            if decision.duplicate_similarity is None
            else _round(decision.duplicate_similarity, 4),
        }
    evidence = {
        "top_source": source_labels[decision.top_source_index],
        "second_source": source_labels[decision.second_source_index],
        **_peak_payload(decision.top),
        "source_margin": _round(decision.source_margin),
        "return_rms": _round(decision.return_rms, 8),
        "return_peak": _round(decision.return_peak, 8),
        "all_source_peaks": source_peaks,
    }
    return {
        "return_channel": decision.return_label,
        "status": decision.status,
        "automatic": automatic,
        "manual_review": decision.status == "review",
        "mapped_source": mapped_source,
        "candidate_source": source_labels[decision.top_source_index]
        if decision.status != "open"
        else None,
        "polarity": polarity,
        "polarity_candidate": polarity_candidate,
        "latency_ms": _round(decision.top.lag_seconds * 1000.0, 3),
        "duplicate": duplicate,
        "evidence": evidence,
        "confidence": _confidence(decision, config),
        "reasons": decision.reasons,
    }


def analyze(
    reference: str | Path | WavData,
    recording: str | Path | WavData,
    config: AnalysisConfig | None = None,
    reference_labels: list[str] | None = None,
    return_labels: list[str] | None = None,
) -> dict[str, Any]:
    """Correlate every source against every return and produce an audit report."""

    settings = config or AnalysisConfig()
    _validate_config(settings)
    reference_wav = _as_wav(reference)
    recording_wav = _as_wav(recording)
    expected = settings.expected_channels
    if reference_wav.channel_count != expected:
        raise ValueError(
            f"reference must contain {expected} channels, found {reference_wav.channel_count}"
        )
    if recording_wav.channel_count != expected:
        raise ValueError(
            f"recording must contain {expected} channels, found {recording_wav.channel_count}"
        )
    if reference_wav.sample_rate != recording_wav.sample_rate:
        raise ValueError("reference and recording sample rates do not match")
    if reference_wav.frame_count == 0 or recording_wav.frame_count == 0:
        raise ValueError("reference and recording must contain audio frames")

    sources = _labels(expected, reference_labels, "SRC")
    returns = _labels(expected, return_labels, "RET")
    for source_index, channel in enumerate(reference_wav.channels):
        if max((abs(value) for value in channel), default=0.0) < 1e-12:
            raise ValueError(
                f"reference channel {sources[source_index]} contains no test signal"
            )
    decisions: list[ChannelDecision] = []
    for return_index in range(expected):
        peaks = [
            normalized_xcorr_peak(
                reference_wav.channels[source_index],
                recording_wav.channels[return_index],
                recording_wav.sample_rate,
                min_overlap=settings.min_overlap,
                max_lag_seconds=settings.max_lag_seconds,
                exclusion_seconds=settings.exclusion_seconds,
            )
            for source_index in range(expected)
        ]
        ranked = sorted(
            range(expected), key=lambda index: abs(peaks[index].signed_score), reverse=True
        )
        top_index = ranked[0]
        second_index = ranked[1]
        decision = ChannelDecision(
            return_index=return_index,
            return_label=returns[return_index],
            top=peaks[top_index],
            top_source_index=top_index,
            second_source_index=second_index,
            source_margin=abs(peaks[top_index].signed_score)
            - abs(peaks[second_index].signed_score),
            all_peaks=peaks,
            return_rms=math.sqrt(
                sum(value * value for value in recording_wav.channels[return_index])
                / recording_wav.frame_count
            ),
            return_peak=max(
                (abs(value) for value in recording_wav.channels[return_index]), default=0.0
            ),
        )
        _base_status(decision, settings)
        decisions.append(decision)

    duplicate_groups = _find_duplicate_groups(decisions, recording_wav, settings)
    shared_top_sources: dict[int, list[ChannelDecision]] = {}
    for decision in decisions:
        if decision.status in ("ok", "inverted", "duplicate"):
            shared_top_sources.setdefault(decision.mapped_source_index, []).append(decision)
    for source_index, owners in shared_top_sources.items():
        if len(owners) > 1 and not any(
            decision.duplicate_group is not None for decision in owners
        ):
            for decision in owners:
                decision.status = "review"
                decision.mapped_source_index = None
                decision.reasons.append(
                    f"多个通道共享候选源 {sources[source_index]}，但重复证据不一致"
                )

    channel_payloads = [
        _channel_payload(decision, sources, settings) for decision in decisions
    ]
    if duplicate_groups:
        channel_payload_by_index = {
            decision.return_index: payload
            for decision, payload in zip(decisions, channel_payloads)
        }
        duplicate_payload = []
        for group in duplicate_groups:
            reference_index = group["reference_channel_index"]
            member_indices = group["member_indices"]
            source_index = group["source_index"]
            duplicate_payload.append(
                {
                    "id": group["id"],
                    "source": sources[source_index],
                    "reference_channel": returns[reference_index],
                    "members": [returns[index] for index in member_indices],
                    "member_statuses": {
                        returns[index]: channel_payload_by_index[index]["status"]
                        for index in member_indices
                    },
                }
            )
        duplicate_groups = duplicate_payload

    mapping = []
    for decision, payload in zip(decisions, channel_payloads):
        if payload["mapped_source"] is not None:
            mapping.append(
                {
                    "source": payload["mapped_source"],
                    "return_channel": payload["return_channel"],
                    "polarity": payload["polarity"],
                    "status": payload["status"],
                    "confidence": payload["confidence"]["level"],
                }
            )
    coverage = []
    for source_index, source_label in enumerate(sources):
        observed_returns = [
            payload["return_channel"]
            for payload in channel_payloads
            if payload["mapped_source"] == source_label
        ]
        coverage.append(
            {
                "source": source_label,
                "observed_returns": observed_returns,
                "observed": bool(observed_returns),
            }
        )
    counts = {
        status: sum(payload["status"] == status for payload in channel_payloads)
        for status in ("ok", "inverted", "duplicate", "open", "review")
    }
    manual_review = [
        {
            "return_channel": payload["return_channel"],
            "candidate_source": payload["candidate_source"],
            "reason": "；".join(payload["reasons"]),
        }
        for payload in channel_payloads
        if payload["manual_review"]
    ]
    input_names = {
        "reference": str(reference) if not isinstance(reference, WavData) else "<memory>",
        "recording": str(recording) if not isinstance(recording, WavData) else "<memory>",
    }
    return {
        "schema_version": 1,
        "tool": "venue-wiring-check",
        "tool_version": "0.1.0",
        "input": {
            **input_names,
            "sample_rate": recording_wav.sample_rate,
            "channel_count": expected,
        },
        "config": asdict(settings),
        "summary": {
            "channel_count": expected,
            "source_labels": sources,
            "return_labels": returns,
            **counts,
            "automatic_claims": counts["ok"]
            + counts["inverted"]
            + counts["duplicate"]
            + counts["open"],
            "manual_review_required": counts["review"] > 0,
            "duplicate_group_count": len(duplicate_groups),
            "unobserved_sources": [
                item["source"] for item in coverage if not item["observed"]
            ],
        },
        "mapping": mapping,
        "duplicate_groups": duplicate_groups,
        "source_coverage": coverage,
        "manual_review": manual_review,
        "channels": channel_payloads,
    }
