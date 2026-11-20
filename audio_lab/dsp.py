"""Delay estimation, alignment, and window-energy primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DelayEstimate:
    delay_samples: int
    delay_ms: float
    confidence: float
    method: str


def mean_square(samples: list[float], start: int, stop: int) -> float:
    start = max(0, start)
    stop = min(len(samples), stop)
    if stop <= start:
        return 0.0
    return sum(value * value for value in samples[start:stop]) / (stop - start)


def energy_envelope(samples: list[float], frame_size: int) -> list[float]:
    frame_size = max(1, frame_size)
    return [
        mean_square(samples, start, min(start + frame_size, len(samples)))
        for start in range(0, len(samples), frame_size)
    ]


def _normalized_lag_score(
    reference: list[float], decoded: list[float], lag: int
) -> tuple[float, int]:
    start = max(0, lag)
    stop = min(len(reference), len(decoded) + lag)
    count = stop - start
    if count <= 0:
        return 0.0, 0
    dot = 0.0
    ref_norm = 0.0
    dec_norm = 0.0
    for index in range(start, stop):
        ref_value = reference[index]
        dec_value = decoded[index - lag]
        dot += ref_value * dec_value
        ref_norm += ref_value * ref_value
        dec_norm += dec_value * dec_value
    denominator = math.sqrt(ref_norm * dec_norm)
    if denominator <= 1e-30:
        return 0.0, count
    return max(-1.0, min(1.0, dot / denominator)), count


def _best_lag_in_envelopes(
    reference: list[float],
    decoded: list[float],
    min_lag: int,
    max_lag: int,
) -> tuple[int, float, int]:
    best_lag = 0
    best_score = -2.0
    best_count = 0
    min_overlap = max(8, min(len(reference), len(decoded)) // 20)
    for lag in range(min_lag, max_lag + 1):
        score, count = _normalized_lag_score(reference, decoded, lag)
        if count < min_overlap:
            continue
        if score > best_score:
            best_lag, best_score, best_count = lag, score, count
    return best_lag, max(0.0, best_score), best_count


def _sample_refine(
    reference: list[float],
    decoded: list[float],
    sample_rate: int,
    delay_samples: int,
    search_radius: int,
) -> int:
    """Refine a coarse delay with absolute-amplitude correlation near an attack."""
    coarse_size = max(1, round(sample_rate * 0.004))
    reference_envelope = energy_envelope(reference, coarse_size)
    anchor = max(range(len(reference_envelope)), key=reference_envelope.__getitem__)
    anchor_sample = anchor * coarse_size
    start = max(0, anchor_sample - round(sample_rate * 0.015))
    stop = min(len(reference), anchor_sample + round(sample_rate * 0.060))
    aligned = align_signal(decoded, len(reference), delay_samples)
    best_score = -1.0
    best_residual = 0
    for residual in range(-search_radius, search_radius + 1):
        overlap_start = max(start, -residual, 0)
        overlap_stop = min(stop, len(reference), len(aligned) - residual)
        if overlap_stop <= overlap_start:
            continue
        dot = 0.0
        ref_norm = 0.0
        dec_norm = 0.0
        for index in range(overlap_start, overlap_stop):
            ref_value = abs(reference[index])
            dec_value = abs(aligned[index + residual])
            dot += ref_value * dec_value
            ref_norm += ref_value * ref_value
            dec_norm += dec_value * dec_value
        denominator = math.sqrt(ref_norm * dec_norm)
        if denominator <= 1e-30:
            continue
        score = dot / denominator
        if score > best_score:
            best_score = score
            best_residual = residual
    return best_residual


def estimate_delay(
    reference: list[float],
    decoded: list[float],
    sample_rate: int,
    max_delay_s: float = 0.35,
) -> DelayEstimate:
    """Estimate how many samples late the decoded signal is.

    A positive delay means decoded[delay:] aligns with reference[0:]. The
    search uses a coarse energy envelope, then refines over one coarse bin at
    sub-millisecond resolution. This is deliberately insensitive to waveform
    phase changes introduced by lossy codecs.
    """
    if not reference or not decoded:
        raise ValueError("reference and decoded signals must be non-empty")

    coarse_size = max(1, round(sample_rate * 0.004))
    fine_size = max(1, round(sample_rate * 0.00025))
    max_coarse_lag = max(1, round(max_delay_s * sample_rate / coarse_size))
    coarse_ref = energy_envelope(reference, coarse_size)
    coarse_dec = energy_envelope(decoded, coarse_size)
    coarse_lag, coarse_score, _ = _best_lag_in_envelopes(
        coarse_ref, coarse_dec, -max_coarse_lag, max_coarse_lag
    )
    fine_ref = energy_envelope(reference, fine_size)
    fine_dec = energy_envelope(decoded, fine_size)
    center = round(coarse_lag * coarse_size / fine_size)
    radius = max(1, math.ceil(coarse_size / fine_size))
    fine_lag, fine_score, _ = _best_lag_in_envelopes(
        fine_ref, fine_dec, center - radius, center + radius
    )
    # The correlation compares reference[i] with decoded[i-lag]. If decoded
    # content is delayed by D samples, its best lag is -D.
    delay_samples = -fine_lag * fine_size
    delay_samples += _sample_refine(
        reference, decoded, sample_rate, delay_samples, fine_size
    )
    confidence = fine_score if fine_score > 0.0 else coarse_score
    return DelayEstimate(
        delay_samples=delay_samples,
        delay_ms=delay_samples * 1000.0 / sample_rate,
        confidence=confidence,
        method=f"envelope-xcorr {coarse_size / sample_rate * 1000:.1f}ms->"
        f"{fine_size / sample_rate * 1000:.3f}ms + sample-refine",
    )


def align_signal(
    decoded: list[float], reference_length: int, delay_samples: int
) -> list[float]:
    """Remove a positive codec delay and return exactly reference_length frames."""
    if delay_samples >= 0:
        aligned = decoded[delay_samples : delay_samples + reference_length]
    else:
        aligned = [0.0] * (-delay_samples) + decoded[: reference_length + delay_samples]
    if len(aligned) < reference_length:
        aligned.extend([0.0] * (reference_length - len(aligned)))
    return aligned[:reference_length]


def rms_envelope_db(
    samples: list[float],
    start: int,
    stop: int,
    frame_size: int,
    floor_db: float = -100.0,
) -> list[float]:
    start = max(0, start)
    stop = min(len(samples), stop)
    frame_size = max(1, frame_size)
    values: list[float] = []
    for offset in range(start, stop, frame_size):
        energy = mean_square(samples, offset, min(offset + frame_size, stop))
        values.append(
            floor_db
            if energy <= 0.0
            else max(floor_db, 10.0 * math.log10(energy))
        )
    return values
