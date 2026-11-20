"""Dependency-free FSK marker detector with carrier, sync and CRC evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .protocol import (
    ASSET_BY_ID,
    KNOWN_ASSETS,
    PROTOCOL,
    bits_to_hex,
    decode_body_bits,
    sync_distance,
)
from .wav_io import read_wav, write_wave_header_preview


@dataclass
class ToneTrack:
    energies: list[float]
    rate: int
    hop_seconds: float
    window_seconds: float

    def at(self, seconds: float) -> float:
        index = round((seconds - self.window_seconds / 2.0) / self.hop_seconds)
        if index < 0 or index >= len(self.energies):
            return 0.0
        return self.energies[index]


def tone_track(
    samples: list[float],
    rate: int,
    freq: float,
    window_seconds: float = 0.006,
    hop_seconds: float = 0.001,
) -> ToneTrack:
    count = max(8, int(window_seconds * rate))
    hop = max(1, int(hop_seconds * rate))
    if count >= len(samples):
        return ToneTrack([], rate, hop_seconds, window_seconds)

    omega = 2.0 * math.pi * freq / rate
    step_cos = math.cos(omega)
    step_sin = math.sin(omega)
    phase_cos = 1.0
    phase_sin = 0.0
    prefix_real = [0.0] * (len(samples) + 1)
    prefix_imag = [0.0] * (len(samples) + 1)
    real_sum = 0.0
    imag_sum = 0.0
    for index, sample in enumerate(samples):
        real_sum += sample * phase_cos
        imag_sum += sample * phase_sin
        prefix_real[index + 1] = real_sum
        prefix_imag[index + 1] = imag_sum
        next_cos = phase_cos * step_cos - phase_sin * step_sin
        phase_sin = phase_sin * step_cos + phase_cos * step_sin
        phase_cos = next_cos
        if (index & 0x3FFF) == 0:
            norm = math.hypot(phase_cos, phase_sin)
            phase_cos /= norm
            phase_sin /= norm

    length = 1 + (len(samples) - count) // hop
    scale = 1.0 / (count * count)
    energies = [0.0] * length
    for frame in range(length):
        start = frame * hop
        real = prefix_real[start + count] - prefix_real[start]
        imag = prefix_imag[start + count] - prefix_imag[start]
        energies[frame] = (real * real + imag * imag) * scale
    return ToneTrack(energies, rate, hop / rate, count / rate)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(fraction * (len(ordered) - 1))))
    return ordered[index]


def find_leader_regions(track: ToneTrack) -> list[dict]:
    if not track.energies:
        return []
    smooth_width = max(3, int(0.055 / track.hop_seconds))
    half = smooth_width // 2
    smooth = []
    running = 0.0
    for index, value in enumerate(track.energies):
        running += value
        if index >= smooth_width:
            running -= track.energies[index - smooth_width]
        if index >= smooth_width - 1:
            smooth.append((index - half, running / smooth_width))

    values = [value for _, value in smooth]
    floor = _percentile(values, 0.50)
    ceiling = _percentile(values, 0.995)
    threshold = max(floor * 2.15, ceiling * 0.065, 1e-10)
    active = [value >= threshold for _, value in smooth]

    runs = []
    start = None
    last = None
    merged_gap = max(2, int(0.026 / track.hop_seconds))
    for index, flag in enumerate(active):
        if flag:
            if start is None:
                start = index
            elif last is not None and index - last > merged_gap:
                runs.append((start, last))
                start = index
            last = index
    if start is not None and last is not None:
        runs.append((start, last))

    regions = []
    minimum = max(4, int(0.045 / track.hop_seconds))
    for first, final in runs:
        if final - first < minimum:
            continue
        window = smooth[first : final + 1]
        _, (_, peak_value) = max(
            enumerate(window), key=lambda item: item[1][1]
        )
        onset_frame = smooth[first][0]
        onset = onset_frame * track.hop_seconds + track.window_seconds / 2.0
        regions.append(
            {
                "onset": onset,
                "first": first,
                "last": final,
                "peak": peak_value,
                "length_seconds": (final - first) * track.hop_seconds,
            }
        )
    return regions


def _tone_power(
    samples: list[float],
    rate: int,
    center_seconds: float,
    window_seconds: float,
    freq: float,
) -> float:
    count = max(8, int(window_seconds * rate))
    first = int(round(center_seconds * rate - count / 2.0))
    first = max(0, min(len(samples) - count, first))
    if first < 0 or count > len(samples):
        return 0.0
    omega = 2.0 * math.pi * freq / rate
    real = 0.0
    imag = 0.0
    window_sum = 0.0
    for index in range(count):
        window = 0.5 - 0.5 * math.cos(2.0 * math.pi * index / (count - 1))
        value = samples[first + index] * window
        phase = omega * index
        real += value * math.cos(phase)
        imag -= value * math.sin(phase)
        window_sum += value * value
    return (real * real + imag * imag) / (count * count)


def _measure_frequency(
    samples: list[float],
    rate: int,
    centers: list[float],
    nominal: float,
    span: float = 0.08,
    window_seconds: float = 0.007,
) -> tuple[float, float]:
    if not centers:
        return nominal, 0.0
    best_freq = nominal
    best_power = -1.0
    first = nominal * (1.0 - span)
    last = nominal * (1.0 + span)
    step = 5.0 if span >= 0.05 else 2.5
    freq = first
    while freq <= last + 1e-9:
        power = sum(
            _tone_power(samples, rate, center, window_seconds, freq)
            for center in centers
        )
        if power > best_power:
            best_power = power
            best_freq = freq
        freq += step
    return best_freq, best_power


def _decode_symbols(
    track_zero: ToneTrack,
    track_one: ToneTrack,
    start: float,
    symbol_seconds: float,
    count: int,
) -> tuple[list[int], list[float]]:
    bits = []
    margins = []
    for index in range(count):
        center = start + (index + 0.5) * symbol_seconds
        energy_zero = track_zero.at(center)
        energy_one = track_one.at(center)
        total = energy_zero + energy_one + 1e-15
        bits.append(1 if energy_one > energy_zero else 0)
        margins.append((energy_one - energy_zero) / total)
    return bits, margins


def _scan_sync(
    track_zero: ToneTrack,
    track_one: ToneTrack,
    sync_guess: float,
    symbol_seconds: float,
) -> tuple[float, int, float]:
    proto = PROTOCOL
    best = (sync_guess, proto.sync_bits + 1, -1.0)
    offset = -0.065
    while offset <= 0.065 + 1e-9:
        candidate = sync_guess + offset
        bits, margins = _decode_symbols(
            track_zero, track_one, candidate, symbol_seconds, proto.sync_bits
        )
        distance = sync_distance(bits)
        quality = sum(abs(value) for value in margins) / len(margins)
        if distance < best[1] or (distance == best[1] and quality > best[2]):
            best = (candidate, distance, quality)
        offset += 0.001
    return best


def _contrast_db(
    samples: list[float],
    rate: int,
    centers: list[float],
    frequency: float,
    window_seconds: float,
) -> float:
    if not centers:
        return 0.0
    on_power = sum(
        _tone_power(samples, rate, center, window_seconds, frequency)
        for center in centers
    )
    offs = []
    for delta in (-260.0, -170.0, 170.0, 260.0):
        offs.append(
            sum(
                _tone_power(samples, rate, center, window_seconds, frequency + delta)
                for center in centers
            )
        )
    off_power = sum(offs) / len(offs)
    return 10.0 * math.log10((on_power + 1e-12) / (off_power + 1e-12))


def _build_detection(
    samples: list[float],
    rate: int,
    track_zero: ToneTrack,
    track_one: ToneTrack,
    speed: float,
    sync_start: float,
    hamming: int,
    quality: float,
) -> dict:
    proto = PROTOCOL
    symbol_seconds = proto.symbol_seconds / speed
    bits, margins = _decode_symbols(
        track_zero,
        track_one,
        sync_start,
        symbol_seconds,
        proto.frame_bits,
    )
    body = bits[proto.sync_bits :]
    decoded = decode_body_bits(body)
    crc_ok = decoded["crc_expected"] == decoded["crc_received"]
    asset = ASSET_BY_ID.get(decoded["asset_id"])

    sync_centers = [
        sync_start + (index + 0.5) * symbol_seconds
        for index in range(proto.sync_bits)
    ]
    sync_bits = bits[: proto.sync_bits]
    mark_centers = [sync_centers[index] for index, bit in enumerate(sync_bits) if bit == 1][:4]
    space_centers = [sync_centers[index] for index, bit in enumerate(sync_bits) if bit == 0][:4]
    leader_center = (
        sync_start
        - (proto.gap_seconds + proto.leader_seconds / 2.0) / speed
    )
    leader_hz, _ = _measure_frequency(
        samples, rate, [leader_center], proto.leader_hz * speed, span=0.06, window_seconds=0.035
    )
    mark_hz, _ = _measure_frequency(
        samples, rate, mark_centers, proto.mark_hz * speed, span=0.06
    )
    space_hz, _ = _measure_frequency(
        samples, rate, space_centers, proto.space_hz * speed, span=0.06
    )
    leader_contrast = _contrast_db(samples, rate, [leader_center], leader_hz, 0.035)
    mark_contrast = _contrast_db(samples, rate, mark_centers, mark_hz, 0.007)
    space_contrast = _contrast_db(samples, rate, space_centers, space_hz, 0.007)
    contrast = (leader_contrast + mark_contrast + space_contrast) / 3.0
    measured_speed = (
        leader_hz / proto.leader_hz
        + mark_hz / proto.mark_hz
        + space_hz / proto.space_hz
    ) / 3.0
    carrier_locked = (
        abs(leader_hz - proto.leader_hz * speed) <= 70.0
        and abs(mark_hz - proto.mark_hz * speed) <= 90.0
        and abs(space_hz - proto.space_hz * speed) <= 70.0
        and min(leader_contrast, mark_contrast, space_contrast) > 1.0
    )

    marker_time = sync_start - proto.sync_to_marker_seconds / speed
    if crc_ok and asset is not None:
        kind = "hit"
        reason = None
    elif crc_ok:
        kind = "candidate"
        reason = "CRC 通过，但编号不在登记库"
    else:
        kind = "candidate"
        reason = "同步已锁定，CRC 不匹配，仅记录候选"

    return {
        "kind": kind,
        "candidate_reason": reason,
        "asset_id": f"{decoded['asset_id']:04X}",
        "asset_label": asset.label if asset else f"UNKNOWN-{decoded['asset_id']:04X}",
        "asset_title": asset.title if asset else "未登记编号",
        "marker_seconds": round(marker_time, 6),
        "sync_seconds": round(sync_start, 6),
        "frame_end_seconds": round(sync_start + symbol_seconds * proto.frame_bits, 6),
        "carrier": {
            "leader_hz": round(leader_hz, 1),
            "space_hz": round(space_hz, 1),
            "mark_hz": round(mark_hz, 1),
            "leader_nominal_hz": round(proto.leader_hz * speed, 1),
            "speed_factor": round(measured_speed, 4),
            "speed_hypothesis": round(speed, 4),
            "contrast_db": round(contrast, 2),
            "leader_contrast_db": round(leader_contrast, 2),
            "mark_contrast_db": round(mark_contrast, 2),
            "space_contrast_db": round(space_contrast, 2),
            "locked": carrier_locked,
        },
        "sync": {
            "word": f"{proto.sync_word:04X}",
            "hamming": hamming,
            "locked": hamming <= 3,
            "soft_margin": round(quality, 4),
        },
        "data": {
            "version": decoded["version"],
            "meta": decoded["meta"],
            "header_hex": decoded["header_hex"],
            "body_hex": bits_to_hex(body),
            "crc_expected": f"{decoded['crc_expected']:04X}",
            "crc_received": f"{decoded['crc_received']:04X}",
            "crc_ok": crc_ok,
            "verified": crc_ok and asset is not None,
        },
        "_score": (1 if crc_ok and asset else 0, 1 if crc_ok else 0, -hamming, quality),
    }


def _speed_grid() -> list[float]:
    values = []
    speed = 0.94
    while speed <= 1.0601:
        values.append(round(speed, 5))
        speed += 0.005
    return values


def _deduplicate(detections: list[dict]) -> list[dict]:
    ordered = sorted(detections, key=lambda item: item["marker_seconds"])
    groups: list[list[dict]] = []
    for detection in ordered:
        if groups and abs(detection["marker_seconds"] - groups[-1][0]["marker_seconds"]) < 0.34:
            groups[-1].append(detection)
        else:
            groups.append([detection])
    winners = [max(group, key=lambda item: item["_score"]) for group in groups]
    for detection in winners:
        detection.pop("_score", None)
    return sorted(winners, key=lambda item: item["marker_seconds"])


def analyze_samples(
    samples: list[float],
    rate: int,
    source_name: str = "recording.wav",
) -> dict:
    proto = PROTOCOL
    leader_track = tone_track(samples, rate, proto.leader_hz, 0.024, 0.002)
    regions = find_leader_regions(leader_track)
    candidates: list[dict] = []

    for speed in _speed_grid():
        track_zero = tone_track(samples, rate, proto.space_hz * speed)
        track_one = tone_track(samples, rate, proto.mark_hz * speed)
        symbol_seconds = proto.symbol_seconds / speed
        for region in regions:
            sync_guess = region["onset"] + proto.sync_to_marker_seconds / speed
            sync_start, hamming, quality = _scan_sync(
                track_zero, track_one, sync_guess, symbol_seconds
            )
            if hamming > 3:
                continue
            detection = _build_detection(
                samples,
                rate,
                track_zero,
                track_one,
                speed,
                sync_start,
                hamming,
                quality,
            )
            candidates.append(detection)

    detections = _deduplicate(candidates)
    for detection in detections:
        detection["false_positive"] = False
        detection["matched_truth_index"] = None
        detection["timing_offset_ms"] = None

    return {
        "schema_version": 1,
        "source": {"name": source_name, "sample_rate": rate, "duration_seconds": round(len(samples) / rate, 6)},
        "protocol": {
            "name": proto.name,
            "leader_hz": proto.leader_hz,
            "space_hz": proto.space_hz,
            "mark_hz": proto.mark_hz,
            "symbol_ms": proto.symbol_seconds * 1000.0,
            "frame_ms": proto.frame_seconds * 1000.0,
            "sync_word": f"{proto.sync_word:04X}",
            "crc": "CRC-16/CCITT-FALSE",
            "known_assets": [
                {"id": f"{asset.asset_id:04X}", "label": asset.label, "title": asset.title}
                for asset in KNOWN_ASSETS
            ],
        },
        "summary": {
            "hits": sum(1 for item in detections if item["kind"] == "hit"),
            "candidates": sum(1 for item in detections if item["kind"] == "candidate"),
            "false_positives": None,
            "timing_mae_ms": None,
            "timing_max_abs_ms": None,
        },
        "detections": detections,
        "false_positive_detail": [],
        "overview": write_wave_header_preview(samples, 720),
        "leader_regions_scanned": len(regions),
    }


def analyze_file(path: str, source_name: str | None = None) -> dict:
    samples, rate = read_wav(path)
    return analyze_samples(samples, rate, source_name or path)
