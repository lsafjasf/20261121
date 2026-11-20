"""Procedural museum-music sample and a reproducible damage chain."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

from .protocol import ASSET_BY_ID, PROTOCOL, Asset, build_frame_bits
from .wav_io import (
    apply_speed,
    lowpass,
    normalize,
    resample_linear,
    rms,
    write_wav,
)

SOURCE_RATE = 48000
OUTPUT_RATE = 11025
SAMPLE_SPEED = 1.018


@dataclass(frozen=True)
class MarkerPlacement:
    asset: Asset
    start_seconds: float
    damaged: bool = False


SAMPLE_PLACEMENTS = (
    MarkerPlacement(ASSET_BY_ID[0x1042], 2.20),
    MarkerPlacement(ASSET_BY_ID[0x7781], 4.30),
    MarkerPlacement(ASSET_BY_ID[0x1042], 7.15),
    MarkerPlacement(ASSET_BY_ID[0x7781], 9.20),
    MarkerPlacement(ASSET_BY_ID[0x1042], 12.35),
    MarkerPlacement(ASSET_BY_ID[0x7781], 14.40),
    MarkerPlacement(ASSET_BY_ID[0x1042], 16.55, damaged=True),
)


def _oscillator(freq: float, seconds: float, rate: int, amplitude: float, fade: float) -> list[float]:
    count = max(1, int(seconds * rate))
    fade_count = max(1, int(fade * rate))
    phase_step = 2.0 * math.pi * freq / rate
    result = [0.0] * count
    for index in range(count):
        edge = min(index, count - 1 - index)
        envelope = min(1.0, edge / fade_count) if fade_count else 1.0
        envelope = 0.5 - 0.5 * math.cos(math.pi * envelope)
        result[index] = amplitude * envelope * math.sin(phase_step * index)
    return result


def marker_waveform(asset: Asset, rate: int, damaged: bool = False) -> list[float]:
    proto = PROTOCOL
    total = int(proto.frame_seconds * rate) + int(0.01 * rate)
    result = [0.0] * total
    leader = _oscillator(proto.leader_hz, proto.leader_seconds, rate, 0.24, 0.008)
    for index, sample in enumerate(leader):
        result[index] += sample

    bits, _ = build_frame_bits(asset.asset_id)
    start = int(proto.sync_to_marker_seconds * rate)
    symbol_count = int(proto.symbol_seconds * rate)
    fade_count = max(2, int(0.0014 * rate))
    for bit_index, bit in enumerate(bits):
        freq = proto.mark_hz if bit else proto.space_hz
        offset = start + bit_index * symbol_count
        phase_step = 2.0 * math.pi * freq / rate
        for index in range(symbol_count):
            edge = min(index, symbol_count - 1 - index)
            envelope = min(1.0, edge / fade_count)
            envelope = 0.5 - 0.5 * math.cos(math.pi * envelope)
            result[offset + index] += 0.25 * envelope * math.sin(phase_step * index)

    if damaged:
        rng = random.Random(1042)
        body_start = start + int(proto.sync_bits * proto.symbol_seconds * rate)
        body_end = min(total, body_start + int(0.24 * rate))
        for index in range(body_start, body_end):
            noise = rng.uniform(-1.0, 1.0)
            interference = math.sin(2.0 * math.pi * 1730.0 * index / rate)
            result[index] = result[index] * 0.18 + 0.42 * noise + 0.25 * interference
    return result


def _mix_at(destination: list[float], source: list[float], start: int, gain: float = 1.0) -> None:
    end = min(len(destination), start + len(source))
    for offset in range(max(0, end - start)):
        destination[start + offset] += source[offset] * gain


def _add_pad(out: list[float], rate: int, start: float, end: float, freqs: list[float]) -> None:
    first = int(start * rate)
    last = min(len(out), int(end * rate))
    length = max(1, last - first)
    fade = int(0.35 * rate)
    for freq in freqs:
        for detune in (-0.28, 0.0, 0.31):
            step = 2.0 * math.pi * (freq + detune) / rate
            for local in range(length):
                edge = min(local, length - 1 - local)
                env = min(1.0, edge / fade)
                env = 0.5 - 0.5 * math.cos(math.pi * env)
                out[first + local] += 0.022 * env * math.sin(step * local)


def _add_pluck(out: list[float], rate: int, start: float, freq: float, amp: float) -> None:
    first = int(start * rate)
    length = min(len(out) - first, int(1.1 * rate))
    for local in range(max(0, length)):
        env = math.exp(-4.3 * local / rate)
        attack = min(1.0, local / max(1, int(0.012 * rate)))
        base = math.sin(2.0 * math.pi * freq * local / rate)
        overtone = 0.32 * math.sin(4.0 * math.pi * freq * local / rate)
        out[first + local] += amp * env * attack * (base + overtone)


def generate_music(duration: float, rate: int) -> list[float]:
    out = [0.0] * int(duration * rate)
    chords = [
        (0.0, 3.2, [146.83, 174.61, 220.00]),
        (3.2, 6.4, [116.54, 146.83, 174.61]),
        (6.4, 9.6, [174.61, 220.00, 261.63]),
        (9.6, 12.8, [130.81, 164.81, 196.00]),
        (12.8, 16.0, [146.83, 174.61, 220.00]),
        (16.0, 19.0, [116.54, 146.83, 174.61]),
    ]
    for start, end, freqs in chords:
        _add_pad(out, rate, start, min(duration, end), freqs)

    melody = [
        (0.55, 587.33, 0.10), (1.18, 698.46, 0.080), (1.82, 880.00, 0.075),
        (2.55, 783.99, 0.070), (3.58, 523.25, 0.090), (4.25, 698.46, 0.075),
        (5.08, 587.33, 0.070), (5.72, 466.16, 0.080), (6.68, 659.25, 0.078),
        (7.45, 880.00, 0.068), (8.28, 783.99, 0.074), (9.02, 523.25, 0.090),
        (9.88, 698.46, 0.070), (10.52, 622.25, 0.070), (11.28, 523.25, 0.078),
        (12.66, 783.99, 0.070), (13.35, 880.00, 0.065), (14.02, 698.46, 0.075),
        (15.22, 587.33, 0.072), (16.05, 523.25, 0.082), (17.10, 466.16, 0.070),
    ]
    for start, freq, amp in melody:
        _add_pluck(out, rate, start, freq, amp)

    for freq, amp in ((73.42, 0.028), (110.00, 0.016)):
        step = 2.0 * math.pi * freq / rate
        for index in range(len(out)):
            out[index] += amp * math.sin(step * index)

    rng = random.Random(7741)
    hits = (0.18, 1.62, 3.18, 4.72, 6.20, 7.92, 9.44, 10.88, 12.62, 14.18, 15.72, 17.34)
    for start in hits:
        first = int(start * rate)
        length = min(len(out) - first, int(0.28 * rate))
        state = 0.0
        for local in range(max(0, length)):
            state = 0.86 * state + 0.14 * rng.uniform(-1.0, 1.0)
            env = math.exp(-22.0 * local / rate)
            out[first + local] += 0.035 * env * state
    return out


def _comb(samples: list[float], rate: int, delay: float, feedback: float, damp: float) -> list[float]:
    delay_count = max(1, int(delay * rate))
    buffer = [0.0] * delay_count
    result = [0.0] * len(samples)
    state = 0.0
    index = 0
    for pos, sample in enumerate(samples):
        delayed = buffer[index]
        state = delayed * (1.0 - damp) + state * damp
        buffer[index] = sample + state * feedback
        result[pos] = delayed
        index = (index + 1) % delay_count
    return result


def _allpass(samples: list[float], rate: int, delay: float, gain: float) -> list[float]:
    delay_count = max(1, int(delay * rate))
    buffer = [0.0] * delay_count
    result = [0.0] * len(samples)
    index = 0
    for pos, sample in enumerate(samples):
        delayed = buffer[index]
        output = delayed - gain * sample
        buffer[index] = sample + gain * output
        result[pos] = output
        index = (index + 1) % delay_count
    return result


def add_reverb(samples: list[float], rate: int, wet: float = 0.22) -> list[float]:
    taps = [_comb(samples, rate, delay, 0.59, 0.24) for delay in (0.0297, 0.0371, 0.0411, 0.0437)]
    reverb = [sum(values) * 0.25 for values in zip(*taps)]
    reverb = _allpass(reverb, rate, 0.0050, 0.50)
    reverb = _allpass(reverb, rate, 0.0017, 0.50)
    return [dry + wet * reflected for dry, reflected in zip(samples, reverb)]


def add_noise(samples: list[float], snr_db: float, seed: int = 57) -> list[float]:
    level = rms(samples)
    if level <= 0:
        return samples
    noise_rms = level / (10.0 ** (snr_db / 20.0))
    rng = random.Random(seed)
    result = [0.0] * len(samples)
    for index, sample in enumerate(samples):
        result[index] = sample + rng.gauss(0.0, noise_rms)
    return result


def build_sample(
    output: str | Path,
    speed: float = SAMPLE_SPEED,
    snr_db: float = 27.0,
) -> dict:
    duration = 18.8
    track = generate_music(duration, SOURCE_RATE)
    for placement in SAMPLE_PLACEMENTS:
        marker = marker_waveform(placement.asset, SOURCE_RATE, placement.damaged)
        _mix_at(track, marker, int(placement.start_seconds * SOURCE_RATE), 0.95)

    track = add_reverb(track, SOURCE_RATE, wet=0.23)
    track = lowpass(track, SOURCE_RATE, 4850.0)
    track = resample_linear(track, SOURCE_RATE, OUTPUT_RATE)
    track, rate = apply_speed(track, OUTPUT_RATE, speed)
    track = add_noise(track, snr_db=snr_db)
    track = normalize(track, 0.93)
    write_wav(output, track, rate)

    proto = PROTOCOL
    truth = []
    for index, placement in enumerate(SAMPLE_PLACEMENTS):
        truth.append(
            {
                "index": index,
                "asset_id": f"{placement.asset.asset_id:04X}",
                "label": placement.asset.label,
                "title": placement.asset.title,
                "source_seconds": placement.start_seconds,
                "expected_seconds": round(placement.start_seconds / speed, 6),
                "damaged": placement.damaged,
                "expected_kind": "candidate" if placement.damaged else "hit",
            }
        )
    return {
        "audio_file": str(output),
        "sample_rate": rate,
        "duration_seconds": round(len(track) / rate, 6),
        "speed_factor": speed,
        "downsample_from": SOURCE_RATE,
        "reverb_wet": 0.23,
        "snr_db": snr_db,
        "frame_seconds": proto.frame_seconds,
        "truth": truth,
    }
