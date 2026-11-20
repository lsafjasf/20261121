"""Generate reproducible room responses and an eight-channel wiring scenario."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import random

from .spectral import fft_convolve
from .wavio import write_wav


@dataclass
class GenerationConfig:
    sample_rate: int = 8000
    duration_seconds: float = 3.2
    reverb_seconds: float = 0.6
    seed: int = 20261121
    snr_db: float = 32.0
    source_lead_seconds: float = 0.12
    source_tail_seconds: float = 0.20
    start_offsets_ms: list[float] = field(
        default_factory=lambda: [0.0, 34.0, 91.0, 17.0, 123.0, 58.0, 7.0, 76.0]
    )


def _band_limited_noise(
    frame_count: int,
    sample_rate: int,
    rng: random.Random,
    lead_samples: int,
    tail_samples: int,
) -> list[float]:
    low_cut = 140.0
    high_cut = min(4200.0, sample_rate * 0.40)
    low_alpha = 1.0 - math.exp(-2.0 * math.pi * low_cut / sample_rate)
    high_alpha = 1.0 - math.exp(-2.0 * math.pi * high_cut / sample_rate)
    low_state = 0.0
    high_state = 0.0
    signal = [0.0] * frame_count
    active_frames = max(1, frame_count - lead_samples - tail_samples)
    fade_frames = max(1, int(0.025 * sample_rate))
    for index in range(frame_count):
        if index < lead_samples or index >= frame_count - tail_samples:
            continue
        low_state += high_alpha * (rng.uniform(-1.0, 1.0) - low_state)
        high_state += low_alpha * (low_state - high_state)
        shaped = low_state - high_state
        position = index - lead_samples
        envelope = 1.0
        if position < fade_frames:
            envelope = position / fade_frames
        elif position >= active_frames - fade_frames:
            envelope = max(0.0, (active_frames - position) / fade_frames)
        signal[index] = shaped * envelope
    rms = math.sqrt(sum(value * value for value in signal) / max(1, active_frames))
    if rms > 0.0:
        signal = [value / rms for value in signal]
    return signal


def _scale_ir_segment(
    values: list[float], positions: list[int], target_energy: float
) -> list[tuple[int, float]]:
    energy = sum(value * value for value in values)
    if energy <= 1e-20:
        return list(zip(positions, values))
    scale = math.sqrt(target_energy / energy)
    return [(position, value * scale) for position, value in zip(positions, values)]


def _make_impulse_response(
    sample_rate: int, reverb_seconds: float, rng: random.Random
) -> list[float]:
    delay = int(round((0.018 + 0.055 * rng.random()) * sample_rate))
    tail_samples = max(1, int(round(reverb_seconds * sample_rate)))
    length = delay + tail_samples
    response = [0.0] * length
    response[delay] = 1.0

    early_times = [int(round(value * sample_rate)) for value in (0.009, 0.019, 0.033, 0.052, 0.071)]
    early_positions = [delay + value for value in early_times if delay + value < length]
    early_values = [
        0.42 * math.exp(-index / 2.3) * (1.0 if rng.random() >= 0.35 else -1.0)
        * (0.75 + 0.5 * rng.random())
        for index in range(len(early_positions))
    ]
    for position, value in _scale_ir_segment(early_values, early_positions, 0.16):
        response[position] += value

    diffuse_start = min(length, delay + int(0.024 * sample_rate))
    diffuse_positions = list(range(diffuse_start, length))
    if diffuse_positions:
        decay = [
            math.exp(-6.9078 * (position - diffuse_start) / max(1, tail_samples - diffuse_start))
            for position in diffuse_positions
        ]
        diffuse_values = [rng.uniform(-1.0, 1.0) * value for value in decay]
        for position, value in _scale_ir_segment(
            diffuse_values, diffuse_positions, 0.22
        ):
            response[position] += value
    return response


def _pad(values: list[float], length: int) -> list[float]:
    if len(values) >= length:
        return values[:length]
    return values + [0.0] * (length - len(values))


def _noise_floor(
    frame_count: int, sample_rate: int, amplitude: float, rng: random.Random
) -> list[float]:
    hum_phase = 2.0 * math.pi * 50.0 / sample_rate
    return [
        amplitude * rng.uniform(-1.0, 1.0)
        + 0.35 * amplitude * math.sin(hum_phase * index)
        for index in range(frame_count)
    ]


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def generate_scenario(
    output_dir: str | Path, config: GenerationConfig | None = None
) -> dict:
    """Create two WAV files plus machine-readable ground truth."""

    settings = config or GenerationConfig()
    if settings.sample_rate < 2000:
        raise ValueError("sample_rate must be at least 2000 for this scenario")
    if settings.duration_seconds <= 0.5:
        raise ValueError("duration_seconds must be greater than 0.5")
    if settings.reverb_seconds <= 0.0:
        raise ValueError("reverb_seconds must be positive")
    if settings.snr_db < 0.0:
        raise ValueError("snr_db must not be negative")
    if len(settings.start_offsets_ms) != 8:
        raise ValueError("start_offsets_ms must contain exactly eight entries")
    if any(value < 0.0 for value in settings.start_offsets_ms):
        raise ValueError("start offsets must not be negative")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame_count = int(round(settings.duration_seconds * settings.sample_rate))
    lead_samples = int(round(settings.source_lead_seconds * settings.sample_rate))
    tail_samples = int(round(settings.source_tail_seconds * settings.sample_rate))
    if lead_samples + tail_samples >= frame_count:
        raise ValueError("source lead and tail leave no active test sequence")

    source_labels = [f"SRC-{index + 1}" for index in range(8)]
    return_labels = [f"SPK-{letter}" for letter in "ABCDEFGH"]
    source_signals: list[list[float]] = []
    for index in range(8):
        rng = random.Random(settings.seed * 101 + index)
        signal = _band_limited_noise(
            frame_count,
            settings.sample_rate,
            rng,
            lead_samples,
            tail_samples,
        )
        source_signals.append([0.18 * value for value in signal])

    ir_rng = random.Random(settings.seed * 103)
    impulse_responses = [
        _make_impulse_response(
            settings.sample_rate, settings.reverb_seconds, ir_rng
        )
        for _ in range(8)
    ]

    drive_sources: list[int | None] = [0, 1, 5, 3, 4, 2, 1, None]
    inverted_channels = {3}
    duplicate_channels = {6}
    open_channels = {7}

    dry_outputs: list[list[float]] = []
    for channel_index, source_index in enumerate(drive_sources):
        if source_index is None:
            dry_outputs.append([0.0] * frame_count)
            continue
        response = fft_convolve(source_signals[source_index], impulse_responses[channel_index])
        if channel_index in inverted_channels:
            response = [-value for value in response]
        dry_outputs.append(response)

    active_peak = max(
        (max((abs(value) for value in response), default=0.0) for response in dry_outputs),
        default=0.0,
    )
    output_scale = 0.72 / active_peak if active_peak > 1e-12 else 1.0
    returns: list[list[float]] = []
    common_length = 0
    for channel_index, response in enumerate(dry_outputs):
        scaled = [output_scale * value for value in response]
        offset = int(
            round(settings.start_offsets_ms[channel_index] * settings.sample_rate / 1000.0)
        )
        prepared = [0.0] * offset + scaled
        common_length = max(common_length, len(prepared))
        returns.append(prepared)

    open_noise_scale = 0.0025
    for channel_index in range(8):
        rng = random.Random(settings.seed * 107 + channel_index)
        if channel_index in open_channels:
            amplitude = open_noise_scale
        else:
            signal_rms = math.sqrt(
                sum(value * value for value in returns[channel_index])
                / max(1, len(returns[channel_index]))
            )
            amplitude = signal_rms * 10.0 ** (-settings.snr_db / 20.0)
        returns[channel_index] = _pad(returns[channel_index], common_length)
        noise = _noise_floor(common_length, settings.sample_rate, amplitude, rng)
        returns[channel_index] = [
            signal + floor for signal, floor in zip(returns[channel_index], noise)
        ]

    reference_path = output / "reference.wav"
    recording_path = output / "recording.wav"
    ir_path = output / "impulse_responses.wav"
    write_wav(reference_path, source_signals, settings.sample_rate)
    write_wav(recording_path, returns, settings.sample_rate)
    ir_length = max(len(response) for response in impulse_responses)
    write_wav(
        ir_path,
        [_pad(response, ir_length) for response in impulse_responses],
        settings.sample_rate,
    )

    channel_truth = []
    for channel_index, source_index in enumerate(drive_sources):
        if channel_index in open_channels:
            status = "open"
            source_label = None
            polarity = "unknown"
        elif channel_index in duplicate_channels:
            status = "duplicate"
            source_label = source_labels[source_index] if source_index is not None else None
            polarity = "inverted" if channel_index in inverted_channels else "normal"
        else:
            status = "ok"
            source_label = source_labels[source_index] if source_index is not None else None
            polarity = "inverted" if channel_index in inverted_channels else "normal"
        channel_truth.append(
            {
                "return_channel": return_labels[channel_index],
                "source": source_label,
                "polarity": polarity,
                "status": status,
            }
        )

    observed_sources = {value for value in drive_sources if value is not None}
    unobserved_sources = [
        source_labels[index]
        for index in range(8)
        if index not in observed_sources
    ]
    ground_truth = {
        "schema_version": 1,
        "scenario": "mixed_faults_and_open",
        "sample_rate": settings.sample_rate,
        "reference_file": reference_path.name,
        "recording_file": recording_path.name,
        "impulse_response_file": ir_path.name,
        "source_labels": source_labels,
        "return_labels": return_labels,
        "source_lead_seconds": settings.source_lead_seconds,
        "start_offsets_ms": settings.start_offsets_ms,
        "channels": channel_truth,
        "duplicate_groups": [[return_labels[1], return_labels[6]]],
        "open_channels": [return_labels[7]],
        "unobserved_sources": unobserved_sources,
        "faults": {
            "mapping": "SPK-C receives SRC-6; SPK-F receives SRC-3",
            "polarity": "SPK-D is inverted",
            "duplicate": "SPK-G duplicates the SRC-2 feed of SPK-B",
            "open": "SPK-H has no test-source signal",
        },
        "forbidden_auto_claims": [
            "SPK-D is not inverted if its polarity is ambiguous",
            "SPK-G is not a duplicate without matching source evidence",
        ],
    }
    manifest = {
        "schema_version": 1,
        "generator": "venue_wiring.synth",
        "config": asdict(settings),
        "files": {
            "reference": reference_path.name,
            "recording": recording_path.name,
            "impulse_responses": ir_path.name,
        },
        "notes": [
            "Each source is an independent band-limited noise sequence.",
            "Every return includes an independently generated room impulse response.",
            "Channel start offsets and reverb tails are intentionally unequal.",
        ],
    }
    _write_json(output / "scenario.json", manifest)
    _write_json(output / "ground_truth.json", ground_truth)
    return ground_truth
