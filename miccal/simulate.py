from __future__ import annotations

import json
import math
from pathlib import Path
import random
from typing import Any

from .config import Scene, Vec3
from .dsp import (
    exponential_sweep,
    ifft,
    next_power_of_two,
    real_fft,
    write_wav,
)
from .geometry import distance, image_source, relative_delays


def _event_spectrum(
    events: list[tuple[float, float]],
    size: int,
    sample_rate: int,
) -> list[complex]:
    spectrum: list[complex] = [0j] * size
    delays = [(time * sample_rate, amplitude) for time, amplitude in events]
    for bin_index in range(size // 2 + 1):
        angle = -2.0 * math.pi * bin_index / size
        value = 0j
        for delay_samples, amplitude in delays:
            value += amplitude * complex(
                math.cos(angle * delay_samples),
                math.sin(angle * delay_samples),
            )
        spectrum[bin_index] = value
    spectrum[size // 2] = complex(spectrum[size // 2].real, 0.0)
    for bin_index in range(1, size // 2):
        spectrum[size - bin_index] = spectrum[bin_index].conjugate()
    return spectrum


def channel_events(
    scene: Scene,
    source: Vec3,
    channel: int,
) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    physical_index = scene.installation.channel_order[channel]
    mic = scene.installed_microphones()[physical_index]
    direct_distance = distance(source, mic)
    direct_time = direct_distance / scene.sound_speed
    gain = scene.installation.gain[channel]
    polarity = scene.installation.polarity[channel]
    events = [(direct_time, gain * polarity / direct_distance)]
    details: dict[str, Any] = {
        "direct_time_s": direct_time,
        "direct_amplitude": gain * polarity / direct_distance,
        "reflection_time_s": None,
        "reflection_amplitude": None,
        "reflection_peak_delay_s": None,
    }
    if scene.wall is not None:
        image = image_source(source, scene.wall.point, scene.wall.normal)
        reflection_distance = distance(image, mic)
        reflection_time = reflection_distance / scene.sound_speed
        coefficient = scene.wall.reflection_amplitudes[channel]
        reflection_amplitude = gain * polarity * coefficient / reflection_distance
        events.append((reflection_time, reflection_amplitude))
        details.update(
            {
                "reflection_time_s": reflection_time,
                "reflection_amplitude": reflection_amplitude,
                "reflection_peak_delay_s": reflection_time - direct_time,
            }
        )
    return events, details


def synthesize_recording(
    scene: Scene,
    source_index: int,
    seed: int | None = None,
) -> list[list[float]]:
    source = scene.sources[source_index].position
    size = next_power_of_two(scene.recording_samples)
    sweep = exponential_sweep(scene.sweep, scene.sample_rate)
    excitation = real_fft(sweep, size)
    rng = random.Random(
        seed if seed is not None else 1000 + source_index
    )
    outputs: list[list[float]] = []
    for channel in range(4):
        events, _ = channel_events(scene, source, channel)
        transfer = _event_spectrum(events, size, scene.sample_rate)
        product = [
            excitation[index] * transfer[index] for index in range(size)
        ]
        impulse_response = ifft(product)
        recording = [value.real for value in impulse_response[: scene.recording_samples]]
        if scene.sweep.noise_rms > 0:
            recording = [
                value + rng.gauss(0.0, scene.sweep.noise_rms)
                for value in recording
            ]
        outputs.append(recording)
    peak = max(max(abs(value) for value in channel) for channel in outputs)
    if peak > 0.98:
        gain = 0.98 / peak
        outputs = [[value * gain for value in channel] for channel in outputs]
    return outputs


def truth_manifest(scene: Scene) -> dict[str, Any]:
    installed = []
    for channel in range(4):
        physical_index = scene.installation.channel_order[channel]
        installed.append(
            {
                "channel": channel,
                "nominal_microphone": physical_index,
                "installed_position": list(
                    scene.installed_microphones()[physical_index]
                ),
                "gain": scene.installation.gain[channel],
                "polarity": scene.installation.polarity[channel],
            }
        )
    sources = []
    for source_index, source in enumerate(scene.sources):
        ordered_mics = tuple(
            scene.installed_microphones()[index]
            for index in scene.installation.channel_order
        )
        direct = relative_delays(
            ordered_mics,
            source.position,
            scene.reference_channel,
            scene.sound_speed,
        )
        channels = []
        for channel in range(4):
            events, details = channel_events(scene, source.position, channel)
            details["events"] = [
                {"time_s": time, "amplitude": amplitude}
                for time, amplitude in events
            ]
            channels.append(details)
        sources.append(
            {
                "name": source.name,
                "position": list(source.position),
                "true_relative_delays_s": list(direct),
                "channels": channels,
            }
        )
    return {
        "scene": scene.name,
        "sample_rate": scene.sample_rate,
        "sound_speed": scene.sound_speed,
        "reference_channel": scene.reference_channel,
        "microphones_nominal": [list(m) for m in scene.microphones],
        "channels": installed,
        "sources": sources,
        "reflection_amplitudes": (
            list(scene.wall.reflection_amplitudes) if scene.wall else None
        ),
    }


def simulate_scene(
    scene: Scene,
    output_dir: str | Path,
    seed: int | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    for index, source in enumerate(scene.sources):
        channels = synthesize_recording(scene, index, seed)
        filename = f"{source.name}.wav"
        write_wav(output / filename, channels, scene.sample_rate)
        files.append(filename)
    manifest = truth_manifest(scene)
    manifest["recordings"] = files
    (output / "truth.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    scene.save(output / "scene.json")
    return manifest
