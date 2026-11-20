"""Verification of pan direction, energy continuity and export ordering."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from .audio import RenderConfig, read_wav
from .energy import verify_energy_continuity
from .exporter import ExportResult, build_gain_table
from .geometry import angle_between, az_el_of_unit, normalize
from .layouts import Layout
from .panner import VbapPanner
from .trajectory import Trajectory


@dataclass
class VerificationReport:
    layout_name: str
    checks: dict
    passed: bool


def _direction_from_gains(gains: tuple[float, ...], layout: Layout):
    vec = [0.0, 0.0, 0.0]
    for gain, speaker in zip(gains, layout.speakers):
        unit = speaker.unit
        for axis in range(3):
            vec[axis] += gain * unit[axis]
    magnitude = math.sqrt(sum(v * v for v in vec))
    if magnitude <= 1e-12:
        return None, 0.0
    return normalize(tuple(vec)), magnitude


def verify_pan_direction(
    layout: Layout,
    trajectory: Trajectory,
    config: RenderConfig,
    num_samples: int,
) -> dict:
    """Target vs reconstructed direction, per block (covered blocks only)."""
    panner = VbapPanner(layout)
    gain_table, _, _, _ = build_gain_table(
        panner, trajectory, config, num_samples
    )
    errors: list[float] = []
    degraded_blocks = 0
    samples: list[dict] = []
    for block, gains in enumerate(gain_table):
        center = min(
            num_samples - 1,
            block * config.block_size + config.block_size // 2,
        )
        time_s = center / config.sample_rate
        target = trajectory.direction_at(time_s)
        reconstructed, _ = _direction_from_gains(gains, layout)
        result = panner.pan(target)
        if result.degraded or reconstructed is None:
            degraded_blocks += 1
            continue
        error = math.degrees(angle_between(target, reconstructed))
        errors.append(error)
        if block % max(1, len(gain_table) // 8) == 0:
            target_az_el = az_el_of_unit(target)
            rec_az_el = az_el_of_unit(reconstructed)
            samples.append(
                {
                    "block": block,
                    "time_s": round(time_s, 3),
                    "target_az_el_deg": [
                        round(target_az_el[0], 2),
                        round(target_az_el[1], 2),
                    ],
                    "rendered_az_el_deg": [
                        round(rec_az_el[0], 2),
                        round(rec_az_el[1], 2),
                    ],
                    "angular_error_deg": round(error, 4),
                    "active_speakers": [
                        layout.speakers[i].name
                        for i, g in enumerate(gains)
                        if g > 1e-9
                    ],
                }
            )
    return {
        "covered_blocks": len(errors),
        "degraded_blocks_excluded": degraded_blocks,
        "max_angular_error_deg": round(max(errors), 6),
        "mean_angular_error_deg": round(sum(errors) / len(errors), 6),
        "sample_blocks": samples,
    }



def verify_export_order(
    layout: Layout, result: ExportResult, config: RenderConfig
) -> dict:
    """Manifest order, mono files and the interleaved WAV must all agree."""
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    expected_order = list(layout.channel_order)
    manifest_matches = manifest["channel_order"] == expected_order

    multi_channels, rate, bit_depth = read_wav(result.multichannel_wav)
    header_ok = (
        rate == config.sample_rate
        and bit_depth == config.bit_depth
        and len(multi_channels) == len(expected_order)
    )

    per_channel_match = True
    max_sample_diff = 0.0
    for index, mono_path in enumerate(result.per_channel_wavs):
        mono_channels, _, _ = read_wav(mono_path)
        if len(mono_channels) != 1:
            per_channel_match = False
            continue
        for a, b in zip(mono_channels[0], multi_channels[index]):
            max_sample_diff = max(max_sample_diff, abs(a - b))
        if mono_channels[0] != multi_channels[index]:
            per_channel_match = False

    filename_tags_ok = all(
        f"ch{index:02d}_{name}.wav" in path.name
        for index, (name, path) in enumerate(
            zip(expected_order, result.per_channel_wavs)
        )
    )
    return {
        "expected_order": expected_order,
        "manifest_order": manifest["channel_order"],
        "manifest_matches": manifest_matches,
        "wav_header_ok": header_ok,
        "interleaved_vs_mono_identical": per_channel_match,
        "max_sample_diff": round(max_sample_diff, 10),
        "filename_channel_tags_ok": filename_tags_ok,
    }


def verify_all(
    layout: Layout,
    trajectory: Trajectory,
    config: RenderConfig,
    result: ExportResult,
) -> VerificationReport:
    pan = verify_pan_direction(
        layout, trajectory, config, result.num_samples
    )
    channels, _, _ = read_wav(result.multichannel_wav)
    energy = verify_energy_continuity(
        channels,
        config.block_size,
        config.sample_rate,
        config.amplitude,
        result.gain_table,
    )
    order = verify_export_order(layout, result, config)
    checks = {
        "pan_direction": pan,
        "energy_continuity": energy,
        "export_order": order,
        "degradation": {
            "degraded_blocks": result.degradation_blocks,
            "degraded_block_ratio": round(
                result.degradation_blocks / len(result.gain_table), 6
            ),
            "events": [
                {
                    "time_s": event.time_s,
                    "azimuth_deg": event.azimuth_deg,
                    "elevation_deg": event.elevation_deg,
                    "nearest_speaker": event.nearest_speaker,
                    "angular_gap_deg": event.angular_gap_deg,
                }
                for event in result.degradation_events
            ],
        },
    }
    passed = (
        pan["max_angular_error_deg"] < 1.0
        and energy["active_blocks"] == energy["blocks"]
        and abs(energy["global_rendered_to_source_energy_ratio"] - 1.0) < 0.02
        and energy["max_block_energy_ratio_deviation_from_unity"] < 0.03
        and energy["max_block_to_block_energy_ratio_jump"] < 0.03
        and order["manifest_matches"]
        and order["wav_header_ok"]
        and order["interleaved_vs_mono_identical"]
        and order["filename_channel_tags_ok"]
    )
    return VerificationReport(layout_name=layout.name, checks=checks, passed=passed)
