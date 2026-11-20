"""End-to-end offline rendering pipeline and export manifest generation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .audio import (
    RenderConfig,
    render_channels,
    synthesize_mono,
    write_mono_wav,
    write_multichannel_wav,
)
from .geometry import az_el_of_unit
from .layouts import Layout
from .panner import DegradationEvent, VbapPanner
from .trajectory import Trajectory


@dataclass
class ExportResult:
    layout_name: str
    multichannel_wav: Path
    per_channel_wavs: list[Path]
    manifest_path: Path
    num_samples: int
    gain_table: list[tuple[float, ...]]
    degradation_events: list[DegradationEvent]
    degradation_blocks: int
    pan_errors: list[float]


def build_gain_table(
    panner: VbapPanner,
    trajectory: Trajectory,
    config: RenderConfig,
    num_samples: int,
) -> tuple[
    list[tuple[float, ...]],
    list[DegradationEvent],
    list[float],
    list[bool],
]:
    num_blocks = (num_samples + config.block_size - 1) // config.block_size
    gains: list[tuple[float, ...]] = []
    events: list[DegradationEvent] = []
    errors: list[float] = []
    degraded_flags: list[bool] = []
    previous_degraded = False
    for block in range(num_blocks):
        center_sample = min(
            num_samples - 1, block * config.block_size + config.block_size // 2
        )
        time_s = center_sample / config.sample_rate
        direction = trajectory.direction_at(time_s)
        result = panner.pan(direction)
        gains.append(result.gains)
        degraded_flags.append(result.degraded)
        if not result.degraded:
            errors.append(result.angular_error_deg)
            if previous_degraded and events:
                events[-1].end_time_s = round(time_s, 5)
                az_end, el_end, _ = az_el_of_unit(direction)
                events[-1].end_azimuth_deg = round(az_end, 3)
            previous_degraded = False
        else:
            az, el, _ = az_el_of_unit(direction)
            nearest = result.active[0]
            if not previous_degraded:
                events.append(
                    DegradationEvent(
                        frame=center_sample,
                        time_s=round(time_s, 5),
                        azimuth_deg=round(az, 3),
                        elevation_deg=round(el, 3),
                        nearest_speaker=panner.layout.speakers[nearest].name,
                        angular_gap_deg=round(result.angular_error_deg, 3),
                    )
                )
            previous_degraded = True
    return gains, events, errors, degraded_flags


def export_layout(
    layout: Layout,
    trajectory: Trajectory,
    config: RenderConfig,
    output_dir: Path,
) -> ExportResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem_prefix = f"{layout.name}_{trajectory.name}"
    for stale in output_dir.glob(f"{stem_prefix}*"):
        stale.unlink()
    duration_s = trajectory.keyframes[-1].time_s
    num_samples = int(round(duration_s * config.sample_rate))

    signal = synthesize_mono(num_samples, config.sample_rate, config.amplitude)
    panner = VbapPanner(layout)
    gain_table, events, errors, degraded_flags = build_gain_table(
        panner, trajectory, config, num_samples
    )
    channels = render_channels(
        signal, gain_table, config.block_size, len(layout.speakers)
    )

    stem = f"{layout.name}_{trajectory.name}"
    multi_path = output_dir / f"{stem}_multichannel.wav"
    write_multichannel_wav(
        multi_path, channels, config.sample_rate, config.bit_depth
    )

    per_channel_paths: list[Path] = []
    for index, (name, buffer) in enumerate(zip(layout.channel_order, channels)):
        mono_path = output_dir / f"{stem}_ch{index:02d}_{name}.wav"
        write_mono_wav(
            mono_path, buffer, config.sample_rate, config.bit_depth
        )
        per_channel_paths.append(mono_path)

    degradation_blocks = sum(1 for flag in degraded_flags if flag)
    manifest = {
        "layout": layout.name,
        "layout_description": layout.description,
        "trajectory": trajectory.name,
        "sample_rate_hz": config.sample_rate,
        "bit_depth": config.bit_depth,
        "block_size_samples": config.block_size,
        "num_samples": num_samples,
        "duration_s": round(num_samples / config.sample_rate, 5),
        "channel_order": list(layout.channel_order),
        "speaker_positions": {
            speaker.name: list(speaker.position) for speaker in layout.speakers
        },
        "multichannel_wav": multi_path.name,
        "per_channel_wavs": [path.name for path in per_channel_paths],
        "valid_vbap_triplets": len(panner.triplets),
        "degradation": {
            "policy": (
                "directions outside the convex cone of speaker unit vectors "
                "are routed to the nearest single speaker; angular error is "
                "reported per event"
            ),
            "degraded_blocks": degradation_blocks,
            "degraded_block_ratio": round(
                degradation_blocks / len(gain_table), 6
            ),
            "events": [asdict(event) for event in events],
        },
        "mean_pan_error_deg_when_covered": round(
            sum(errors) / len(errors), 6
        )
        if errors
        else None,
        "max_pan_error_deg_when_covered": round(max(errors), 6)
        if errors
        else None,
    }
    manifest_path = output_dir / f"{stem}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return ExportResult(
        layout_name=layout.name,
        multichannel_wav=multi_path,
        per_channel_wavs=per_channel_paths,
        manifest_path=manifest_path,
        num_samples=num_samples,
        gain_table=gain_table,
        degradation_events=events,
        degradation_blocks=degradation_blocks,
        pan_errors=errors,
    )
