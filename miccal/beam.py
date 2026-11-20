from __future__ import annotations

from dataclasses import dataclass
import html
import math
from pathlib import Path
from typing import Sequence

from .config import Scene
from .estimate import CalibrationResult
from .geometry import add, centroid, distance, dot, norm, scale, sub, unit


@dataclass(frozen=True)
class BeamPattern:
    angles_deg: tuple[float, ...]
    before_db: tuple[float, ...]
    after_db: tuple[float, ...]
    before_peak_deg: float
    after_peak_deg: float
    after_minus6db_width_deg: float | None


def _plane_basis(scene: Scene, source_position: tuple[float, float, float]):
    center = centroid(scene.microphones)
    forward = unit(sub(source_position, center))
    helper = (0.0, 0.0, 1.0)
    cross = (
        helper[1] * forward[2] - helper[2] * forward[1],
        helper[2] * forward[0] - helper[0] * forward[2],
        helper[0] * forward[1] - helper[1] * forward[0],
    )
    if norm(cross) < 1e-6:
        helper = (0.0, 1.0, 0.0)
        cross = (
            helper[1] * forward[2] - helper[2] * forward[1],
            helper[2] * forward[0] - helper[0] * forward[2],
            helper[0] * forward[1] - helper[1] * forward[0],
        )
    side = unit(cross)
    vertical = unit(
        (
            side[1] * forward[2] - side[2] * forward[1],
            side[2] * forward[0] - side[0] * forward[2],
            side[0] * forward[1] - side[1] * forward[0],
        )
    )
    return center, forward, vertical


def compute_beam_pattern(
    scene: Scene,
    result: CalibrationResult,
    angle_step_deg: float = 2.0,
    source_index: int = 0,
) -> BeamPattern:
    source = scene.sources[source_index].position
    center, forward, vertical = _plane_basis(scene, source)
    radius = distance(source, center)
    order = result.best_map
    mics = tuple(scene.installed_microphones()[index] for index in order)
    reference = scene.reference_channel
    gains = result.gain_relative
    max_gain = max(abs(value) for value in gains)
    if max_gain <= 1e-15:
        max_gain = 1.0
    raw_weights = [gain / max_gain for gain in gains]
    corrected_weights = [
        max_gain / value if abs(value) > 1e-15 else max_gain
        for value in gains
    ]
    measured_delays = result.source_estimates[source_index].relative_delays_s
    frequencies = scene.analysis.beam_frequencies
    count = int(round(360.0 / angle_step_deg))
    angles = tuple(-180.0 + index * angle_step_deg for index in range(count + 1))
    before: list[float] = []
    after: list[float] = []
    for angle in angles:
        radians = math.radians(angle)
        direction = add(
            scale(forward, math.cos(radians)),
            scale(vertical, math.sin(radians)),
        )
        virtual_source = add(center, scale(direction, radius))
        delays = []
        base = distance(virtual_source, mics[reference])
        for mic in mics:
            delays.append((distance(virtual_source, mic) - base) / scene.sound_speed)
        raw_power = 0.0
        corrected_power = 0.0
        for frequency in frequencies:
            omega = 2.0 * math.pi * frequency
            raw_sum = 0j
            corrected_sum = 0j
            for channel in range(4):
                transfer = complex(
                    math.cos(-omega * delays[channel]),
                    math.sin(-omega * delays[channel]),
                ) * raw_weights[channel]
                steering = complex(
                    math.cos(omega * measured_delays[channel]),
                    math.sin(omega * measured_delays[channel]),
                )
                raw_sum += transfer
                corrected_sum += corrected_weights[channel] * steering * transfer
            raw_power += (abs(raw_sum) / 4.0) ** 2
            corrected_power += (abs(corrected_sum) / 4.0) ** 2
        before.append(_power_db(raw_power / len(frequencies)))
        after.append(_power_db(corrected_power / len(frequencies)))
    before_peak = angles[max(range(len(before)), key=lambda index: before[index])]
    after_peak = angles[max(range(len(after)), key=lambda index: after[index])]
    width = _minus6db_width(angles, after)
    return BeamPattern(
        angles_deg=angles,
        before_db=tuple(before),
        after_db=tuple(after),
        before_peak_deg=before_peak,
        after_peak_deg=after_peak,
        after_minus6db_width_deg=width,
    )


def _power_db(power: float) -> float:
    return 10.0 * math.log10(max(power, 1e-12))


def _minus6db_width(angles: Sequence[float], values: Sequence[float]) -> float | None:
    center_index = min(range(len(angles)), key=lambda index: abs(angles[index]))
    if values[center_index] < -6.0:
        return None
    left = center_index
    while left > 0 and values[left - 1] >= -6.0:
        left -= 1
    right = center_index
    while right + 1 < len(values) and values[right + 1] >= -6.0:
        right += 1
    return angles[right] - angles[left]


def write_pattern_csv(pattern: BeamPattern, path: str | Path) -> None:
    lines = ["angle_deg,before_db,after_db"]
    for angle, before, after in zip(
        pattern.angles_deg, pattern.before_db, pattern.after_db
    ):
        lines.append(f"{angle:.3f},{before:.6f},{after:.6f}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pattern_svg(
    pattern: BeamPattern,
    path: str | Path,
    title: str,
    subtitle: str,
    floor_db: float = -30.0,
) -> None:
    width, height = 980, 760
    cx, cy, radius = 360.0, 390.0, 285.0
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            'role="img" aria-labelledby="title desc">'
        ),
        f'<title id="title">{html.escape(title)}</title>',
        f'<desc id="desc">{html.escape(subtitle)}</desc>',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        (
            '<text x="42" y="48" font-family="sans-serif" font-size="25" '
            'font-weight="700" fill="#17202a">'
            f"{html.escape(title)}</text>"
        ),
        (
            '<text x="42" y="78" font-family="sans-serif" font-size="14" '
            'fill="#566573">'
            f"{html.escape(subtitle)}</text>"
        ),
    ]
    for db_value in (0, -10, -20, -30):
        ring = radius * (db_value - floor_db) / (-floor_db)
        parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="{ring:.2f}" fill="none" '
            'stroke="#d5d8dc" stroke-width="1"/>'
        )
        label_y = cy - ring + 16
        parts.append(
            f'<text x="{cx + 5}" y="{label_y:.2f}" font-family="sans-serif" '
            f'font-size="11" fill="#7b7d7d">{db_value} dB</text>'
        )
    for angle in range(0, 360, 30):
        radians = math.radians(angle)
        x = cx + radius * math.sin(radians)
        y = cy - radius * math.cos(radians)
        parts.append(
            f'<line x1="{cx}" y1="{cy}" x2="{x:.2f}" y2="{y:.2f}" '
            'stroke="#e5e7e9" stroke-width="1"/>'
        )
        label_x = cx + (radius + 22) * math.sin(radians)
        label_y = cy - (radius + 22) * math.cos(radians) + 4
        parts.append(
            f'<text x="{label_x:.2f}" y="{label_y:.2f}" text-anchor="middle" '
            'font-family="sans-serif" font-size="11" fill="#707b7c">'
            f"{angle}°</text>"
        )
    parts.extend(
        [
            (
                f'<line x1="{cx}" y1="{cy - radius - 12}" '
                f'x2="{cx}" y2="{cy - radius + 10}" stroke="#7f8c8d" '
                'stroke-width="1.5" stroke-dasharray="4 4"/>'
            ),
            (
                f'<text x="{cx}" y="{cy - radius - 20}" text-anchor="middle" '
                'font-family="sans-serif" font-size="12" fill="#566573">'
                "source direction (0°)</text>"
            ),
        ]
    )
    parts.append(_polar_path(pattern.angles_deg, pattern.before_db, cx, cy, radius, floor_db, "#c0392b"))
    parts.append(_polar_path(pattern.angles_deg, pattern.after_db, cx, cy, radius, floor_db, "#1f618d"))
    legend_x, legend_y = 690, 165
    parts.extend(
        [
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 36}" y2="{legend_y}" stroke="#c0392b" stroke-width="3"/>',
            f'<text x="{legend_x + 48}" y="{legend_y + 5}" font-family="sans-serif" font-size="14" fill="#283747">before calibration</text>',
            f'<line x1="{legend_x}" y1="{legend_y + 34}" x2="{legend_x + 36}" y2="{legend_y + 34}" stroke="#1f618d" stroke-width="3"/>',
            f'<text x="{legend_x + 48}" y="{legend_y + 39}" font-family="sans-serif" font-size="14" fill="#283747">after calibration</text>',
        ]
    )
    width_text = (
        f"{pattern.after_minus6db_width_deg:.1f}°"
        if pattern.after_minus6db_width_deg is not None
        else "n/a"
    )
    info_lines = [
        f"raw main-lobe peak: {pattern.before_peak_deg:+.1f}°",
        f"calibrated peak: {pattern.after_peak_deg:+.1f}°",
        f"calibrated -6 dB width: {width_text}",
        "near-field polar cut, broadband 1.2-4.8 kHz",
    ]
    for index, line in enumerate(info_lines):
        parts.append(
            f'<text x="{legend_x}" y="{legend_y + 88 + index * 24}" '
            'font-family="sans-serif" font-size="13" fill="#566573">'
            f"{html.escape(line)}</text>"
        )
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts) + "\n", encoding="utf-8")


def _polar_path(
    angles: Sequence[float],
    values: Sequence[float],
    cx: float,
    cy: float,
    radius: float,
    floor_db: float,
    color: str,
) -> str:
    commands: list[str] = []
    for index, (angle, value) in enumerate(zip(angles, values)):
        clipped = max(floor_db, min(0.0, value))
        radial = radius * (clipped - floor_db) / (-floor_db)
        radians = math.radians(angle)
        x = cx + radial * math.sin(radians)
        y = cy - radial * math.cos(radians)
        commands.append(
            f"{'M' if index == 0 else 'L'} {x:.2f} {y:.2f}"
        )
    return (
        f'<path d="{" ".join(commands)}" fill="none" stroke="{color}" '
        'stroke-width="2.6" stroke-linejoin="round" stroke-linecap="round"/>'
    )
