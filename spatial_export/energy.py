"""Energy conservation and continuity checks for rendered channel sets."""

from __future__ import annotations

from .audio import synthesize_mono


def verify_energy_continuity(
    channels: list[list[float]],
    block_size: int,
    sample_rate: int,
    amplitude: float,
    gain_table: list[tuple[float, ...]] | None = None,
) -> dict:
    """Total energy is conserved; block-to-block energy changes are smooth.

    Energy is measured as summed squared samples (not block RMS), which is
    immune to the beating envelope of the multi-sine test signal. Per-block
    rendered/source energy ratios stay near 1 with equal-power panning, and
    target gain powers are reported as an independent crossfade check.
    """
    frame_count = min(len(ch) for ch in channels)
    source = synthesize_mono(frame_count, sample_rate, amplitude)
    source_energy = sum(x * x for x in source)
    rendered_energy = sum(
        x * x for buffer in channels for x in buffer[:frame_count]
    )

    block_count = (frame_count + block_size - 1) // block_size
    ratios: list[float] = []
    active_blocks = 0
    for block in range(block_count):
        start = block * block_size
        end = min(start + block_size, frame_count)
        src_e = sum(source[i] ** 2 for i in range(start, end))
        if src_e < 1e-10:
            continue
        active_blocks += 1
        ren_e = sum(
            buffer[i] ** 2
            for buffer in channels
            for i in range(start, end)
        )
        ratios.append(ren_e / src_e)
    ratio_deviation = [abs(r - 1.0) for r in ratios]
    ratio_jumps = [
        abs(ratios[i] - ratios[i - 1]) for i in range(1, len(ratios))
    ]

    gain_power_stats = None
    if gain_table is not None:
        powers = [sum(g * g for g in gains) for gains in gain_table]
        gain_power_stats = {
            "min_block_target_gain_power": round(min(powers), 10),
            "max_block_target_gain_power": round(max(powers), 10),
        }
    return {
        "blocks": block_count,
        "active_blocks": active_blocks,
        "global_rendered_to_source_energy_ratio": round(
            rendered_energy / source_energy, 8
        ),
        "max_block_energy_ratio_deviation_from_unity": round(
            max(ratio_deviation), 8
        ),
        "max_block_to_block_energy_ratio_jump": round(max(ratio_jumps), 8),
        "target_gain_power": gain_power_stats,
    }
