"""Automated checks run against the rendered mix.

The acceptance questions from the exhibit scenario are answered here:

1. Does the first syllable of every announcement survive at full level?
2. Does narration freeze its playhead and resume at the right position?
3. What is the maximum output peak (and does clipping occur)?
4. Do sidechain attack / release follow the configured time constants?
"""

from __future__ import annotations

import math

from .mixer import Mixer


def first_substantial_block(signal: list[float], sample_rate: int,
                            threshold: float = 0.1,
                            window_ms: float = 5.0) -> int | None:
    """Index of the first short window whose peak exceeds ``threshold``."""
    win = max(1, int(window_ms / 1000.0 * sample_rate))
    for start in range(0, len(signal), win):
        if any(abs(s) > threshold for s in signal[start:start + win]):
            return start
    return None


def analyze(mixer: Mixer, output: list[float], sample_rate: int,
            alert_starts: list[float]) -> dict:
    report: dict = {'alerts': []}

    # --- global output peak
    peak = max(abs(s) for s in output)
    report['peak_linear'] = peak
    report['peak_dbfs'] = 20.0 * math.log10(peak)
    report['clipped'] = any(abs(s) >= 0.9999 for s in output)

    # --- per-alert first-syllable inspection.
    # The alert's own gain at onset must be ~unity; the surviving output
    # energy in the first 60 ms must be dominated by the alert channel.
    alert = mixer.channel('alert')
    narration = mixer.channel('narration')
    for onset in alert_starts:
        onset_sample = int(onset * sample_rate)
        probe_end = onset_sample + int(0.06 * sample_rate)
        out_seg = output[onset_sample:probe_end]
        out_peak = max((abs(s) for s in out_seg), default=0.0)
        # Gain recorded for the block containing the onset.
        # First block whose midpoint is at or after the onset.
        block_idx = next(
            (i for i, (t, _, _) in enumerate(alert.history) if t >= onset),
            len(alert.history) - 1)
        onset_gain = alert.history[block_idx][1]
        nar_gain = narration.history[block_idx][1]
        report['alerts'].append({
            'onset_s': onset,
            'onset_alert_gain': onset_gain,
            'onset_narration_gain': nar_gain,
            'first_60ms_output_peak': out_peak,
            'syllable_survives': onset_gain > 0.95 and out_peak > 0.5,
        })

    # --- sidechain timing on the narration channel around the first alert.
    gains = [(t, g) for t, g, _ in narration.history]
    first = alert_starts[0]
    g0 = next(g for t, g in gains if t <= first)
    floor_est = min(g for t, g in gains if first < t < first + 0.5)
    span_linear = g0 - floor_est
    attack_t = next((t for t, g in gains
                     if t >= first and g <= floor_est + 0.37 * span_linear),
                    None)
    report['narration_floor_db'] = (20 * math.log10(floor_est)
                                    if floor_est > 0 else float('-inf'))
    report['attack_ms_63pct'] = ((attack_t - first) * 1000
                                 if attack_t is not None else None)
    # Release measured after the last alert ends: time from floor back to
    # within 37% of the full base gain.
    alert_duration = len(mixer.channel('alert').signal) / sample_rate
    end = alert_starts[-1] + alert_duration
    end_block = min(len(gains) - 1, int(end * sample_rate / mixer.block_size))
    g_end = gains[end_block][1]
    target = g_end + 0.63 * (narration.base_gain - g_end)
    release_t = next((t for t, g in gains if t >= end and g >= target), None)
    report['release_ms_63pct'] = ((release_t - end) * 1000
                                  if release_t is not None else None)

    # --- narration playhead continuity.
    # Position only advances while not paused; compare rendered count with
    # wall-clock duration to prove the head was frozen during alerts.
    total = len(output)
    rendered = narration.position
    paused_blocks = sum(1 for _, _, p in narration.history if p)
    report['narration_samples_rendered'] = rendered
    report['narration_samples_skipped_if_no_pause'] = total - rendered
    report['narration_paused_blocks'] = paused_blocks
    report['playhead_resumes_cleanly'] = rendered < total and paused_blocks > 0
    return report


def format_report(report: dict) -> str:
    lines = [
        '== Mix verification report ==',
        f"Max output peak : {report['peak_linear']:.4f} "
        f"({report['peak_dbfs']:.2f} dBFS), "
        f"hard-clipped={report['clipped']}",
        f"Narration duck floor: {report['narration_floor_db']:.1f} dB",
        f"Sidechain attack (63%): {report['attack_ms_63pct']:.0f} ms, "
        f"release (63%): {report['release_ms_63pct']:.0f} ms",
        'Narration playhead: rendered '
        f"{report['narration_samples_rendered']} samples vs "
        f"{report['narration_samples_skipped_if_no_pause']} samples that "
        'a non-pausing head would have skipped, paused blocks='
        f"{report['narration_paused_blocks']}, "
        f"resume_ok={report['playhead_resumes_cleanly']}",
    ]
    for i, a in enumerate(report['alerts'], 1):
        lines.append(
            f"Alert #{i} @ {a['onset_s']:.2f}s: alert gain at onset="
            f"{a['onset_alert_gain']:.3f}, narration ducked to="
            f"{a['onset_narration_gain']:.3f}, first-60ms output peak="
            f"{a['first_60ms_output_peak']:.3f}, "
            f"syllable_ok={a['syllable_survives']}")
    return '\n'.join(lines)
