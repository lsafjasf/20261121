"""WAV rendering, gain-curve CSV logs and SVG plots (stdlib only)."""

from __future__ import annotations

import csv
import math
import wave
from array import array

from .mixer import Mixer


def write_wav(path: str, samples: list[float], sample_rate: int) -> float:
    """Write 16-bit PCM mono WAV; return the peak amplitude in dBFS."""
    pcm = array('h', [0] * len(samples))
    peak = 0.0
    for i, s in enumerate(samples):
        clamped = max(-1.0, min(1.0, s))
        value = int(round(clamped * 32767))
        pcm[i] = value
        peak = max(peak, abs(clamped))
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return 20.0 * math.log10(peak) if peak > 0 else float('-inf')


def write_gain_csv(path: str, mixer: Mixer) -> None:
    """One row per processed block: every channel gain + output peaks."""
    names = [ch.name for ch in mixer.channels]
    with open(path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['time_s']
                        + [f'gain_{n}' for n in names]
                        + [f'paused_{n}' for n in names]
                        + ['raw_peak', 'clipped_peak'])
        for i, (time_s, raw_peak, clip_peak) in enumerate(mixer.output_log):
            row = [f'{time_s:.5f}']
            for ch in mixer.channels:
                row.append(f'{ch.history[i][1]:.6f}')
            for ch in mixer.channels:
                row.append(str(int(ch.history[i][2])))
            row += [f'{raw_peak:.6f}', f'{clip_peak:.6f}']
            writer.writerow(row)


def write_svg(path: str, mixer: Mixer, output: list[float],
              sample_rate: int, title: str = '') -> None:
    """Two-panel SVG: channel gain curves above, output waveform below."""
    width, height = 1100, 620
    margin_l, margin_r, margin_t = 70, 30, 50
    panel_h = 230
    gap = 70
    plot_w = width - margin_l - margin_r
    total_t = len(output) / sample_rate
    colors = {'ambient': '#2e7d32', 'narration': '#1565c0', 'alert': '#c62828'}

    def x_of(t: float) -> float:
        return margin_l + t / total_t * plot_w

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        'font-family="sans-serif">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" font-size="20" text-anchor="middle" '
        f'font-weight="bold">{title}</text>',
    ]

    # ---- Panel 1: gain curves (log-ish dB scale 0 dB .. -30 dB)
    g_top = margin_t
    parts.append(f'<text x="{margin_l}" y="{g_top - 12}" font-size="14" '
                 'font-weight="bold">Channel gain (dB)</text>')
    for db in (0, -6, -12, -20, -30):
        y = g_top + (-db) / 30.0 * panel_h
        parts.append(f'<line x1="{margin_l}" y1="{y:.1f}" '
                     f'x2="{margin_l + plot_w}" y2="{y:.1f}" '
                     'stroke="#e0e0e0"/>')
        parts.append(f'<text x="{margin_l - 8}" y="{y + 4:.1f}" '
                     f'font-size="11" text-anchor="end">{db} dB</text>')

    def db_y(gain: float, top: float) -> float:
        db = 20 * math.log10(max(gain, 1e-4))
        return top + min(30.0, max(0.0, -db)) / 30.0 * panel_h

    for ch in mixer.channels:
        color = colors.get(ch.name, '#444')
        pts = ' '.join(f'{x_of(t):.1f},{db_y(g, g_top):.1f}'
                       for t, g, _ in ch.history)
        parts.append(f'<polyline points="{pts}" fill="none" '
                     f'stroke="{color}" stroke-width="1.8"/>')
    for i, ch in enumerate(mixer.channels):
        color = colors.get(ch.name, '#444')
        lx = margin_l + plot_w - 210 + i * 72
        parts.append(f'<rect x="{lx}" y="{g_top - 10}" width="12" '
                     f'height="12" fill="{color}"/>')
        parts.append(f'<text x="{lx + 16}" y="{g_top}" '
                     f'font-size="12">{ch.name}</text>')

    # ---- Panel 2: output waveform (min/max per bucket)
    w_top = g_top + panel_h + gap
    parts.append(f'<text x="{margin_l}" y="{w_top - 12}" font-size="14" '
                 'font-weight="bold">Mixed output waveform (linear)</text>')
    mid = w_top + panel_h / 2
    for level in (-1, -0.5, 0, 0.5, 1):
        y = mid - level * panel_h / 2
        parts.append(f'<line x1="{margin_l}" y1="{y:.1f}" '
                     f'x2="{margin_l + plot_w}" y2="{y:.1f}" '
                     'stroke="#e0e0e0"/>')
        parts.append(f'<text x="{margin_l - 8}" y="{y + 4:.1f}" '
                     f'font-size="11" text-anchor="end">{level:.1f}</text>')
    buckets = plot_w
    bucket_size = max(1, len(output) // buckets)
    pts_top, pts_bot = [], []
    for b in range(buckets):
        seg = output[b * bucket_size:(b + 1) * bucket_size]
        if not seg:
            continue
        hi = max(seg)
        lo = min(seg)
        x = margin_l + b
        pts_top.append(f'{x:.1f},{mid - hi * panel_h / 2:.1f}')
        pts_bot.append(f'{x:.1f},{mid - lo * panel_h / 2:.1f}')
    parts.append(f'<polygon points="{" ".join(pts_top)} '
                 f'{" ".join(reversed(pts_bot))}" fill="#546e7a" '
                 'opacity="0.8"/>')

    # Shared time axis ticks.
    for sec in range(int(total_t) + 1):
        x = x_of(sec)
        parts.append(f'<line x1="{x:.1f}" y1="{w_top}" x2="{x:.1f}" '
                     f'y2="{w_top + panel_h}" stroke="#eeeeee"/>')
        parts.append(f'<text x="{x:.1f}" y="{w_top + panel_h + 18}" '
                     f'font-size="11" text-anchor="middle">{sec}s</text>')
    parts.append('</svg>')
    with open(path, 'w') as fh:
        fh.write('\n'.join(parts))
