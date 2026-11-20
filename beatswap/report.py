"""Alignment evidence: JSON report + self-contained SVG plots."""

from __future__ import annotations

import json

import numpy as np


def _polyline(xs, ys, x0, y0, w, h, xmin, xmax, ymin, ymax):
    if xmax <= xmin:
        xmax = xmin + 1
    if ymax <= ymin:
        ymax = ymin + 1
    pts = []
    for x, y in zip(xs, ys):
        px = x0 + (x - xmin) / (xmax - xmin) * w
        py = y0 + h - (y - ymin) / (ymax - ymin) * h
        pts.append(f"{px:.1f},{py:.1f}")
    return " ".join(pts)


def _series_svg(xs, ys, x0, y0, w, h, color, xmin, xmax, ymin, ymax):
    pts = _polyline(xs, ys, x0, y0, w, h, xmin, xmax, ymin, ymax)
    return (f'<polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="1"/>')


def _decimate(x, y, n=800):
    if len(x) <= n:
        return x, y
    idx = np.linspace(0, len(x) - 1, n).astype(int)
    return np.asarray(x)[idx], np.asarray(y)[idx]


def plot_envelopes(path, t_a, env_a, t_b, env_b, cand_windows_a, title):
    """Two stacked onset-envelope panels with candidate windows shaded."""
    W, H, PH = 900, 420, 180
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" '
             f'height="{H}" font-family="monospace" font-size="12">',
             f'<rect width="{W}" height="{H}" fill="white"/>',
             f'<text x="20" y="20">{title}</text>']
    xa, ya = _decimate(np.asarray(t_a), np.asarray(env_a))
    xb, yb = _decimate(np.asarray(t_b), np.asarray(env_b))
    for row, (xs, ys, color, label) in enumerate([
            (xa, ya, "#1f77b4", "A onset envelope"),
            (xb, yb, "#d62728", "B onset envelope")]):
        y0 = 40 + row * (PH + 20)
        x0, w = 60, W - 90
        ymax = float(ys.max()) if ys.size else 1.0
        xmax = float(xs.max()) if xs.size else 1.0
        parts.append(f'<text x="10" y="{y0 + 15}" fill="{color}">{label}</text>')
        parts.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{PH}" '
                     f'fill="none" stroke="#ccc"/>')
        if row == 0:
            for (s, e) in cand_windows_a:
                px0 = x0 + s / xmax * w
                px1 = x0 + e / xmax * w
                parts.append(f'<rect x="{px0:.1f}" y="{y0}" '
                             f'width="{max(px1 - px0, 1.0):.1f}" height="{PH}" '
                             f'fill="#2ca02c" fill-opacity="0.18"/>')
        parts.append(_series_svg(xs, ys, x0, y0, w, PH, color,
                                 0.0, xmax, 0.0, ymax))
        parts.append(f'<text x="{x0}" y="{y0 + PH + 14}" fill="#888">0s</text>')
        parts.append(f'<text x="{x0 + w - 40}" y="{y0 + PH + 14}" '
                     f'fill="#888">{xmax:.1f}s</text>')
    parts.append('</svg>')
    with open(path, "w") as f:
        f.write("\n".join(parts))


def plot_dtw(path, pa, pb, fps, title):
    """DTW warping path: A time (x) vs B time (y), plus identity line."""
    W = H = 460
    x0, y0, w, h = 60, 30, W - 90, H - 80
    xs, ys = _decimate(pa / fps, pb / fps)
    xmax = float(xs.max())
    ymax = float(max(ys.max(), xmax))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" '
             f'height="{H}" font-family="monospace" font-size="12">',
             f'<rect width="{W}" height="{H}" fill="white"/>',
             f'<text x="20" y="20">{title}</text>',
             f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="none" '
             f'stroke="#ccc"/>']
    parts.append(_series_svg([0, ymax], [0, ymax], x0, y0, w, h, "#bbbbbb",
                             0.0, xmax, 0.0, ymax))
    parts.append(_series_svg(xs, ys, x0, y0, w, h, "#1f77b4",
                             0.0, xmax, 0.0, ymax))
    parts.append(f'<text x="{x0}" y="{y0 + h + 16}" fill="#888">A time (s)'
                 f'</text>')
    parts.append(f'<text x="8" y="{y0 + 12}" fill="#888">B (s)</text>')
    parts.append('</svg>')
    with open(path, "w") as f:
        f.write("\n".join(parts))


def plot_scores(path, entries, scores, chosen_entries, title):
    """Candidate score vs entry beat index."""
    W, H = 900, 260
    x0, y0, w, h = 60, 30, W - 90, H - 70
    xs = np.asarray(entries, float)
    ys = np.asarray(scores, float)
    xmax = float(xs.max()) if xs.size else 1.0
    ymin = float(min(0.0, ys.min())) if ys.size else 0.0
    ymax = float(max(1e-6, ys.max())) if ys.size else 1.0
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" '
             f'height="{H}" font-family="monospace" font-size="12">',
             f'<rect width="{W}" height="{H}" fill="white"/>',
             f'<text x="20" y="20">{title}</text>',
             f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="none" '
             f'stroke="#ccc"/>']
    parts.append(_series_svg(xs, ys, x0, y0, w, h, "#1f77b4",
                             0.0, xmax, ymin, ymax))
    for e in chosen_entries:
        px = x0 + e / xmax * w
        parts.append(f'<line x1="{px:.1f}" y1="{y0}" x2="{px:.1f}" '
                     f'y2="{y0 + h}" stroke="#2ca02c" stroke-width="2"/>')
    parts.append(f'<text x="{x0}" y="{y0 + h + 16}" fill="#888">entry beat '
                 f'index (green = reported candidates)</text>')
    parts.append('</svg>')
    with open(path, "w") as f:
        f.write("\n".join(parts))


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
