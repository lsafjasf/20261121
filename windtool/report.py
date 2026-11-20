"""Figures and self-contained HTML audition report."""

from __future__ import annotations

import base64
import html
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .dsp import FrameFeatures
from .reduction import band_rms_db
from .segmentation import LABEL_NAMES, Segment

LABEL_COLORS = {
    "drum": "#2c7fb8",
    "wind": "#d95f0e",
    "bump": "#7a0177",
    "mixed_wind": "#e6550d",
    "clean": "#31a354",
    "uncertain": "#969696",
}


def plot_overview(
    features: FrameFeatures,
    segments: list[Segment],
    gain: np.ndarray,
    path: str,
) -> None:
    t = features.times
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(11, 9))

    for seg in segments:
        color = LABEL_COLORS[seg.label]
        for ax in axes:
            ax.axvspan(seg.start_sec, seg.end_sec, color=color, alpha=0.15)

    ax = axes[0]
    for name, style in (
        ("wind_20_200", "-"),
        ("bass_40_120", "--"),
        ("mid_200_2000", ":"),
    ):
        ax.plot(t, features.band_energy_db[name], style, label=name, lw=1.2)
    ax.set_ylabel("band energy (dBFS)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Per-frame band energy with classified segments")

    ax = axes[1]
    fmask = (features.freqs >= 20) & (features.freqs <= 500)
    im = ax.pcolormesh(
        t,
        features.freqs[fmask],
        features.coherence[:, fmask].T,
        shading="auto",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    ax.set_ylabel("frequency (Hz)")
    ax.set_title("Cross-channel magnitude-squared coherence (20-500 Hz)")
    fig.colorbar(im, ax=ax, label="|C|$^2$")

    ax = axes[2]
    ax.plot(t, features.ild_db, color="#7a0177", lw=1.2)
    ax.axhline(12, color="k", ls="--", lw=0.8)
    ax.axhline(-12, color="k", ls="--", lw=0.8)
    ax.set_ylabel("ILD (dB)")
    ax.set_title("Low-band inter-channel level difference ch1 - ch2")

    ax = axes[3]
    low_gain = gain[:, features.freqs <= 200].min(axis=1)
    ax.plot(t, 20 * np.log10(np.maximum(low_gain, 1e-6)), color="#d95f0e", lw=1.2)
    ax.set_ylabel("gain (dB)")
    ax.set_xlabel("time (s)")
    ax.set_ylim(-20, 1)
    ax.set_title("Applied low-band adaptive gain")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _b64(path: str) -> str:
    with open(path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("ascii")


def _segment_rows(segments: list[Segment]) -> str:
    rows = []
    for seg in segments:
        energy = seg.metrics.get(
            "band_energy_dbfs_median",
            {"wind_20_200": float("nan"), "bass_40_120": float("nan"),
             "mid_200_2000": float("nan")},
        )
        reason = "<br>".join(html.escape(r) for r in seg.reasons)
        rows.append(
            "<tr>"
            f"<td>{seg.index}</td>"
            f"<td>{seg.start_sec:.2f}–{seg.end_sec:.2f}</td>"
            f"<td style='color:{LABEL_COLORS[seg.label]}'>"
            f"<b>{html.escape(LABEL_NAMES[seg.label])}</b></td>"
            f"<td>{seg.attenuation_db:.0f} dB</td>"
            f"<td>{seg.metrics['coherence_wind_band']:.2f}/"
            f"{seg.metrics['coherence_bass_band']:.2f}</td>"
            f"<td>{seg.metrics['ild_p80_db']:.1f}</td>"
            f"<td>{seg.metrics['crest_db']:.1f}</td>"
            f"<td>{seg.metrics['low_band_fraction']:.2f}</td>"
            f"<td>{energy['wind_20_200']:.1f} / "
            f"{energy['bass_40_120']:.1f} / "
            f"{energy['mid_200_2000']:.1f}</td>"
            f"<td style='font-size:12px'>{reason}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def write_html(
    path: str,
    title: str,
    segments: list[Segment],
    figure_path: str,
    original_wav: str,
    processed_wav: str,
    wind_rows: list[dict],
    drum_rows: list[dict],
    file_band_energy: dict,
) -> None:
    wind_html = "".join(
        f"<tr><td>{r['segment']}</td><td>{r['label']}</td>"
        f"<td>{r['start_sec']:.2f}–{r['end_sec']:.2f}</td>"
        f"<td>{r['requested_db']:.0f}</td>"
        f"<td><b>{r['measured_low_band_reduction_db']:.2f}</b></td>"
        f"<td>{r['before_dbfs']:.1f} → {r['after_dbfs']:.1f}</td></tr>"
        for r in wind_rows
    ) or "<tr><td colspan='6'>No wind/bump segments detected.</td></tr>"
    drum_html = "".join(
        f"<tr><td>{r['segment']}</td><td>{r['start_sec']:.2f}–{r['end_sec']:.2f}</td>"
        f"<td>{r['mean_bass_phase_deg']:.1f}°</td>"
        f"<td>{r['phase_concentration']:.2f}</td>"
        f"<td>{r['group_delay_ms']:.2f} ms</td>"
        f"<td>{r['sample_offset_at_sr']:.1f} samples</td>"
        f"<td>{r['mean_bass_coherence']:.2f}</td>"
        f"<td>{html.escape(r['verdict'])}</td></tr>"
        for r in drum_rows
    ) or "<tr><td colspan='8'>No drum segments detected.</td></tr>"
    band_html = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v:.2f} dBFS</td></tr>"
        for k, v in file_band_energy.items()
    )
    img = _b64(figure_path)
    orig_b64 = _b64(original_wav)
    proc_b64 = _b64(processed_wav)

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: -apple-system, Segoe UI, sans-serif; margin: 24px; color:#222; }}
h1 {{ font-size: 20px; }} h2 {{ font-size: 16px; margin-top: 28px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left;
  vertical-align: top; }}
th {{ background: #f0f0f0; }}
.players {{ display: flex; gap: 32px; flex-wrap: wrap; }}
.players figure {{ margin: 0; }}
caption {{ caption-side: top; text-align: left; font-weight: 600; padding: 4px 0; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p>Offline two-channel wind-noise annotation. Classification uses low-frequency
cross-channel coherence, inter-channel level difference, event persistence and
crest factor; attenuation is adaptive per segment and limited to the low band.</p>

<h2>Audition</h2>
<div class="players">
<figure><figcaption>Original</figcaption>
<audio controls src="data:audio/wav;base64,{orig_b64}"></audio></figure>
<figure><figcaption>Processed (adaptive low-band reduction)</figcaption>
<audio controls src="data:audio/wav;base64,{proc_b64}"></audio></figure>
</div>

<h2>File-level band energy</h2>
<table><tr><th>band</th><th>RMS (dBFS)</th></tr>{band_html}</table>

<h2>Segments: energy and classification evidence</h2>
<table><tr><th>#</th><th>time (s)</th><th>label</th><th>target cut</th>
<th>coh wind/bass</th><th>ILD p80</th><th>crest</th><th>low share</th>
<th>energy wind/bass/mid (dBFS)</th><th>classification basis</th></tr>
{_segment_rows(segments)}
</table>

<h2>Drum-hit phase cross-check</h2>
<table><tr><th>seg</th><th>time</th><th>mean phase</th><th>concentration</th>
<th>group delay</th><th>offset</th><th>coherence</th><th>verdict</th></tr>
{drum_html}</table>

<h2>Wind-segment reduction check (measured on rendered audio)</h2>
<table><tr><th>seg</th><th>label</th><th>time</th><th>requested</th>
<th>measured 20-200 Hz cut</th><th>before → after</th></tr>
{wind_html}</table>

<h2>Overview figure</h2>
<img alt="analysis overview" style="max-width:100%"
 src="data:image/png;base64,{img}">
</body></html>"""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(doc)


def file_band_energy(audio: np.ndarray, sr: int, bands: dict) -> dict:
    return {
        name: round(band_rms_db(audio, sr, band), 2) for name, band in bands.items()
    }
