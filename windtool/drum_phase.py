"""Independent cross-check of drum-hit inter-channel phase behaviour."""

from __future__ import annotations

import numpy as np

from .config import AnalysisConfig
from .dsp import FrameFeatures
from .segmentation import Segment


def circular_mean(angles: np.ndarray) -> float:
    return float(np.angle(np.mean(np.exp(1j * angles))))


def circular_concentration(angles: np.ndarray) -> float:
    return float(np.abs(np.mean(np.exp(1j * angles))))


def check_drum_segments(
    segments: list[Segment],
    features: FrameFeatures,
    cfg: AnalysisConfig,
    sr: int,
) -> list[dict]:
    """For each drum segment report phase lag, sample offset and concentration.

    A genuine shared source (kick) should keep a stable, near-zero inter-channel
    phase in the bass band across every hit. Wind produces drifting phase and
    low circular concentration, which is what this check guards against.
    """
    fmask = (features.freqs >= cfg.bass_band_hz[0]) & (
        features.freqs <= cfg.bass_band_hz[1]
    )
    bass_freqs = features.freqs[fmask]
    rows = []
    for seg in segments:
        if seg.label != "drum":
            continue
        sl = (features.times >= seg.start_sec) & (features.times <= seg.end_sec)
        phase = features.cross_phase[np.ix_(sl, fmask)]
        coh = features.coherence[np.ix_(sl, fmask)]

        # Keep only the hit onset frames (top-coherence half). The decaying
        # tail is dominated by ambience whose random phase would otherwise
        # dilute concentration even for a perfectly shared kick source.
        frame_strength = coh.mean(axis=1)
        keep = frame_strength >= np.median(frame_strength)
        phase = phase[keep]
        coh = coh[keep]

        # Phase vs frequency slope => group delay between capsules.
        mean_phase_per_bin = np.array(
            [circular_mean(phase[:, j]) for j in range(phase.shape[1])]
        )
        slope, _ = np.polyfit(bass_freqs, np.unwrap(mean_phase_per_bin), 1)
        group_delay_sec = -slope / (2.0 * np.pi)
        sample_offset = group_delay_sec * sr

        flat_phase = phase.reshape(-1)
        mean_phase = circular_mean(flat_phase)
        concentration = circular_concentration(flat_phase)
        rows.append(
            {
                "segment": seg.index,
                "start_sec": round(seg.start_sec, 3),
                "end_sec": round(seg.end_sec, 3),
                "mean_bass_phase_deg": round(np.degrees(mean_phase), 1),
                "phase_concentration": round(concentration, 3),
                "group_delay_ms": round(group_delay_sec * 1000.0, 3),
                "sample_offset_at_sr": round(sample_offset, 2),
                "mean_bass_coherence": round(float(coh.mean()), 3),
                "verdict": (
                    "stable shared-source phase"
                    if concentration >= 0.7 and abs(mean_phase) <= np.pi / 2
                    else "phase drift - review classification"
                ),
            }
        )
    return rows
