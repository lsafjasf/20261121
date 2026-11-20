"""Tunable analysis parameters.

All thresholds are expressed in physical units (Hz, dB, seconds) so they can be
reviewed without reading the DSP code.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AnalysisConfig:
    # STFT framing (seconds/hop). 2048-point window at 48 kHz gives ~23 Hz
    # bins, fine enough to resolve the 20-200 Hz wind band.
    frame_sec: float = 0.0427
    hop_sec: float = 0.0107

    # Frequency bands used both for energy reporting and classification.
    wind_band_hz: tuple[float, float] = (20.0, 200.0)
    bass_band_hz: tuple[float, float] = (40.0, 120.0)
    mid_band_hz: tuple[float, float] = (200.0, 2000.0)

    # Coherence estimation: cross-spectrum is averaged over this many frames
    # (~43 ms at the default hop) before magnitude-squared coherence is taken.
    coherence_smooth_frames: int = 4

    # Classification thresholds.
    coherent_threshold: float = 0.55       # mean |C|^2 above this => correlated
    peak_coherent_threshold: float = 0.85  # peak-frame coherence (drum onset)
    bump_ild_db: float = 12.0              # inter-channel level difference
    wind_persistence_sec: float = 0.25     # sustained incoherent low energy
    transient_persistence_sec: float = 0.12
    drum_crest_db: float = 6.0
    bump_peak_ild_db: float = 15.0         # ILD spike splitting a wind segment
    low_band_fraction: float = 0.35        # low-band share of total energy

    # Adaptive attenuation (dB), keyed by label. None means "no reduction".
    attenuation_db: dict = field(
        default_factory=lambda: {
            "wind": 15.0,
            "bump": 12.0,
            "mixed_wind": 9.0,
            "drum": 0.0,
            "clean": 0.0,
            "uncertain": 3.0,
        }
    )
    # Gain interpolation anchors (Hz, fraction of full attenuation). Full
    # cut up to 200 Hz (the wind band), linear fade across 200-500 Hz so
    # voice/instrument content above 500 Hz is never touched.
    gain_anchors_hz: tuple[float, float, float] = (20.0, 200.0, 500.0)
    gain_anchor_fractions: tuple[float, float, float] = (1.0, 1.0, 0.0)
    attack_sec: float = 0.05
    release_sec: float = 0.15

    # Segment building: frames with low-band energy above this percentile of
    # the file-level floor start a candidate event.
    activity_relative_threshold_db: float = 8.0
    min_segment_sec: float = 0.08
    merge_gap_sec: float = 0.10

    @property
    def bands(self) -> dict[str, tuple[float, float]]:
        return {
            "wind_20_200": self.wind_band_hz,
            "bass_40_120": self.bass_band_hz,
            "mid_200_2000": self.mid_band_hz,
        }
