"""Turn frame features into classified, persistent low-frequency events."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import AnalysisConfig
from .dsp import FrameFeatures


@dataclass
class Segment:
    index: int
    start_sec: float
    end_sec: float
    label: str = "uncertain"
    attenuation_db: float = 0.0
    metrics: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


LABEL_NAMES = {
    "drum": "correlated bass / kick",
    "wind": "independent wind noise",
    "bump": "one-sided mic bump",
    "mixed_wind": "wind with partial correlation",
    "clean": "quiet / clean",
    "uncertain": "uncertain",
}


def _activity_frames(features: FrameFeatures, cfg: AnalysisConfig) -> np.ndarray:
    low = features.band_energy_db["wind_20_200"]
    floor = np.percentile(low, 20)
    return low > (floor + cfg.activity_relative_threshold_db)


def build_segments(
    features: FrameFeatures, cfg: AnalysisConfig
) -> list[Segment]:
    """Gate low-band activity, merge brief gaps, enforce minimum duration."""
    active = _activity_frames(features, cfg)
    hop = float(np.median(np.diff(features.times)))
    min_frames = max(1, int(round(cfg.min_segment_sec / hop)))
    gap_frames = max(1, int(round(cfg.merge_gap_sec / hop)))

    filled = active.copy()
    gap = 0
    run_start = None
    for i, flag in enumerate(active):
        if flag:
            if run_start is not None and 0 < gap <= gap_frames:
                filled[run_start + 1 : i] = True
            run_start = i
            gap = 0
        elif run_start is not None:
            gap += 1

    segments: list[Segment] = []
    idx = 0
    start = None
    for i, flag in enumerate(filled):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_frames:
                segments.append(
                    Segment(
                        index=idx,
                        start_sec=float(features.times[start]),
                        end_sec=float(features.times[i - 1]),
                    )
                )
                idx += 1
            start = None
    if start is not None and len(filled) - start >= min_frames:
        segments.append(
            Segment(
                index=idx,
                start_sec=float(features.times[start]),
                end_sec=float(features.times[-1]),
            )
        )
    return segments


def _segment_slice(features: FrameFeatures, seg: Segment) -> np.ndarray:
    return (features.times >= seg.start_sec) & (features.times <= seg.end_sec)


def _band_mean_coherence(
    features: FrameFeatures, sl: np.ndarray, band: tuple[float, float]
) -> float:
    fmask = (features.freqs >= band[0]) & (features.freqs <= band[1])
    return float(features.coherence[np.ix_(sl, fmask)].mean())


def classify_segments(
    segments: list[Segment], features: FrameFeatures, cfg: AnalysisConfig
) -> list[Segment]:
    segments = split_bumps_within_wind(segments, features, cfg)
    return _classify(segments, features, cfg)


def split_bumps_within_wind(
    segments: list[Segment],
    features: FrameFeatures,
    cfg: AnalysisConfig,
) -> list[Segment]:
    """Carve one-sided ILD spikes out as dedicated bump sub-segments.

    A handling bump can sit inside a longer wind gust; energy gating alone
    merges them. Frames whose ILD exceeds the stricter bump threshold form
    short sub-segments that get classified separately.
    """
    result: list[Segment] = []
    next_index = 0
    hop = float(np.median(np.diff(features.times)))
    min_bump_frames = max(2, int(round(0.05 / hop)))
    for seg in segments:
        sl = _segment_slice(features, seg)
        frame_idx = np.flatnonzero(sl)
        spike = np.abs(features.ild_db[sl]) >= cfg.bump_peak_ild_db
        if not np.any(spike):
            seg.index = next_index
            next_index += 1
            result.append(seg)
            continue

        runs: list[tuple[int, int]] = []
        start = None
        for i, flag in enumerate(spike):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                runs.append((start, i))
                start = None
        if start is not None:
            runs.append((start, len(spike)))

        cursor = frame_idx[0]
        for run_start, run_end in runs:
            a = frame_idx[run_start]
            b = frame_idx[run_end - 1] + 1
            if b - a < min_bump_frames:
                continue
            if a > cursor:
                result.append(
                    Segment(next_index, float(features.times[cursor]),
                            float(features.times[a - 1]))
                )
                next_index += 1
            result.append(
                Segment(next_index, float(features.times[a]),
                        float(features.times[min(b - 1, len(features.times) - 1)]))
            )
            next_index += 1
            cursor = b
        if cursor <= frame_idx[-1]:
            result.append(
                Segment(next_index, float(features.times[cursor]),
                        float(features.times[frame_idx[-1]]))
            )
            next_index += 1
    return result


def _classify(
    segments: list[Segment], features: FrameFeatures, cfg: AnalysisConfig
) -> list[Segment]:
    """Rule-based classifier with an explicit, auditable reason list."""
    for seg in segments:
        sl = _segment_slice(features, seg)
        wind_coh = _band_mean_coherence(features, sl, cfg.wind_band_hz)
        bass_coh = _band_mean_coherence(features, sl, cfg.bass_band_hz)
        fmask = (features.freqs >= cfg.bass_band_hz[0]) & (
            features.freqs <= cfg.bass_band_hz[1]
        )
        peak_coh = float(np.percentile(features.coherence[np.ix_(sl, fmask)], 90))
        ild_abs = float(np.percentile(np.abs(features.ild_db[sl]), 80))
        ild_peak = float(np.max(np.abs(features.ild_db[sl])))
        ild_mean = float(np.mean(features.ild_db[sl]))
        crest = float(np.median(features.crest[sl]))
        low_frac = float(np.median(features.low_fraction[sl]))
        duration = seg.duration_sec

        energy = {
            name: float(np.median(features.band_energy_db[name][sl]))
            for name in cfg.bands
        }

        reasons: list[str] = []
        reasons.append(
            f"low-band coherence {wind_coh:.2f} (bass {bass_coh:.2f}), "
            f"threshold {cfg.coherent_threshold:.2f}"
        )
        reasons.append(
            f"p90 bass coherence {peak_coh:.2f} "
            f"(peak threshold {cfg.peak_coherent_threshold:.2f})"
        )
        reasons.append(
            f"ILD p80 {ild_abs:.1f} dB / peak {ild_peak:.1f} dB "
            f"(mean {ild_mean:+.1f} dB), "
            f"bump threshold {cfg.bump_ild_db:.1f} dB"
        )
        reasons.append(
            f"duration {duration:.2f} s vs wind persistence "
            f"{cfg.wind_persistence_sec:.2f} s / transient "
            f"{cfg.transient_persistence_sec:.2f} s"
        )
        reasons.append(f"low-band crest {crest:.1f} dB, low-band share {low_frac:.2f}")

        # Decision order matters: one-sided handling first (extreme ILD can
        # still look coherent on the loud channel), then coherent bass, then
        # persistence-driven wind.
        short_event = duration < cfg.transient_persistence_sec
        sharp_ild_spike = (
            ild_peak >= cfg.bump_peak_ild_db
            and short_event
            and ild_abs >= cfg.bump_ild_db
        )
        if sharp_ild_spike or (
            ild_abs >= cfg.bump_ild_db and wind_coh < cfg.coherent_threshold
        ):
            label = "bump"
            reasons.append(
                "-> bump: dominant energy on one capsule with weak cross-channel lock"
            )
        elif (
            peak_coh >= cfg.peak_coherent_threshold
            and bass_coh >= cfg.coherent_threshold
            and crest >= cfg.drum_crest_db
        ):
            label = "drum"
            reasons.append(
                "-> drum: coherent low-frequency phase lock plus high crest transient"
            )
        elif (
            wind_coh >= cfg.coherent_threshold
            and duration >= cfg.wind_persistence_sec
            and crest < cfg.drum_crest_db
        ):
            label = "mixed_wind"
            reasons.append(
                "-> mixed wind: sustained partly-coherent low energy, reduced gently"
            )
        elif (
            wind_coh < cfg.coherent_threshold
            and low_frac >= cfg.low_band_fraction
            and (duration >= cfg.wind_persistence_sec or not short_event)
        ):
            label = "wind"
            reasons.append(
                "-> wind: incoherent low-band energy concentrated below 200 Hz"
            )
        elif short_event and wind_coh < cfg.coherent_threshold and ild_abs >= 8.0:
            label = "bump"
            reasons.append(
                "-> bump: short incoherent one-sided event without phase lock"
            )
        else:
            label = "uncertain"
            reasons.append("-> uncertain: no rule matched, conservative light reduction")

        seg.label = label
        seg.attenuation_db = float(cfg.attenuation_db[label])
        seg.metrics = {
            "coherence_wind_band": wind_coh,
            "coherence_bass_band": bass_coh,
            "peak_coherence_bass_p90": peak_coh,
            "ild_p80_db": ild_abs,
            "ild_peak_db": ild_peak,
            "ild_mean_db": ild_mean,
            "crest_db": crest,
            "low_band_fraction": low_frac,
            "duration_sec": duration,
            "band_energy_dbfs_median": energy,
        }
        seg.reasons = reasons
    return segments
