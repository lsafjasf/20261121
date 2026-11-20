"""Reference-pilot tracking for tape wow/flutter restoration.

The pilot is a near-sinusoidal reference tone whose instantaneous frequency
is modulated by the same time-base error (wow/flutter + slow drift) as the
program audio.  A software phase-locked loop (PLL) locks onto the analytic
signal of the pilot and returns, on a regular frame grid:

* instantaneous pilot frequency (Hz),
* unwrapped pilot phase (rad),
* a per-frame confidence in [0, 1].

Confidence combines local pilot energy (a lock detector) and phase-error
statistics.  Frames whose confidence is below threshold are reported as
dropouts so the warp estimator can extrapolate instead of trusting noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass
class PilotTrack:
    """Frame-rate pilot observations."""

    frame_rate: float
    frame_times: np.ndarray
    freq: np.ndarray
    phase: np.ndarray
    confidence: np.ndarray
    reliable: np.ndarray
    nominal_freq: float


def _bandpass_analytic(x: np.ndarray, fs: float, f0: float, bw: float) -> np.ndarray:
    """Band-pass around the pilot and return the analytic signal."""
    low = max(f0 - bw / 2.0, 1.0)
    high = min(f0 + bw / 2.0, fs / 2.0 - 1.0)
    if high <= low:
        raise ValueError("pilot band does not fit inside the Nyquist band")
    sos = signal.butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    filtered = signal.sosfiltfilt(sos, x)
    return signal.hilbert(filtered)


def _frame_stats(
    raw: np.ndarray, analytic: np.ndarray, hop: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """In-band analytic amplitude, in-band RMS and raw RMS per frame."""
    n = len(analytic)
    n_frames = max(1, (n - hop) // hop + 1)
    end = (n_frames - 1) * hop + hop
    a_frames = analytic[:end].reshape(n_frames, hop)
    x_frames = raw[:end].reshape(n_frames, hop)
    amp = np.mean(np.abs(a_frames), axis=1)
    in_band_rms = np.sqrt(np.mean(np.abs(a_frames) ** 2, axis=1)) / np.sqrt(2.0)
    raw_rms = np.sqrt(np.mean(x_frames ** 2, axis=1))
    return amp, in_band_rms, raw_rms


def track_pilot(
    pilot: np.ndarray,
    fs: float,
    nominal_freq: float,
    *,
    frame_hop_ms: float = 10.0,
    bandwidth_hz: float = 200.0,
    pll_bandwidth_hz: float = 8.0,
    confidence_threshold: float = 0.25,
) -> PilotTrack:
    """Track a pilot tone and estimate frame-rate frequency/phase/confidence.

    Parameters
    ----------
    pilot:
        Pilot-channel samples (mono).
    fs:
        Sample rate in Hz.
    nominal_freq:
        Expected pilot frequency when the machine runs at nominal speed.
    frame_hop_ms:
        Output frame spacing in milliseconds.
    bandwidth_hz:
        Band-pass width (Hz) around the pilot used for lock detection.
    pll_bandwidth_hz:
        Closed-loop bandwidth of the software PLL in Hz.  Must be wide enough
        to follow the fastest flutter to be corrected but well below the
        pilot frequency.
    confidence_threshold:
        Frames below this confidence are flagged as unreliable.
    """
    pilot = np.asarray(pilot, dtype=np.float64)
    if pilot.ndim != 1:
        raise ValueError("pilot must be mono")
    if not 0.0 < nominal_freq < fs / 2.0:
        raise ValueError("nominal_freq must lie between 0 and Nyquist")
    if not 0.0 < pll_bandwidth_hz < nominal_freq / 2.0:
        raise ValueError("PLL bandwidth must be well below the pilot frequency")

    # The band-pass must be wide enough to pass the FM side-bands created by
    # wow/flutter; a narrow filter would erase the very modulation we measure.
    analytic = _bandpass_analytic(pilot, fs, nominal_freq, bandwidth_hz)
    hop = max(1, int(round(fs * frame_hop_ms / 1000.0)))
    frame_rate = fs / hop

    # Analytic phase carries the full FM modulation.  A first-order software
    # PLL smooths the instantaneous frequency and *holds* its last estimate
    # through dropouts instead of tracking noise.  The raw pilot amplitude
    # provides a per-sample gate used both here and for frame confidence.
    raw_phase = np.unwrap(np.angle(analytic))
    raw_freq = np.empty_like(raw_phase)
    raw_freq[0] = nominal_freq
    raw_freq[1:] = np.diff(raw_phase) * fs / (2.0 * np.pi)
    inst_amp = np.abs(analytic)
    amp_floor = np.median(inst_amp) * 0.3
    sample_gate = inst_amp > amp_floor

    alpha = 2.0 * np.pi * pll_bandwidth_hz / fs
    n = len(analytic)
    nco_freq = float(nominal_freq)
    inst_freq = np.empty(n)
    for i in range(n):
        if sample_gate[i]:
            nco_freq += alpha * (raw_freq[i] - nco_freq)
        # When gated off (dropout), freeze: no update at all.
        inst_freq[i] = nco_freq

    n_frames = max(1, (n - hop) // hop + 1)
    end = (n_frames - 1) * hop + hop

    def frame_mean(a: np.ndarray) -> np.ndarray:
        return a[:end].reshape(n_frames, hop).mean(axis=1)

    amp, in_band_rms, raw_rms = _frame_stats(pilot, analytic, hop)
    freq = frame_mean(inst_freq)
    gate_frames = sample_gate[:end].reshape(n_frames, hop).mean(axis=1)
    phase_err = frame_mean(raw_freq - inst_freq) / nominal_freq * (np.pi / 2.0)
    times = (np.arange(n_frames) + 0.5) * hop / fs

    # Lock detector from absolute pilot level (robust to dropouts): the
    # analytic amplitude of band-limited noise stays constant, so an energy
    # ratio alone cannot distinguish silence/noise from a real locked tone.
    floor = np.median(amp) * 0.25
    level_score = np.clip((amp - floor) / (np.median(amp) - floor + 1e-12), 0.0, 1.0)
    # SNR cross-check: pilot present => in-band power dominates raw noise.
    snr = in_band_rms / np.maximum(raw_rms, 1e-12)
    snr_score = np.clip((snr - 0.2) / 0.5, 0.0, 1.0)
    err_score = np.clip(1.0 - np.abs(phase_err) / (np.pi / 2.0), 0.0, 1.0)
    confidence = np.clip(
        0.5 * level_score
        + 0.2 * snr_score
        + 0.15 * err_score
        + 0.15 * gate_frames,
        0.0,
        1.0,
    )
    reliable = confidence >= confidence_threshold

    # Unwrapped phase of the tracked pilot, anchored at zero at t=0.
    phase = np.concatenate(([0.0], np.cumsum(2.0 * np.pi * freq[:-1]) / frame_rate))

    return PilotTrack(
        frame_rate=frame_rate,
        frame_times=times,
        freq=freq,
        phase=phase,
        confidence=confidence,
        reliable=reliable,
        nominal_freq=float(nominal_freq),
    )
