//! Round-trip verification: playback curve followed by the guarded inverse.

use crate::engine::{
    build_chain, multitone_signal, run_chain, time_error, CurveKind, EngineConfig, TimeError,
};
use crate::response::{analog_playback_db, analog_playback_db_rel_1k, to_db, warped_analog_hz};
use crate::wav::WavData;

/// Frequencies used by the multitone time-domain probe.
pub const PROBE_FREQUENCIES_HZ: [f64; 7] = [30.0, 60.0, 120.0, 250.0, 500.0, 1_000.0, 4_000.0];

/// Probe duration and edge fade, milliseconds.
pub const PROBE_DURATION_MS: u32 = 1_000;
pub const PROBE_FADE_MS: u32 = 8;

/// One row of the frequency-domain round-trip table.
#[derive(Debug, Clone, Copy)]
pub struct FrequencyRow {
    pub digital_hz: f64,
    pub warped_analog_hz: f64,
    pub playback_db: f64,
    pub playback_rel_1k_db: f64,
    pub inverse_db: f64,
    pub roundtrip_db: f64,
    pub analog_playback_db: f64,
    pub analog_rel_1k_db: f64,
}

/// Default audit points, including all three RIAA turnovers and 1 kHz.
pub fn audit_frequencies_hz() -> Vec<f64> {
    let mut rows = vec![
        20.0, 50.0, 100.0, 200.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 5_000.0, 10_000.0, 15_000.0,
        20_000.0,
    ];
    rows.extend_from_slice(&crate::riaa::riaa_corners_hz());
    rows.sort_by(|a, b| a.partial_cmp(b).unwrap());
    rows.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    rows
}

/// Build the full playback -> inverse round-trip table.
pub fn frequency_table(config: EngineConfig) -> Vec<FrequencyRow> {
    let fs = config.sample_rate;
    let playback_chain = build_chain(CurveKind::Playback, fs, config.guard_fraction, 1);
    let inverse_chain = build_chain(CurveKind::Inverse, fs, config.guard_fraction, 1);
    audit_frequencies_hz()
        .into_iter()
        .filter(|hz| *hz < fs / 2.0)
        .map(|hz| {
            let theta = 2.0 * std::f64::consts::PI * hz / fs;
            let playback_db = playback_chain[0].coeffs.response_db(hz, fs);
            let inverse_db = inverse_chain[0].coeffs.response_db(hz, fs);
            let roundtrip_mag = playback_chain[0].coeffs.response(theta).abs()
                * inverse_chain[0].coeffs.response(theta).abs();
            FrequencyRow {
                digital_hz: hz,
                warped_analog_hz: warped_analog_hz(hz, fs),
                playback_db,
                playback_rel_1k_db: playback_chain[0].coeffs.response_db_rel_1k(hz, fs),
                inverse_db,
                roundtrip_db: to_db(roundtrip_mag),
                analog_playback_db: analog_playback_db(hz),
                analog_rel_1k_db: analog_playback_db_rel_1k(hz),
            }
        })
        .collect()
}

/// Round-trip result for one tail-handling policy.
#[derive(Debug, Clone)]
pub struct RoundtripResult {
    pub flush: bool,
    pub time_error: TimeError,
    pub output_peak: f64,
    pub flush_taps: Vec<usize>,
}

/// Feed a multitone through playback then inverse and compare to the source.
pub fn roundtrip_time_domain(config: EngineConfig, flush: bool) -> RoundtripResult {
    let fs = config.sample_rate;
    let source: WavData = multitone_signal(
        fs,
        PROBE_DURATION_MS,
        &PROBE_FREQUENCIES_HZ,
        PROBE_FADE_MS,
        1,
    );
    let mut playback_chain = build_chain(CurveKind::Playback, fs, config.guard_fraction, 1);
    let played = run_chain(&mut playback_chain, &source, config, false);
    let equalized = WavData {
        sample_rate: fs as u32,
        channels: 1,
        format: source.format,
        samples: played.samples,
    };
    let mut inverse_chain = build_chain(CurveKind::Inverse, fs, config.guard_fraction, 1);
    let outcome = run_chain(&mut inverse_chain, &equalized, config, flush);
    let error = if flush {
        let appended: usize = outcome.flush_taps.iter().sum();
        let body_len = outcome.samples.len() - appended;
        let trim = (fs * PROBE_FADE_MS as f64 / 1000.0) as usize;
        time_error(
            &outcome.samples[trim..body_len.saturating_sub(trim)],
            &source.samples[trim..body_len.saturating_sub(trim)],
            0,
        )
    } else {
        let trim = (fs * PROBE_FADE_MS as f64 / 1000.0) as usize;
        time_error(&outcome.samples, &source.samples, trim)
    };
    RoundtripResult {
        flush,
        time_error: error,
        output_peak: outcome.peak,
        flush_taps: outcome.flush_taps,
    }
}

/// Pass criterion: in the core audio band (<=4 kHz) round-trip deviation
/// stays below 0.2 dB (the expected guard-pole contribution at 4 kHz is
/// ~0.15 dB), and steady-state component amplitude error stays below 2%.
pub fn passes(config: EngineConfig) -> (bool, f64, f64) {
    let max_db_deviation = frequency_table(config)
        .iter()
        .filter(|row| row.digital_hz <= 4_000.0)
        .map(|row| row.roundtrip_db.abs())
        .fold(0.0_f64, f64::max);
    let amplitude = amplitude_roundtrip(config);
    let pass = max_db_deviation < 0.2 && amplitude.max_relative < 0.02;
    (pass, max_db_deviation, amplitude.max_relative)
}

/// Per-component amplitude estimate by single-bin DFT correlation, robust
/// against the small group delay of the IIR cascade (which shifts samples and
/// makes pointwise peak comparison meaningless).
pub fn component_amplitudes(signal: &[f64], frequencies_hz: &[f64], sample_rate: f64) -> Vec<f64> {
    let n = signal.len() as f64;
    frequencies_hz
        .iter()
        .map(|&frequency| {
            let (mut cos_sum, mut sin_sum) = (0.0_f64, 0.0_f64);
            for (index, sample) in signal.iter().enumerate() {
                let phase = 2.0 * std::f64::consts::PI * frequency * index as f64 / sample_rate;
                cos_sum += sample * phase.cos();
                sin_sum += sample * phase.sin();
            }
            2.0 * (cos_sum * cos_sum + sin_sum * sin_sum).sqrt() / n
        })
        .collect()
}

/// Amplitude-domain round-trip error per probe component.
#[derive(Debug, Clone)]
pub struct AmplitudeResult {
    pub relative_errors: Vec<f64>,
    pub measured_amplitudes: Vec<f64>,
    pub reference_amplitudes: Vec<f64>,
    pub max_relative: f64,
    pub pointwise_rms: f64,
}

/// Run the multitone through playback -> inverse and compare each spectral
/// component's amplitude against the source, inside the faded steady window.
pub fn amplitude_roundtrip(config: EngineConfig) -> AmplitudeResult {
    let fs = config.sample_rate;
    let source: WavData = multitone_signal(
        fs,
        PROBE_DURATION_MS,
        &PROBE_FREQUENCIES_HZ,
        PROBE_FADE_MS,
        1,
    );
    let mut playback_chain = build_chain(CurveKind::Playback, fs, config.guard_fraction, 1);
    let played = run_chain(&mut playback_chain, &source, config, false);
    let equalized = WavData {
        sample_rate: fs as u32,
        channels: 1,
        format: source.format,
        samples: played.samples,
    };
    let mut inverse_chain = build_chain(CurveKind::Inverse, fs, config.guard_fraction, 1);
    let outcome = run_chain(&mut inverse_chain, &equalized, config, false);
    let trim = (fs * PROBE_FADE_MS as f64 / 1000.0) as usize;
    let end = outcome.samples.len() - trim;
    let reference_amplitudes =
        component_amplitudes(&source.samples[trim..end], &PROBE_FREQUENCIES_HZ, fs);
    let measured_amplitudes =
        component_amplitudes(&outcome.samples[trim..end], &PROBE_FREQUENCIES_HZ, fs);
    let relative_errors: Vec<f64> = reference_amplitudes
        .iter()
        .zip(measured_amplitudes.iter())
        .map(|(reference, measured)| (measured - reference).abs() / reference.max(1e-12))
        .collect();
    let max_relative = relative_errors.iter().copied().fold(0.0_f64, f64::max);
    let pointwise_rms = time_error(&outcome.samples, &source.samples, trim).rms;
    AmplitudeResult {
        relative_errors,
        measured_amplitudes,
        reference_amplitudes,
        max_relative,
        pointwise_rms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steady_tone_survives_roundtrip_within_tolerance() {
        let config = EngineConfig::default();
        let fs = config.sample_rate;
        let frames = 240_000;
        let samples: Vec<f64> = (0..frames)
            .map(|index| (2.0 * std::f64::consts::PI * 1_000.0 * index as f64 / fs).cos())
            .collect();
        let source = WavData {
            sample_rate: fs as u32,
            channels: 1,
            format: crate::wav::SampleFormat::Float32,
            samples,
        };
        let mut playback_chain = build_chain(CurveKind::Playback, fs, config.guard_fraction, 1);
        let played = run_chain(&mut playback_chain, &source, config, false);
        let equalized = WavData {
            sample_rate: fs as u32,
            channels: 1,
            format: source.format,
            samples: played.samples,
        };
        let mut inverse_chain = build_chain(CurveKind::Inverse, fs, config.guard_fraction, 1);
        let outcome = run_chain(&mut inverse_chain, &equalized, config, false);
        let window_start = frames - 48_000;
        let measured = component_amplitudes(&outcome.samples[window_start..], &[1_000.0], fs)[0];
        assert!((measured - 1.0).abs() < 0.002, "measured {measured}");
    }

    #[test]
    fn amplitude_report_passes_core_band_criteria() {
        let (pass, max_db, max_rel) = passes(EngineConfig::default());
        assert!(pass, "db={max_db}, rel={max_rel}");
        assert!(max_db < 0.2);
        assert!(max_rel < 0.02);
    }

    #[test]
    fn head_state_is_cold_and_tail_state_is_nonzero_for_signal() {
        let config = EngineConfig::default();
        let source = multitone_signal(config.sample_rate, 500, &[500.0], 8, 1);
        let mut chain = build_chain(CurveKind::Playback, config.sample_rate, 0.45, 1);
        assert!(chain[0].states[0] == crate::filter::State::default());
        let outcome = run_chain(&mut chain, &source, config, false);
        assert_eq!(outcome.head_states[0][0].s1, 0.0);
        assert_eq!(outcome.head_states[0][0].s2, 0.0);
        assert!(
            outcome.tail_states[0][0].s1.abs() > 0.0 || outcome.tail_states[0][0].s2.abs() > 0.0
        );
    }
}
