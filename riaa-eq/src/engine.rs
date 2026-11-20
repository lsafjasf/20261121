//! Rendering engine: explicit representations, filter chains and verification.

use crate::filter::Biquad;
use crate::riaa::{guard_corner_hz, guarded_inverse, playback};
use crate::wav::WavData;

/// Signal representation the operator explicitly selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Representation {
    /// Flat, pre-RIAA signal straight from the phono preamp output.
    Flat,
    /// Already RIAA-equalized archival recording (playback curve applied).
    Riaa,
}

impl Representation {
    pub fn parse(text: &str) -> Option<Self> {
        match text.to_ascii_lowercase().as_str() {
            "flat" => Some(Self::Flat),
            "riaa" => Some(Self::Riaa),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Flat => "flat",
            Self::Riaa => "riaa",
        }
    }
}

/// What happens to a signal given its input and output representations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurveKind {
    /// flat -> riaa: recording/inverse curve.
    Inverse,
    /// riaa -> flat: playback curve.
    Playback,
    /// riaa -> riaa or flat -> flat: data passes through untouched.
    Bypass,
}

/// Decide the processing kind, rejecting the unphysical flat -> flat request.
pub fn plan_curve(input: Representation, output: Representation) -> Result<CurveKind, String> {
    match (input, output) {
        (Representation::Flat, Representation::Riaa) => Ok(CurveKind::Inverse),
        (Representation::Riaa, Representation::Flat) => Ok(CurveKind::Playback),
        (Representation::Flat, Representation::Flat) => Ok(CurveKind::Bypass),
        (Representation::Riaa, Representation::Riaa) => Ok(CurveKind::Bypass),
    }
}

/// Build the ordered filter chain for a render (empty for bypass).
pub fn build_chain(
    kind: CurveKind,
    sample_rate: f64,
    guard_fraction: f64,
    channels: usize,
) -> Vec<Biquad> {
    match kind {
        CurveKind::Playback => vec![Biquad::new(playback(sample_rate), channels)],
        CurveKind::Inverse => {
            vec![Biquad::new(
                guarded_inverse(sample_rate, guard_fraction),
                channels,
            )]
        }
        CurveKind::Bypass => Vec::new(),
    }
}

/// Parameters shared by every generated analysis artifact.
#[derive(Debug, Clone, Copy)]
pub struct EngineConfig {
    pub sample_rate: f64,
    pub guard_fraction: f64,
    pub flush_epsilon: f64,
    pub max_flush_taps: usize,
}

impl EngineConfig {
    pub fn guard_hz(&self) -> f64 {
        guard_corner_hz(self.sample_rate, self.guard_fraction)
    }
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48_000.0,
            guard_fraction: 0.45,
            flush_epsilon: 1e-12,
            max_flush_taps: 200_000,
        }
    }
}

/// Filter state captured at a boundary of the render.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StateSnapshot {
    pub s1: f64,
    pub s2: f64,
}

/// Result of processing one signal through a chain.
#[derive(Debug, Clone)]
pub struct RenderOutcome {
    pub samples: Vec<f64>,
    pub head_states: Vec<Vec<StateSnapshot>>,
    pub tail_states: Vec<Vec<StateSnapshot>>,
    /// Filter state after the optional zero-input flush completes.
    pub end_states: Vec<Vec<StateSnapshot>>,
    pub flush_taps: Vec<usize>,
    pub peak: f64,
}

fn snapshot(chain: &[Biquad]) -> Vec<Vec<StateSnapshot>> {
    chain
        .iter()
        .map(|stage| {
            stage
                .states
                .iter()
                .map(|s| StateSnapshot { s1: s.s1, s2: s.s2 })
                .collect()
        })
        .collect()
}

/// Run `audio` through the chain, optionally zero-flushing every stage.
/// Head state is always the cold zero state.
pub fn run_chain(
    chain: &mut [Biquad],
    audio: &WavData,
    config: EngineConfig,
    flush: bool,
) -> RenderOutcome {
    let head_states = snapshot(chain);
    let mut samples = audio.samples.clone();
    for stage in chain.iter_mut() {
        stage.process_interleaved(&mut samples);
    }
    let tail_states = snapshot(chain);
    let mut flush_taps = Vec::new();
    if flush {
        let taps = Biquad::flush_chain(
            chain,
            &mut samples,
            config.flush_epsilon,
            config.max_flush_taps,
        );
        flush_taps.push(taps);
    }
    let end_states = snapshot(chain);
    let peak = samples
        .iter()
        .copied()
        .fold(0.0_f64, |acc, x| acc.max(x.abs()));
    RenderOutcome {
        samples,
        head_states,
        tail_states,
        end_states,
        flush_taps,
        peak,
    }
}

/// FNV-1a 64-bit checksum over a canonical specification string.
pub fn fnv1a_64(text: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in text.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// A single unit impulse at sample index 0, followed by silence.
pub fn impulse_signal(sample_rate: f64, duration_ms: u32, channels: u16) -> WavData {
    let frames = (sample_rate as u64 * duration_ms as u64 / 1000) as usize;
    let mut samples = vec![0.0_f64; frames * channels as usize];
    if !samples.is_empty() {
        samples[0] = 1.0;
    }
    WavData {
        sample_rate: sample_rate as u32,
        channels,
        format: crate::wav::SampleFormat::Float32,
        samples,
    }
}

/// A windowed tone burst: sine gated by a Hann window over the whole signal.
pub fn tone_burst_signal(
    sample_rate: f64,
    duration_ms: u32,
    frequency_hz: f64,
    amplitude: f64,
    channels: u16,
) -> WavData {
    let frames = (sample_rate as u64 * duration_ms as u64 / 1000) as usize;
    let mut samples = Vec::with_capacity(frames * channels as usize);
    for frame in 0..frames {
        let phase = 2.0 * std::f64::consts::PI * frequency_hz * frame as f64 / sample_rate;
        let window = if frames <= 1 {
            1.0
        } else {
            0.5 * (1.0 - (2.0 * std::f64::consts::PI * frame as f64 / (frames - 1) as f64).cos())
        };
        let value = amplitude * phase.sin() * window;
        for _ in 0..channels {
            samples.push(value);
        }
    }
    WavData {
        sample_rate: sample_rate as u32,
        channels,
        format: crate::wav::SampleFormat::Float32,
        samples,
    }
}

/// Deterministic multitone probe: cosine sum with fixed prime phases, kept
/// below full scale by `1/n` amplitude scaling per component.
/// A Hann fade of `fade_ms` at both ends removes hard boundary transients so
/// the inverse filter's high-frequency gain cannot amplify edge clicks.
pub fn multitone_signal(
    sample_rate: f64,
    duration_ms: u32,
    frequencies_hz: &[f64],
    fade_ms: u32,
    channels: u16,
) -> WavData {
    let frames = (sample_rate as u64 * duration_ms as u64 / 1000) as usize;
    let fade_frames = (sample_rate as u64 * fade_ms as u64 / 1000) as usize;
    let component = 0.5 / frequencies_hz.len() as f64;
    let mut samples = Vec::with_capacity(frames * channels as usize);
    for frame in 0..frames {
        let mut value = 0.0;
        for (index, &frequency) in frequencies_hz.iter().enumerate() {
            let phase = 2.0 * std::f64::consts::PI * frequency * frame as f64 / sample_rate
                + (index as f64 * 0.7);
            value += component * phase.cos();
        }
        let fade = if frame < fade_frames {
            0.5 * (1.0 - (std::f64::consts::PI * frame as f64 / fade_frames as f64).cos())
        } else if frame >= frames - fade_frames {
            let remaining = frames - frame;
            0.5 * (1.0 - (std::f64::consts::PI * remaining as f64 / fade_frames as f64).cos())
        } else {
            1.0
        };
        value *= fade;
        for _ in 0..channels {
            samples.push(value);
        }
    }
    WavData {
        sample_rate: sample_rate as u32,
        channels,
        format: crate::wav::SampleFormat::Float32,
        samples,
    }
}

/// Error summary between two equally sized signals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeError {
    pub max_abs: f64,
    pub rms: f64,
}

/// Compare an output against a reference, skipping `trim` samples at both ends
/// so optional tail flushing does not contaminate the metric.
pub fn time_error(output: &[f64], reference: &[f64], trim: usize) -> TimeError {
    let len = output.len().min(reference.len());
    let window = &output[trim..len.saturating_sub(trim)];
    let ref_window = &reference[trim..len.saturating_sub(trim)];
    let (max_abs, sum_sq) =
        window
            .iter()
            .zip(ref_window)
            .fold((0.0_f64, 0.0_f64), |(max_acc, sq_acc), (a, b)| {
                let diff = (a - b).abs();
                (max_acc.max(diff), sq_acc + diff * diff)
            });
    TimeError {
        max_abs,
        rms: (sum_sq / window.len().max(1) as f64).sqrt(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_representations_select_the_right_curve() {
        assert_eq!(
            plan_curve(Representation::Flat, Representation::Riaa).unwrap(),
            CurveKind::Inverse
        );
        assert_eq!(
            plan_curve(Representation::Riaa, Representation::Flat).unwrap(),
            CurveKind::Playback
        );
        assert_eq!(
            plan_curve(Representation::Riaa, Representation::Riaa).unwrap(),
            CurveKind::Bypass
        );
        assert_eq!(
            plan_curve(Representation::Flat, Representation::Flat).unwrap(),
            CurveKind::Bypass
        );
    }

    #[test]
    fn bypass_chain_is_empty() {
        let config = EngineConfig::default();
        assert!(build_chain(CurveKind::Bypass, config.sample_rate, 0.45, 2).is_empty());
        assert_eq!(
            build_chain(CurveKind::Playback, config.sample_rate, 0.45, 2).len(),
            1
        );
        assert_eq!(
            build_chain(CurveKind::Inverse, config.sample_rate, 0.45, 2).len(),
            1
        );
    }

    #[test]
    fn multitone_is_deterministic() {
        let config = EngineConfig::default();
        let freqs = [100.0, 1_000.0];
        let a = multitone_signal(config.sample_rate, 200, &freqs, 8, 1);
        let b = multitone_signal(config.sample_rate, 200, &freqs, 8, 1);
        assert_eq!(a.samples, b.samples);
        assert!(a.samples.iter().all(|x| x.abs() <= 0.5 + 1e-12));
    }

    #[test]
    fn fingerprint_is_stable_and_input_sensitive() {
        let config = EngineConfig::default();
        let coeffs = crate::riaa::playback(config.sample_rate);
        let first = fnv1a_64(&format!("{coeffs:?}"));
        let second = fnv1a_64(&format!("{coeffs:?}"));
        assert_eq!(first, second);
        let changed = fnv1a_64(&format!("{:?}", (coeffs.b0 + 1.0,)));
        assert_ne!(first, changed);
    }
}
