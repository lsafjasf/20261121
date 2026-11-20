use std::fmt;

/// Configuration for the repeated low-amplitude coded probe.
#[derive(Clone, Debug, PartialEq)]
pub struct ProbeConfig {
    pub sample_rate: u32,
    pub probe_len: usize,
    pub period: usize,
    pub amplitude: f32,
    pub seed: u64,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48_000,
            probe_len: 1_023,
            period: 2_048,
            amplitude: 0.008,
            seed: 0x5eed_1234_abcd_9876,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProbeError {
    InvalidSampleRate,
    ProbeTooShort,
    PeriodTooShort,
    InvalidAmplitude,
}

impl fmt::Display for ProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSampleRate => write!(f, "sample_rate must be greater than zero"),
            Self::ProbeTooShort => write!(f, "probe_len must be at least 64 samples"),
            Self::PeriodTooShort => {
                write!(f, "period must be greater than probe_len")
            }
            Self::InvalidAmplitude => {
                write!(f, "amplitude must be finite and in (0, 0.5]")
            }
        }
    }
}

impl std::error::Error for ProbeError {}

/// A deterministic, balanced binary code emitted once per period.
///
/// The reference code itself is normalized to +/-1.  The emitted stream is
/// multiplied by `amplitude`, keeping the probe low level while preserving the
/// matched-filter shape.  A short raised-cosine edge avoids hard clicks.
#[derive(Clone, Debug)]
pub struct Probe {
    config: ProbeConfig,
    code: Vec<f32>,
    edges: Vec<f32>,
}

impl Probe {
    pub fn new(config: ProbeConfig) -> Result<Self, ProbeError> {
        if config.sample_rate == 0 {
            return Err(ProbeError::InvalidSampleRate);
        }
        if config.probe_len < 64 {
            return Err(ProbeError::ProbeTooShort);
        }
        if config.period <= config.probe_len {
            return Err(ProbeError::PeriodTooShort);
        }
        if !config.amplitude.is_finite() || config.amplitude <= 0.0 || config.amplitude > 0.5 {
            return Err(ProbeError::InvalidAmplitude);
        }

        let code = generate_balanced_code(config.probe_len, config.seed);
        let edge_len = (config.probe_len / 32).clamp(4, 32);
        let mut edges = vec![1.0; config.probe_len];
        for i in 0..edge_len {
            let phase = std::f32::consts::FRAC_PI_2 * (i as f32 + 0.5) / edge_len as f32;
            let gain = phase.sin();
            edges[i] = gain;
            edges[config.probe_len - 1 - i] = gain;
        }

        Ok(Self {
            config,
            code,
            edges,
        })
    }

    /// Default probe.
    pub fn default_probe() -> Self {
        Self::new(ProbeConfig::default()).expect("default probe config is valid")
    }

    pub fn config(&self) -> &ProbeConfig {
        &self.config
    }

    pub fn reference(&self) -> &[f32] {
        &self.code
    }

    pub fn amplitude(&self) -> f32 {
        self.config.amplitude
    }

    pub fn period(&self) -> usize {
        self.config.period
    }

    pub fn probe_len(&self) -> usize {
        self.config.probe_len
    }

    pub fn max_lag(&self) -> usize {
        self.config.period - self.config.probe_len
    }

    pub fn sample_at(&self, source_sample: usize) -> f32 {
        let offset = source_sample % self.config.period;
        if offset >= self.config.probe_len {
            return 0.0;
        }
        self.config.amplitude * self.code[offset] * self.edges[offset]
    }

    pub fn emit_into(&self, source_start: usize, output: &mut [f32]) {
        for (i, sample) in output.iter_mut().enumerate() {
            *sample = self.sample_at(source_start + i);
        }
    }

    pub fn emitted_len_for_frames(&self, frames: usize) -> usize {
        frames * self.config.period
    }
}

/// xorshift64* is used only to generate a repeatable code, not for security.
fn generate_balanced_code(len: usize, seed: u64) -> Vec<f32> {
    let mut state = if seed == 0 {
        0x9e37_79b9_7f4a_7c15
    } else {
        seed
    };
    let mut code = Vec::with_capacity(len);
    let mut balance = 0_i64;
    for i in 0..len {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let value = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
        let mut sign = if value & (1 << 63) == 0 { -1.0 } else { 1.0 };
        if i + 1 == len && balance > 0 {
            sign = -1.0;
        } else if i + 1 == len && balance < 0 {
            sign = 1.0;
        }
        balance += sign as i64;
        code.push(sign);
    }
    code
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_is_balanced() {
        let probe = Probe::default_probe();
        let sum: f32 = probe.reference().iter().sum();
        assert!(sum.abs() <= 1.0, "code sum was {sum}");
    }

    #[test]
    fn repeated_probe_has_guard_interval() {
        let probe = Probe::default_probe();
        let period = probe.period();
        let code_end = probe.sample_at(probe.probe_len() - 1).abs();
        let guard = probe.sample_at(probe.probe_len());
        let next = probe.sample_at(period + 10);
        assert!(code_end > 0.0);
        assert_eq!(guard, 0.0);
        assert_ne!(next, 0.0);
    }

    #[test]
    fn invalid_configs_are_rejected() {
        let mut config = ProbeConfig::default();
        config.period = config.probe_len;
        assert!(matches!(
            Probe::new(config),
            Err(ProbeError::PeriodTooShort)
        ));
    }
}
