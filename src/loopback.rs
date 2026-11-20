use std::fmt;

use crate::code::Probe;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoopbackError {
    NoFrames,
    DelayCountMismatch,
    InvalidDelay,
    InvalidChunkSize,
    InvalidLevel,
    InvalidFalseBurst,
}

impl fmt::Display for LoopbackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoFrames => write!(f, "frames must be greater than zero"),
            Self::DelayCountMismatch => write!(f, "one delay is required per frame"),
            Self::InvalidDelay => {
                write!(f, "delays must be finite and in [0, period - probe_len)")
            }
            Self::InvalidChunkSize => write!(f, "chunk sizes must be greater than zero"),
            Self::InvalidLevel => write!(f, "noise/interference levels must be finite"),
            Self::InvalidFalseBurst => write!(f, "false burst is outside its frame"),
        }
    }
}

impl std::error::Error for LoopbackError {}

/// A one-frame, probe-shaped false peak at a deliberately wrong delay.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FalseBurst {
    pub frame: usize,
    pub lag: usize,
    pub amplitude: f32,
}

/// Captured buffers are returned in these varying lengths, cyclically.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkPattern {
    sizes: Vec<usize>,
}

impl ChunkPattern {
    pub fn new(sizes: Vec<usize>) -> Result<Self, LoopbackError> {
        if sizes.is_empty() || sizes.contains(&0) {
            return Err(LoopbackError::InvalidChunkSize);
        }
        Ok(Self { sizes })
    }

    pub fn regular(size: usize) -> Result<Self, LoopbackError> {
        Self::new(vec![size])
    }

    pub fn sizes(&self) -> &[usize] {
        &self.sizes
    }

    pub fn chunks_for(&self, total_samples: usize) -> Vec<usize> {
        let mut chunks = Vec::new();
        let mut remaining = total_samples;
        let mut index = 0;
        while remaining > 0 {
            let size = self.sizes[index % self.sizes.len()].min(remaining);
            chunks.push(size);
            remaining -= size;
            index += 1;
        }
        chunks
    }
}

impl Default for ChunkPattern {
    fn default() -> Self {
        Self {
            sizes: vec![257, 31, 1_024, 93, 511, 7, 333],
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LoopbackConfig {
    pub frames: usize,
    pub delays: Vec<f32>,
    pub echo_gain: f32,
    pub noise_rms: f32,
    pub interference_rms: f32,
    pub interference_seed: u64,
    pub chunks: ChunkPattern,
    pub dropout_frames: Vec<usize>,
    pub false_bursts: Vec<FalseBurst>,
}

impl LoopbackConfig {
    pub fn uniform(frames: usize, delay: f32) -> Self {
        Self {
            frames,
            delays: vec![delay; frames],
            echo_gain: 0.8,
            noise_rms: 0.008,
            interference_rms: 0.04,
            interference_seed: 0x4d55_5349_435f_3031,
            chunks: ChunkPattern::default(),
            dropout_frames: Vec::new(),
            false_bursts: Vec::new(),
        }
    }

    pub fn validate(&self, probe: &Probe) -> Result<(), LoopbackError> {
        if self.frames == 0 {
            return Err(LoopbackError::NoFrames);
        }
        if self.delays.len() != self.frames {
            return Err(LoopbackError::DelayCountMismatch);
        }
        let max_delay = probe.max_lag() as f32 - 0.001;
        if self
            .delays
            .iter()
            .any(|delay| !delay.is_finite() || *delay < 0.0 || *delay > max_delay)
        {
            return Err(LoopbackError::InvalidDelay);
        }
        if !self.noise_rms.is_finite()
            || !self.interference_rms.is_finite()
            || self.noise_rms < 0.0
            || self.interference_rms < 0.0
        {
            return Err(LoopbackError::InvalidLevel);
        }
        for burst in &self.false_bursts {
            if burst.frame >= self.frames
                || burst.lag + probe.probe_len() > probe.period()
                || !burst.amplitude.is_finite()
            {
                return Err(LoopbackError::InvalidFalseBurst);
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct LoopbackRender {
    pub emitted: Vec<f32>,
    pub capture: Vec<f32>,
    pub chunks: Vec<usize>,
    pub truth_delays: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct SoftwareLoopback {
    probe: Probe,
    config: LoopbackConfig,
}

impl SoftwareLoopback {
    pub fn new(probe: Probe, config: LoopbackConfig) -> Result<Self, LoopbackError> {
        config.validate(&probe)?;
        Ok(Self { probe, config })
    }

    pub fn render(&self) -> LoopbackRender {
        render_capture(&self.probe, &self.config).expect("configuration was validated")
    }

    pub fn config(&self) -> &LoopbackConfig {
        &self.config
    }
}

pub fn render_capture(
    probe: &Probe,
    config: &LoopbackConfig,
) -> Result<LoopbackRender, LoopbackError> {
    config.validate(probe)?;
    let period = probe.period();
    let total = config.frames * period;
    let mut emitted = vec![0.0; total];
    probe.emit_into(0, &mut emitted);

    let interference = render_interference(
        total,
        probe.config().sample_rate,
        config.interference_rms,
        config.interference_seed,
    );
    let noise = render_noise(
        total,
        config.noise_rms,
        config.interference_seed ^ 0xa5a5_5a5a,
    );
    let mut capture = vec![0.0; total];

    for (sample_index, output) in capture.iter_mut().enumerate() {
        let frame = sample_index / period;
        let echo = if config.dropout_frames.contains(&frame) {
            0.0
        } else {
            let source_position = sample_index as f32 - config.delays[frame];
            config.echo_gain * interpolated(&emitted, source_position)
        };
        *output = echo + interference[sample_index] + noise[sample_index];
    }

    for burst in &config.false_bursts {
        let start = burst.frame * period + burst.lag;
        for i in 0..probe.probe_len() {
            let code = probe.sample_at(i) / probe.amplitude();
            capture[start + i] += burst.amplitude * code;
        }
    }
    for sample in &mut capture {
        *sample = sample.clamp(-1.0, 1.0);
    }

    let chunks = config.chunks.chunks_for(total);
    Ok(LoopbackRender {
        emitted,
        capture,
        chunks,
        truth_delays: config.delays.clone(),
    })
}

fn interpolated(emitted: &[f32], position: f32) -> f32 {
    if position < 0.0 {
        return 0.0;
    }
    let index = position.floor() as usize;
    if index >= emitted.len() {
        return 0.0;
    }
    let fraction = position - index as f32;
    let left = emitted[index];
    let right = emitted.get(index + 1).copied().unwrap_or(0.0);
    left + fraction * (right - left)
}

fn render_interference(len: usize, sample_rate: u32, target_rms: f32, seed: u64) -> Vec<f32> {
    if target_rms == 0.0 {
        return vec![0.0; len];
    }
    let rate = sample_rate as f64;
    let frequencies = [97.0, 211.0, 389.0, 733.0, 1_307.0, 2_113.0];
    let mut raw = Vec::with_capacity(len);
    for n in 0..len {
        let time = n as f64 / rate;
        let envelope = 0.65 + 0.35 * (2.0 * std::f64::consts::PI * 0.37 * time).sin();
        let mut value = 0.0;
        for (index, frequency) in frequencies.iter().enumerate() {
            let phase = 2.0 * std::f64::consts::PI * frequency * time + index as f64 * 0.73;
            value += (phase.sin() + 0.23 * (1.71 * phase).sin()) / (index + 1) as f64;
        }
        let hash = splitmix64(n as u64 ^ seed);
        let grit = ((hash >> 40) as f64 / (1u64 << 24) as f64) - 0.5;
        raw.push((envelope * value * 0.55 + 0.08 * grit) as f32);
    }
    let rms = rms(&raw).max(1.0e-12);
    let scale = target_rms / rms;
    raw.into_iter().map(|value| value * scale).collect()
}

fn render_noise(len: usize, rms_level: f32, seed: u64) -> Vec<f32> {
    let mut rng = XorShift64::new(seed);
    let mut output = Vec::with_capacity(len);
    while output.len() < len {
        let u1 = rng.next_f64().max(1.0e-12);
        let u2 = rng.next_f64();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * u2;
        output.push((radius * angle.cos() * rms_level as f64) as f32);
        if output.len() < len {
            output.push((radius * angle.sin() * rms_level as f64) as f32);
        }
    }
    output
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mean_square = samples
        .iter()
        .map(|sample| (*sample as f64) * (*sample as f64))
        .sum::<f64>()
        / samples.len() as f64;
    mean_square.sqrt() as f32
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        self.state.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_cover_exactly_once() {
        let pattern = ChunkPattern::default();
        let chunks = pattern.chunks_for(10_000);
        assert_eq!(chunks.iter().sum::<usize>(), 10_000);
        assert!(chunks.windows(2).any(|pair| pair[0] != pair[1]));
    }

    #[test]
    fn rendering_is_chunk_invariant() {
        let probe = Probe::default_probe();
        let mut config = LoopbackConfig::uniform(8, 123.25);
        config.noise_rms = 0.01;
        config.interference_rms = 0.03;
        let regular = render_capture(&probe, &config).unwrap();
        config.chunks = ChunkPattern::new(vec![13, 509, 1, 777, 61]).unwrap();
        let irregular = render_capture(&probe, &config).unwrap();
        assert_eq!(regular.capture, irregular.capture);
        assert_eq!(
            irregular.chunks.iter().sum::<usize>(),
            irregular.capture.len()
        );
    }
}
