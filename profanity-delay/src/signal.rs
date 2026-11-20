//! Test-signal synthesis and marker detection.
//!
//! The "speech substitute signal" is an FSK marker: a fixed, easily
//! recognised bit pattern modulated between two tones. It stands in for the
//! spoken phrase that must never reach the speaker path. Because the pattern
//! is exactly known, its absence from the output WAV is machine-checkable
//! proof that a dumped interval did not leak.

pub const MARKER_F0: f32 = 1800.0;
pub const MARKER_F1: f32 = 2600.0;
pub const MARKER_SYM_SECS: f32 = 0.08;
pub const MARKER_BITS: usize = 12;
pub const MARKER_AMP: f32 = 0.5;

/// The fixed, recognisable pattern (same for every marker instance).
pub fn marker_pattern() -> [u8; MARKER_BITS] {
    [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1]
}

pub fn marker_duration() -> f32 {
    MARKER_BITS as f32 * MARKER_SYM_SECS
}

/// Synthesise one marker (the speech-substitute signal).
pub fn marker_samples(sample_rate: usize) -> Vec<f32> {
    let sym_len = (MARKER_SYM_SECS * sample_rate as f32).round() as usize;
    let total = sym_len * MARKER_BITS;
    let mut out = vec![0.0f32; total];
    let edge = (0.005 * sample_rate as f32) as usize; // 5 ms raised-cosine edges
    let pattern = marker_pattern();
    let mut phase = 0.0f32;
    for (i, s) in out.iter_mut().enumerate() {
        let sym = i / sym_len;
        let f = if pattern[sym] == 1 { MARKER_F1 } else { MARKER_F0 };
        phase += 2.0 * std::f32::consts::PI * f / sample_rate as f32;
        let pos_in_sym = i % sym_len;
        let mut env = 1.0f32;
        if pos_in_sym < edge {
            env = 0.5 - 0.5 * (std::f32::consts::PI * pos_in_sym as f32 / edge as f32).cos();
        } else if pos_in_sym >= sym_len - edge {
            env = 0.5
                - 0.5 * (std::f32::consts::PI * (sym_len - pos_in_sym) as f32 / edge as f32).cos();
        }
        *s = MARKER_AMP * env * phase.sin();
    }
    out
}

/// Overwrite `buf` starting at `start_sec` with the marker signal.
/// (Used by the engine tests; the demo composes markers inline.)
#[allow(dead_code)]
pub fn embed_marker(buf: &mut [f32], start_sec: f32, sample_rate: usize) {
    let start = (start_sec * sample_rate as f32).round() as usize;
    let marker = marker_samples(sample_rate);
    for (i, &m) in marker.iter().enumerate() {
        if start + i < buf.len() {
            buf[start + i] = m;
        }
    }
}

/// Tiny deterministic PRNG (xorshift) so demos are reproducible.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.max(1))
    }
    pub fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Speech-like burst: harmonic stack with syllabic amplitude modulation.
/// Stands in for normal programme audio. Peak level ~0.2.
pub fn speech_burst(dur_secs: f32, sample_rate: usize, rng: &mut Rng) -> Vec<f32> {
    let n = (dur_secs * sample_rate as f32).round() as usize;
    let mut out = vec![0.0f32; n];
    let f0 = 110.0 + 60.0 * rng.next_f32();
    let syll_rate = 3.0 + 3.0 * rng.next_f32();
    let mut phase = [0.0f32; 4];
    let gains = [1.0f32, 0.5, 0.28, 0.12];
    let attack = (0.03 * sample_rate as f32) as usize;
    for (i, s) in out.iter_mut().enumerate() {
        let t = i as f32 / sample_rate as f32;
        let syll = 0.55 + 0.45 * (2.0 * std::f32::consts::PI * syll_rate * t).sin();
        let mut v = 0.0;
        for (h, ph) in phase.iter_mut().enumerate() {
            *ph += 2.0 * std::f32::consts::PI * f0 * (h + 1) as f32 / sample_rate as f32;
            v += gains[h] * ph.sin();
        }
        let mut env = syll;
        if i < attack {
            env *= i as f32 / attack as f32;
        }
        if n - i < attack {
            env *= (n - i) as f32 / attack as f32;
        }
        *s = 0.15 * env * v;
    }
    out
}

/// Low-energy "silence": a quiet noise floor, well below the rebuild
/// threshold, so the engine can safely rebuild delay here.
pub fn silence(dur_secs: f32, sample_rate: usize, rng: &mut Rng) -> Vec<f32> {
    let n = (dur_secs * sample_rate as f32).round() as usize;
    (0..n).map(|_| 0.004 * (rng.next_f32() - 0.5)).collect()
}

fn goertzel_power(window: &[f32], freq: f32, sample_rate: usize) -> f32 {
    let n = window.len();
    if n == 0 {
        return 0.0;
    }
    let w = 2.0 * std::f32::consts::PI * freq / sample_rate as f32;
    let coeff = 2.0 * w.cos();
    let (mut s1, mut s2) = (0.0f32, 0.0f32);
    for &x in window {
        let s0 = x + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    s1 * s1 + s2 * s2 - coeff * s1 * s2
}

/// Scan `samples` for the marker pattern; return (start, end) in seconds.
/// The scan runs at four symbol-phase offsets so markers do not need to be
/// aligned to the 80 ms symbol grid.
pub fn detect_markers(samples: &[f32], sample_rate: usize) -> Vec<(f32, f32)> {
    let sym_len = (MARKER_SYM_SECS * sample_rate as f32).round() as usize;
    let pattern = marker_pattern();
    let mut found: Vec<(f32, f32)> = Vec::new();
    for phase in [0, sym_len / 4, sym_len / 2, 3 * sym_len / 4] {
        let n_sym = samples.len().saturating_sub(phase) / sym_len;
        let mut bits = Vec::with_capacity(n_sym);
        let mut tone_present = Vec::with_capacity(n_sym);
        for k in 0..n_sym {
            let w = &samples[phase + k * sym_len..phase + (k + 1) * sym_len];
            let p0 = goertzel_power(w, MARKER_F0, sample_rate);
            let p1 = goertzel_power(w, MARKER_F1, sample_rate);
            bits.push(if p1 > p0 { 1u8 } else { 0u8 });
            // A marker symbol has strong, decisive tone energy.
            let strong = p0.max(p1);
            let weak = p0.min(p1);
            tone_present.push(strong > 1.0 && strong > 25.0 * weak);
        }
        let mut k = 0;
        while k + MARKER_BITS <= n_sym {
            let is_match = bits[k..k + MARKER_BITS] == pattern
                && tone_present[k..k + MARKER_BITS].iter().all(|&t| t);
            if is_match {
                let start = (phase + k * sym_len) as f32 / sample_rate as f32;
                let end = start + marker_duration();
                let dup = found
                    .iter()
                    .any(|&(s, e)| start < e && s < end);
                if !dup {
                    found.push((start, end));
                }
                k += MARKER_BITS;
            } else {
                k += 1;
            }
        }
    }
    found.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    found
}
