//! Time-domain biquad filtering with explicit, inspectable state.

use super::riaa::Coeffs;

/// Direct Form II Transposed biquad delay state (per channel).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct State {
    pub s1: f64,
    pub s2: f64,
}

/// A biquad processor: fixed coefficients plus mutable per-channel state.
#[derive(Debug, Clone)]
pub struct Biquad {
    pub coeffs: Coeffs,
    pub states: Vec<State>,
}

impl Biquad {
    /// Cold start: every channel begins at the zero state.
    pub fn new(coeffs: Coeffs, channels: usize) -> Self {
        Self {
            coeffs,
            states: vec![State::default(); channels],
        }
    }

    pub fn channels(&self) -> usize {
        self.states.len()
    }

    /// Process one sample on one channel in place (DF2T).
    pub fn process_sample(&mut self, channel: usize, input: f64) -> f64 {
        let c = &self.coeffs;
        let s = &mut self.states[channel];
        let output = c.b0 * input + s.s1;
        s.s1 = c.b1 * input - c.a1 * output + s.s2;
        s.s2 = c.b2 * input - c.a2 * output;
        output
    }

    /// Process an interleaved multi-channel buffer in place.
    pub fn process_interleaved(&mut self, buffer: &mut [f64]) {
        let channels = self.channels();
        for frame in buffer.chunks_mut(channels) {
            for (channel, sample) in frame.iter_mut().enumerate() {
                *sample = self.process_sample(channel, *sample);
            }
        }
    }

    /// Drive the filter with zero input until every state magnitude is below
    /// `epsilon`, or until `max_taps` zero samples have been fed. Returns the
    /// number of zero-input samples that were appended.
    pub fn flush(&mut self, epsilon: f64, max_taps: usize) -> usize {
        let channels = self.channels();
        for taps in 0..max_taps {
            let settled = self
                .states
                .iter()
                .all(|s| s.s1.abs() < epsilon && s.s2.abs() < epsilon);
            if settled {
                return taps;
            }
            for channel in 0..channels {
                self.process_sample(channel, 0.0);
            }
        }
        max_taps
    }

    /// Feed zero frames through a whole cascade, appending the final stage's
    /// ring-out samples to `output`. Stops once every state in every stage is
    /// below `epsilon`, or after `max_taps`; returns the number of frames fed.
    pub fn flush_chain(
        chain: &mut [Biquad],
        output: &mut Vec<f64>,
        epsilon: f64,
        max_taps: usize,
    ) -> usize {
        let channels = chain.first().map(|first| first.channels()).unwrap_or(0);
        for taps in 0..max_taps {
            let settled = chain.iter().all(|stage| {
                stage
                    .states
                    .iter()
                    .all(|s| s.s1.abs() < epsilon && s.s2.abs() < epsilon)
            });
            if settled {
                return taps;
            }
            for channel in 0..channels {
                let mut value = 0.0;
                for stage in chain.iter_mut() {
                    value = stage.process_sample(channel, value);
                }
                output.push(value);
            }
        }
        max_taps
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::riaa::{guarded_inverse, playback};

    #[test]
    fn cold_start_is_zero_state() {
        let f = Biquad::new(playback(48_000.0), 2);
        assert!(f.states.iter().all(|s| *s == State::default()));
    }

    #[test]
    fn inverse_poles_inside_unit_circle() {
        let c = guarded_inverse(48_000.0, 0.45);
        let discriminant = c.a1 * c.a1 - 4.0 * c.a2;
        let radius = if discriminant < 0.0 {
            c.a2.sqrt()
        } else {
            let root = discriminant.sqrt();
            ((-c.a1 + root) / 2.0)
                .abs()
                .max(((-c.a1 - root) / 2.0).abs())
        };
        assert!(radius < 1.0, "pole radius {radius} must be < 1");
    }

    #[test]
    fn flush_settles_to_zero_state() {
        let mut f = Biquad::new(playback(48_000.0), 1);
        let impulse: Vec<f64> = std::iter::once(1.0)
            .chain(std::iter::repeat_n(0.0, 64))
            .collect();
        let mut buffer = impulse;
        f.process_interleaved(&mut buffer);
        let taps = f.flush(1e-12, 100_000);
        assert!(taps < 100_000);
        assert!(f.states[0].s1.abs() < 1e-12);
        assert!(f.states[0].s2.abs() < 1e-12);
    }
}
