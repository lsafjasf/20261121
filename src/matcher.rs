use crate::code::Probe;

/// A normalized matched-filter peak.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    /// Integer sample delay from the analysis-frame start.
    pub lag_samples: usize,
    /// Parabolic interpolation around the integer peak.
    pub lag_fractional: f32,
    /// Normalized correlation coefficient at the integer peak.
    pub correlation: f32,
    /// Correlation consistency between the two probe halves.
    pub split_balance: f32,
    /// Combined acquisition score, always in `[0, 1]`.
    pub confidence: f32,
}

impl Candidate {
    pub fn lag(&self) -> f32 {
        self.lag_samples as f32 + self.lag_fractional
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatchWindow {
    pub frame_index: usize,
    pub capture_start: usize,
    pub capture_end: usize,
    pub min_lag: usize,
    pub max_lag: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Detection {
    pub window: MatchWindow,
    pub candidates: Vec<Candidate>,
}

impl Detection {
    pub fn best(&self) -> Option<&Candidate> {
        self.candidates.first()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MatcherConfig {
    pub min_lag: usize,
    pub max_lag: Option<usize>,
    pub min_confidence: f32,
    pub max_peaks: usize,
    pub peak_separation: usize,
}

impl Default for MatcherConfig {
    fn default() -> Self {
        Self {
            min_lag: 0,
            max_lag: None,
            min_confidence: 0.03,
            max_peaks: 6,
            peak_separation: 4,
        }
    }
}

/// Per-frame normalized matched filter.
pub struct Matcher {
    reference: Vec<f32>,
    reference_energy: f32,
    config: MatcherConfig,
    unambiguous_max_lag: usize,
}

impl Matcher {
    pub fn new(probe: &Probe, config: MatcherConfig) -> Self {
        let reference = probe.reference().to_vec();
        let reference_energy = reference.iter().map(|v| v * v).sum::<f32>();
        Self {
            reference,
            reference_energy,
            config,
            unambiguous_max_lag: probe.max_lag(),
        }
    }

    pub fn config(&self) -> MatcherConfig {
        self.config
    }

    pub fn detect(&self, frame: &[f32], window: MatchWindow) -> Detection {
        let max_by_frame = frame.len().saturating_sub(self.reference.len());
        let max_lag = self
            .config
            .max_lag
            .unwrap_or(self.unambiguous_max_lag)
            .min(self.unambiguous_max_lag)
            .min(max_by_frame);
        let min_lag = self.config.min_lag.min(max_lag);

        if frame.len() < self.reference.len() || max_lag < min_lag {
            return Detection {
                window,
                candidates: Vec::new(),
            };
        }

        let scores = self.correlation_scores(frame, min_lag, max_lag);
        let peaks = self.find_peaks(&scores, min_lag);
        let mut candidates = Vec::with_capacity(peaks.len());

        for peak_index in peaks {
            let correlation = scores[peak_index];
            if correlation <= 0.0 {
                continue;
            }
            let lag = min_lag + peak_index;
            let split_balance = self.split_balance(frame, lag);
            // The full-window correlation carries most of the evidence.  The
            // two-half balance rejects click-like coincidences without making
            // ordinary interference drive an otherwise solid peak below the
            // detection threshold.
            let confidence = (correlation * (0.7 + 0.3 * split_balance)).clamp(0.0, 1.0);
            if confidence < self.config.min_confidence {
                continue;
            }
            let fractional = fractional_peak(
                scores.get(peak_index.wrapping_sub(1)).copied(),
                scores[peak_index],
                scores.get(peak_index + 1).copied(),
            );
            candidates.push(Candidate {
                lag_samples: lag,
                lag_fractional: fractional,
                correlation,
                split_balance,
                confidence,
            });
        }

        candidates.sort_by(|a, b| {
            b.confidence
                .total_cmp(&a.confidence)
                .then_with(|| a.lag_samples.cmp(&b.lag_samples))
        });
        candidates.truncate(self.config.max_peaks);
        Detection { window, candidates }
    }

    fn correlation_scores(&self, frame: &[f32], min_lag: usize, max_lag: usize) -> Vec<f32> {
        let n = self.reference.len();
        let mut scores = Vec::with_capacity(max_lag - min_lag + 1);
        let first = &frame[min_lag..min_lag + n];
        let mut energy = first.iter().map(|v| v * v).sum::<f32>();

        for lag in min_lag..=max_lag {
            let segment = &frame[lag..lag + n];
            let dot = segment
                .iter()
                .zip(&self.reference)
                .map(|(x, c)| x * c)
                .sum::<f32>();
            let denominator = (energy * self.reference_energy).sqrt();
            scores.push(if denominator > 1.0e-12 {
                (dot / denominator).clamp(-1.0, 1.0)
            } else {
                0.0
            });

            if lag < max_lag {
                let leaving = frame[lag];
                let entering = frame[lag + n];
                energy += entering * entering - leaving * leaving;
                energy = energy.max(0.0);
            }
        }
        scores
    }

    fn find_peaks(&self, scores: &[f32], lag_offset: usize) -> Vec<usize> {
        let mut local = Vec::new();
        for (i, score) in scores.iter().enumerate() {
            let left = i
                .checked_sub(1)
                .and_then(|j| scores.get(j))
                .copied()
                .unwrap_or(f32::NEG_INFINITY);
            let right = scores.get(i + 1).copied().unwrap_or(f32::NEG_INFINITY);
            if *score >= left && *score > right && *score > 0.0 {
                local.push((i, *score));
            }
        }
        local.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let mut selected = Vec::new();
        for (index, _) in local {
            let absolute_lag = lag_offset + index;
            if selected
                .iter()
                .all(|chosen: &usize| absolute_lag.abs_diff(*chosen) >= self.config.peak_separation)
            {
                selected.push(index);
                if selected.len() >= self.config.max_peaks * 4 {
                    break;
                }
            }
        }
        selected
    }

    fn split_balance(&self, frame: &[f32], lag: usize) -> f32 {
        let half = self.reference.len() / 2;
        if half == 0 {
            return 0.0;
        }
        let first_signal = &frame[lag..lag + half];
        let second_signal = &frame[lag + half..lag + self.reference.len()];
        let first_code = &self.reference[..half];
        let second_code = &self.reference[half..self.reference.len()];

        let first = normalized_half(first_signal, first_code);
        let second = normalized_half(second_signal, second_code);
        if first + second <= 1.0e-9 {
            0.0
        } else {
            (2.0 * first.min(second) / (first + second)).clamp(0.0, 1.0)
        }
    }
}

fn normalized_half(signal: &[f32], code: &[f32]) -> f32 {
    let dot = signal.iter().zip(code).map(|(x, c)| x * c).sum::<f32>();
    let energy = signal.iter().map(|v| v * v).sum::<f32>();
    let code_energy = code.len() as f32;
    let denominator = (energy * code_energy).sqrt();
    if denominator > 1.0e-12 {
        (dot / denominator).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

fn fractional_peak(left: Option<f32>, center: f32, right: Option<f32>) -> f32 {
    let (Some(left), Some(right)) = (left, right) else {
        return 0.0;
    };
    let curvature = left - 2.0 * center + right;
    if curvature >= -1.0e-9 {
        return 0.0;
    }
    (0.5 * (left - right) / curvature).clamp(-0.5, 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code::Probe;

    fn clean_frame(probe: &Probe, lag: usize) -> Vec<f32> {
        let mut frame = vec![0.0; probe.period()];
        for i in 0..probe.probe_len() {
            frame[lag + i] = probe.sample_at(i);
        }
        frame
    }

    #[test]
    fn finds_clean_delay() {
        let probe = Probe::default_probe();
        let matcher = Matcher::new(&probe, MatcherConfig::default());
        let lag = 317;
        let frame = clean_frame(&probe, lag);
        let window = MatchWindow {
            frame_index: 0,
            capture_start: 0,
            capture_end: frame.len(),
            min_lag: 0,
            max_lag: probe.max_lag(),
        };
        let detection = matcher.detect(&frame, window);
        let best = detection.best().expect("probe should be detected");
        assert_eq!(best.lag_samples, lag);
        assert!(best.confidence > 0.8, "confidence={}", best.confidence);
    }

    #[test]
    fn interpolation_is_sub_sample() {
        let probe = Probe::default_probe();
        let lag = 200;
        let mut frame = clean_frame(&probe, lag);
        let shifted = clean_frame(&probe, lag + 1);
        for (i, value) in frame.iter_mut().enumerate() {
            *value = 0.5 * *value + 0.5 * shifted[i];
        }
        let matcher = Matcher::new(&probe, MatcherConfig::default());
        let window = MatchWindow {
            frame_index: 0,
            capture_start: 0,
            capture_end: frame.len(),
            min_lag: 0,
            max_lag: probe.max_lag(),
        };
        let detection = matcher.detect(&frame, window);
        let best = detection.best().unwrap();
        assert!(
            (best.lag() - lag as f32 - 0.5).abs() < 0.2,
            "lag={}",
            best.lag()
        );
    }
}
