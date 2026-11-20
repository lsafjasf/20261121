use crate::code::Probe;
use crate::matcher::{MatchWindow, Matcher, MatcherConfig};
use crate::tracker::{FrameObservation, Tracker, TrackerConfig};

/// Streaming adapter that is independent of capture buffer boundaries.
pub struct MeasurementSession {
    matcher: Matcher,
    tracker: Tracker,
    period: usize,
    max_lag: usize,
    buffer: Vec<f32>,
    read_offset: usize,
    absolute_start: usize,
    next_frame: usize,
}

impl MeasurementSession {
    pub fn new(
        probe: &Probe,
        matcher_config: MatcherConfig,
        tracker_config: TrackerConfig,
    ) -> Self {
        let max_lag = matcher_config
            .max_lag
            .unwrap_or(probe.max_lag())
            .min(probe.max_lag());
        Self {
            matcher: Matcher::new(probe, matcher_config),
            tracker: Tracker::new(tracker_config),
            period: probe.period(),
            max_lag,
            buffer: Vec::new(),
            read_offset: 0,
            absolute_start: 0,
            next_frame: 0,
        }
    }

    pub fn push(&mut self, chunk: &[f32]) -> Vec<FrameObservation> {
        self.buffer.extend_from_slice(chunk);
        let mut observations = Vec::new();
        while self.buffer.len().saturating_sub(self.read_offset) >= self.period {
            let end = self.read_offset + self.period;
            let frame_start = self.absolute_start + self.read_offset;
            let window = MatchWindow {
                frame_index: self.next_frame,
                capture_start: frame_start,
                capture_end: frame_start + self.period,
                min_lag: self.matcher.config().min_lag,
                max_lag: self.max_lag,
            };
            let detection = self
                .matcher
                .detect(&self.buffer[self.read_offset..end], window);
            observations.push(self.tracker.process(&detection));
            self.read_offset = end;
            self.next_frame += 1;
        }
        self.compact();
        observations
    }

    pub fn tracker(&self) -> &Tracker {
        &self.tracker
    }

    pub fn frames_processed(&self) -> usize {
        self.next_frame
    }

    pub fn samples_processed(&self) -> usize {
        self.absolute_start + self.read_offset
    }

    fn compact(&mut self) {
        if self.read_offset >= self.period * 4 {
            self.buffer.drain(..self.read_offset);
            self.absolute_start += self.read_offset;
            self.read_offset = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn different_chunk_sizes_produce_identical_lags() {
        let probe = Probe::default_probe();
        let tracker_config = TrackerConfig {
            confidence_threshold: 0.03,
            ..TrackerConfig::default()
        };
        let mut session = MeasurementSession::new(&probe, MatcherConfig::default(), tracker_config);
        let mut frame = vec![0.0; probe.period()];
        for i in 0..probe.probe_len() {
            frame[211 + i] = probe.sample_at(i);
        }
        let mut regular = Vec::new();
        for _ in 0..6 {
            regular.extend(session.push(&frame));
        }

        let mut irregular_session =
            MeasurementSession::new(&probe, MatcherConfig::default(), tracker_config);
        let mut stream = Vec::new();
        for _ in 0..6 {
            stream.extend_from_slice(&frame);
        }
        let mut irregular = Vec::new();
        let mut offset = 0;
        for size in [1, 37, 900, 5, 1_574, 41, 2_051, 88, 3_111] {
            let end = (offset + size).min(stream.len());
            irregular.extend(irregular_session.push(&stream[offset..end]));
            offset = end;
            if offset == stream.len() {
                break;
            }
        }
        while offset < stream.len() {
            let end = (offset + 137).min(stream.len());
            irregular.extend(irregular_session.push(&stream[offset..end]));
            offset = end;
        }

        assert_eq!(regular.len(), irregular.len());
        for (left, right) in regular.iter().zip(&irregular) {
            assert_eq!(left.window, right.window);
            assert_eq!(left.reported_lag, right.reported_lag);
            assert_eq!(left.state, right.state);
        }
    }
}
