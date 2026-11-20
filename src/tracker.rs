use std::collections::VecDeque;

use crate::matcher::{Candidate, Detection, MatchWindow};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrackingState {
    Searching,
    Acquiring,
    Locked,
    Holdover,
}

impl TrackingState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Searching => "searching",
            Self::Acquiring => "acquiring",
            Self::Locked => "locked",
            Self::Holdover => "holdover",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackerConfig {
    pub confidence_threshold: f32,
    pub acquire_confirmations: u32,
    pub reacquire_confirmations: u32,
    pub gate_samples: usize,
    pub consistency_samples: f32,
    pub max_holdover_frames: u32,
    pub history_len: usize,
    pub challenge_ratio: f32,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.03,
            acquire_confirmations: 3,
            reacquire_confirmations: 2,
            gate_samples: 8,
            consistency_samples: 3.0,
            max_holdover_frames: 3,
            history_len: 7,
            challenge_ratio: 1.6,
        }
    }
}

/// One frame of tracker output.  `reported_lag` is the robust estimate, while
/// `observed_lag` is the raw candidate used for this decision.
#[derive(Clone, Debug, PartialEq)]
pub struct FrameObservation {
    pub frame_index: usize,
    pub window: MatchWindow,
    pub gate: Option<(usize, usize)>,
    pub confidence: f32,
    pub observed_lag: Option<f32>,
    pub reported_lag: Option<f32>,
    pub delta_from_lock: Option<f32>,
    pub accepted: bool,
    pub state: TrackingState,
    pub reason: &'static str,
}

struct Hypothesis {
    lag: f32,
    count: u32,
}

impl Hypothesis {
    fn new(lag: f32) -> Self {
        Self { lag, count: 1 }
    }

    fn update(&mut self, lag: f32, tolerance: f32) -> bool {
        if (lag - self.lag).abs() <= tolerance {
            self.count += 1;
            self.lag += 0.2 * (lag - self.lag);
            true
        } else {
            self.lag = lag;
            self.count = 1;
            false
        }
    }
}

/// Stateful tracker that turns per-frame correlation peaks into a delay track.
pub struct Tracker {
    config: TrackerConfig,
    state: TrackingState,
    pending: Option<Hypothesis>,
    challenger: Option<Hypothesis>,
    history: VecDeque<f32>,
    reported_lag: Option<f32>,
    misses: u32,
}

impl Tracker {
    pub fn new(config: TrackerConfig) -> Self {
        Self {
            config,
            state: TrackingState::Searching,
            pending: None,
            challenger: None,
            history: VecDeque::new(),
            reported_lag: None,
            misses: 0,
        }
    }

    pub fn config(&self) -> TrackerConfig {
        self.config
    }

    pub fn state(&self) -> TrackingState {
        self.state
    }

    pub fn reported_lag(&self) -> Option<f32> {
        self.reported_lag
    }

    pub fn is_locked(&self) -> bool {
        self.state == TrackingState::Locked
    }

    pub fn process(&mut self, detection: &Detection) -> FrameObservation {
        let mut observation = FrameObservation {
            frame_index: detection.window.frame_index,
            window: detection.window,
            gate: self.current_gate(),
            confidence: 0.0,
            observed_lag: None,
            reported_lag: self.reported_lag,
            delta_from_lock: None,
            accepted: false,
            state: self.state,
            reason: "no_candidate",
        };

        match self.state {
            TrackingState::Locked => self.process_locked(detection, &mut observation),
            TrackingState::Holdover => self.process_holdover(detection, &mut observation),
            TrackingState::Searching | TrackingState::Acquiring => {
                self.process_searching(detection, &mut observation)
            }
        }

        observation.state = self.state;
        observation.gate = self.current_gate();
        observation.reported_lag = self.reported_lag;
        observation
    }

    fn process_locked(&mut self, detection: &Detection, observation: &mut FrameObservation) {
        let Some(reference) = self.reported_lag else {
            self.release_to_searching();
            self.process_searching(detection, observation);
            return;
        };
        observation.gate = self.current_gate();

        let gate_min = (reference - self.config.gate_samples as f32).max(0.0);
        let gate_max = reference + self.config.gate_samples as f32;
        let in_gate = detection
            .candidates
            .iter()
            .filter(|candidate| {
                candidate.confidence >= self.config.confidence_threshold
                    && candidate.lag() >= gate_min
                    && candidate.lag() <= gate_max
            })
            .min_by(|a, b| {
                (a.lag() - reference)
                    .abs()
                    .total_cmp(&(b.lag() - reference).abs())
                    .then_with(|| b.confidence.total_cmp(&a.confidence))
            })
            .copied();

        if let Some(candidate) = in_gate {
            observation.confidence = candidate.confidence;
            observation.observed_lag = Some(candidate.lag());
            observation.delta_from_lock = Some(candidate.lag() - reference);
            self.misses = 0;
            if self.challenger_is_stronger(detection, &candidate, reference) {
                let challenged = self.challenger.as_ref().is_some_and(|hypothesis| {
                    hypothesis.count >= self.config.acquire_confirmations
                });
                if challenged {
                    self.release_to_searching();
                    self.process_searching(detection, observation);
                    observation.reason = "challenger_switch";
                    return;
                }
            } else {
                self.challenger = None;
            }
            self.accept_locked(candidate);
            observation.accepted = true;
            observation.reason = "locked_peak";
            return;
        }

        self.misses += 1;
        self.state = TrackingState::Holdover;
        observation.reason = "gate_miss";
        observation.gate = self.current_gate();

        if self.misses > self.config.max_holdover_frames {
            self.release_to_searching();
            observation.reason = "lost_lock";
            self.process_searching(detection, observation);
        }
    }

    fn process_holdover(&mut self, detection: &Detection, observation: &mut FrameObservation) {
        let gate_min = self
            .reported_lag
            .map(|lag| (lag - self.config.gate_samples as f32).max(0.0));
        let gate_max = self
            .reported_lag
            .map(|lag| lag + self.config.gate_samples as f32);
        let in_gate = detection.candidates.iter().find(|candidate| {
            candidate.confidence >= self.config.confidence_threshold
                && gate_min.is_some_and(|min| candidate.lag() >= min)
                && gate_max.is_some_and(|max| candidate.lag() <= max)
        });

        if let Some(candidate) = in_gate.copied() {
            observation.confidence = candidate.confidence;
            observation.observed_lag = Some(candidate.lag());
            self.state = TrackingState::Locked;
            self.misses = 0;
            self.accept_locked(candidate);
            observation.accepted = true;
            observation.reason = "recovered_inside_gate";
            return;
        }

        self.misses += 1;
        observation.reason = "holdover";
        if self.misses > self.config.max_holdover_frames {
            self.release_to_searching();
            observation.reason = "lost_lock";
            self.process_searching(detection, observation);
        }
    }

    fn process_searching(&mut self, detection: &Detection, observation: &mut FrameObservation) {
        observation.gate = None;
        let Some(candidate) = detection
            .candidates
            .iter()
            .find(|candidate| candidate.confidence >= self.config.confidence_threshold)
        else {
            self.pending = None;
            self.state = TrackingState::Searching;
            observation.reason = "search_no_candidate";
            return;
        };

        observation.confidence = candidate.confidence;
        observation.observed_lag = Some(candidate.lag());
        let required = if self.reported_lag.is_some() {
            self.config.reacquire_confirmations
        } else {
            self.config.acquire_confirmations
        };
        let confirmed = match &mut self.pending {
            Some(hypothesis) => hypothesis.update(candidate.lag(), self.config.consistency_samples),
            None => {
                self.pending = Some(Hypothesis::new(candidate.lag()));
                true
            }
        };
        let count = self
            .pending
            .as_ref()
            .map_or(0, |hypothesis| hypothesis.count);
        observation.reason = if confirmed {
            "acquire_confirm"
        } else {
            "acquire_reset"
        };

        if count >= required {
            let lag = self.pending.as_ref().map_or(candidate.lag(), |h| h.lag);
            self.lock(lag);
            observation.accepted = true;
            observation.reason = "acquired";
        } else {
            self.state = TrackingState::Acquiring;
        }
    }

    fn accept_locked(&mut self, candidate: Candidate) {
        self.push_history(candidate.lag());
        self.reported_lag = Some(median(&self.history));
        self.state = TrackingState::Locked;
    }

    fn lock(&mut self, lag: f32) {
        self.history.clear();
        self.history.push_back(lag);
        self.reported_lag = Some(lag);
        self.misses = 0;
        self.pending = None;
        self.challenger = None;
        self.state = TrackingState::Locked;
    }

    fn release_to_searching(&mut self) {
        self.state = TrackingState::Searching;
        self.pending = None;
        self.challenger = None;
        self.misses = 0;
        // Keep the last robust delay only as diagnostic state.  A new lock
        // clears the history after confirmation, so the old track cannot pull
        // the replacement estimate back to a stale value.
    }

    fn push_history(&mut self, lag: f32) {
        if self.history.len() >= self.config.history_len {
            self.history.pop_front();
        }
        self.history.push_back(lag);
    }

    fn current_gate(&self) -> Option<(usize, usize)> {
        if !matches!(self.state, TrackingState::Locked | TrackingState::Holdover) {
            return None;
        }
        let lag = self.reported_lag?;
        Some((
            (lag - self.config.gate_samples as f32).max(0.0) as usize,
            (lag + self.config.gate_samples as f32) as usize,
        ))
    }

    fn challenger_is_stronger(
        &mut self,
        detection: &Detection,
        in_gate: &Candidate,
        reference: f32,
    ) -> bool {
        let Some(global) = detection.best() else {
            return false;
        };
        if (global.lag() - reference).abs() <= self.config.gate_samples as f32 {
            return false;
        }
        if global.confidence <= in_gate.confidence * self.config.challenge_ratio {
            return false;
        }
        match &mut self.challenger {
            Some(hypothesis) => {
                hypothesis.update(global.lag(), self.config.consistency_samples);
            }
            None => self.challenger = Some(Hypothesis::new(global.lag())),
        }
        true
    }
}

fn median(history: &VecDeque<f32>) -> f32 {
    let mut values: Vec<f32> = history.iter().copied().collect();
    values.sort_by(f32::total_cmp);
    if values.is_empty() {
        return 0.0;
    }
    values[values.len() / 2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matcher::MatchWindow;

    fn detection(frame: usize, candidates: &[(usize, f32)]) -> Detection {
        Detection {
            window: MatchWindow {
                frame_index: frame,
                capture_start: frame * 2048,
                capture_end: (frame + 1) * 2048,
                min_lag: 0,
                max_lag: 1025,
            },
            candidates: candidates
                .iter()
                .map(|&(lag, confidence)| Candidate {
                    lag_samples: lag,
                    lag_fractional: 0.0,
                    correlation: confidence,
                    split_balance: 1.0,
                    confidence,
                })
                .collect(),
        }
    }

    fn lock_at(lag: usize) -> Tracker {
        let mut tracker = Tracker::new(TrackerConfig::default());
        for frame in 0..4 {
            tracker.process(&detection(frame, &[(lag, 0.8)]));
        }
        assert!(tracker.is_locked());
        tracker
    }

    #[test]
    fn acquisition_requires_confirmation() {
        let mut tracker = Tracker::new(TrackerConfig::default());
        let first = tracker.process(&detection(0, &[(100, 0.8)]));
        assert!(!first.accepted);
        assert_eq!(first.state, TrackingState::Acquiring);
        let second = tracker.process(&detection(1, &[(100, 0.8)]));
        assert!(!second.accepted);
        let third = tracker.process(&detection(2, &[(100, 0.8)]));
        assert!(third.accepted);
        assert!(tracker.is_locked());
    }

    #[test]
    fn one_false_peak_does_not_move_reported_delay() {
        let mut tracker = lock_at(100);
        for frame in 4..9 {
            tracker.process(&detection(frame, &[(100, 0.8)]));
        }
        let before = tracker.reported_lag().unwrap();
        let false_frame = tracker.process(&detection(9, &[(800, 0.99)]));
        assert!(!false_frame.accepted);
        assert!((tracker.reported_lag().unwrap() - before).abs() < 0.01);

        let next = tracker.process(&detection(10, &[(100, 0.8)]));
        assert!(next.accepted);
        assert!((next.reported_lag.unwrap() - 100.0).abs() < 0.01);
    }

    #[test]
    fn one_in_gate_false_peak_is_absorbed_by_median() {
        let mut tracker = lock_at(100);
        for frame in 4..9 {
            tracker.process(&detection(frame, &[(100, 0.8)]));
        }
        tracker.process(&detection(9, &[(112, 0.95)]));
        assert!((tracker.reported_lag().unwrap() - 100.0).abs() < 0.01);
        let next = tracker.process(&detection(10, &[(100, 0.8)]));
        assert!(next.accepted);
        assert!((next.reported_lag.unwrap() - 100.0).abs() < 0.01);
    }

    #[test]
    fn large_step_reacquires_after_holdover() {
        let mut tracker = lock_at(100);
        for frame in 4..7 {
            tracker.process(&detection(frame, &[(300, 0.8)]));
        }
        let mut recovered = None;
        for frame in 7..14 {
            let result = tracker.process(&detection(frame, &[(300, 0.85)]));
            if result.accepted && result.state == TrackingState::Locked {
                recovered = Some(result);
                break;
            }
        }
        let recovered = recovered.expect("tracker should reacquire the new delay");
        assert!((recovered.reported_lag.unwrap() - 300.0).abs() < 3.0);
    }

    #[test]
    fn a_consistent_global_challenger_displaces_a_wrong_lock() {
        let mut tracker = lock_at(100);
        for frame in 4..9 {
            tracker.process(&detection(frame, &[(100, 0.8)]));
        }

        let mut switched = None;
        for frame in 9..18 {
            let result = tracker.process(&detection(frame, &[(300, 0.9), (100, 0.2)]));
            if result.accepted
                && result
                    .reported_lag
                    .is_some_and(|lag| (lag - 300.0).abs() < 1.0)
            {
                switched = Some(result);
                break;
            }
        }
        let switched = switched.expect("the consistently stronger global peak should win");
        assert_eq!(switched.state, TrackingState::Locked);
    }
}
