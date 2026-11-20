//! Broadcast profanity-delay engine.
//!
//! A ring buffer holds `target_delay` seconds of programme audio. The output
//! (speaker) path always reads behind the write head. Pressing the dump key
//! crossfades across an operator-chosen interval so the offending audio never
//! reaches the speaker path. The lost delay is rebuilt gradually, and only
//! while the audio at the read head is low energy, by re-outputting the
//! current frame instead of advancing (inaudible during silence).

#[derive(Clone, Copy, Debug)]
pub struct DelayConfig {
    pub sample_rate: usize,
    /// Steady-state delay, seconds (classic "seven second delay").
    pub target_delay: f32,
    /// Interval skipped per dump-key press, seconds.
    pub dump_len: f32,
    /// Crossfade duration straddling the skipped interval, seconds.
    pub crossfade: f32,
    /// Processing frame size, seconds.
    pub frame: f32,
    /// Max fraction of frames that may be "held" to rebuild delay (0..1).
    pub rebuild_ratio: f32,
    /// RMS at the read head below which rebuild is allowed.
    pub silence_threshold: f32,
    /// Seconds of non-silence that must pass before rebuild re-arms.
    pub rebuild_holdoff: f32,
}

impl DelayConfig {
    pub fn broadcast_48k() -> Self {
        Self {
            sample_rate: 48_000,
            target_delay: 7.0,
            dump_len: 4.0,
            crossfade: 0.08,
            frame: 0.01,
            rebuild_ratio: 0.5,
            silence_threshold: 0.02,
            rebuild_holdoff: 0.25,
        }
    }

    fn samples(&self, secs: f32) -> usize {
        (secs * self.sample_rate as f32).round() as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum EngineState {
    /// Delay at target, normal operation.
    #[default]
    Steady,
    /// Crossfading across a dumped interval.
    Dumping,
    /// Delay below target, rebuilding during low-energy frames.
    Rebuilding,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FrameInfo {
    pub water_level: f32,
    pub state: EngineState,
    pub dumped_this_frame: bool,
    pub held_this_frame: bool,
}

pub struct DelayEngine {
    cfg: DelayConfig,
    ring: Vec<f32>,
    /// Absolute sample counters; index into ring via modulo.
    write: usize,
    read: usize,
    /// Samples of crossfade remaining, and total crossfade length.
    xf_left: usize,
    xf_total: usize,
    /// Skip distance of the in-progress dump, in samples.
    xf_skip: usize,
    frame_len: usize,
    /// Rate limiter: hold at most num/den of frames.
    hold_budget_num: usize,
    hold_budget_den: usize,
    hold_phase: usize,
    /// Consecutive low-energy frames seen (rebuild holdoff).
    quiet_frames: usize,
    holdoff_frames: usize,
    /// Total samples discarded by dumps (proof / accounting).
    pub total_dumped: usize,
    pub dump_count: usize,
}

impl DelayEngine {
    pub fn new(cfg: DelayConfig) -> Self {
        let cap_secs = cfg.target_delay + cfg.dump_len + cfg.crossfade + 2.0;
        let cap = cfg.samples(cap_secs);
        let frame_len = cfg.samples(cfg.frame).max(1);
        let holdoff_frames = (cfg.rebuild_holdoff / cfg.frame).ceil() as usize;
        let hold_budget_den = 8;
        let hold_budget_num =
            (cfg.rebuild_ratio.clamp(0.0, 1.0) * hold_budget_den as f32).round() as usize;
        // Arm the unit pre-filled with target_delay seconds of silence, like
        // a real delay line that has been ramped up before going on air.
        let write = cfg.samples(cfg.target_delay);
        Self {
            cfg,
            ring: vec![0.0; cap],
            write,
            read: 0,
            xf_left: 0,
            xf_total: 0,
            xf_skip: 0,
            frame_len,
            hold_budget_num,
            hold_budget_den,
            hold_phase: 0,
            quiet_frames: 0,
            holdoff_frames,
            total_dumped: 0,
            dump_count: 0,
        }
    }

    pub fn water_level(&self) -> f32 {
        (self.write - self.read) as f32 / self.cfg.sample_rate as f32
    }

    pub fn state(&self) -> EngineState {
        if self.xf_left > 0 {
            EngineState::Dumping
        } else if self.water_level() < self.cfg.target_delay - 0.05 {
            EngineState::Rebuilding
        } else {
            EngineState::Steady
        }
    }

    /// Push one frame of input (mono) into the delay line.
    pub fn push_input(&mut self, input: &[f32]) {
        for &s in input {
            // The ring is sized so the write head can never lap the read
            // head, even mid-crossfade when we read xf_skip ahead.
            debug_assert!(self.write - self.read + self.xf_skip < self.ring.len());
            let idx = self.write % self.ring.len();
            self.ring[idx] = s;
            self.write += 1;
        }
    }

    /// Operator pressed the dump key: crossfade across `dump_len` seconds.
    /// The skipped region is discarded and can never be read again because
    /// the read head only moves forward.
    pub fn dump(&mut self) {
        // A dump during an in-progress crossfade commits the previous skip
        // immediately; the read head still only moves forward.
        if self.xf_left > 0 {
            self.read += self.xf_skip;
            self.xf_left = 0;
        }
        let available = self.write - self.read;
        let want = self.cfg.samples(self.cfg.dump_len);
        // Never skip past the write head (consecutive dumps included).
        let skip = want.min(available.saturating_sub(self.frame_len));
        if skip == 0 {
            return;
        }
        let xf = self.cfg.samples(self.cfg.crossfade).min(skip);
        self.xf_total = xf;
        self.xf_left = xf;
        self.xf_skip = skip;
        self.total_dumped += skip;
        self.dump_count += 1;
    }

    fn rms_at(&self, start: usize, len: usize) -> f32 {
        let n = self.ring.len();
        let mut acc = 0.0f32;
        for i in 0..len {
            let s = self.ring[(start + i) % n];
            acc += s * s;
        }
        (acc / len.max(1) as f32).sqrt()
    }

    /// Produce one output frame for the speaker path.
    pub fn output_frame(&mut self) -> (Vec<f32>, FrameInfo) {
        let n = self.ring.len();
        let fl = self.frame_len;
        let mut out = vec![0.0f32; fl];
        let mut dumped_this_frame = false;

        let read_rms = self.rms_at(self.read, fl);

        // Decide whether this frame is "held" (repeated) to rebuild delay.
        // Only during low-energy audio at the read head, rate-limited, and
        // never while a dump crossfade is running.
        let mut held = false;
        if self.xf_left == 0
            && self.water_level() < self.cfg.target_delay
            && read_rms < self.cfg.silence_threshold
            && self.quiet_frames >= self.holdoff_frames
        {
            self.hold_phase = (self.hold_phase + 1) % self.hold_budget_den;
            if self.hold_phase < self.hold_budget_num {
                held = true;
            }
        }

        if read_rms >= self.cfg.silence_threshold {
            self.quiet_frames = 0;
        } else {
            self.quiet_frames = self.quiet_frames.saturating_add(1);
        }

        if held {
            // Re-output the current frame without advancing the read head:
            // the delay line grows by one frame of low-energy audio.
            for i in 0..fl {
                out[i] = self.ring[(self.read + i) % n];
            }
        } else {
            for i in 0..fl {
                let a = self.ring[(self.read + i) % n];
                let s = if self.xf_left > 0 {
                    dumped_this_frame = true;
                    let b = self.ring[(self.read + self.xf_skip + i) % n];
                    let t = 1.0 - self.xf_left as f32 / self.xf_total.max(1) as f32;
                    // Equal-power crossfade from pre-dump to post-dump audio.
                    let g_a = (t * std::f32::consts::FRAC_PI_2).cos();
                    let g_b = (t * std::f32::consts::FRAC_PI_2).sin();
                    self.xf_left -= 1;
                    if self.xf_left == 0 {
                        // Crossfade done: commit the skip. The dumped region
                        // [read, read+skip) is now permanently behind us.
                        self.read += self.xf_skip;
                    }
                    a * g_a + b * g_b
                } else {
                    a
                };
                out[i] = s;
            }
            self.read += fl;
        }

        let info = FrameInfo {
            water_level: self.water_level(),
            state: self.state(),
            dumped_this_frame,
            held_this_frame: held,
        };
        (out, info)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal;

    fn run(engine: &mut DelayEngine, input: &[f32], frame: usize) -> Vec<f32> {
        let mut out = Vec::new();
        for chunk in input.chunks(frame) {
            engine.push_input(chunk);
            let (o, _) = engine.output_frame();
            out.extend_from_slice(&o);
        }
        out
    }

    #[test]
    fn steady_state_delay_is_seven_seconds() {
        let cfg = DelayConfig::broadcast_48k();
        let sr = cfg.sample_rate;
        let frame = (cfg.frame * sr as f32) as usize;
        let mut engine = DelayEngine::new(cfg);
        // Impulse at t=0 of the input stream.
        let mut input = vec![0.0f32; 9 * sr];
        input[0] = 1.0;
        let out = run(&mut engine, &input, frame);
        let peak = out
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let lag = peak as f32 / sr as f32;
        assert!((lag - 7.0).abs() < 0.02, "delay was {lag}s, want ~7s");
    }

    #[test]
    fn dump_crossfades_and_drops_water() {
        let cfg = DelayConfig::broadcast_48k();
        let sr = cfg.sample_rate;
        let frame = (cfg.frame * sr as f32) as usize;
        let mut engine = DelayEngine::new(cfg);
        // Loud programme so rebuild stays disarmed.
        let input: Vec<f32> = (0..12 * sr)
            .map(|i| 0.3 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin())
            .collect();
        // Warm up to steady state, then dump at t=8s.
        let mut out = Vec::new();
        for (fi, chunk) in input.chunks(frame).enumerate() {
            if fi == 800 {
                engine.dump();
            }
            engine.push_input(chunk);
            let (o, _) = engine.output_frame();
            out.extend_from_slice(&o);
        }
        let wl = engine.water_level();
        assert!(
            (wl - (cfg.target_delay - cfg.dump_len)).abs() < 0.1,
            "water after dump {wl}"
        );
        assert_eq!(engine.dump_count, 1);
        // No clipping-level discontinuity from the crossfade.
        assert!(out.iter().all(|s| s.abs() <= 0.65));
    }

    #[test]
    fn rebuild_only_during_low_energy() {
        let cfg = DelayConfig::broadcast_48k();
        let sr = cfg.sample_rate;
        let frame = (cfg.frame * sr as f32) as usize;
        let mut engine = DelayEngine::new(cfg);
        // Loud programme for 14 s, then 20 s of silence. The read head
        // trails the write head by the water level, so the dump at t=8s
        // lands the read head in loud territory until t=17s.
        let mut input: Vec<f32> = (0..14 * sr)
            .map(|i| 0.3 * (2.0 * std::f32::consts::PI * 330.0 * i as f32 / sr as f32).sin())
            .collect();
        input.extend_from_slice(&vec![0.001f32; 20 * sr]);
        let mut water_at_16 = 0.0;
        for (fi, chunk) in input.chunks(frame).enumerate() {
            if fi == 800 {
                engine.dump();
            }
            engine.push_input(chunk);
            let (_o, info) = engine.output_frame();
            if fi == 1600 {
                water_at_16 = info.water_level;
            }
        }
        // While the read head was in loud audio (t=8..17s) no rebuild may
        // happen: water stays at target - dump_len.
        assert!(
            (water_at_16 - (cfg.target_delay - cfg.dump_len)).abs() < 0.1,
            "rebuilt during loud audio: {water_at_16}"
        );
        // Once silence reaches the read head the delay is rebuilt.
        let wl_final = engine.water_level();
        assert!(
            (wl_final - cfg.target_delay).abs() < 0.2,
            "did not rebuild in silence: {wl_final}"
        );
    }

    #[test]
    fn consecutive_dumps_never_reread_discarded_audio() {
        let cfg = DelayConfig::broadcast_48k();
        let sr = cfg.sample_rate;
        let frame = (cfg.frame * sr as f32) as usize;
        let mut engine = DelayEngine::new(cfg);
        // Input: silence with two markers at 2s and 7s.
        let mut input = vec![0.001f32; 20 * sr];
        signal::embed_marker(&mut input, 2.0, sr);
        signal::embed_marker(&mut input, 7.0, sr);
        assert_eq!(signal::detect_markers(&input, sr).len(), 2);
        // Two quick consecutive dumps, one per marker, while both sit in
        // the delay line.
        let mut out = Vec::new();
        for (fi, chunk) in input.chunks(frame).enumerate() {
            if fi == 850 || fi == 950 {
                engine.dump();
            }
            engine.push_input(chunk);
            let (o, _) = engine.output_frame();
            out.extend_from_slice(&o);
        }
        assert_eq!(engine.dump_count, 2);
        assert!(
            signal::detect_markers(&out, sr).is_empty(),
            "dumped marker leaked to the speaker path"
        );
        assert!(engine.water_level() >= 0.0);
    }
}

