//! Interactive loopback mode: synthesised 48 kHz programme runs through the
//! delay engine in real time. The operator watches the shared timeline of
//! audio and buffer water level, and presses the dump key to crossfade
//! across the operator-selected interval. Markers (speech-substitute
//! signals) are injected at random; the proof runs on quit.

use std::io::{self, Write};
use std::path::Path;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event as CtEvent, KeyCode};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};

use crate::delay::{DelayConfig, DelayEngine, EngineState};
use crate::signal::{self, Rng};
use crate::timeline::{Event, Timeline, TimelineRow};
use crate::wav;

const SR: usize = 48_000;

/// Endless programme generator: speech bursts, low-energy gaps, and the
/// occasional marker ("bad word") with a heads-up for the operator.
struct LiveProgram {
    rng: Rng,
    seg: Vec<f32>,
    pos: usize,
    marker_starts_at: Option<f32>,
    now: f32,
}

impl LiveProgram {
    fn new(seed: u64) -> Self {
        Self {
            rng: Rng::new(seed),
            seg: Vec::new(),
            pos: 0,
            marker_starts_at: None,
            now: 0.0,
        }
    }

    fn refill(&mut self) {
        let roll = self.rng.next_f32();
        if roll < 0.12 && self.now > 10.0 {
            self.seg = signal::marker_samples(SR);
            self.marker_starts_at = Some(self.now);
        } else if roll < 0.65 {
            let dur = 1.0 + 2.0 * self.rng.next_f32();
            self.seg = signal::speech_burst(dur, SR, &mut self.rng);
        } else {
            let dur = 1.0 + 2.5 * self.rng.next_f32();
            self.seg = signal::silence(dur, SR, &mut self.rng);
        }
        self.pos = 0;
    }

    fn frame(&mut self, len: usize) -> (Vec<f32>, Option<f32>) {
        let mut out = Vec::with_capacity(len);
        let mut marker_at = None;
        while out.len() < len {
            if self.pos >= self.seg.len() {
                self.refill();
                if let Some(t) = self.marker_starts_at.take() {
                    marker_at = Some(t);
                }
            }
            let take = (self.seg.len() - self.pos).min(len - out.len());
            out.extend_from_slice(&self.seg[self.pos..self.pos + take]);
            self.pos += take;
        }
        self.now += len as f32 / SR as f32;
        (out, marker_at)
    }
}

pub fn run(out_dir: &Path) -> io::Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let cfg = DelayConfig::broadcast_48k();
    let frame_len = (cfg.frame * SR as f32).round() as usize;
    let mut engine = DelayEngine::new(cfg);
    let mut program = LiveProgram::new(0xC0FFEE);
    let mut tl = Timeline::default();
    let mut input_all: Vec<f32> = Vec::new();
    let mut output_all: Vec<f32> = Vec::new();

    enable_raw_mode()?;
    println!("profanity-delay live loopback @ 48 kHz");
    println!("  [d] dump {:.1}s interval   [q] quit & write proof", cfg.dump_len);
    println!("  dump key skips {:.1}s of air audio from the moment it is pressed", cfg.dump_len);

    let start = Instant::now();
    let frame_dur = Duration::from_secs_f32(cfg.frame);
    let mut frame_idx = 0u64;
    let mut quit = false;

    while !quit {
        let deadline = start + frame_dur * (frame_idx as u32 + 1);
        let t = frame_idx as f32 * cfg.frame;

        while event::poll(Duration::ZERO)? {
            if let CtEvent::Key(k) = event::read()? {
                match k.code {
                    KeyCode::Char('d') | KeyCode::Char('D') => {
                        engine.dump();
                        tl.events.push(Event::Dump { t });
                        println!("\n[dump] pressed at {t:.2}s — crossfading across interval");
                    }
                    KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => quit = true,
                    _ => {}
                }
            }
        }

        let (input, marker_at) = program.frame(frame_len);
        if let Some(m) = marker_at {
            tl.events.push(Event::Marker {
                start: m,
                end: m + signal::marker_duration(),
            });
            let lo = m + cfg.target_delay - cfg.dump_len;
            let hi = m + cfg.target_delay;
            println!("\n[marker] bad word in input at {m:.2}s — press [d] between {lo:.2}s and {hi:.2}s to cut it");
        }
        engine.push_input(&input);
        let (out, info) = engine.output_frame();
        input_all.extend_from_slice(&input);
        let out_rms = (out.iter().map(|s| s * s).sum::<f32>() / out.len() as f32).sqrt();
        tl.push(TimelineRow {
            t,
            water_level: info.water_level,
            in_rms: (input.iter().map(|s| s * s).sum::<f32>() / input.len() as f32).sqrt(),
            out_rms,
            state: info.state,
            dumped: info.dumped_this_frame,
            held: info.held_this_frame,
        });
        output_all.extend_from_slice(&out);

        if frame_idx % 10 == 0 {
            let state = match info.state {
                EngineState::Steady => "STEADY ",
                EngineState::Dumping => "DUMPING",
                EngineState::Rebuilding => "REBUILD",
            };
            let bars = (info.water_level / cfg.target_delay * 20.0).round() as usize;
            let meter: String = "#".repeat(bars.min(20));
            print!(
                "\rt={t:6.1}s  water={:5.2}s [{meter:<20}] {state}  out={:5.1} dB   ",
                info.water_level,
                20.0 * out_rms.max(1e-6).log10()
            );
            io::stdout().flush()?;
        }

        frame_idx += 1;
        let now = Instant::now();
        if deadline > now {
            std::thread::sleep(deadline - now);
        }
    }
    disable_raw_mode()?;
    println!("\nwriting proof artifacts to {}", out_dir.display());

    wav::write_wav(&out_dir.join("live_input.wav"), &input_all, SR as u32)?;
    wav::write_wav(&out_dir.join("live_output.wav"), &output_all, SR as u32)?;
    tl.write_csv(&out_dir.join("live_timeline.csv"))?;
    tl.write_svg(&out_dir.join("live_timeline.svg"), cfg.target_delay)?;
    tl.write_png(&out_dir.join("live_timeline.png"), cfg.target_delay)?;

    let in_markers = signal::detect_markers(&input_all, SR);
    let out_markers = signal::detect_markers(&output_all, SR);
    println!("markers in input : {in_markers:?}");
    println!("markers in output: {out_markers:?}");
    println!(
        "dumps: {}, discarded: {:.2}s",
        engine.dump_count,
        engine.total_dumped as f32 / SR as f32
    );
    if out_markers.is_empty() {
        println!("PROOF: PASS — no dumped marker reached the speaker path");
    } else {
        println!("PROOF: FAIL — a marker leaked to the speaker path");
    }
    Ok(())
}
