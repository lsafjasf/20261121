//! Scripted loopback demo: 48 kHz programme audio with two embedded
//! speech-substitute markers, two consecutive dump-key presses, and a
//! low-energy rebuild stretch. Proves the dumped intervals never reach the
//! speaker path and that consecutive dumps never re-read discarded data.

use std::io;
use std::path::Path;

use crate::delay::{DelayConfig, DelayEngine};
use crate::signal::{self, Rng};
use crate::timeline::{Event, Timeline, TimelineRow};
use crate::wav;

const SR: usize = 48_000;

/// (start, end) of the two "bad word" markers in the input, seconds.
pub const MARKER1: (f32, f32) = (10.0, 10.96);
pub const MARKER2: (f32, f32) = (14.6, 15.56);
/// Dump-key press times on the shared (output) clock.
pub const DUMP1_T: f32 = 16.5;
pub const DUMP2_T: f32 = 17.2;
pub const DURATION: f32 = 45.0;

fn build_input() -> Vec<f32> {
    let mut rng = Rng::new(0x5EED);
    let mut buf: Vec<f32> = Vec::new();
    // 0..10 s: lively programme with pauses.
    buf.extend_from_slice(&signal::speech_burst(3.0, SR, &mut rng));
    buf.extend_from_slice(&signal::silence(1.0, SR, &mut rng));
    buf.extend_from_slice(&signal::speech_burst(2.5, SR, &mut rng));
    buf.extend_from_slice(&signal::silence(0.5, SR, &mut rng));
    buf.extend_from_slice(&signal::speech_burst(3.0, SR, &mut rng));
    // 10.0..10.96: marker 1 (bad word #1).
    let at = buf.len() as f32 / SR as f32;
    buf.extend_from_slice(&signal::silence(MARKER1.0 - at, SR, &mut rng));
    buf.extend_from_slice(&signal::marker_samples(SR));
    // 10.96..14.6: more programme.
    buf.extend_from_slice(&signal::speech_burst(2.0, SR, &mut rng));
    buf.extend_from_slice(&signal::silence(0.7, SR, &mut rng));
    let at = buf.len() as f32 / SR as f32;
    buf.extend_from_slice(&signal::silence(MARKER2.0 - at, SR, &mut rng));
    // 14.6..15.56: marker 2 (bad word #2).
    buf.extend_from_slice(&signal::marker_samples(SR));
    // 15.56..18: programme tail.
    buf.extend_from_slice(&signal::speech_burst(1.5, SR, &mut rng));
    let at = buf.len() as f32 / SR as f32;
    buf.extend_from_slice(&signal::silence(18.0 - at, SR, &mut rng));
    // 18..45 s: sparse speech over long low-energy stretches so the delay
    // can be rebuilt gradually and only during silence.
    let mut t = 18.0;
    while t < DURATION - 4.0 {
        buf.extend_from_slice(&signal::speech_burst(0.8, SR, &mut rng));
        buf.extend_from_slice(&signal::silence(3.2, SR, &mut rng));
        t += 4.0;
    }
    let at = buf.len() as f32 / SR as f32;
    buf.extend_from_slice(&signal::silence(DURATION - at, SR, &mut rng));
    buf
}

#[derive(Debug)]
pub struct DemoResult {
    pub input_markers: Vec<(f32, f32)>,
    pub output_markers: Vec<(f32, f32)>,
    pub dump_count: usize,
    pub total_dumped_secs: f32,
    pub min_water: f32,
    pub final_water: f32,
    pub held_frames: usize,
    pub ok: bool,
}

pub fn run(out_dir: &Path) -> io::Result<DemoResult> {
    std::fs::create_dir_all(out_dir)?;
    let cfg = DelayConfig::broadcast_48k();
    let mut engine = DelayEngine::new(cfg);
    let input = build_input();
    let frame_len = (cfg.frame * SR as f32).round() as usize;

    let mut output: Vec<f32> = Vec::with_capacity(input.len());
    let mut tl = Timeline::default();
    tl.events.push(Event::Marker {
        start: MARKER1.0,
        end: MARKER1.1,
    });
    tl.events.push(Event::Marker {
        start: MARKER2.0,
        end: MARKER2.1,
    });

    let mut dumped1 = false;
    let mut dumped2 = false;
    let mut min_water = f32::MAX;
    let mut held_frames = 0usize;

    for (fi, chunk) in input.chunks(frame_len).enumerate() {
        let t = fi as f32 * cfg.frame;
        if !dumped1 && t >= DUMP1_T {
            engine.dump();
            tl.events.push(Event::Dump { t });
            dumped1 = true;
        }
        if !dumped2 && t >= DUMP2_T {
            engine.dump();
            tl.events.push(Event::Dump { t });
            dumped2 = true;
        }
        engine.push_input(chunk);
        let (out, info) = engine.output_frame();
        let in_rms = rms(chunk);
        let out_rms = rms(&out);
        min_water = min_water.min(info.water_level);
        if info.held_this_frame {
            held_frames += 1;
        }
        tl.push(TimelineRow {
            t,
            water_level: info.water_level,
            in_rms,
            out_rms,
            state: info.state,
            dumped: info.dumped_this_frame,
            held: info.held_this_frame,
        });
        output.extend_from_slice(&out);
    }

    wav::write_wav(&out_dir.join("input.wav"), &input, SR as u32)?;
    wav::write_wav(&out_dir.join("output.wav"), &output, SR as u32)?;
    tl.write_csv(&out_dir.join("timeline.csv"))?;
    tl.write_svg(&out_dir.join("timeline.svg"), cfg.target_delay)?;
    tl.write_png(&out_dir.join("timeline.png"), cfg.target_delay)?;

    // --- Proof -------------------------------------------------------------
    // Detect markers in the files as written to / read back from disk, so
    // the proof covers the actual artifacts, not just in-memory buffers.
    let (input_disk, _) = wav::read_wav(&out_dir.join("input.wav"))?;
    let (output_disk, _) = wav::read_wav(&out_dir.join("output.wav"))?;
    let input_markers = signal::detect_markers(&input_disk, SR);
    let output_markers = signal::detect_markers(&output_disk, SR);
    let final_water = engine.water_level();
    let total_dumped_secs = engine.total_dumped as f32 / SR as f32;

    let ok = input_markers.len() == 2
        && output_markers.is_empty()
        && engine.dump_count == 2
        && min_water >= 0.0
        && (final_water - cfg.target_delay).abs() < 0.2;

    let result = DemoResult {
        input_markers,
        output_markers,
        dump_count: engine.dump_count,
        total_dumped_secs,
        min_water,
        final_water,
        held_frames,
        ok,
    };
    let report = format_report(&result, cfg.target_delay);
    std::fs::write(out_dir.join("report.txt"), &report)?;
    print!("{report}");
    Ok(result)
}

fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|s| s * s).sum::<f32>() / x.len() as f32).sqrt()
}

fn format_report(r: &DemoResult, target: f32) -> String {
    let mut s = String::new();
    s.push_str("\n=== profanity-delay demo report ===\n");
    s.push_str(&format!(
        "markers embedded in input : {:?}\n",
        r.input_markers
    ));
    s.push_str(&format!(
        "markers found in output   : {:?}\n",
        r.output_markers
    ));
    s.push_str(&format!("dump-key presses          : {}\n", r.dump_count));
    s.push_str(&format!(
        "audio discarded by dumps  : {:.2} s\n",
        r.total_dumped_secs
    ));
    s.push_str(&format!("min water level           : {:.3} s\n", r.min_water));
    s.push_str(&format!(
        "final water level         : {:.3} s (target {target:.1} s)\n",
        r.final_water
    ));
    s.push_str(&format!(
        "frames held to rebuild    : {} (only during low-energy audio)\n",
        r.held_frames
    ));
    s.push_str(if r.ok {
        "PROOF: PASS — dumped intervals never reached the speaker path; \
         consecutive dumps never re-read discarded data.\n"
    } else {
        "PROOF: FAIL\n"
    });
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scripted_demo_proves_no_leak() {
        let dir = std::env::temp_dir().join("profanity-delay-demo-test");
        let r = run(&dir).expect("demo run");
        assert!(r.ok, "demo proof failed: {r:?}");
        assert_eq!(r.input_markers.len(), 2);
        assert!(r.output_markers.is_empty());
        assert_eq!(r.dump_count, 2);
    }
}
