use std::env;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::process::ExitCode;

use roundtrip_probe::{
    ChunkPattern, FalseBurst, FrameObservation, LoopbackConfig, MatcherConfig, MeasurementSession,
    Probe, ProbeConfig, SoftwareLoopback, TrackerConfig,
};

const HELP: &str = "\
roundtrip-probe - coded audio round-trip delay/jitter measurement

Usage:
  roundtrip-probe [run] [options]
  roundtrip-probe self-test

Options:
  --frames N                 Number of analysis periods (default: from schedule)
  --delay-schedule SPEC      lag:frames,... (default: 80:40,140:40,65:40)
  --delays CSV               Explicit per-frame delays; overrides schedule
  --jitter-samples X         Deterministic sine-mix jitter added to delays
  --probe-len N              Probe length in samples (default: 1023)
  --period N                 Probe period in samples (default: 2048)
  --amplitude X              Probe amplitude (default: 0.008)
  --sample-rate N            Sample rate metadata (default: 48000)
  --echo-gain X              Loopback return gain (default: 0.8)
  --noise-rms X              Additive white-noise RMS (default: 0.008)
  --interference-rms X       Concurrent source RMS (default: 0.04)
  --seed N                   Deterministic source/noise seed
  --chunks CSV               Cyclic capture chunk sizes (default: irregular)
  --regular-chunks N         Use one constant capture chunk size
  --dropout-frames CSV       Frames where the true return is muted
  --false-burst F:L:A        Inject one probe-shaped false peak (repeatable)
  --confidence X             Detector/tracker confidence threshold (default: 0.03)
  --gate-samples N           Locked-track jitter gate in samples (default: 8)
  --jsonl PATH               Write JSONL observations to a file
  --quiet                    Write no JSONL to stdout
  -h, --help                 Show this help
";

struct Options {
    frames: Option<usize>,
    delay_schedule: String,
    delays: Option<Vec<f32>>,
    jitter_samples: f32,
    probe_len: usize,
    period: usize,
    amplitude: f32,
    sample_rate: u32,
    echo_gain: f32,
    noise_rms: f32,
    interference_rms: f32,
    seed: u64,
    chunks: Option<Vec<usize>>,
    regular_chunks: Option<usize>,
    dropout_frames: Vec<usize>,
    false_bursts: Vec<FalseBurst>,
    confidence: f32,
    gate_samples: usize,
    jsonl: Option<String>,
    quiet: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            frames: None,
            delay_schedule: "80:40,140:40,65:40".to_string(),
            delays: None,
            jitter_samples: 0.0,
            probe_len: 1_023,
            period: 2_048,
            amplitude: 0.008,
            sample_rate: 48_000,
            echo_gain: 0.8,
            noise_rms: 0.008,
            interference_rms: 0.04,
            seed: 0x5eed_1234_abcd_9876,
            chunks: None,
            regular_chunks: None,
            dropout_frames: Vec::new(),
            false_bursts: Vec::new(),
            confidence: 0.03,
            gate_samples: 8,
            jsonl: None,
            quiet: false,
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let Some(first) = args.next() else {
        return simulate(Options::default());
    };
    if matches!(first.as_str(), "-h" | "--help" | "help") {
        print!("{HELP}");
        return Ok(());
    }
    if first == "self-test" {
        return self_test();
    }

    let mut values = Vec::new();
    if first != "run" {
        values.push(first);
    }
    values.extend(args);
    simulate(parse_options(values)?)
}

fn parse_options(values: Vec<String>) -> Result<Options, Box<dyn std::error::Error>> {
    let mut options = Options::default();
    let mut index = 0;
    while index < values.len() {
        let flag = &values[index];
        if matches!(flag.as_str(), "-h" | "--help") {
            print!("{HELP}");
            std::process::exit(0);
        }
        if flag == "--quiet" {
            options.quiet = true;
            index += 1;
            continue;
        }
        index += 1;
        let Some(value) = values.get(index) else {
            return Err(usage_error(format!("missing value for {flag}")));
        };
        index += 1;
        match flag.as_str() {
            "--frames" => options.frames = Some(parse_number(value, flag)?),
            "--delay-schedule" => options.delay_schedule = value.clone(),
            "--delays" => options.delays = Some(parse_f32_list(value, flag)?),
            "--jitter-samples" => options.jitter_samples = parse_number(value, flag)?,
            "--probe-len" => options.probe_len = parse_number(value, flag)?,
            "--period" => options.period = parse_number(value, flag)?,
            "--amplitude" => options.amplitude = parse_number(value, flag)?,
            "--sample-rate" => options.sample_rate = parse_number(value, flag)?,
            "--echo-gain" => options.echo_gain = parse_number(value, flag)?,
            "--noise-rms" => options.noise_rms = parse_number(value, flag)?,
            "--interference-rms" => options.interference_rms = parse_number(value, flag)?,
            "--seed" => options.seed = parse_number(value, flag)?,
            "--chunks" => options.chunks = Some(parse_usize_list(value, flag)?),
            "--regular-chunks" => options.regular_chunks = Some(parse_number(value, flag)?),
            "--dropout-frames" => options.dropout_frames = parse_usize_list(value, flag)?,
            "--false-burst" => options.false_bursts.push(parse_false_burst(value)?),
            "--confidence" => options.confidence = parse_number(value, flag)?,
            "--gate-samples" => options.gate_samples = parse_number(value, flag)?,
            "--jsonl" => options.jsonl = Some(value.clone()),
            _ => return Err(usage_error(format!("unknown option {flag}"))),
        }
    }
    Ok(options)
}

fn usage_error(message: String) -> Box<dyn std::error::Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidInput, message))
}

fn parse_number<T>(value: &str, flag: &str) -> Result<T, Box<dyn std::error::Error>>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .parse()
        .map_err(|error| usage_error(format!("invalid {flag} value {value:?}: {error}")))
}

fn parse_f32_list(value: &str, flag: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    value
        .split(',')
        .map(|item| parse_number(item.trim(), flag))
        .collect()
}

fn parse_usize_list(value: &str, flag: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    value
        .split(',')
        .map(|item| parse_number(item.trim(), flag))
        .collect()
}

fn parse_false_burst(value: &str) -> Result<FalseBurst, Box<dyn std::error::Error>> {
    let fields: Vec<&str> = value.split(':').collect();
    if fields.len() != 3 {
        return Err(usage_error(
            "--false-burst expects FRAME:LAG:AMPLITUDE".to_string(),
        ));
    }
    Ok(FalseBurst {
        frame: parse_number(fields[0], "--false-burst frame")?,
        lag: parse_number(fields[1], "--false-burst lag")?,
        amplitude: parse_number(fields[2], "--false-burst amplitude")?,
    })
}

fn build_delays(options: &Options) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut delays = if let Some(delays) = &options.delays {
        delays.clone()
    } else {
        let mut delays = Vec::new();
        for segment in options.delay_schedule.split(',') {
            let fields: Vec<&str> = segment.split(':').collect();
            if fields.len() != 2 {
                return Err(usage_error(format!(
                    "invalid delay schedule segment {segment:?}"
                )));
            }
            let delay: f32 = parse_number(fields[0], "--delay-schedule lag")?;
            let count: usize = parse_number(fields[1], "--delay-schedule frames")?;
            delays.extend(std::iter::repeat_n(delay, count));
        }
        delays
    };

    if let Some(frames) = options.frames {
        if delays.is_empty() {
            return Err(usage_error("cannot expand an empty delay plan".to_string()));
        }
        delays.resize(frames, *delays.last().unwrap());
    }
    if delays.is_empty() {
        return Err(usage_error("delay plan is empty".to_string()));
    }
    if options.jitter_samples != 0.0 {
        for (index, delay) in delays.iter_mut().enumerate() {
            let frame = index as f32;
            let jitter = options.jitter_samples
                * (0.65 * (frame * 1.37).sin() + 0.35 * (frame * 0.31).sin());
            *delay = (*delay + jitter).max(0.0);
        }
    }
    Ok(delays)
}

fn simulate(options: Options) -> Result<(), Box<dyn std::error::Error>> {
    let delays = build_delays(&options)?;
    let probe = Probe::new(ProbeConfig {
        sample_rate: options.sample_rate,
        probe_len: options.probe_len,
        period: options.period,
        amplitude: options.amplitude,
        seed: options.seed,
    })?;
    let chunks = if let Some(size) = options.regular_chunks {
        ChunkPattern::regular(size)?
    } else if let Some(sizes) = options.chunks.clone() {
        ChunkPattern::new(sizes)?
    } else {
        ChunkPattern::default()
    };
    let config = LoopbackConfig {
        frames: delays.len(),
        delays: delays.clone(),
        echo_gain: options.echo_gain,
        noise_rms: options.noise_rms,
        interference_rms: options.interference_rms,
        interference_seed: options.seed,
        chunks,
        dropout_frames: options.dropout_frames.clone(),
        false_bursts: options.false_bursts.clone(),
    };
    let render = SoftwareLoopback::new(probe.clone(), config)?.render();

    let matcher_config = MatcherConfig {
        min_lag: 0,
        max_lag: Some(probe.max_lag()),
        min_confidence: options.confidence,
        max_peaks: 6,
        peak_separation: 4,
    };
    let tracker_config = TrackerConfig {
        confidence_threshold: options.confidence,
        gate_samples: options.gate_samples,
        ..TrackerConfig::default()
    };
    let mut session = MeasurementSession::new(&probe, matcher_config, tracker_config);
    let mut observations = Vec::with_capacity(delays.len());
    let mut offset = 0;
    for size in &render.chunks {
        let end = offset + size;
        observations.extend(session.push(&render.capture[offset..end]));
        offset = end;
    }

    let mut file = options
        .jsonl
        .as_ref()
        .map(File::create)
        .transpose()?
        .map(BufWriter::new);
    let mut stdout = BufWriter::new(io::stdout().lock());
    for observation in &observations {
        let line = observation_json(observation);
        if let Some(writer) = &mut file {
            writeln!(writer, "{line}")?;
        }
        if !options.quiet {
            writeln!(stdout, "{line}")?;
        }
    }
    let summary = summary_json(
        &observations,
        &render.truth_delays,
        &render.chunks,
        &options,
    );
    if let Some(writer) = &mut file {
        writeln!(writer, "{summary}")?;
        writer.flush()?;
    }
    if !options.quiet {
        writeln!(stdout, "{summary}")?;
        stdout.flush()?;
    }
    Ok(())
}

fn observation_json(observation: &FrameObservation) -> String {
    let (gate_min, gate_max) = observation.gate.map_or_else(
        || ("null".to_string(), "null".to_string()),
        |(min, max)| (min.to_string(), max.to_string()),
    );
    format!(
        concat!(
            "{{\"record\":\"observation\",\"frame\":{},",
            "\"capture_start\":{},\"capture_end\":{},",
            "\"lag_min\":{},\"lag_max\":{},",
            "\"gate_min\":{},\"gate_max\":{},",
            "\"confidence\":{:.6},\"observed_lag\":{},",
            "\"reported_lag\":{},\"delta_from_lock\":{},",
            "\"accepted\":{},\"state\":\"{}\",\"reason\":\"{}\"}}"
        ),
        observation.frame_index,
        observation.window.capture_start,
        observation.window.capture_end,
        observation.window.min_lag,
        observation.window.max_lag,
        gate_min,
        gate_max,
        observation.confidence,
        format_option(observation.observed_lag),
        format_option(observation.reported_lag),
        format_option(observation.delta_from_lock),
        observation.accepted,
        observation.state.as_str(),
        observation.reason,
    )
}

fn format_option(value: Option<f32>) -> String {
    value
        .map(|value| format!("{value:.6}"))
        .unwrap_or_else(|| "null".to_string())
}

fn summary_json(
    observations: &[FrameObservation],
    truth: &[f32],
    chunks: &[usize],
    options: &Options,
) -> String {
    let observed_confidences: Vec<f32> = observations
        .iter()
        .filter(|observation| observation.observed_lag.is_some())
        .map(|observation| observation.confidence)
        .collect();
    let accepted_confidences: Vec<f32> = observations
        .iter()
        .filter(|observation| observation.accepted)
        .map(|observation| observation.confidence)
        .collect();
    let locked_frames = observations
        .iter()
        .filter(|observation| observation.accepted)
        .filter(|observation| matches!(observation.state, roundtrip_probe::TrackingState::Locked))
        .count();
    let mut errors = Vec::new();
    for observation in observations {
        if !observation.accepted {
            continue;
        }
        let (Some(reported), Some(expected)) =
            (observation.reported_lag, truth.get(observation.frame_index))
        else {
            continue;
        };
        errors.push((reported - expected).abs());
    }
    let error_rms = if errors.is_empty() {
        0.0
    } else {
        (errors.iter().map(|error| error * error).sum::<f32>() / errors.len() as f32).sqrt()
    };
    let max_error = errors.iter().copied().fold(0.0_f32, f32::max);
    let p95_error = percentile(&errors, 0.95);
    let jitter: Vec<f32> = observations
        .iter()
        .filter(|observation| observation.accepted)
        .filter_map(|observation| observation.delta_from_lock)
        .filter(|delta| delta.abs() <= options.gate_samples as f32)
        .map(f32::abs)
        .collect();
    let jitter_rms = if jitter.is_empty() {
        0.0
    } else {
        (jitter.iter().map(|value| value * value).sum::<f32>() / jitter.len() as f32).sqrt()
    };
    let jitter_p95 = percentile(&jitter, 0.95);
    let first_accepted = observations
        .iter()
        .find(|observation| observation.accepted)
        .map(|observation| observation.frame_index);
    let min_chunk = chunks.iter().copied().min().unwrap_or(0);
    let max_chunk = chunks.iter().copied().max().unwrap_or(0);

    format!(
        concat!(
            "{{\"record\":\"summary\",\"frames\":{},",
            "\"accepted_frames\":{},\"locked_frames\":{},",
            "\"first_accepted_frame\":{},",
            "\"mean_confidence\":{:.6},\"max_confidence\":{:.6},",
            "\"mean_accepted_confidence\":{:.6},",
            "\"delay_error_rms_samples\":{:.6},",
            "\"delay_error_p95_samples\":{:.6},",
            "\"max_abs_delay_error_samples\":{:.6},",
            "\"jitter_rms_samples\":{:.6},",
            "\"jitter_p95_samples\":{:.6},",
            "\"chunks\":{},\"min_chunk_size\":{},\"max_chunk_size\":{},",
            "\"noise_rms\":{:.6},\"interference_rms\":{:.6},",
            "\"probe_amplitude\":{:.6},\"confidence_threshold\":{:.6}}}"
        ),
        observations.len(),
        observations.iter().filter(|item| item.accepted).count(),
        locked_frames,
        first_accepted.map_or_else(|| "null".to_string(), |value| value.to_string()),
        mean(&observed_confidences),
        observed_confidences.iter().copied().fold(0.0, f32::max),
        mean(&accepted_confidences),
        error_rms,
        p95_error,
        max_error,
        jitter_rms,
        jitter_p95,
        chunks.len(),
        min_chunk,
        max_chunk,
        options.noise_rms,
        options.interference_rms,
        options.amplitude,
        options.confidence,
    )
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn percentile(values: &[f32], percentile: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    let index = ((sorted.len() - 1) as f32 * percentile).round() as usize;
    sorted[index]
}

fn self_test() -> Result<(), Box<dyn std::error::Error>> {
    let options = Options {
        quiet: true,
        frames: Some(120),
        ..Options::default()
    };
    simulate(options)?;
    println!("self-test: rendered and analyzed 120 frames");
    Ok(())
}
