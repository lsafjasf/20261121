use riaa_eq::cli::{self, Command, TransientKind};
use riaa_eq::engine::{
    build_chain, impulse_signal, plan_curve, run_chain, tone_burst_signal, EngineConfig,
    RenderOutcome,
};
use riaa_eq::wav::{read_wav, write_wav, OverflowReport, SampleFormat, WavData};
use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let command = match cli::parse(raw) {
        Ok(command) => command,
        Err(message) => {
            if message.starts_with("riaa-eq") {
                println!("{message}");
                return ExitCode::SUCCESS;
            }
            eprintln!("error: {message}");
            return ExitCode::from(2);
        }
    };
    match dispatch(command) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

fn dispatch(command: Command) -> Result<(), String> {
    match command {
        Command::Analyze(args) => {
            let config = EngineConfig {
                sample_rate: args.sample_rate,
                guard_fraction: args.guard_fraction,
                ..EngineConfig::default()
            };
            if args.json {
                riaa_eq::report::print_analysis_json(config);
            } else {
                riaa_eq::report::print_analysis(config);
            }
            Ok(())
        }
        Command::Render(args) => render(args),
        Command::Transient(args) => transient(args),
        Command::Verify(args) => {
            let config = EngineConfig {
                sample_rate: args.sample_rate,
                guard_fraction: args.guard_fraction,
                ..EngineConfig::default()
            };
            if args.json {
                print_verify_json(config);
            } else {
                riaa_eq::report::print_verify(config);
            }
            Ok(())
        }
    }
}

fn render(args: cli::RenderArgs) -> Result<(), String> {
    let kind = plan_curve(args.input_repr, args.output_repr)?;
    let input = read_wav(Path::new(&args.input)).map_err(|err| err.to_string())?;
    let config = EngineConfig {
        sample_rate: input.sample_rate as f64,
        guard_fraction: args.guard_fraction,
        ..EngineConfig::default()
    };
    let mut chain = build_chain(
        kind,
        config.sample_rate,
        config.guard_fraction,
        input.channels as usize,
    );
    let outcome = run_chain(&mut chain, &input, config, args.flush);
    let report = write_wav(
        Path::new(&args.output),
        input.sample_rate,
        input.channels,
        args.output_format,
        &outcome.samples,
    )
    .map_err(|err| err.to_string())?;

    println!("render complete");
    println!(
        "  curve        : {} -> {} ({:?})",
        args.input_repr.label(),
        args.output_repr.label(),
        kind
    );
    println!(
        "  input        : {} Hz, {} ch, {} frames, {:?}",
        input.sample_rate,
        input.channels,
        input.frames(),
        input.format
    );
    println!(
        "  output       : {} ({:?})",
        args.output, args.output_format
    );
    println!(
        "  output peak  : {:.6} ({:+.3} dBFS)",
        outcome.peak,
        dbfs(outcome.peak)
    );
    print_states(&outcome, args.flush);
    print_overflow(report, args.output_format);
    Ok(())
}

fn transient(args: cli::TransientArgs) -> Result<(), String> {
    let audio: WavData = match args.kind {
        TransientKind::Impulse => impulse_signal(args.sample_rate, args.duration_ms, args.channels),
        TransientKind::ToneBurst => tone_burst_signal(
            args.sample_rate,
            args.duration_ms,
            args.tone_hz,
            args.amplitude,
            args.channels,
        ),
    };
    let samples = match args.kind {
        TransientKind::Impulse => audio
            .samples
            .iter()
            .map(|x| x * args.amplitude)
            .collect::<Vec<f64>>(),
        TransientKind::ToneBurst => audio.samples.clone(),
    };
    let report = write_wav(
        Path::new(&args.output),
        audio.sample_rate,
        audio.channels,
        args.output_format,
        &samples,
    )
    .map_err(|err| err.to_string())?;
    let peak = samples.iter().copied().fold(0.0_f64, f64::max);
    println!("transient written: {}", args.output);
    println!(
        "  kind         : {:?}, {} Hz, {} ms, {} ch, {:?}",
        args.kind, audio.sample_rate, args.duration_ms, args.channels, args.output_format
    );
    println!("  peak         : {:.6} ({:+.3} dBFS)", peak, dbfs(peak));
    print_overflow(report, args.output_format);
    Ok(())
}

fn print_states(outcome: &RenderOutcome, flush: bool) {
    if outcome.head_states.is_empty() {
        println!("  filter state : bypass (no state)");
        return;
    }
    for (stage, states) in outcome.head_states.iter().enumerate() {
        for (channel, state) in states.iter().enumerate() {
            println!(
                "  head state   : stage {stage} ch {channel} s1 = {:+.3e}, s2 = {:+.3e}",
                state.s1, state.s2
            );
        }
    }
    for (stage, states) in outcome.tail_states.iter().enumerate() {
        for (channel, state) in states.iter().enumerate() {
            println!(
                "  tail state   : stage {stage} ch {channel} s1 = {:+.3e}, s2 = {:+.3e}",
                state.s1, state.s2
            );
        }
    }
    if flush {
        for (stage, states) in outcome.end_states.iter().enumerate() {
            for (channel, state) in states.iter().enumerate() {
                println!(
                    "  end state    : stage {stage} ch {channel} s1 = {:+.3e}, s2 = {:+.3e}",
                    state.s1, state.s2
                );
            }
        }
        let taps: usize = outcome.flush_taps.iter().sum();
        println!("  flush taps   : {taps} zero-input frames appended");
    }
}

fn print_overflow(report: OverflowReport, format: SampleFormat) {
    println!(
        "  over FS      : {} sample(s) exceeded |1.0|",
        report.over_full_scale
    );
    match format {
        SampleFormat::Float32 => {
            println!("  handling     : float32 preserves over-full-scale values, no clipping");
        }
        integer => {
            println!(
                "  handling     : hard saturation to {:?} extremes, {} clipped sample(s)",
                integer, report.clipped
            );
        }
    }
}

fn dbfs(amplitude: f64) -> f64 {
    20.0 * amplitude.max(1e-300).log10()
}

fn print_verify_json(config: EngineConfig) {
    let rows = riaa_eq::verify::frequency_table(config);
    let truncated = riaa_eq::verify::roundtrip_time_domain(config, false);
    let flushed = riaa_eq::verify::roundtrip_time_domain(config, true);
    let amplitude = riaa_eq::verify::amplitude_roundtrip(config);
    let (pass, max_dev, max_err) = riaa_eq::verify::passes(config);
    println!("{{");
    println!("  \"sample_rate_hz\": {},", config.sample_rate);
    println!("  \"guard_fraction\": {},", config.guard_fraction);
    println!("  \"max_roundtrip_db_4k\": {:.9},", max_dev);
    println!("  \"max_relative_amplitude_error\": {:.9e},", max_err);
    println!("  \"pass\": {pass},");
    println!(
        "  \"truncated\": {{\"max_abs\": {:.9e}, \"rms\": {:.9e}, \"peak\": {:.9e}}},",
        truncated.time_error.max_abs, truncated.time_error.rms, truncated.output_peak
    );
    println!(
        "  \"flushed\": {{\"max_abs\": {:.9e}, \"rms\": {:.9e}, \"peak\": {:.9e}, \"taps\": {}}},",
        flushed.time_error.max_abs,
        flushed.time_error.rms,
        flushed.output_peak,
        flushed.flush_taps.iter().sum::<usize>()
    );
    println!("  \"amplitude_components\": [");
    for (index, frequency) in riaa_eq::verify::PROBE_FREQUENCIES_HZ.iter().enumerate() {
        let comma = if index + 1 == riaa_eq::verify::PROBE_FREQUENCIES_HZ.len() {
            ""
        } else {
            ","
        };
        println!(
            "    {{\"f_hz\": {:.1}, \"reference\": {:.9e}, \"measured\": {:.9e}, \"relative_error\": {:.9e}}}{}",
            frequency,
            amplitude.reference_amplitudes[index],
            amplitude.measured_amplitudes[index],
            amplitude.relative_errors[index],
            comma
        );
    }
    println!("  ],");
    println!("  \"frequency_rows\": [");
    for (index, row) in rows.iter().enumerate() {
        let comma = if index + 1 == rows.len() { "" } else { "," };
        println!(
            "    {{\"f_digital_hz\": {:.4}, \"roundtrip_db\": {:.9}}}{}",
            row.digital_hz, row.roundtrip_db, comma
        );
    }
    println!("  ]");
    println!("}}");
}
