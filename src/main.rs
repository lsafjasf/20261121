use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use flac_trim::{
    TrimOptions, decode_flac, decode_flac_file, inspect_flac, trim_flac_file, verify_flac_file,
};

#[derive(Parser)]
#[command(
    name = "flac-trim",
    about = "Sample-accurate FLAC trimming with encoded-frame reuse"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Copy a sample range into a new, independently decodable FLAC stream.
    Trim {
        input: PathBuf,
        output: PathBuf,
        /// First sample to keep (zero based, inclusive).
        #[arg(long)]
        start: u64,
        /// First sample after the kept range (exclusive). Defaults to EOF.
        #[arg(long)]
        end: Option<u64>,
        #[arg(long)]
        json: bool,
    },
    /// Print stream metadata and integrity information.
    Info {
        input: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Decode two FLAC files and compare PCM over one or more ranges.
    Verify {
        expected: PathBuf,
        actual: PathBuf,
        /// Sample range START:END. May be supplied more than once. Without
        /// this option, beginning, seek-point, and ending ranges are probed.
        #[arg(long = "range", value_name = "START:END")]
        ranges: Vec<String>,
        #[arg(long)]
        json: bool,
    },
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    match Cli::parse().command {
        Command::Trim {
            input,
            output,
            start,
            end,
            json,
        } => trim(input, output, start, end, json),
        Command::Info { input, json } => info(input, json),
        Command::Verify {
            expected,
            actual,
            ranges,
            json,
        } => verify(expected, actual, ranges, json),
    }
}

fn trim(input: PathBuf, output: PathBuf, start: u64, end: Option<u64>, json: bool) -> Result<()> {
    let report = trim_flac_file(&input, &output, &TrimOptions::new(start, end))?;
    if json {
        println!(
            "{{\"source_samples\":{},\"output_samples\":{},\"start\":{},\"end\":{},\
             \"source_frames\":{},\"output_frames\":{},\"reused_frames\":{},\
             \"reencoded_frames\":{},\"seek_points\":{},\"md5\":\"{}\",\
             \"blocking_strategy\":\"{}\"}}",
            report.source_samples,
            report.output_samples,
            report.start,
            report.end,
            report.source_frames,
            report.output_frames,
            report.reused_frames,
            report.reencoded_frames,
            report.seek_points.len(),
            hex(&report.md5),
            report.blocking_strategy,
        );
    } else {
        println!(
            "trimmed samples {}..{} ({} samples) -> {}",
            report.start,
            report.end,
            report.output_samples,
            output.display()
        );
        println!(
            "frames: {} output, {} reused, {} re-encoded; seek points: {}",
            report.output_frames,
            report.reused_frames,
            report.reencoded_frames,
            report.seek_points.len()
        );
        println!(
            "blocking: {}; STREAMINFO MD5: {}",
            report.blocking_strategy,
            hex(&report.md5)
        );
    }
    Ok(())
}

fn info(input: PathBuf, json: bool) -> Result<()> {
    let data = fs::read(&input).with_context(|| format!("failed to read {}", input.display()))?;
    let summary = inspect_flac(&data)?;
    let decoded = decode_flac(&data)?;
    let total = decoded.total_samples();
    let md5 = decoded.pcm_md5(0, total)?;
    if summary.stream_info.md5 != [0; 16] && summary.stream_info.md5 != md5 {
        bail!("decoded PCM does not match the STREAMINFO MD5");
    }

    if json {
        println!(
            "{{\"sample_rate\":{},\"channels\":{},\"bits_per_sample\":{},\
             \"total_samples\":{},\"frames\":{},\"seek_points\":{},\
             \"md5\":\"{}\"}}",
            summary.stream_info.sample_rate,
            summary.stream_info.channels,
            summary.stream_info.bits_per_sample,
            total,
            summary.frame_count,
            summary.seek_points.len(),
            hex(&md5),
        );
    } else {
        println!(
            "format: {} Hz, {} channels, {}-bit",
            summary.stream_info.sample_rate,
            summary.stream_info.channels,
            summary.stream_info.bits_per_sample
        );
        println!(
            "samples: {}; frames: {}; seek points: {}",
            total,
            summary.frame_count,
            summary.seek_points.len()
        );
        println!("MD5: {}", hex(&md5));
    }
    Ok(())
}

fn verify(expected: PathBuf, actual: PathBuf, raw_ranges: Vec<String>, json: bool) -> Result<()> {
    let ranges = if raw_ranges.is_empty() {
        auto_ranges(&expected, &actual)?
    } else {
        raw_ranges
            .iter()
            .map(|value| parse_range(value))
            .collect::<Result<Vec<_>>>()?
    };
    let report = verify_flac_file(&expected, &actual, &ranges)?;

    if json {
        let ranges_json = report
            .ranges
            .iter()
            .map(|range| {
                format!(
                    "{{\"start\":{},\"end\":{},\"samples\":{},\"equal\":{}}}",
                    range.start, range.end, range.samples_compared, range.equal
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "{{\"format_match\":{},\"length_match\":{},\"match\":{},\"ranges\":[{}]}}",
            report.format_match,
            report.length_match,
            report.is_match(),
            ranges_json
        );
    } else {
        println!(
            "format match: {}; length match: {}",
            report.format_match, report.length_match
        );
        for range in &report.ranges {
            println!(
                "samples {}..{} ({} samples): {}",
                range.start,
                range.end,
                range.samples_compared,
                if range.equal { "equal" } else { "MISMATCH" }
            );
        }
    }

    if !report.is_match() {
        bail!("decoded PCM verification failed");
    }
    Ok(())
}

fn auto_ranges(expected: &PathBuf, actual: &PathBuf) -> Result<Vec<(u64, u64)>> {
    let expected_audio = decode_flac_file(expected)?;
    let actual_audio = decode_flac_file(actual)?;
    let total = expected_audio
        .total_samples()
        .min(actual_audio.total_samples());
    if total == 0 {
        bail!("cannot verify an empty stream");
    }

    let edge = total.min(256);
    let expected_data =
        fs::read(expected).with_context(|| format!("failed to read {}", expected.display()))?;
    let actual_data =
        fs::read(actual).with_context(|| format!("failed to read {}", actual.display()))?;
    let expected_seeks = inspect_flac(&expected_data)?.seek_points;
    let actual_seeks = inspect_flac(&actual_data)?.seek_points;
    let seek_target = expected_seeks
        .iter()
        .chain(actual_seeks.iter())
        .map(|point| point.sample)
        .find(|&sample| sample > total / 3 && sample < total)
        .unwrap_or(total / 2);

    Ok(vec![
        (0, edge),
        (
            seek_target.saturating_sub(128),
            (seek_target + 128).min(total),
        ),
        (total - edge, total),
    ])
}

fn parse_range(value: &str) -> Result<(u64, u64)> {
    let (start, end) = value
        .split_once(':')
        .ok_or_else(|| anyhow::anyhow!("range must use START:END syntax"))?;
    let start = start
        .parse::<u64>()
        .with_context(|| format!("invalid range start in {value:?}"))?;
    let end = end
        .parse::<u64>()
        .with_context(|| format!("invalid range end in {value:?}"))?;
    if start >= end {
        bail!("range {value:?} is empty");
    }
    Ok((start, end))
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut output, "{byte:02x}").expect("writing to a string cannot fail");
    }
    output
}
