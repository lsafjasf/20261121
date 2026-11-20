use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::decode::{DecodedAudio, decode_flac};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RangeVerification {
    pub start: u64,
    pub end: u64,
    pub samples_compared: u64,
    pub equal: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerificationReport {
    pub format_match: bool,
    pub length_match: bool,
    pub ranges: Vec<RangeVerification>,
}

impl VerificationReport {
    pub fn is_match(&self) -> bool {
        self.format_match && self.length_match && self.ranges.iter().all(|range| range.equal)
    }
}

pub fn verify_flac(
    expected: &[u8],
    actual: &[u8],
    ranges: &[(u64, u64)],
) -> Result<VerificationReport> {
    let expected = decode_flac(expected).context("failed to decode expected source")?;
    let actual = decode_flac(actual).context("failed to decode actual source")?;
    verify_decoded(&expected, &actual, ranges)
}

pub fn verify_flac_file(
    expected: impl AsRef<Path>,
    actual: impl AsRef<Path>,
    ranges: &[(u64, u64)],
) -> Result<VerificationReport> {
    let expected_path = expected.as_ref();
    let actual_path = actual.as_ref();
    let expected = fs::read(expected_path)
        .with_context(|| format!("failed to read {}", expected_path.display()))?;
    let actual = fs::read(actual_path)
        .with_context(|| format!("failed to read {}", actual_path.display()))?;
    verify_flac(&expected, &actual, ranges)
}

fn verify_decoded(
    expected: &DecodedAudio,
    actual: &DecodedAudio,
    ranges: &[(u64, u64)],
) -> Result<VerificationReport> {
    let format_match = expected.sample_rate == actual.sample_rate
        && expected.channels == actual.channels
        && expected.bits_per_sample == actual.bits_per_sample;
    let length_match = expected.total_samples() == actual.total_samples();
    if !format_match {
        return Ok(VerificationReport {
            format_match,
            length_match,
            ranges: Vec::new(),
        });
    }
    if ranges.is_empty() {
        bail!("at least one sample range is required");
    }

    let mut results = Vec::with_capacity(ranges.len());
    for &(start, end) in ranges {
        if start >= end {
            bail!("verification range {start}..{end} is empty");
        }
        let expected_samples = expected.slice(start, end)?;
        let actual_samples = actual.slice(start, end)?;
        results.push(RangeVerification {
            start,
            end,
            samples_compared: end - start,
            equal: expected_samples == actual_samples,
        });
    }
    Ok(VerificationReport {
        format_match,
        length_match,
        ranges: results,
    })
}
