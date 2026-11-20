use std::fs;
use std::io::Cursor;
use std::path::Path;

use anyhow::{Context, Result, bail};
use claxon::FlacReader;
use md5::{Digest, Md5};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecodedAudio {
    pub sample_rate: u32,
    pub channels: u8,
    pub bits_per_sample: u8,
    pub samples: Vec<i32>,
}

impl DecodedAudio {
    pub fn total_samples(&self) -> u64 {
        self.samples.len() as u64 / u64::from(self.channels)
    }

    pub fn frame(&self, sample: u64) -> Result<&[i32]> {
        let channels = usize::from(self.channels);
        let start = usize::try_from(sample)
            .ok()
            .and_then(|value| value.checked_mul(channels))
            .ok_or_else(|| anyhow::anyhow!("sample offset overflow"))?;
        let end = start
            .checked_add(channels)
            .ok_or_else(|| anyhow::anyhow!("sample offset overflow"))?;
        self.samples
            .get(start..end)
            .ok_or_else(|| anyhow::anyhow!("sample {sample} is past the end of the stream"))
    }

    pub fn slice(&self, start: u64, end: u64) -> Result<&[i32]> {
        if end > self.total_samples() || start > end {
            bail!(
                "sample range {start}..{end} is outside 0..{}",
                self.total_samples()
            );
        }
        let channels = u64::from(self.channels);
        let first = usize::try_from(start * channels)
            .map_err(|_| anyhow::anyhow!("sample range is too large"))?;
        let last = usize::try_from(end * channels)
            .map_err(|_| anyhow::anyhow!("sample range is too large"))?;
        Ok(&self.samples[first..last])
    }

    pub fn pcm_md5(&self, start: u64, end: u64) -> Result<[u8; 16]> {
        let pcm = self.slice(start, end)?;
        let bytes_per_sample = (usize::from(self.bits_per_sample) + 7) / 8;
        let mut hasher = Md5::new();
        for &sample in pcm {
            let encoded = i64::from(sample).to_le_bytes();
            hasher.update(&encoded[..bytes_per_sample]);
        }
        Ok(hasher.finalize().into())
    }
}

pub fn decode_flac(data: &[u8]) -> Result<DecodedAudio> {
    let cursor = Cursor::new(data);
    let mut reader = FlacReader::new(cursor).context("failed to parse FLAC stream")?;
    let stream_info = reader.streaminfo();
    if !(1..=8).contains(&stream_info.channels) {
        bail!("unsupported channel count {}", stream_info.channels);
    }
    let channels = stream_info.channels as u8;
    let capacity = stream_info
        .samples
        .and_then(|samples| usize::try_from(samples).ok())
        .and_then(|samples| samples.checked_mul(usize::from(channels)))
        .unwrap_or(0);
    let mut samples = Vec::with_capacity(capacity);
    for sample in reader.samples() {
        samples.push(sample.context("failed to decode FLAC audio")?);
    }
    if samples.len() % usize::from(channels) != 0 {
        bail!("decoder produced a partial inter-channel sample");
    }

    Ok(DecodedAudio {
        sample_rate: stream_info.sample_rate,
        channels,
        bits_per_sample: stream_info.bits_per_sample as u8,
        samples,
    })
}

pub fn decode_flac_file(path: impl AsRef<Path>) -> Result<DecodedAudio> {
    let path = path.as_ref();
    let data = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    decode_flac(&data).with_context(|| format!("failed to decode {}", path.display()))
}
