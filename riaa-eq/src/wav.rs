//! Minimal RIFF/WAVE reader and writer for the formats this tool needs:
//! PCM 16/24/32-bit integer and IEEE 754 32-bit float.

use std::fs;
use std::io::{self, Write};
use std::path::Path;

/// Supported sample encodings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFormat {
    Pcm16,
    Pcm24,
    Pcm32,
    Float32,
}

impl SampleFormat {
    pub fn parse(text: &str) -> Option<Self> {
        match text.to_ascii_lowercase().as_str() {
            "pcm16" | "s16" => Some(Self::Pcm16),
            "pcm24" | "s24" => Some(Self::Pcm24),
            "pcm32" | "s32" => Some(Self::Pcm32),
            "float32" | "f32" => Some(Self::Float32),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Pcm16 => "pcm16",
            Self::Pcm24 => "pcm24",
            Self::Pcm32 => "pcm32",
            Self::Float32 => "float32",
        }
    }

    fn bytes_per_sample(self) -> usize {
        match self {
            Self::Pcm16 => 2,
            Self::Pcm24 => 3,
            Self::Pcm32 | Self::Float32 => 4,
        }
    }
}

/// Decoded audio: interleaved `f64` samples in the [-1, 1] nominal range.
#[derive(Debug, Clone)]
pub struct WavData {
    pub sample_rate: u32,
    pub channels: u16,
    pub format: SampleFormat,
    pub samples: Vec<f64>,
}

impl WavData {
    pub fn frames(&self) -> usize {
        self.samples.len() / self.channels as usize
    }
}

/// Errors surfaced by WAV parsing.
#[derive(Debug)]
pub enum WavError {
    Io(io::Error),
    Malformed(String),
}

impl std::fmt::Display for WavError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::Malformed(msg) => write!(f, "malformed WAV: {msg}"),
        }
    }
}

impl std::error::Error for WavError {}

impl From<io::Error> for WavError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

fn take<'a>(
    bytes: &'a [u8],
    at: &mut usize,
    count: usize,
    what: &str,
) -> Result<&'a [u8], WavError> {
    if *at + count > bytes.len() {
        return Err(WavError::Malformed(format!("truncated {what}")));
    }
    let slice = &bytes[*at..*at + count];
    *at += count;
    Ok(slice)
}

fn u16le(bytes: &[u8]) -> u16 {
    u16::from_le_bytes([bytes[0], bytes[1]])
}

fn u32le(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

/// Read and decode a WAV file. Unknown chunks are skipped.
pub fn read_wav(path: &Path) -> Result<WavData, WavError> {
    let bytes = fs::read(path)?;
    let mut pos = 0;
    let riff = take(&bytes, &mut pos, 4, "RIFF tag")?;
    if riff != b"RIFF" {
        return Err(WavError::Malformed("missing RIFF tag".into()));
    }
    pos += 4; // overall size, ignored
    let wave = take(&bytes, &mut pos, 4, "WAVE tag")?;
    if wave != b"WAVE" {
        return Err(WavError::Malformed("missing WAVE tag".into()));
    }

    let mut audio_format = 0u16;
    let mut channels = 0u16;
    let mut sample_rate = 0u32;
    let mut bits = 0u16;
    let mut data: Option<&[u8]> = None;

    while pos + 8 <= bytes.len() {
        let chunk_id = take(&bytes, &mut pos, 4, "chunk id")?;
        let chunk_size = u32le(take(&bytes, &mut pos, 4, "chunk size")?) as usize;
        let chunk = take(&bytes, &mut pos, chunk_size, "chunk body")?;
        if chunk_size % 2 == 1 && pos < bytes.len() {
            pos += 1; // RIFF padding byte
        }
        match chunk_id {
            b"fmt " => {
                if chunk.len() < 16 {
                    return Err(WavError::Malformed("short fmt chunk".into()));
                }
                audio_format = u16le(&chunk[0..2]);
                channels = u16le(&chunk[2..4]);
                sample_rate = u32le(&chunk[4..8]);
                bits = u16le(&chunk[14..16]);
            }
            b"data" => data = Some(chunk),
            _ => {}
        }
    }

    let format = match (audio_format, bits) {
        (1, 16) => SampleFormat::Pcm16,
        (1, 24) => SampleFormat::Pcm24,
        (1, 32) => SampleFormat::Pcm32,
        (3, 32) => SampleFormat::Float32,
        (tag, depth) => {
            return Err(WavError::Malformed(format!(
                "unsupported format tag {tag} / bit depth {depth}"
            )))
        }
    };
    if channels == 0 || sample_rate == 0 {
        return Err(WavError::Malformed("missing fmt metadata".into()));
    }
    let data = data.ok_or_else(|| WavError::Malformed("missing data chunk".into()))?;
    let frame_bytes = format.bytes_per_sample() * channels as usize;
    if frame_bytes == 0 || data.len() % frame_bytes != 0 {
        return Err(WavError::Malformed(
            "data length does not align to frames".into(),
        ));
    }
    let samples = decode_samples(data, format);
    Ok(WavData {
        sample_rate,
        channels,
        format,
        samples,
    })
}

fn decode_samples(data: &[u8], format: SampleFormat) -> Vec<f64> {
    let stride = format.bytes_per_sample();
    let count = data.len() / stride;
    let mut out = Vec::with_capacity(count);
    for index in 0..count {
        let base = index * stride;
        let value = match format {
            SampleFormat::Pcm16 => {
                i16::from_le_bytes([data[base], data[base + 1]]) as f64 / 32_768.0
            }
            SampleFormat::Pcm24 => {
                let raw = (data[base] as i32)
                    | ((data[base + 1] as i32) << 8)
                    | ((data[base + 2] as i32) << 16);
                let signed = if raw & 0x0080_0000 != 0 {
                    raw | -0x0100_0000
                } else {
                    raw
                };
                signed as f64 / 8_388_608.0
            }
            SampleFormat::Pcm32 => {
                i32::from_le_bytes([data[base], data[base + 1], data[base + 2], data[base + 3]])
                    as f64
                    / 2_147_483_648.0
            }
            SampleFormat::Float32 => {
                f32::from_le_bytes([data[base], data[base + 1], data[base + 2], data[base + 3]])
                    as f64
            }
        };
        out.push(value);
    }
    out
}

/// Outcome of encoding potentially out-of-range samples.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OverflowReport {
    /// Samples whose absolute value exceeded 1.0 before output handling.
    pub over_full_scale: usize,
    /// Integer samples that had to be hard-clipped to the codec extremes.
    pub clipped: usize,
}

fn encode_pcm16(sample: f64, report: &mut OverflowReport) -> [u8; 2] {
    if sample.abs() > 1.0 {
        report.over_full_scale += 1;
    }
    let scaled = (sample * 32_768.0).round();
    let limited = scaled.clamp(-32_768.0, 32_767.0);
    if scaled != limited {
        report.clipped += 1;
    }
    (limited as i16).to_le_bytes()
}

fn encode_pcm24(sample: f64, report: &mut OverflowReport) -> [u8; 3] {
    if sample.abs() > 1.0 {
        report.over_full_scale += 1;
    }
    let scaled = (sample * 8_388_608.0).round();
    let clamped = scaled.clamp(-8_388_608.0, 8_388_607.0);
    if scaled != clamped {
        report.clipped += 1;
    }
    let value = clamped as i32;
    [
        (value & 0xff) as u8,
        ((value >> 8) & 0xff) as u8,
        ((value >> 16) & 0xff) as u8,
    ]
}

fn encode_pcm32(sample: f64, report: &mut OverflowReport) -> [u8; 4] {
    if sample.abs() > 1.0 {
        report.over_full_scale += 1;
    }
    let scaled = (sample * 2_147_483_648.0).round();
    let clamped = scaled.clamp(-2_147_483_648.0, 2_147_483_647.0);
    if scaled != clamped {
        report.clipped += 1;
    }
    (clamped as i32).to_le_bytes()
}

fn encode_float32(sample: f64, report: &mut OverflowReport) -> [u8; 4] {
    if sample.abs() > 1.0 {
        report.over_full_scale += 1;
    }
    (sample as f32).to_le_bytes()
}

/// Encode interleaved samples into the body of a `data` chunk.
pub fn encode_samples(samples: &[f64], format: SampleFormat) -> (Vec<u8>, OverflowReport) {
    let stride = format.bytes_per_sample();
    let mut out = Vec::with_capacity(samples.len() * stride);
    let mut report = OverflowReport::default();
    for &sample in samples {
        let bytes: &[u8] = match format {
            SampleFormat::Pcm16 => &encode_pcm16(sample, &mut report),
            SampleFormat::Pcm24 => &encode_pcm24(sample, &mut report),
            SampleFormat::Pcm32 => &encode_pcm32(sample, &mut report),
            SampleFormat::Float32 => &encode_float32(sample, &mut report),
        };
        out.extend_from_slice(bytes);
    }
    (out, report)
}

/// Write a WAV file, emitting a `fact` chunk for float output.
pub fn write_wav(
    path: &Path,
    sample_rate: u32,
    channels: u16,
    format: SampleFormat,
    samples: &[f64],
) -> io::Result<OverflowReport> {
    let (data, report) = encode_samples(samples, format);
    let bits = (format.bytes_per_sample() * 8) as u16;
    let audio_format: u16 = match format {
        SampleFormat::Float32 => 3,
        _ => 1,
    };
    let block_align = channels * format.bytes_per_sample() as u16;
    let byte_rate = sample_rate * block_align as u32;
    let fact_len = if format == SampleFormat::Float32 {
        12u32
    } else {
        0
    };
    let riff_size = 4 + (8 + 16) + fact_len + (8 + data.len() as u32);

    let mut file = fs::File::create(path)?;
    file.write_all(b"RIFF")?;
    file.write_all(&riff_size.to_le_bytes())?;
    file.write_all(b"WAVE")?;
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(&audio_format.to_le_bytes())?;
    file.write_all(&channels.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&block_align.to_le_bytes())?;
    file.write_all(&bits.to_le_bytes())?;
    if format == SampleFormat::Float32 {
        file.write_all(b"fact")?;
        file.write_all(&4u32.to_le_bytes())?;
        file.write_all(&(samples.len() as u32).to_le_bytes())?;
    }
    file.write_all(b"data")?;
    file.write_all(&(data.len() as u32).to_le_bytes())?;
    file.write_all(&data)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm16_overflow_is_counted_and_clipped() {
        let (bytes, report) = encode_samples(&[1.5, -1.5, 0.5], SampleFormat::Pcm16);
        assert_eq!(report.over_full_scale, 2);
        assert_eq!(report.clipped, 2);
        assert_eq!(bytes.len(), 6);
        assert_eq!(i16::from_le_bytes([bytes[0], bytes[1]]), 32_767);
    }

    #[test]
    fn float32_preserves_over_full_scale() {
        let (_, report) = encode_samples(&[1.25], SampleFormat::Float32);
        assert_eq!(report.over_full_scale, 1);
        assert_eq!(report.clipped, 0);
    }

    #[test]
    fn roundtrip_pcm24() {
        let samples = vec![-0.75, 0.0, 0.999];
        let tmp = std::env::temp_dir().join("riaa_eq_roundtrip_pcm24.wav");
        write_wav(&tmp, 48_000, 1, SampleFormat::Pcm24, &samples).unwrap();
        let loaded = read_wav(&tmp).unwrap();
        assert_eq!(loaded.sample_rate, 48_000);
        assert_eq!(loaded.channels, 1);
        for (a, b) in samples.iter().zip(loaded.samples.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        let _ = fs::remove_file(&tmp);
    }
}
