use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::decode::{DecodedAudio, decode_flac};
use crate::encode::{encode_verbatim_frame, rebuild_reused_frame};
use crate::frame::{BlockingStrategy, Frame, parse_frames, parse_header};
use crate::metadata::{ParsedFlac, StreamInfo, build_flac};

const MAX_SEEK_POINTS: usize = 1000;

#[derive(Clone, Debug)]
pub struct TrimOptions {
    pub start: u64,
    pub end: Option<u64>,
}

impl TrimOptions {
    pub fn new(start: u64, end: Option<u64>) -> Self {
        Self { start, end }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SeekPointInfo {
    pub sample: u64,
    pub offset: u64,
    pub frame_samples: u16,
}

#[derive(Clone, Debug)]
pub struct TrimReport {
    pub source_samples: u64,
    pub output_samples: u64,
    pub start: u64,
    pub end: u64,
    pub source_frames: usize,
    pub output_frames: usize,
    pub reused_frames: usize,
    pub reencoded_frames: usize,
    pub seek_points: Vec<SeekPointInfo>,
    pub md5: [u8; 16],
    pub blocking_strategy: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FlacSummary {
    pub stream_info: StreamInfo,
    pub frame_count: usize,
    pub seek_points: Vec<SeekPointInfo>,
}

#[derive(Clone, Copy, Debug)]
enum Action {
    Reuse(usize),
    Encode { start: u64, end: u64 },
}

#[derive(Clone, Debug)]
struct OutputFrame {
    sample_start: u64,
    offset: u64,
    block_size: u32,
    length: u32,
}

pub fn trim_flac(data: &[u8], options: &TrimOptions) -> Result<(Vec<u8>, TrimReport)> {
    let parsed = ParsedFlac::parse(data.to_vec())?;
    let frames = parse_frames(&parsed.data, parsed.audio_offset, &parsed.stream_info)?;
    let decoded = decode_flac(data)?;
    validate_source(&parsed.stream_info, &frames, &decoded)?;

    let total_samples = decoded.total_samples();
    let end = options.end.unwrap_or(total_samples);
    if options.start >= end {
        bail!("trim range must not be empty");
    }
    if end > total_samples {
        bail!("trim range ends at sample {end}, beyond the source's {total_samples} samples");
    }

    let actions = plan_actions(&frames, options.start, end)?;
    let strategy = choose_strategy(&frames, &parsed.stream_info, &actions, options.start);
    let (audio, output_frames) = render_frames(
        &parsed.data,
        &frames,
        &actions,
        &decoded,
        &parsed.stream_info,
        strategy,
    )?;
    let output_samples = end - options.start;
    let md5 = decoded.pcm_md5(options.start, end)?;
    let (seek_table, seek_points) = make_seek_table(&output_frames);

    let actual_min_block = output_frames
        .iter()
        .map(|frame| frame.block_size)
        .min()
        .ok_or_else(|| anyhow::anyhow!("trim produced no frames"))?;
    let actual_max_block = output_frames
        .iter()
        .map(|frame| frame.block_size)
        .max()
        .expect("output frames are non-empty");
    let min_frame_size = output_frames
        .iter()
        .map(|frame| frame.length)
        .min()
        .expect("output frames are non-empty");
    let max_frame_size = output_frames
        .iter()
        .map(|frame| frame.length)
        .max()
        .expect("output frames are non-empty");

    let (min_block_size, max_block_size) = if strategy == BlockingStrategy::Fixed {
        let fixed_size = parsed.stream_info.max_block_size;
        (fixed_size, fixed_size)
    } else {
        (
            actual_min_block.max(16) as u16,
            actual_max_block.max(16) as u16,
        )
    };
    let output_info = StreamInfo {
        min_block_size,
        max_block_size,
        min_frame_size,
        max_frame_size,
        sample_rate: parsed.stream_info.sample_rate,
        channels: parsed.stream_info.channels,
        bits_per_sample: parsed.stream_info.bits_per_sample,
        total_samples: output_samples,
        md5,
    };

    // Cue sheets describe absolute sample positions and would be stale after
    // trimming. Other user metadata (comments, pictures, applications) remains
    // valid and is retained.
    let preserved_metadata: Vec<_> = parsed
        .metadata
        .iter()
        .filter(|block| block.block_type != 5)
        .cloned()
        .collect();
    let output = build_flac(&output_info, &seek_table, &preserved_metadata, &audio)?;

    let check = decode_flac(&output).context("trimmed output failed to decode")?;
    validate_trimmed_output(
        &check,
        decoded.slice(options.start, end)?,
        output_samples,
        md5,
        &output_info,
    )?;

    let reused_frames = actions
        .iter()
        .filter(|action| matches!(action, Action::Reuse(_)))
        .count();
    let report = TrimReport {
        source_samples: total_samples,
        output_samples,
        start: options.start,
        end,
        source_frames: frames.len(),
        output_frames: output_frames.len(),
        reused_frames,
        reencoded_frames: output_frames.len() - reused_frames,
        seek_points,
        md5,
        blocking_strategy: match strategy {
            BlockingStrategy::Fixed => "fixed",
            BlockingStrategy::Variable => "variable",
        },
    };
    Ok((output, report))
}

pub fn trim_flac_file(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    options: &TrimOptions,
) -> Result<TrimReport> {
    let input = input.as_ref();
    let output = output.as_ref();
    let data = fs::read(input).with_context(|| format!("failed to read {}", input.display()))?;
    let (trimmed, report) =
        trim_flac(&data, options).with_context(|| format!("failed to trim {}", input.display()))?;
    fs::write(output, trimmed).with_context(|| format!("failed to write {}", output.display()))?;
    Ok(report)
}

pub fn inspect_flac(data: &[u8]) -> Result<FlacSummary> {
    let parsed = ParsedFlac::parse(data.to_vec())?;
    let frames = parse_frames(&parsed.data, parsed.audio_offset, &parsed.stream_info)?;
    let mut seek_points = Vec::new();
    for block in &parsed.metadata {
        if block.block_type != crate::metadata::SEEKTABLE_BLOCK {
            continue;
        }
        if block.data.len() % 18 != 0 {
            bail!("malformed SEEKTABLE block");
        }
        for entry in block.data.chunks_exact(18) {
            let sample = u64::from_be_bytes(entry[0..8].try_into().expect("8 bytes"));
            if sample == u64::MAX {
                continue;
            }
            seek_points.push(SeekPointInfo {
                sample,
                offset: u64::from_be_bytes(entry[8..16].try_into().expect("8 bytes")),
                frame_samples: u16::from_be_bytes(entry[16..18].try_into().expect("2 bytes")),
            });
        }
    }
    Ok(FlacSummary {
        stream_info: parsed.stream_info,
        frame_count: frames.len(),
        seek_points,
    })
}

fn validate_source(info: &StreamInfo, frames: &[Frame], decoded: &DecodedAudio) -> Result<()> {
    if info.sample_rate != decoded.sample_rate
        || info.channels != decoded.channels
        || info.bits_per_sample != decoded.bits_per_sample
    {
        bail!("STREAMINFO and decoded audio disagree on the audio format");
    }
    if info.total_samples != 0 && info.total_samples != decoded.total_samples() {
        bail!(
            "STREAMINFO declares {} samples, decoder produced {}",
            info.total_samples,
            decoded.total_samples()
        );
    }
    let frame_samples = frames
        .last()
        .map(Frame::sample_end)
        .ok_or_else(|| anyhow::anyhow!("source contains no frames"))?;
    if frame_samples != decoded.total_samples() {
        bail!(
            "frame headers cover {frame_samples} samples, decoder produced {}",
            decoded.total_samples()
        );
    }
    if info.md5 != [0; 16] {
        let actual = decoded.pcm_md5(0, decoded.total_samples())?;
        if actual != info.md5 {
            bail!("source PCM does not match its STREAMINFO MD5");
        }
    }
    Ok(())
}

fn plan_actions(frames: &[Frame], start: u64, end: u64) -> Result<Vec<Action>> {
    let mut index = frames.partition_point(|frame| frame.sample_end() <= start);
    if index >= frames.len() {
        bail!("trim start is past the last audio frame");
    }
    let mut position = start;
    let mut actions = Vec::new();
    while position < end {
        let frame = &frames[index];
        if position < frame.sample_start {
            bail!("sample gap before frame {index}");
        }
        let segment_end = frame.sample_end().min(end);
        if position == frame.sample_start && segment_end == frame.sample_end() {
            actions.push(Action::Reuse(index));
        } else {
            actions.push(Action::Encode {
                start: position,
                end: segment_end,
            });
        }
        position = segment_end;
        index += 1;
        if position < end && index == frames.len() {
            bail!("trim range extends beyond the final frame");
        }
    }
    Ok(actions)
}

fn action_block_size(action: Action, frames: &[Frame]) -> u32 {
    match action {
        Action::Reuse(index) => frames[index].header.block_size,
        Action::Encode { start, end } => (end - start) as u32,
    }
}

fn choose_strategy(
    frames: &[Frame],
    info: &StreamInfo,
    actions: &[Action],
    start: u64,
) -> BlockingStrategy {
    let source_is_fixed = frames
        .first()
        .is_some_and(|frame| frame.header.strategy == BlockingStrategy::Fixed);
    let starts_on_frame = frames.iter().any(|frame| frame.sample_start == start);
    if !source_is_fixed || !starts_on_frame {
        return BlockingStrategy::Variable;
    }

    let fixed_size = u32::from(info.max_block_size);
    let fits_fixed = actions.iter().enumerate().all(|(index, action)| {
        let size = action_block_size(*action, frames);
        size == fixed_size || index + 1 == actions.len()
    });
    if fits_fixed {
        BlockingStrategy::Fixed
    } else {
        BlockingStrategy::Variable
    }
}

fn render_frames(
    source: &[u8],
    source_frames: &[Frame],
    actions: &[Action],
    decoded: &DecodedAudio,
    info: &StreamInfo,
    strategy: BlockingStrategy,
) -> Result<(Vec<u8>, Vec<OutputFrame>)> {
    let mut audio = Vec::new();
    let mut output_frames = Vec::with_capacity(actions.len());
    let mut sample_position = 0u64;
    for (index, action) in actions.iter().enumerate() {
        let sample_start = sample_position;
        if strategy == BlockingStrategy::Fixed
            && sample_start != index as u64 * u64::from(info.max_block_size)
        {
            bail!("internal error: fixed-block frame position is not aligned");
        }
        let (bytes, block_size) = match *action {
            Action::Reuse(source_index) => {
                let frame = &source_frames[source_index];
                let number = match strategy {
                    BlockingStrategy::Fixed => index as u64,
                    BlockingStrategy::Variable => sample_start,
                };
                (
                    rebuild_reused_frame(source, frame, strategy, number, info.bits_per_sample)?,
                    frame.header.block_size,
                )
            }
            Action::Encode { start, end } => {
                let pcm = decoded.slice(start, end)?;
                let number = match strategy {
                    BlockingStrategy::Fixed => index as u64,
                    BlockingStrategy::Variable => sample_start,
                };
                (
                    encode_verbatim_frame(
                        pcm,
                        info.channels,
                        info.bits_per_sample,
                        strategy,
                        number,
                    )?,
                    (end - start) as u32,
                )
            }
        };
        let parsed = parse_header(&bytes, 0)?;
        if parsed.block_size != block_size {
            bail!("internal error: encoded block-size mismatch");
        }
        if bytes.len() >= (1 << 24) {
            bail!("frame is too large for STREAMINFO");
        }
        output_frames.push(OutputFrame {
            sample_start,
            offset: audio.len() as u64,
            block_size,
            length: bytes.len() as u32,
        });
        audio.extend_from_slice(&bytes);
        sample_position += u64::from(block_size);
    }
    Ok((audio, output_frames))
}

fn make_seek_table(frames: &[OutputFrame]) -> (Vec<u8>, Vec<SeekPointInfo>) {
    let mut selected = Vec::new();
    if frames.len() <= MAX_SEEK_POINTS {
        selected.extend(0..frames.len());
    } else {
        for index in 0..MAX_SEEK_POINTS {
            selected.push(index * frames.len() / MAX_SEEK_POINTS);
        }
        selected.dedup();
        if selected.last().copied() != Some(frames.len() - 1) {
            selected.push(frames.len() - 1);
        }
    }

    let points: Vec<_> = selected
        .into_iter()
        .map(|index| {
            let frame = &frames[index];
            SeekPointInfo {
                sample: frame.sample_start,
                offset: frame.offset,
                frame_samples: frame.block_size.min(u32::from(u16::MAX)) as u16,
            }
        })
        .collect();

    let mut table = Vec::with_capacity(points.len() * 18);
    for point in &points {
        table.extend_from_slice(&point.sample.to_be_bytes());
        table.extend_from_slice(&point.offset.to_be_bytes());
        table.extend_from_slice(&point.frame_samples.to_be_bytes());
    }
    (table, points)
}

fn validate_trimmed_output(
    decoded: &DecodedAudio,
    expected: &[i32],
    output_samples: u64,
    md5: [u8; 16],
    info: &StreamInfo,
) -> Result<()> {
    if decoded.total_samples() != output_samples {
        bail!("trimmed output contains the wrong number of samples");
    }
    if decoded.samples != expected {
        bail!("trimmed output PCM differs from the requested source range");
    }
    if decoded.pcm_md5(0, output_samples)? != md5 {
        bail!("trimmed output MD5 differs from the generated digest");
    }
    if info.md5 != md5 {
        bail!("internal error: STREAMINFO MD5 mismatch");
    }
    Ok(())
}
