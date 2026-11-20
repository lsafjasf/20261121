use anyhow::Result;

use crate::decode::{DecodedAudio, decode_flac};
use crate::encode::encode_verbatim_frame;
use crate::frame::{BlockingStrategy, Frame, parse_frames};
use crate::metadata::{MetadataBlock, ParsedFlac, StreamInfo, build_flac};
use crate::trim::{TrimOptions, trim_flac};
use crate::verify::verify_flac;

const CHANNELS: u8 = 2;
const BITS: u8 = 16;
const RATE: u32 = 44_100;

fn make_pcm(sample_count: usize) -> Vec<i32> {
    let mut pcm = Vec::with_capacity(sample_count * usize::from(CHANNELS));
    for sample in 0..sample_count {
        let left = ((sample as i32 * 7919 + 12345) % 65_536) - 32_768;
        let right = ((sample as i32 * 3571 + 655) % 65_536) - 32_768;
        pcm.push(left);
        pcm.push(right);
    }
    pcm
}

fn encode_source(
    pcm: &[i32],
    block_sizes: &[usize],
    strategy: BlockingStrategy,
) -> Result<Vec<u8>> {
    assert_eq!(pcm.len() % usize::from(CHANNELS), 0);
    let total_samples = pcm.len() / usize::from(CHANNELS);
    assert_eq!(block_sizes.iter().sum::<usize>(), total_samples);

    let mut audio = Vec::new();
    let mut frames = Vec::new();
    let mut sample = 0usize;
    for (index, &block_size) in block_sizes.iter().enumerate() {
        let number = match strategy {
            BlockingStrategy::Fixed => index as u64,
            BlockingStrategy::Variable => sample as u64,
        };
        let first = sample * usize::from(CHANNELS);
        let last = (sample + block_size) * usize::from(CHANNELS);
        let frame = encode_verbatim_frame(&pcm[first..last], CHANNELS, BITS, strategy, number)?;
        frames.push((
            sample as u64,
            audio.len() as u64,
            block_size as u16,
            frame.len() as u32,
        ));
        audio.extend_from_slice(&frame);
        sample += block_size;
    }

    let mut seek_table = Vec::with_capacity(frames.len() * 18);
    for (sample, offset, block_size, _) in &frames {
        seek_table.extend_from_slice(&sample.to_be_bytes());
        seek_table.extend_from_slice(&offset.to_be_bytes());
        seek_table.extend_from_slice(&block_size.to_be_bytes());
    }

    let decoded = DecodedAudio {
        sample_rate: RATE,
        channels: CHANNELS,
        bits_per_sample: BITS,
        samples: pcm.to_vec(),
    };
    let info = StreamInfo {
        min_block_size: block_sizes.iter().copied().min().unwrap() as u16,
        max_block_size: block_sizes.iter().copied().max().unwrap() as u16,
        min_frame_size: frames.iter().map(|frame| frame.3).min().unwrap(),
        max_frame_size: frames.iter().map(|frame| frame.3).max().unwrap(),
        sample_rate: RATE,
        channels: CHANNELS,
        bits_per_sample: BITS,
        total_samples: total_samples as u64,
        md5: decoded.pcm_md5(0, total_samples as u64)?,
    };
    build_flac(&info, &seek_table, &[] as &[MetadataBlock], &audio)
}

fn fixed_blocks(total: usize, block_size: usize) -> Vec<usize> {
    let mut blocks = vec![block_size; total / block_size];
    let remainder = total % block_size;
    if remainder != 0 {
        blocks.push(remainder);
    }
    blocks
}

fn variable_blocks(total: usize) -> Vec<usize> {
    let pattern = [300usize, 700, 512, 999, 401, 1024, 256, 808];
    let mut blocks = Vec::new();
    let mut remaining = total;
    let mut index = 0;
    while remaining > 0 {
        let block = pattern[index % pattern.len()].min(remaining);
        blocks.push(block);
        remaining -= block;
        index += 1;
    }
    blocks
}

fn output_frames(bytes: &[u8]) -> Result<Vec<Frame>> {
    let parsed = ParsedFlac::parse(bytes.to_vec())?;
    parse_frames(&parsed.data, parsed.audio_offset, &parsed.stream_info)
}

#[test]
fn fixed_and_variable_headers_resolve_to_the_same_sample_positions() -> Result<()> {
    let total = 5_000;
    let pcm = make_pcm(total);
    let fixed = encode_source(&pcm, &fixed_blocks(total, 512), BlockingStrategy::Fixed)?;
    let variable = encode_source(&pcm, &variable_blocks(total), BlockingStrategy::Variable)?;

    let fixed_frames = output_frames(&fixed)?;
    let variable_frames = output_frames(&variable)?;
    assert!(
        fixed_frames
            .iter()
            .all(|frame| { frame.header.strategy == BlockingStrategy::Fixed })
    );
    assert!(
        variable_frames
            .iter()
            .all(|frame| { frame.header.strategy == BlockingStrategy::Variable })
    );
    assert_eq!(fixed_frames.last().unwrap().sample_end(), total as u64);
    assert_eq!(variable_frames.last().unwrap().sample_end(), total as u64);
    assert_eq!(decode_flac(&fixed)?.samples, pcm);
    assert_eq!(decode_flac(&variable)?.samples, pcm);
    Ok(())
}

#[test]
fn non_aligned_trim_matches_independent_reference_at_three_ranges() -> Result<()> {
    let total = 5_000usize;
    let pcm = make_pcm(total);
    let start = 137u64;
    let end = 4_683u64;

    let sources = [
        encode_source(&pcm, &fixed_blocks(total, 512), BlockingStrategy::Fixed)?,
        encode_source(&pcm, &variable_blocks(total), BlockingStrategy::Variable)?,
    ];

    for source in sources {
        let (trimmed, report) = trim_flac(&source, &TrimOptions::new(start, Some(end)))?;
        assert_eq!(report.output_samples, end - start);
        assert!(report.reused_frames > 0, "interior frames should be reused");
        assert!(
            report.reencoded_frames >= 1,
            "partial boundary needs re-encoding"
        );
        assert_eq!(
            report.reused_frames + report.reencoded_frames,
            report.output_frames
        );

        let decoded = decode_flac(&trimmed)?;
        let expected_start = start as usize * usize::from(CHANNELS);
        let expected_end = end as usize * usize::from(CHANNELS);
        assert_eq!(decoded.samples, pcm[expected_start..expected_end]);

        // Build an independent reference stream with a different block layout.
        let reference = encode_source(
            &decoded.samples,
            &variable_blocks(report.output_samples as usize),
            BlockingStrategy::Variable,
        )?;
        let cross_seek = report.seek_points[report.seek_points.len() / 2].sample;
        let ranges = [
            (0, 127),
            (
                cross_seek.saturating_sub(41),
                (cross_seek + 73).min(report.output_samples),
            ),
            (report.output_samples - 91, report.output_samples),
        ];
        let verification = verify_flac(&reference, &trimmed, &ranges)?;
        assert!(verification.is_match(), "{verification:?}");
        assert_eq!(verification.ranges.len(), 3);
        assert_eq!(verification.ranges[1].samples_compared, 114);

        let parsed = ParsedFlac::parse(trimmed.clone())?;
        assert_eq!(parsed.stream_info.total_samples, report.output_samples);
        assert_eq!(
            parsed.stream_info.md5,
            decoded.pcm_md5(0, report.output_samples)?
        );
        let frames = output_frames(&trimmed)?;
        for point in &report.seek_points {
            assert!(frames.iter().any(|frame| {
                frame.sample_start == point.sample
                    && frame.offset as u64 == parsed.audio_offset as u64 + point.offset
                    && frame.header.block_size as u16 == point.frame_samples
            }));
        }
    }
    Ok(())
}

#[test]
fn aligned_trim_reuses_every_interior_frame_payload() -> Result<()> {
    let total = 5_000usize;
    let pcm = make_pcm(total);
    let fixed = encode_source(&pcm, &fixed_blocks(total, 512), BlockingStrategy::Fixed)?;
    let variable_blocks = variable_blocks(total);
    let variable = encode_source(&pcm, &variable_blocks, BlockingStrategy::Variable)?;

    let cases = [
        (
            fixed,
            BlockingStrategy::Fixed,
            1_024u64,
            3_072u64,
            2usize,
            4usize,
        ),
        (
            variable,
            BlockingStrategy::Variable,
            variable_blocks[..2].iter().sum::<usize>() as u64,
            variable_blocks[..5].iter().sum::<usize>() as u64,
            2,
            3,
        ),
    ];

    for (source, strategy, start, end, first_frame, frame_count) in cases {
        let source_frames = output_frames(&source)?;
        let (trimmed, report) = trim_flac(&source, &TrimOptions::new(start, Some(end)))?;
        assert_eq!(report.reused_frames, frame_count);
        assert_eq!(report.reencoded_frames, 0);
        assert_eq!(
            report.blocking_strategy,
            match strategy {
                BlockingStrategy::Fixed => "fixed",
                BlockingStrategy::Variable => "variable",
            }
        );

        let trimmed_frames = output_frames(&trimmed)?;
        assert_eq!(trimmed_frames.len(), frame_count);
        for (output_index, output_frame) in trimmed_frames.iter().enumerate() {
            let source_frame = &source_frames[first_frame + output_index];
            let source_payload =
                &source[source_frame.offset + source_frame.header.header_len..source_frame.end - 2];
            let output_payload = &trimmed
                [output_frame.offset + output_frame.header.header_len..output_frame.end - 2];
            assert_eq!(source_payload, output_payload);
        }

        let decoded = decode_flac(&trimmed)?;
        let expected_start = start as usize * usize::from(CHANNELS);
        let expected_end = end as usize * usize::from(CHANNELS);
        assert_eq!(decoded.samples, pcm[expected_start..expected_end]);
        assert_eq!(decoded.total_samples(), end - start);
    }
    Ok(())
}

#[test]
fn tiny_non_aligned_range_keeps_a_decodable_streaminfo() -> Result<()> {
    let total = 1_000usize;
    let pcm = make_pcm(total);
    let source = encode_source(&pcm, &fixed_blocks(total, 256), BlockingStrategy::Fixed)?;
    let (trimmed, report) = trim_flac(&source, &TrimOptions::new(101, Some(108)))?;
    assert_eq!(report.output_samples, 7);
    assert_eq!(report.reencoded_frames, 1);
    let decoded = decode_flac(&trimmed)?;
    let expected = &pcm[101 * usize::from(CHANNELS)..108 * usize::from(CHANNELS)];
    assert_eq!(decoded.samples, expected);
    let parsed = ParsedFlac::parse(trimmed)?;
    assert_eq!(parsed.stream_info.total_samples, 7);
    assert_eq!(parsed.stream_info.min_block_size, 16);
    Ok(())
}
