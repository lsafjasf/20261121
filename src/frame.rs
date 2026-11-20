use anyhow::{Context, Result, bail};

use crate::crc::{crc8, crc16};
use crate::metadata::StreamInfo;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BlockingStrategy {
    Fixed,
    Variable,
}

#[derive(Clone, Debug)]
pub(crate) struct FrameHeader {
    pub strategy: BlockingStrategy,
    pub block_size: u32,
    pub channel_assignment: u8,
    pub number: u64,
    pub header_len: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct Frame {
    pub offset: usize,
    pub end: usize,
    pub header: FrameHeader,
    pub sample_start: u64,
}

impl Frame {
    pub fn sample_count(&self) -> u64 {
        u64::from(self.header.block_size)
    }

    pub fn sample_end(&self) -> u64 {
        self.sample_start + self.sample_count()
    }
}

pub(crate) fn parse_frames(
    data: &[u8],
    audio_offset: usize,
    stream_info: &StreamInfo,
) -> Result<Vec<Frame>> {
    if audio_offset >= data.len() {
        bail!("FLAC stream contains no audio frames");
    }

    let mut frames = Vec::new();
    let mut offset = audio_offset;
    let mut cumulative = 0u64;
    while offset < data.len() {
        let header = parse_header(data, offset)
            .with_context(|| format!("invalid FLAC frame header at byte {offset}"))?;
        let end = find_frame_end(data, offset, header.header_len)
            .with_context(|| format!("could not determine the end of frame at byte {offset}"))?;
        let expected = match header.strategy {
            BlockingStrategy::Fixed => header
                .number
                .checked_mul(u64::from(stream_info.max_block_size))
                .ok_or_else(|| anyhow::anyhow!("fixed-block sample position overflow"))?,
            BlockingStrategy::Variable => header.number,
        };
        if expected != cumulative {
            bail!(
                "frame position mismatch at byte {offset}: header says sample {expected}, \
                 preceding frames end at sample {cumulative}"
            );
        }
        cumulative = cumulative
            .checked_add(u64::from(header.block_size))
            .ok_or_else(|| anyhow::anyhow!("sample position overflow"))?;
        frames.push(Frame {
            offset,
            end,
            header,
            sample_start: expected,
        });
        offset = end;
    }

    if frames.is_empty() {
        bail!("FLAC stream contains no audio frames");
    }
    Ok(frames)
}

pub(crate) fn parse_header(data: &[u8], offset: usize) -> Result<FrameHeader> {
    let fixed = data
        .get(offset..offset + 4)
        .ok_or_else(|| anyhow::anyhow!("truncated frame header"))?;
    if fixed[0] != 0xff || fixed[1] & 0xfe != 0xf8 {
        bail!("missing FLAC frame sync code");
    }
    if fixed[1] & 0x02 != 0 {
        bail!("reserved frame-header bit is set");
    }
    if fixed[3] & 0x01 != 0 {
        bail!("reserved frame-header bit is set");
    }

    let strategy = if fixed[1] & 0x01 == 0 {
        BlockingStrategy::Fixed
    } else {
        BlockingStrategy::Variable
    };
    let block_code = fixed[2] >> 4;
    let sample_rate_code = fixed[2] & 0x0f;
    let channel_assignment = fixed[3] >> 4;
    let sample_size_code = (fixed[3] >> 1) & 0x07;

    if block_code == 0 {
        bail!("reserved block-size code");
    }
    if sample_rate_code == 15 {
        bail!("reserved sample-rate code");
    }
    if channel_assignment > 10 {
        bail!("reserved channel assignment");
    }
    if sample_size_code == 3 {
        bail!("reserved sample-size code");
    }

    let (number, number_len) = parse_utf8_number(data, offset + 4)?;
    let mut pos = offset + 4 + number_len;
    let block_size = match block_code {
        1 => 192,
        2 => 576,
        3 => 1152,
        4 => 2304,
        5 => 4608,
        6 => {
            let value = *data
                .get(pos)
                .ok_or_else(|| anyhow::anyhow!("truncated 8-bit block size"))?;
            pos += 1;
            u32::from(value) + 1
        }
        7 => {
            let bytes = data
                .get(pos..pos + 2)
                .ok_or_else(|| anyhow::anyhow!("truncated 16-bit block size"))?;
            pos += 2;
            u32::from(u16::from_be_bytes([bytes[0], bytes[1]])) + 1
        }
        8 => 256,
        9 => 512,
        10 => 1024,
        11 => 2048,
        12 => 4096,
        13 => 8192,
        14 => 16384,
        15 => 32768,
        _ => unreachable!(),
    };

    match sample_rate_code {
        12 => {
            data.get(pos)
                .ok_or_else(|| anyhow::anyhow!("truncated sample rate"))?;
            pos += 1;
        }
        13 | 14 => {
            data.get(pos..pos + 2)
                .ok_or_else(|| anyhow::anyhow!("truncated sample rate"))?;
            pos += 2;
        }
        _ => {}
    }

    let expected_crc = *data
        .get(pos)
        .ok_or_else(|| anyhow::anyhow!("missing frame-header CRC"))?;
    if crc8(&data[offset..pos]) != expected_crc {
        bail!("frame-header CRC mismatch");
    }
    pos += 1;

    if block_size == 0 || block_size > 65_535 {
        bail!("invalid frame block size {block_size}");
    }

    Ok(FrameHeader {
        strategy,
        block_size,
        channel_assignment,
        number,
        header_len: pos - offset,
    })
}

fn find_frame_end(data: &[u8], offset: usize, header_len: usize) -> Result<usize> {
    let mut candidate = offset + header_len;
    while candidate + 2 <= data.len() {
        if data[candidate] == 0xff
            && data[candidate + 1] & 0xfe == 0xf8
            && parse_header(data, candidate).is_ok()
        {
            let crc_position = candidate - 2;
            let stored = u16::from_be_bytes([data[crc_position], data[crc_position + 1]]);
            if crc16(&data[offset..crc_position]) == stored {
                return Ok(candidate);
            }
        }
        candidate += 1;
    }

    if data.len() < offset + header_len + 2 {
        bail!("truncated final frame");
    }
    let crc_position = data.len() - 2;
    let stored = u16::from_be_bytes([data[crc_position], data[crc_position + 1]]);
    if crc16(&data[offset..crc_position]) != stored {
        bail!("final frame CRC mismatch");
    }
    Ok(data.len())
}

pub(crate) fn parse_utf8_number(data: &[u8], offset: usize) -> Result<(u64, usize)> {
    let first = *data
        .get(offset)
        .ok_or_else(|| anyhow::anyhow!("missing frame/sample number"))?;
    let (len, mask) = match first {
        0x00..=0x7f => return Ok((u64::from(first), 1)),
        0xc0..=0xdf => (2, 0x1f),
        0xe0..=0xef => (3, 0x0f),
        0xf0..=0xf7 => (4, 0x07),
        0xf8..=0xfb => (5, 0x03),
        0xfc..=0xfd => (6, 0x01),
        0xfe => (7, 0x00),
        _ => bail!("invalid UTF-8-like frame number"),
    };
    let mut value = u64::from(first & mask);
    for index in 1..len {
        let byte = *data
            .get(offset + index)
            .ok_or_else(|| anyhow::anyhow!("truncated frame/sample number"))?;
        if byte & 0xc0 != 0x80 {
            bail!("invalid UTF-8-like continuation byte");
        }
        value = (value << 6) | u64::from(byte & 0x3f);
    }
    let minimum = [
        0,
        0,
        0x80,
        0x800,
        0x1_0000,
        0x20_0000,
        0x400_0000,
        0x8000_0000,
    ][len];
    if value < minimum {
        bail!("overlong UTF-8-like frame number");
    }
    Ok((value, len))
}

pub(crate) fn encode_utf8_number(value: u64) -> Result<Vec<u8>> {
    let (len, prefix) = if value < 0x80 {
        return Ok(vec![value as u8]);
    } else if value < 0x800 {
        (2, 0xc0)
    } else if value < 0x1_0000 {
        (3, 0xe0)
    } else if value < 0x20_0000 {
        (4, 0xf0)
    } else if value < 0x400_0000 {
        (5, 0xf8)
    } else if value < 0x8000_0000 {
        (6, 0xfc)
    } else if value < 0x10_0000_0000 {
        (7, 0xfe)
    } else {
        bail!("frame/sample number does not fit FLAC's 36-bit field");
    };

    let payload_bits = [0, 0, 5, 4, 3, 2, 1, 0][len];
    let mut out = vec![0u8; len];
    out[0] = prefix | ((value >> (6 * (len - 1))) as u8 & ((1 << payload_bits) - 1));
    for index in 1..len {
        out[index] = 0x80 | ((value >> (6 * (len - 1 - index))) as u8 & 0x3f);
    }
    Ok(out)
}

pub(crate) fn build_header(
    strategy: BlockingStrategy,
    block_size: u32,
    channel_assignment: u8,
    number: u64,
    bits_per_sample: u8,
) -> Result<Vec<u8>> {
    if !(1..=65_535).contains(&block_size) {
        bail!("block size {block_size} cannot be encoded in a FLAC frame");
    }
    if channel_assignment > 10 {
        bail!("invalid channel assignment {channel_assignment}");
    }
    let sample_size_code = match bits_per_sample {
        8 => 1,
        12 => 2,
        16 => 4,
        20 => 5,
        24 => 6,
        32 => 7,
        _ => bail!("{bits_per_sample}-bit samples cannot be described in a FLAC frame header"),
    };

    let (block_code, block_extra) = match block_size {
        192 => (1, Vec::new()),
        576 => (2, Vec::new()),
        1152 => (3, Vec::new()),
        2304 => (4, Vec::new()),
        4608 => (5, Vec::new()),
        256 => (8, Vec::new()),
        512 => (9, Vec::new()),
        1024 => (10, Vec::new()),
        2048 => (11, Vec::new()),
        4096 => (12, Vec::new()),
        8192 => (13, Vec::new()),
        16384 => (14, Vec::new()),
        32768 => (15, Vec::new()),
        1..=256 => (6, vec![(block_size - 1) as u8]),
        _ => (7, ((block_size - 1) as u16).to_be_bytes().to_vec()),
    };
    let strategy_bit = match strategy {
        BlockingStrategy::Fixed => 0,
        BlockingStrategy::Variable => 1,
    };

    let mut header = Vec::with_capacity(16);
    header.push(0xff);
    header.push(0xf8 | strategy_bit);
    header.push((block_code << 4) | 0);
    header.push((channel_assignment << 4) | (sample_size_code << 1));
    header.extend_from_slice(&encode_utf8_number(number)?);
    header.extend_from_slice(&block_extra);
    header.push(crc8(&header));
    Ok(header)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_numbers_round_trip_at_boundaries() {
        for value in [
            0,
            0x7f,
            0x80,
            0x7ff,
            0x800,
            0xffff,
            0x1_0000,
            0x1f_ffff,
            0x20_0000,
            0x3ff_ffff,
            0x400_0000,
            0x7fff_ffff,
            0x8000_0000,
            0xf_ffff_ffff,
        ] {
            let encoded = encode_utf8_number(value).unwrap();
            let (decoded, len) = parse_utf8_number(&encoded, 0).unwrap();
            assert_eq!(decoded, value);
            assert_eq!(len, encoded.len());
        }
    }

    #[test]
    fn generated_header_has_valid_crc() {
        let header = build_header(BlockingStrategy::Variable, 333, 1, 12_345, 16).unwrap();
        let parsed = parse_header(&header, 0).unwrap();
        assert_eq!(parsed.block_size, 333);
        assert_eq!(parsed.number, 12_345);
        assert_eq!(parsed.channel_assignment, 1);
    }
}
