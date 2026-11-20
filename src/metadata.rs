use anyhow::{Context, Result, bail};

pub const STREAMINFO_BLOCK: u8 = 0;
pub const PADDING_BLOCK: u8 = 1;
pub const SEEKTABLE_BLOCK: u8 = 3;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamInfo {
    pub min_block_size: u16,
    pub max_block_size: u16,
    pub min_frame_size: u32,
    pub max_frame_size: u32,
    pub sample_rate: u32,
    pub channels: u8,
    pub bits_per_sample: u8,
    pub total_samples: u64,
    pub md5: [u8; 16],
}

impl StreamInfo {
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() != 34 {
            bail!("STREAMINFO must be 34 bytes, got {}", data.len());
        }

        let packed = ((data[10] as u64) << 56)
            | ((data[11] as u64) << 48)
            | ((data[12] as u64) << 40)
            | ((data[13] as u64) << 32)
            | ((data[14] as u64) << 24)
            | ((data[15] as u64) << 16)
            | ((data[16] as u64) << 8)
            | data[17] as u64;

        let sample_rate = (packed >> 44) as u32;
        let channels = (((packed >> 41) & 0x7) + 1) as u8;
        let bits_per_sample = (((packed >> 36) & 0x1f) + 1) as u8;
        let total_samples = packed & 0x0f_ff_ff_ff_ff;

        let mut md5 = [0u8; 16];
        md5.copy_from_slice(&data[18..34]);

        let info = Self {
            min_block_size: u16::from_be_bytes([data[0], data[1]]),
            max_block_size: u16::from_be_bytes([data[2], data[3]]),
            min_frame_size: u32::from_be_bytes([0, data[4], data[5], data[6]]),
            max_frame_size: u32::from_be_bytes([0, data[7], data[8], data[9]]),
            sample_rate,
            channels,
            bits_per_sample,
            total_samples,
            md5,
        };
        info.validate()?;
        Ok(info)
    }

    pub fn to_bytes(&self) -> Result<[u8; 34]> {
        self.validate()?;
        if self.total_samples >= (1u64 << 36) {
            bail!("total sample count does not fit in STREAMINFO");
        }
        if self.sample_rate >= (1u32 << 20) {
            bail!("sample rate does not fit in STREAMINFO");
        }
        if self.min_frame_size >= (1u32 << 24) || self.max_frame_size >= (1u32 << 24) {
            bail!("frame size does not fit in STREAMINFO");
        }

        let mut out = [0u8; 34];
        out[0..2].copy_from_slice(&self.min_block_size.to_be_bytes());
        out[2..4].copy_from_slice(&self.max_block_size.to_be_bytes());
        out[4..7].copy_from_slice(&self.min_frame_size.to_be_bytes()[1..]);
        out[7..10].copy_from_slice(&self.max_frame_size.to_be_bytes()[1..]);

        let packed = ((self.sample_rate as u64) << 44)
            | ((u64::from(self.channels - 1) & 0x7) << 41)
            | ((u64::from(self.bits_per_sample - 1) & 0x1f) << 36)
            | (self.total_samples & 0x0f_ff_ff_ff_ff);
        out[10..18].copy_from_slice(&packed.to_be_bytes());
        out[18..34].copy_from_slice(&self.md5);
        Ok(out)
    }

    pub fn fixed_block_size(&self) -> Option<u32> {
        if self.min_block_size == self.max_block_size {
            Some(u32::from(self.max_block_size))
        } else {
            None
        }
    }

    fn validate(&self) -> Result<()> {
        if self.min_block_size < 16 {
            bail!("minimum block size {} is below 16", self.min_block_size);
        }
        if self.max_block_size < self.min_block_size {
            bail!(
                "maximum block size {} is below minimum {}",
                self.max_block_size,
                self.min_block_size
            );
        }
        if self.sample_rate == 0 {
            bail!("sample rate must not be zero");
        }
        if !(1..=8).contains(&self.channels) {
            bail!("unsupported channel count {}", self.channels);
        }
        if !(4..=32).contains(&self.bits_per_sample) {
            bail!("unsupported bits per sample {}", self.bits_per_sample);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetadataBlock {
    pub block_type: u8,
    pub data: Vec<u8>,
}

pub(crate) struct ParsedFlac {
    pub data: Vec<u8>,
    pub stream_info: StreamInfo,
    pub metadata: Vec<MetadataBlock>,
    pub audio_offset: usize,
}

impl ParsedFlac {
    pub fn parse(data: Vec<u8>) -> Result<Self> {
        let flac_start = id3v2_size(&data).unwrap_or(0);
        if data.get(flac_start..flac_start + 4) != Some(b"fLaC") {
            bail!("input is not a native FLAC stream");
        }

        let mut pos = flac_start + 4;
        let mut stream_info = None;
        let mut metadata = Vec::new();
        let mut saw_last = false;
        let mut first = true;

        while !saw_last {
            let header = data
                .get(pos..pos + 4)
                .ok_or_else(|| anyhow::anyhow!("truncated metadata block header"))?;
            let is_last = header[0] & 0x80 != 0;
            let block_type = header[0] & 0x7f;
            let length =
                (usize::from(header[1]) << 16) | (usize::from(header[2]) << 8) | header[3] as usize;
            let body_start = pos + 4;
            let body_end = body_start
                .checked_add(length)
                .ok_or_else(|| anyhow::anyhow!("metadata length overflow"))?;
            let body = data
                .get(body_start..body_end)
                .ok_or_else(|| anyhow::anyhow!("truncated metadata block"))?;

            if first && block_type != STREAMINFO_BLOCK {
                bail!("the first metadata block is not STREAMINFO");
            }
            if block_type == STREAMINFO_BLOCK {
                if stream_info.is_some() {
                    bail!("duplicate STREAMINFO block");
                }
                stream_info = Some(StreamInfo::parse(body).context("invalid STREAMINFO")?);
            } else {
                metadata.push(MetadataBlock {
                    block_type,
                    data: body.to_vec(),
                });
            }

            first = false;
            saw_last = is_last;
            pos = body_end;
        }

        let stream_info = stream_info.ok_or_else(|| anyhow::anyhow!("missing STREAMINFO"))?;
        Ok(Self {
            data,
            stream_info,
            metadata,
            audio_offset: pos,
        })
    }
}

pub(crate) fn build_flac(
    stream_info: &StreamInfo,
    seek_table: &[u8],
    metadata: &[MetadataBlock],
    audio: &[u8],
) -> Result<Vec<u8>> {
    let mut blocks = Vec::with_capacity(metadata.len() + 2);
    blocks.push(MetadataBlock {
        block_type: STREAMINFO_BLOCK,
        data: stream_info.to_bytes()?.to_vec(),
    });
    blocks.push(MetadataBlock {
        block_type: SEEKTABLE_BLOCK,
        data: seek_table.to_vec(),
    });
    for block in metadata {
        if matches!(
            block.block_type,
            STREAMINFO_BLOCK | SEEKTABLE_BLOCK | PADDING_BLOCK
        ) {
            continue;
        }
        blocks.push(block.clone());
    }

    let metadata_len: usize = blocks.iter().map(|b| b.data.len() + 4).sum();
    let mut out = Vec::with_capacity(4 + metadata_len + audio.len());
    out.extend_from_slice(b"fLaC");
    for (index, block) in blocks.iter().enumerate() {
        if block.data.len() >= (1 << 24) {
            bail!("metadata block is too large");
        }
        let last = if index + 1 == blocks.len() { 0x80 } else { 0 };
        out.push(last | (block.block_type & 0x7f));
        out.extend_from_slice(&(block.data.len() as u32).to_be_bytes()[1..]);
        out.extend_from_slice(&block.data);
    }
    out.extend_from_slice(audio);
    Ok(out)
}

fn id3v2_size(data: &[u8]) -> Option<usize> {
    if data.get(0..3) != Some(b"ID3") || data.len() < 10 {
        return None;
    }
    let size = ((data[6] & 0x7f) as usize) << 21
        | ((data[7] & 0x7f) as usize) << 14
        | ((data[8] & 0x7f) as usize) << 7
        | (data[9] & 0x7f) as usize;
    let footer = if data[5] & 0x10 != 0 { 10 } else { 0 };
    Some(10 + size + footer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaminfo_round_trip_preserves_all_fields() {
        let info = StreamInfo {
            min_block_size: 256,
            max_block_size: 4096,
            min_frame_size: 12,
            max_frame_size: 0x12_34_56,
            sample_rate: 48_000,
            channels: 6,
            bits_per_sample: 24,
            total_samples: 0x1_2345_6789,
            md5: [0x5a; 16],
        };
        let encoded = info.to_bytes().unwrap();
        assert_eq!(StreamInfo::parse(&encoded).unwrap(), info);
    }
}
