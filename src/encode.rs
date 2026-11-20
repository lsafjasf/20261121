use anyhow::{Result, bail};

use crate::bitio::BitWriter;
use crate::crc::crc16;
use crate::frame::{BlockingStrategy, Frame, build_header};

pub(crate) fn encode_verbatim_frame(
    pcm: &[i32],
    channels: u8,
    bits_per_sample: u8,
    strategy: BlockingStrategy,
    number: u64,
) -> Result<Vec<u8>> {
    if channels == 0 || channels > 8 {
        bail!("verbatim frame encoder supports 1 through 8 channels");
    }
    if !(4..=32).contains(&bits_per_sample) {
        bail!("verbatim frame encoder does not support {bits_per_sample}-bit samples");
    }
    let channels_usize = usize::from(channels);
    if pcm.len() % channels_usize != 0 {
        bail!("PCM buffer contains a partial inter-channel sample");
    }
    let block_size = pcm.len() / channels_usize;
    if block_size == 0 || block_size > 65_535 {
        bail!("invalid verbatim block size {block_size}");
    }

    let channel_assignment = channels - 1;
    let header = build_header(
        strategy,
        block_size as u32,
        channel_assignment,
        number,
        bits_per_sample,
    )?;
    let estimated = header.len() + pcm.len() * usize::from(bits_per_sample) / 8 + 2;
    let mut writer = BitWriter::with_capacity(estimated);
    writer.write_bytes(&header);

    for channel in 0..channels_usize {
        // Zero padding bit, subframe type 1 (verbatim), no wasted bits.
        writer.write_u64(0b0000_0010, 8);
        for sample in pcm.chunks_exact(channels_usize) {
            writer.write_signed(i64::from(sample[channel]), u32::from(bits_per_sample));
        }
    }

    let mut frame = writer.finish();
    let crc = crc16(&frame);
    frame.extend_from_slice(&crc.to_be_bytes());
    Ok(frame)
}

pub(crate) fn rebuild_reused_frame(
    source: &[u8],
    frame: &Frame,
    strategy: BlockingStrategy,
    number: u64,
    bits_per_sample: u8,
) -> Result<Vec<u8>> {
    let header = build_header(
        strategy,
        frame.header.block_size,
        frame.header.channel_assignment,
        number,
        bits_per_sample,
    )?;
    let payload = source
        .get(frame.offset + frame.header.header_len..frame.end - 2)
        .ok_or_else(|| anyhow::anyhow!("invalid source frame bounds"))?;
    let mut rebuilt = Vec::with_capacity(header.len() + payload.len() + 2);
    rebuilt.extend_from_slice(&header);
    rebuilt.extend_from_slice(payload);
    let crc = crc16(&rebuilt);
    rebuilt.extend_from_slice(&crc.to_be_bytes());
    Ok(rebuilt)
}
