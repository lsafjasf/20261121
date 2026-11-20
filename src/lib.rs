//! Sample-accurate FLAC trimming with encoded-frame reuse.
//!
//! The public API is intentionally small:
//! [`trim_flac`] reads a source file and writes a new, independently
//! decodable FLAC stream; [`verify_flac`] compares the decoded PCM of two
//! streams, optionally over a sample range.

mod bitio;
mod crc;
mod decode;
mod encode;
mod frame;
mod metadata;
mod trim;
mod verify;

pub use decode::{DecodedAudio, decode_flac, decode_flac_file};
pub use metadata::{MetadataBlock, StreamInfo};
pub use trim::{
    FlacSummary, SeekPointInfo, TrimOptions, TrimReport, inspect_flac, trim_flac, trim_flac_file,
};
pub use verify::{RangeVerification, VerificationReport, verify_flac, verify_flac_file};

#[cfg(test)]
mod tests;
