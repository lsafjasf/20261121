//! `riaa_eq`: dependency-free RIAA equalization rendering library.

pub mod cli;
pub mod engine;
pub mod filter;
pub mod report;
pub mod response;
pub mod riaa;
pub mod verify;
pub mod wav;

pub use engine::{CurveKind, Representation};
pub use riaa::{guard_corner_hz, guarded_inverse, playback, riaa_corners_hz};
