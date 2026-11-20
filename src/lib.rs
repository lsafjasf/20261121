//! Coded audio round-trip delay and jitter measurement.
//!
//! The signal is a repeated, deterministic broadband code.  Detection uses a
//! normalized matched filter over the complete unambiguous delay range.  The
//! tracker then applies acquisition confirmation, a local gate, robust history,
//! and explicit loss of lock.  A single bad correlation peak can therefore
//! never become the permanent tracking reference.

pub mod code;
pub mod loopback;
pub mod matcher;
pub mod stream;
pub mod tracker;

pub use code::{Probe, ProbeConfig};
pub use loopback::{
    ChunkPattern, FalseBurst, LoopbackConfig, LoopbackRender, SoftwareLoopback, render_capture,
};
pub use matcher::{Candidate, Detection, MatchWindow, Matcher, MatcherConfig};
pub use stream::MeasurementSession;
pub use tracker::{FrameObservation, Tracker, TrackerConfig, TrackingState};
