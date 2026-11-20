//! Hand-rolled command line parsing (no external crates).

use crate::engine::Representation;
use crate::wav::SampleFormat;

/// Parsed command line: one subcommand plus its options.
#[derive(Debug)]
pub enum Command {
    Analyze(AnalyzeArgs),
    Render(RenderArgs),
    Transient(TransientArgs),
    Verify(VerifyArgs),
}

#[derive(Debug, Clone)]
pub struct AnalyzeArgs {
    pub sample_rate: f64,
    pub guard_fraction: f64,
    pub json: bool,
}

#[derive(Debug, Clone)]
pub struct RenderArgs {
    pub input: String,
    pub output: String,
    pub input_repr: Representation,
    pub output_repr: Representation,
    pub output_format: SampleFormat,
    pub guard_fraction: f64,
    pub flush: bool,
}

#[derive(Debug, Clone)]
pub struct TransientArgs {
    pub output: String,
    pub kind: TransientKind,
    pub sample_rate: f64,
    pub duration_ms: u32,
    pub channels: u16,
    pub amplitude: f64,
    pub tone_hz: f64,
    pub output_format: SampleFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransientKind {
    Impulse,
    ToneBurst,
}

#[derive(Debug, Clone)]
pub struct VerifyArgs {
    pub sample_rate: f64,
    pub guard_fraction: f64,
    pub json: bool,
}

pub const HELP: &str = "\
riaa-eq — archive-room RIAA equalization renderer

USAGE:
    riaa-eq <SUBCOMMAND> [OPTIONS]

SUBCOMMANDS:
    analyze     Print coefficients, poles/zeros, reference frequency response
    render      Equalize a WAV file with explicit input/output representation
    transient   Generate a short impulse or tone-burst WAV test signal
    verify      Validate playback -> inverse round trip (frequency + time)

COMMON OPTIONS:
    --sample-rate <HZ>       Sample rate for analysis/generation (default 48000)
    --guard <FRACTION>       Guard pole as fraction of sample rate (default 0.45)
    --json                   Emit machine-readable JSON instead of tables
    -h, --help               Show this help

RENDER OPTIONS:
    -i, --input <PATH>       Input WAV (pcm16/pcm24/pcm32/float32)
    -o, --output <PATH>      Output WAV path
    --from <flat|riaa>       Input representation (required)
    --to <flat|riaa>         Output representation (required)
    --format <FMT>           Output encoding: pcm16 (default), pcm24,
                             pcm32, float32
    --flush                  Append zero-input ring-out until state settles

TRANSIENT OPTIONS:
    -o, --output <PATH>      Output WAV path
    --kind <impulse|tone>    Signal kind (default impulse)
    --duration-ms <MS>       Duration (default 100)
    --channels <N>           Channel count (default 1)
    --amplitude <A>          Peak amplitude (default 1.0 impulse, 0.9 tone)
    --tone-hz <HZ>           Tone burst frequency (default 1000)
";

struct Parser {
    args: Vec<String>,
    pos: usize,
}

impl Parser {
    fn new(args: Vec<String>) -> Self {
        Self { args, pos: 0 }
    }

    fn flag(&mut self, name: &str) -> bool {
        if let Some(index) = self.args.iter().position(|arg| arg == name) {
            self.args.remove(index);
            true
        } else {
            false
        }
    }

    fn value(&mut self, name: &str) -> Result<Option<String>, String> {
        match self.args.iter().position(|arg| arg == name) {
            Some(index) => {
                if index + 1 >= self.args.len() {
                    return Err(format!("option {name} requires a value"));
                }
                let value = self.args.remove(index + 1);
                self.args.remove(index);
                Ok(Some(value))
            }
            None => Ok(None),
        }
    }

    fn require(&mut self, name: &str) -> Result<String, String> {
        self.value(name)?
            .ok_or_else(|| format!("missing required option {name}"))
    }

    fn remaining(&self) -> &[String] {
        &self.args[self.pos..]
    }
}

fn parse_f64(name: &str, text: &str) -> Result<f64, String> {
    text.parse()
        .map_err(|_| format!("option {name} expects a number, got '{text}'"))
}

fn parse_u32(name: &str, text: &str) -> Result<u32, String> {
    text.parse()
        .map_err(|_| format!("option {name} expects an integer, got '{text}'"))
}

fn parse_u16(name: &str, text: &str) -> Result<u16, String> {
    text.parse()
        .map_err(|_| format!("option {name} expects an integer, got '{text}'"))
}

fn parse_repr(name: &str, text: &str) -> Result<Representation, String> {
    Representation::parse(text)
        .ok_or_else(|| format!("option {name} expects 'flat' or 'riaa', got '{text}'"))
}

fn parse_format(text: &str) -> Result<SampleFormat, String> {
    SampleFormat::parse(text)
        .ok_or_else(|| format!("unknown sample format '{text}' (pcm16/pcm24/pcm32/float32)"))
}

/// Parse the process arguments (excluding the executable name).
pub fn parse(raw: Vec<String>) -> Result<Command, String> {
    if raw.is_empty() || raw.iter().any(|arg| arg == "-h" || arg == "--help") {
        return Err(HELP.to_string());
    }
    let subcommand = raw[0].clone();
    let mut parser = Parser::new(raw[1..].to_vec());
    match subcommand.as_str() {
        "analyze" => {
            let sample_rate = parser
                .value("--sample-rate")?
                .map(|text| parse_f64("--sample-rate", &text))
                .transpose()?
                .unwrap_or(48_000.0);
            let guard_fraction = parser
                .value("--guard")?
                .map(|text| parse_f64("--guard", &text))
                .transpose()?
                .unwrap_or(0.45);
            let json = parser.flag("--json");
            reject_leftovers(parser.remaining())?;
            Ok(Command::Analyze(AnalyzeArgs {
                sample_rate,
                guard_fraction,
                json,
            }))
        }
        "render" => {
            let input = parser
                .require("--input")
                .or_else(|_| parser.require("-i"))?;
            let output = parser
                .require("--output")
                .or_else(|_| parser.require("-o"))?;
            let input_repr = parse_repr("--from", &parser.require("--from")?)?;
            let output_repr = parse_repr("--to", &parser.require("--to")?)?;
            let output_format = parser
                .value("--format")?
                .map(|text| parse_format(&text))
                .transpose()?
                .unwrap_or(SampleFormat::Pcm16);
            let guard_fraction = parser
                .value("--guard")?
                .map(|text| parse_f64("--guard", &text))
                .transpose()?
                .unwrap_or(0.45);
            let flush = parser.flag("--flush");
            reject_leftovers(parser.remaining())?;
            Ok(Command::Render(RenderArgs {
                input,
                output,
                input_repr,
                output_repr,
                output_format,
                guard_fraction,
                flush,
            }))
        }
        "transient" => {
            let output = parser
                .require("--output")
                .or_else(|_| parser.require("-o"))?;
            let kind = match parser.value("--kind")?.as_deref() {
                None | Some("impulse") => TransientKind::Impulse,
                Some("tone") | Some("tone-burst") => TransientKind::ToneBurst,
                Some(other) => return Err(format!("unknown transient kind '{other}'")),
            };
            let sample_rate = parser
                .value("--sample-rate")?
                .map(|text| parse_f64("--sample-rate", &text))
                .transpose()?
                .unwrap_or(48_000.0);
            let duration_ms = parser
                .value("--duration-ms")?
                .map(|text| parse_u32("--duration-ms", &text))
                .transpose()?
                .unwrap_or(100);
            let channels = parser
                .value("--channels")?
                .map(|text| parse_u16("--channels", &text))
                .transpose()?
                .unwrap_or(1);
            let default_amp = if kind == TransientKind::ToneBurst {
                0.9
            } else {
                1.0
            };
            let amplitude = parser
                .value("--amplitude")?
                .map(|text| parse_f64("--amplitude", &text))
                .transpose()?
                .unwrap_or(default_amp);
            let tone_hz = parser
                .value("--tone-hz")?
                .map(|text| parse_f64("--tone-hz", &text))
                .transpose()?
                .unwrap_or(1_000.0);
            let output_format = parser
                .value("--format")?
                .map(|text| parse_format(&text))
                .transpose()?
                .unwrap_or(SampleFormat::Float32);
            reject_leftovers(parser.remaining())?;
            Ok(Command::Transient(TransientArgs {
                output,
                kind,
                sample_rate,
                duration_ms,
                channels,
                amplitude,
                tone_hz,
                output_format,
            }))
        }
        "verify" => {
            let sample_rate = parser
                .value("--sample-rate")?
                .map(|text| parse_f64("--sample-rate", &text))
                .transpose()?
                .unwrap_or(48_000.0);
            let guard_fraction = parser
                .value("--guard")?
                .map(|text| parse_f64("--guard", &text))
                .transpose()?
                .unwrap_or(0.45);
            let json = parser.flag("--json");
            reject_leftovers(parser.remaining())?;
            Ok(Command::Verify(VerifyArgs {
                sample_rate,
                guard_fraction,
                json,
            }))
        }
        other => Err(format!("unknown subcommand '{other}'\n\n{HELP}")),
    }
}

fn reject_leftovers(leftovers: &[String]) -> Result<(), String> {
    if leftovers.is_empty() {
        Ok(())
    } else {
        Err(format!("unrecognized arguments: {}", leftovers.join(" ")))
    }
}
