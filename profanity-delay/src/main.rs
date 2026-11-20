mod delay;
mod demo;
mod live;
mod png;
mod signal;
mod timeline;
mod wav;

use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("demo");
    let out_dir = args
        .get(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("out"));

    let result = match mode {
        "demo" => demo::run(&out_dir).map(|r| r.ok),
        "live" => live::run(&out_dir).map(|()| true),
        _ => {
            eprintln!(
                "usage: profanity-delay [demo|live] [out_dir]\n\
                 \n\
                 demo  scripted loopback: 7 s delay, two dumped markers,\n\
                 \x20       silence-gated rebuild, WAV/CSV/SVG proof artifacts\n\
                 live  interactive loopback: press [d] to dump, [q] to quit"
            );
            return ExitCode::from(2);
        }
    };

    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
