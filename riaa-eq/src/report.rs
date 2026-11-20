//! Human-readable and JSON reports for analysis and verification.

use crate::engine::{fnv1a_64, EngineConfig};
use crate::riaa::{
    guard_corner_hz, guarded_inverse, playback, riaa_corners_hz, Coeffs, TAU1, TAU2, TAU3,
};
use crate::verify::{
    amplitude_roundtrip, frequency_table, roundtrip_time_domain, PROBE_FREQUENCIES_HZ,
};

fn canonical_spec(config: EngineConfig) -> String {
    format!(
        "riaa-eq|fs={}|guard={}|tau1={}|tau2={}|tau3={}|f64|blt",
        config.sample_rate as u64, config.guard_fraction, TAU1, TAU2, TAU3
    )
}

/// Reproducibility fingerprint for a coefficient set.
pub fn coefficient_fingerprint(config: EngineConfig, label: &str, coeffs: &Coeffs) -> u64 {
    let spec = format!(
        "{}|{}|b0={:.17e}|b1={:.17e}|b2={:.17e}|a1={:.17e}|a2={:.17e}",
        canonical_spec(config),
        label,
        coeffs.b0,
        coeffs.b1,
        coeffs.b2,
        coeffs.a1,
        coeffs.a2
    );
    fnv1a_64(&spec)
}

/// Full text analysis report.
pub fn print_analysis(config: EngineConfig) {
    let play = playback(config.sample_rate);
    let inverse = guarded_inverse(config.sample_rate, config.guard_fraction);
    let corners = riaa_corners_hz();
    println!("RIAA coefficient analysis");
    println!("=========================");
    println!("sample rate      : {} Hz", config.sample_rate as u64);
    println!("time constants   : tau1 = {TAU1} s, tau2 = {TAU2} s, tau3 = {TAU3} s");
    println!(
        "turnover f (tau) : {:.3} Hz, {:.3} Hz, {:.3} Hz",
        corners[0], corners[1], corners[2]
    );
    println!(
        "guard pole       : {:.1} Hz (fraction {:.3} of sample rate)",
        guard_corner_hz(config.sample_rate, config.guard_fraction),
        config.guard_fraction
    );
    println!("transform        : bilinear, s = 2*fs*(z-1)/(z+1)");
    println!();
    print_biquad("playback (riaa -> flat)", &play);
    print_biquad("guarded inverse (flat -> riaa)", &inverse);
    println!();
    println!("fingerprints (FNV-1a over canonical coefficient text):");
    println!(
        "  playback : {:016x}",
        coefficient_fingerprint(config, "playback", &play)
    );
    println!(
        "  inverse  : {:016x}",
        coefficient_fingerprint(config, "inverse", &inverse)
    );
    println!();
    print_frequency_table(config);
}

fn print_biquad(label: &str, coeffs: &Coeffs) {
    println!("{label} (a0 = 1.0):");
    println!("  b0 = {:+.17e}", coeffs.b0);
    println!("  b1 = {:+.17e}", coeffs.b1);
    println!("  b2 = {:+.17e}", coeffs.b2);
    println!("  a1 = {:+.17e}", coeffs.a1);
    println!("  a2 = {:+.17e}", coeffs.a2);
    let dc = coeffs.response_db(0.0, 1.0);
    println!("  DC gain check  : {dc:.6} dB");
}

fn print_frequency_table(config: EngineConfig) {
    println!("reference frequency response (absolute dB, DC gain = 0):");
    println!(
        "{:>10} {:>12} {:>11} {:>11} {:>11} {:>11}",
        "f_dig Hz", "f_ana Hz", "play dB", "inv dB", "round dB", "analog dB"
    );
    for row in frequency_table(config) {
        println!(
            "{:>10.2} {:>12.2} {:>11.4} {:>11.4} {:>11.4} {:>11.4}",
            row.digital_hz,
            row.warped_analog_hz,
            row.playback_db,
            row.inverse_db,
            row.roundtrip_db,
            row.analog_playback_db
        );
    }
    println!();
    println!("playback response referenced to 1 kHz (0 dB at 1 kHz):");
    println!(
        "{:>10} {:>14} {:>14}",
        "f_dig Hz", "digital dB", "analog dB"
    );
    for row in frequency_table(config) {
        println!(
            "{:>10.2} {:>14.4} {:>14.4}",
            row.digital_hz, row.playback_rel_1k_db, row.analog_rel_1k_db
        );
    }
}

/// JSON analysis report (one line per key, no external serializer).
pub fn print_analysis_json(config: EngineConfig) {
    let play = playback(config.sample_rate);
    let inverse = guarded_inverse(config.sample_rate, config.guard_fraction);
    println!("{{");
    println!("  \"sample_rate_hz\": {},", config.sample_rate);
    println!("  \"guard_fraction\": {},", config.guard_fraction);
    println!("  \"guard_corner_hz\": {:.6},", config.guard_hz());
    println!("  \"time_constants_s\": [{TAU1}, {TAU2}, {TAU3}],");
    println!("  \"playback\": {{");
    print_coeffs_json(&play);
    println!(
        "    \"fingerprint\": \"{:016x}\"",
        coefficient_fingerprint(config, "playback", &play)
    );
    println!("  }},");
    println!("  \"inverse\": {{");
    print_coeffs_json(&inverse);
    println!(
        "    \"fingerprint\": \"{:016x}\"",
        coefficient_fingerprint(config, "inverse", &inverse)
    );
    println!("  }},");
    println!("  \"frequency_rows\": [");
    let rows = frequency_table(config);
    for (index, row) in rows.iter().enumerate() {
        let comma = if index + 1 == rows.len() { "" } else { "," };
        println!(
            "    {{\"f_digital_hz\": {:.4}, \"f_analog_hz\": {:.4}, \"playback_db\": {:.9}, \"playback_rel_1k_db\": {:.9}, \"inverse_db\": {:.9}, \"roundtrip_db\": {:.9}, \"analog_playback_db\": {:.9}, \"analog_rel_1k_db\": {:.9}}}{}",
            row.digital_hz,
            row.warped_analog_hz,
            row.playback_db,
            row.playback_rel_1k_db,
            row.inverse_db,
            row.roundtrip_db,
            row.analog_playback_db,
            row.analog_rel_1k_db,
            comma
        );
    }
    println!("  ]");
    println!("}}");
}

fn print_coeffs_json(coeffs: &Coeffs) {
    println!("    \"b0\": {:.17e},", coeffs.b0);
    println!("    \"b1\": {:.17e},", coeffs.b1);
    println!("    \"b2\": {:.17e},", coeffs.b2);
    println!("    \"a1\": {:.17e},", coeffs.a1);
    println!("    \"a2\": {:.17e},", coeffs.a2);
}

/// Text verification report for both tail policies.
pub fn print_verify(config: EngineConfig) {
    println!("playback -> inverse round-trip verification");
    println!("===========================================");
    print_frequency_table(config);
    println!();
    for flush in [false, true] {
        let result = roundtrip_time_domain(config, flush);
        let policy = if flush {
            "flush ring-out"
        } else {
            "truncated tail"
        };
        println!("pointwise time residual ({policy}, includes IIR group delay):");
        println!("  max abs residual : {:.3e}", result.time_error.max_abs);
        println!("  rms residual     : {:.3e}", result.time_error.rms);
        println!("  output peak   : {:.6}", result.output_peak);
        if flush {
            println!(
                "  flush taps    : {}",
                result.flush_taps.iter().sum::<usize>()
            );
        }
    }
    let amplitude = amplitude_roundtrip(config);
    println!();
    println!("steady-state amplitude per probe component (DFT correlation):");
    println!(
        "{:>10} {:>12} {:>12} {:>10}",
        "f Hz", "source", "roundtrip", "rel err"
    );
    for (index, frequency) in PROBE_FREQUENCIES_HZ.iter().enumerate() {
        println!(
            "{:>10.1} {:>12.6} {:>12.6} {:>9.2e}",
            frequency,
            amplitude.reference_amplitudes[index],
            amplitude.measured_amplitudes[index],
            amplitude.relative_errors[index]
        );
    }
    println!(
        "max relative amplitude error: {:.3e} (pointwise rms incl. group delay: {:.3e})",
        amplitude.max_relative, amplitude.pointwise_rms
    );
    let (pass, max_dev, max_err) = crate::verify::passes(config);
    println!();
    println!(
        "criteria (<=4 kHz core band): max |roundtrip dB| = {:.4} dB, max relative amplitude error = {:.3e}",
        max_dev, max_err
    );
    println!(
        "note: deviation above ~{:.0} Hz is the explicit guard pole (at {:.0} Hz),",
        config.guard_hz() * 0.5,
        config.guard_hz()
    );
    println!("      which keeps the inverse biquad strictly stable above the audio band.");
    println!("verdict: {}", if pass { "PASS" } else { "FAIL" });
}
