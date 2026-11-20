//! RIAA equalization core: time constants, bilinear-transform biquads and
//! frequency response helpers.
//!
//! All math is `f64` and dependency free. The playback curve and the guarded
//! inverse curve are derived from the same analog prototype and the same
//! bilinear transform, so their cascade is the exact bilinear transform of a
//! single guard pole (see [`guarded_inverse`]).

/// RIAA standard turnover time constants (IEC 60098 / RIAA E4), seconds.
pub const TAU1: f64 = 3_180e-6;
pub const TAU2: f64 = 318e-6;
pub const TAU3: f64 = 75e-6;

/// Turnover frequency derived from a time constant: f = 1 / (2 pi tau).
pub fn corner_hz(tau: f64) -> f64 {
    1.0 / (2.0 * std::f64::consts::PI * tau)
}

/// The three canonical RIAA turnover frequencies in hertz.
pub fn riaa_corners_hz() -> [f64; 3] {
    [corner_hz(TAU1), corner_hz(TAU2), corner_hz(TAU3)]
}

/// Canonical biquad coefficients, normalized so that `a0 == 1.0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coeffs {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a1: f64,
    pub a2: f64,
}

impl Coeffs {
    fn normalize(b0: f64, b1: f64, b2: f64, a0: f64, a1: f64, a2: f64) -> Self {
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }
}

/// Playback (reproduction) biquad at sample rate `sample_rate`.
///
/// Analog prototype: `H(s) = (1 + s*tau2) / ((1 + s*tau1)(1 + s*tau3))`,
/// discretized with the bilinear transform `s = 2*fs*(z-1)/(z+1)`.
/// DC gain is exactly 1.0; the zero lands on z = -1 (Nyquist).
pub fn playback(sample_rate: f64) -> Coeffs {
    let c = 2.0 * sample_rate;
    let (t1, t2, t3) = (TAU1, TAU2, TAU3);
    let b0 = 1.0 + c * t2;
    let b1 = 2.0;
    let b2 = 1.0 - c * t2;
    let a0 = 1.0 + c * t1 + c * c * t1 * t3;
    let a1 = 2.0 * (1.0 - c * c * t1 * t3);
    let a2 = 1.0 - c * t1 + c * c * t1 * t3;
    Coeffs::normalize(b0, b1, b2, a0, a1, a2)
}

/// Inverse (recording) biquad with an explicit guard pole.
///
/// The pure inverse `1/H_playback(s)` is unstable under the bilinear transform
/// because its pole at `tau2` maps onto the unit circle (`z = -1`). A guard
/// pole at angular frequency `omega_guard` (`H *= 1/(1 + s/omega_guard)`)
/// pulls every pole strictly inside the unit circle. With `guard_fraction` of
/// the sample rate (clamped to `(0, 0.5]`), the default 0.45 puts the pole
/// above the audio band for 44.1/48 kHz sample rates.
pub fn guarded_inverse(sample_rate: f64, guard_fraction: f64) -> Coeffs {
    let g = guard_fraction.clamp(f64::MIN_POSITIVE, 0.5);
    let omega_guard = 2.0 * std::f64::consts::PI * (g * sample_rate);
    let tg = 1.0 / omega_guard;
    let c = 2.0 * sample_rate;
    let (t1, t2, t3) = (TAU1, TAU2, TAU3);
    let b0 = 1.0 + c * t1 + c * c * t1 * t3;
    let b1 = 2.0 * (1.0 - c * c * t1 * t3);
    let b2 = 1.0 - c * t1 + c * c * t1 * t3;
    let a0 = (1.0 + c * t2) * (1.0 + c * tg);
    let a1 = 2.0 * (1.0 - c * c * t2 * tg);
    let a2 = (1.0 - c * t2) * (1.0 - c * tg);
    Coeffs::normalize(b0, b1, b2, a0, a1, a2)
}

/// Guard-pole corner frequency in hertz, matching [`guarded_inverse`].
pub fn guard_corner_hz(sample_rate: f64, guard_fraction: f64) -> f64 {
    let g = guard_fraction.clamp(f64::MIN_POSITIVE, 0.5);
    g * sample_rate
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pole_radius(coeffs: &Coeffs) -> f64 {
        let discriminant = coeffs.a1 * coeffs.a1 - 4.0 * coeffs.a2;
        if discriminant < 0.0 {
            coeffs.a2.sqrt()
        } else {
            let root = discriminant.sqrt();
            ((-coeffs.a1 + root) / 2.0)
                .abs()
                .max(((-coeffs.a1 - root) / 2.0).abs())
        }
    }

    #[test]
    fn turnover_frequencies_match_riaa_standard() {
        let corners = riaa_corners_hz();
        assert!((corners[0] - 50.05).abs() < 0.01);
        assert!((corners[1] - 500.5).abs() < 0.1);
        assert!((corners[2] - 2122.1).abs() < 0.1);
    }

    #[test]
    fn playback_has_unity_dc_gain_and_nyquist_zero() {
        let coeffs = playback(48_000.0);
        let dc = (coeffs.b0 + coeffs.b1 + coeffs.b2) / (1.0 + coeffs.a1 + coeffs.a2);
        assert!((dc - 1.0).abs() < 1e-12);
        let nyquist = (coeffs.b0 - coeffs.b1 + coeffs.b2) / (1.0 - coeffs.a1 + coeffs.a2);
        assert!(nyquist.abs() < 1e-12);
    }

    #[test]
    fn inverse_is_stable_at_common_sample_rates() {
        for sample_rate in [44_100.0, 48_000.0, 96_000.0, 192_000.0] {
            let coeffs = guarded_inverse(sample_rate, 0.45);
            assert!(pole_radius(&coeffs) < 1.0, "unstable at {sample_rate}");
            let dc = (coeffs.b0 + coeffs.b1 + coeffs.b2) / (1.0 + coeffs.a1 + coeffs.a2);
            assert!((dc - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn guard_fraction_is_clamped() {
        let coeffs = guarded_inverse(48_000.0, 10.0);
        let clamped = guarded_inverse(48_000.0, 0.5);
        assert_eq!(coeffs, clamped);
    }
}
