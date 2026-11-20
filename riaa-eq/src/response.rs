//! Frequency response evaluation for analog prototypes and digital biquads.

use super::riaa::{Coeffs, TAU1, TAU2, TAU3};

/// Minimal complex number (only the operations needed here).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn plus(self, other: Self) -> Self {
        Self::new(self.re + other.re, self.im + other.im)
    }

    pub fn times(self, other: Self) -> Self {
        Self::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }

    pub fn scale(self, factor: f64) -> Self {
        Self::new(self.re * factor, self.im * factor)
    }

    pub fn over(self, other: Self) -> Self {
        let denom = other.re * other.re + other.im * other.im;
        Self::new(
            (self.re * other.re + self.im * other.im) / denom,
            (self.im * other.re - self.re * other.im) / denom,
        )
    }

    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
}

/// Magnitude in decibels, guarding against log of zero.
pub fn to_db(magnitude: f64) -> f64 {
    20.0 * magnitude.max(1e-300).log10()
}

/// Analog RIAA playback prototype evaluated at physical frequency `hz`.
pub fn analog_playback_db(hz: f64) -> f64 {
    let jw = Complex::new(0.0, 2.0 * std::f64::consts::PI * hz);
    let num = Complex::new(1.0, 0.0).plus(jw.scale(TAU2));
    let den1 = Complex::new(1.0, 0.0).plus(jw.scale(TAU1));
    let den2 = Complex::new(1.0, 0.0).plus(jw.scale(TAU3));
    to_db(num.over(den1.times(den2)).abs())
}

/// Analog prototype response referenced to 1 kHz (0 dB at 1 kHz).
pub fn analog_playback_db_rel_1k(hz: f64) -> f64 {
    analog_playback_db(hz) - analog_playback_db(1_000.0)
}

impl Coeffs {
    /// Digital response `H(z)` at normalized frequency `theta = 2*pi*f/fs`.
    pub fn response(&self, theta: f64) -> Complex {
        let z = Complex::new(theta.cos(), theta.sin());
        let z2 = z.times(z);
        let num = Complex::new(self.b0, 0.0)
            .plus(z.scale(self.b1))
            .plus(z2.scale(self.b2));
        let den = Complex::new(1.0, 0.0)
            .plus(z.scale(self.a1))
            .plus(z2.scale(self.a2));
        num.over(den)
    }

    /// Magnitude response in dB at physical frequency `hz`.
    pub fn response_db(&self, hz: f64, sample_rate: f64) -> f64 {
        let theta = 2.0 * std::f64::consts::PI * hz / sample_rate;
        to_db(self.response(theta).abs())
    }

    /// Magnitude response in dB referenced to the 1 kHz response (0 dB at 1 kHz),
    /// matching standard RIAA plot conventions.
    pub fn response_db_rel_1k(&self, hz: f64, sample_rate: f64) -> f64 {
        self.response_db(hz, sample_rate) - self.response_db(1_000.0, sample_rate)
    }
}

/// Analog frequency that the bilinear transform maps to digital frequency
/// `digital_hz`: `f_analog = (fs/pi) * tan(pi * f_digital / fs)`.
pub fn warped_analog_hz(digital_hz: f64, sample_rate: f64) -> f64 {
    let theta = std::f64::consts::PI * digital_hz / sample_rate;
    sample_rate / std::f64::consts::PI * theta.tan()
}
