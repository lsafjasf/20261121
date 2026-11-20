"""Forward photometric model for an LED flickering at a known frequency.

Conventions
-----------
All time quantities are expressed in seconds and angles in radians.

The LED drives a periodic square wave with period ``T = 1 / f_led``:

* the LED is "on" for the first ``duty * T`` of every period
  (brightness 1) and "off" otherwise (brightness 0);
* a camera row integrates the light over an exposure window of length
  ``t_exp`` starting at time ``t0``.

A rolling shutter means every sensor row starts its exposure at a
different time.  For readout direction ``s`` in {-1, +1} the exposure
start time of sensor row ``r`` (0-indexed from the image top) is::

    t0(r) = frame_time + t0_frame - s * r * t_line

with ``t_line`` the per-row readout (line) time.  ``s = +1`` means the
top row is read out first ("top-down"), ``s = -1`` means bottom-up.
"""

from __future__ import annotations

import numpy as np


def normalized_sinc(x: np.ndarray | float) -> np.ndarray | float:
    """Normalized sinc ``sin(pi x) / (pi x)`` with sinc(0) = 1."""
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(x == 0.0, 1.0, np.sin(np.pi * x) / (np.pi * x))


def led_harmonic(k: int, duty: float) -> complex:
    """Complex Fourier coefficient of the LED square wave.

    The square wave is 1 on ``[0, duty*T)`` and 0 on the rest of the
    period.  Coefficients use the convention
    ``led(t) = sum_k c_k exp(j 2 pi k t / T)`` with
    ``c_k = (1/T) integral_0^T led(t) exp(-j 2 pi k t / T) dt``.
    """
    if k == 0:
        return complex(duty, 0.0)
    phase = np.pi * k * duty
    return complex(duty * normalized_sinc(k * duty) * np.exp(-1j * phase), 0.0)


def exposure_transfer(k: int, t_exp: float, period: float) -> complex:
    """Exposure-window transfer factor for harmonic ``k``.

    Integrating ``exp(j 2 pi k t / T)`` over a window of length ``t_exp``
    multiplies harmonic ``k`` by ``sinc(k t_exp / T)``; the window start
    only contributes a phase term that is handled separately.
    """
    if t_exp < 0.0:
        raise ValueError("t_exp must be non-negative")
    return complex(normalized_sinc(k * t_exp / period), 0.0)


def exposure_fraction(
    t0: np.ndarray | float, t_exp: float, period: float, duty: float = 0.5
) -> np.ndarray:
    """Fraction of the exposure window during which the LED is on.

    Exact evaluation of the integral of the square wave over
    ``[t0, t0 + t_exp]`` divided by ``t_exp``.  Works with arrays.
    """
    if t_exp < 0.0:
        raise ValueError("t_exp must be non-negative")
    if t_exp == 0.0:
        return _instantaneous(np.asarray(t0, dtype=np.float64), period, duty)

    t0 = np.asarray(t0, dtype=np.float64)
    on_until = _integral_on_time(t0 + t_exp, period, duty) - _integral_on_time(
        t0, period, duty
    )
    return np.clip(on_until / t_exp, 0.0, 1.0)


def _instantaneous(t: np.ndarray, period: float, duty: float) -> np.ndarray:
    phase = np.mod(t, period)
    return (phase < duty * period).astype(np.float64)


def _integral_on_time(t: np.ndarray, period: float, duty: float) -> np.ndarray:
    """Cumulative time the LED has been on over (-inf, t].

    For a 50% (or general ``duty``) square wave aligned at t = 0.
    """
    t = np.asarray(t, dtype=np.float64)
    n = np.floor(t / period)
    phase = t - n * period
    on_phase = np.minimum(phase, duty * period)
    return n * duty * period + on_phase


def harmonic_amplitude(
    k: int, t_exp: float, period: float, duty: float
) -> float:
    """Magnitude of harmonic k observed after an exposure of length t_exp."""
    return abs(led_harmonic(k, duty)) * abs(exposure_transfer(k, t_exp, period))
