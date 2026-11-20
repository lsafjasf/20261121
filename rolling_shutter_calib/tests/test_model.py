import numpy as np

from rolling_shutter_calib.model import (
    exposure_fraction,
    exposure_transfer,
    harmonic_amplitude,
    led_harmonic,
    normalized_sinc,
)


def test_sinc_zeros_and_unity():
    assert normalized_sinc(0.0) == 1.0
    assert abs(normalized_sinc(1.0)) < 1e-12
    assert abs(normalized_sinc(2.0)) < 1e-12


def test_full_period_exposure_is_duty():
    starts = np.linspace(0.0, 1.0, 11)
    fraction = exposure_fraction(starts, 1.0, 1.0, duty=0.5)
    assert np.allclose(fraction, 0.5)
    fraction_30 = exposure_fraction(starts, 1.0, 1.0, duty=0.3)
    assert np.allclose(fraction_30, 0.3)


def test_zero_exposure_matches_square_wave():
    t = np.array([0.0, 0.1, 0.6])
    on = exposure_fraction(t, 0.0, 1.0, duty=0.5)
    assert np.allclose(on, [1.0, 1.0, 0.0])


def test_led_harmonics_duty_half_even_orders_vanish():
    assert abs(led_harmonic(0, 0.5) - 0.5) < 1e-12
    assert abs(led_harmonic(2, 0.5)) < 1e-9
    assert abs(led_harmonic(4, 0.5)) < 1e-9
    assert abs(led_harmonic(1, 0.5)) > 0.3


def test_exposure_null_kills_harmonic():
    # t_exp = T makes every harmonic transfer factor vanish.
    assert abs(exposure_transfer(1, 1.0, 1.0)) < 1e-12
    assert harmonic_amplitude(1, 1.0, 1.0, 0.5) < 1e-9


def test_exposure_fraction_bounds():
    starts = np.linspace(-3.0, 7.0, 50)
    fraction = exposure_fraction(starts, 0.37, 1.0, 0.5)
    assert np.all(fraction >= 0.0)
    assert np.all(fraction <= 1.0)
