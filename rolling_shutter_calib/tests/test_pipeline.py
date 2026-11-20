import numpy as np
import pytest

from rolling_shutter_calib.calibrate import CalibrationConfig, calibrate_video
from rolling_shutter_calib.geometry import (
    estimate_stripe_angle,
    row_profile,
    signed_spatial_frequency,
)
from rolling_shutter_calib.synth import SynthConfig, write_video


def _run(tmp_path, **overrides):
    cfg_kwargs = dict(
        frames=8,
        genlock=True,
        rotation_deg=12.0,
        scan_direction=1,
        t_line_us=20.0,
        t_exp_us=120.0,
        gaussian_sigma=4.0,
    )
    cfg_kwargs.update(overrides)
    cfg = SynthConfig(**cfg_kwargs)
    video, truth = write_video(cfg, tmp_path / "clip.avi")
    calib_cfg = CalibrationConfig(
        f_led_hz=cfg.f_led_hz,
        duty=cfg.duty,
        genlock=cfg.genlock,
        brightness_span=cfg.brightness - cfg.black_level,
    )
    result = calibrate_video(video, calib_cfg)
    return cfg, result


def test_line_time_recovered(tmp_path):
    cfg, result = _run(tmp_path)
    assert result.solvable
    assert result.t_line_s == pytest.approx(cfg.t_line_s, rel=0.03)


def test_exposure_recovered(tmp_path):
    cfg, result = _run(tmp_path)
    assert result.t_exp_s == pytest.approx(cfg.t_exp_s, rel=0.10)


def test_scan_direction_both_ways(tmp_path):
    for direction in (1, -1):
        cfg, result = _run(tmp_path, scan_direction=direction)
        assert result.scan_direction == direction


def test_rotation_estimate(tmp_path):
    cfg, result = _run(tmp_path, rotation_deg=15.0)
    assert result.rotation_deg == pytest.approx(-cfg.rotation_deg, abs=1.5)


def test_full_period_exposure_is_unsolvable(tmp_path):
    # t_exp = LED period: the fundamental sinc is exactly zero.
    cfg, result = _run(tmp_path, t_exp_us=1000.0)
    assert not result.solvable
    assert result.reasons


def test_near_null_exposure_flagged(tmp_path):
    cfg, result = _run(tmp_path, t_exp_us=750.0)
    assert not result.solvable
    assert any("null" in reason for reason in result.reasons)


def test_extreme_noise_unsolvable(tmp_path):
    cfg, result = _run(tmp_path, gaussian_sigma=150.0)
    assert not result.solvable


def test_free_runner_direction_ambiguous(tmp_path):
    cfg_kwargs = dict(
        frames=8, genlock=False, rotation_deg=10.0, scan_direction=1,
        t_exp_us=120.0, gaussian_sigma=4.0,
    )
    cfg = SynthConfig(**cfg_kwargs)
    video, _ = write_video(cfg, tmp_path / "free.avi")
    result = calibrate_video(
        video, CalibrationConfig(f_led_hz=cfg.f_led_hz, genlock=False)
    )
    assert result.direction_ambiguous
    assert result.scan_direction is None


def test_signed_frequency_flips_with_direction():
    profiles = {}
    for direction in (1, -1):
        cfg = SynthConfig(
            frames=1, genlock=True, rotation_deg=0, scan_direction=direction,
            gaussian_sigma=0.0, phase_jitter_us=0.0,
        )
        frame = __import__(
            "rolling_shutter_calib.synth", fromlist=["render_frame"]
        ).render_frame(cfg, 0, np.random.default_rng(0))
        profiles[direction] = row_profile(frame)
    f_down, _, _ = signed_spatial_frequency(profiles[1])
    f_up, _, _ = signed_spatial_frequency(profiles[-1])
    assert f_down * f_up < 0.0


def test_stripe_angle_recovers_rotation():
    cfg = SynthConfig(frames=1, rotation_deg=20.0, gaussian_sigma=2.0)
    frame = __import__(
        "rolling_shutter_calib.synth", fromlist=["render_frame"]
    ).render_frame(cfg, 0, np.random.default_rng(0))
    geometry = estimate_stripe_angle(frame)
    assert geometry.angle_deg == pytest.approx(-20.0, abs=2.0)
