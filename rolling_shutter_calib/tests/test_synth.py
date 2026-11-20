import numpy as np

from rolling_shutter_calib.synth import SynthConfig, render_frame, rotate_frame


def test_render_shape_and_dtype():
    cfg = SynthConfig(width=160, height=120, frames=1)
    frame = render_frame(cfg, 0, np.random.default_rng(0))
    assert frame.shape == (120, 160)
    assert frame.dtype == np.uint8


def test_deterministic_seed():
    cfg = SynthConfig(frames=1)
    a = render_frame(cfg, 0, np.random.default_rng(42))
    b = render_frame(cfg, 0, np.random.default_rng(42))
    assert np.array_equal(a, b)


def test_zero_rotation_is_identity():
    frame = np.arange(100, dtype=np.uint8).reshape(10, 10)
    assert np.array_equal(rotate_frame(frame, 0.0), frame)


def test_scan_direction_changes_phase():
    down = SynthConfig(frames=1, scan_direction=1, rotation_deg=0,
                       gaussian_sigma=0.0, phase_jitter_us=0.0, genlock=True)
    up = SynthConfig(frames=1, scan_direction=-1, rotation_deg=0,
                     gaussian_sigma=0.0, phase_jitter_us=0.0, genlock=True)
    frame_down = render_frame(down, 0, np.random.default_rng(0))
    frame_up = render_frame(up, 0, np.random.default_rng(0))
    assert not np.array_equal(frame_down, frame_up)
