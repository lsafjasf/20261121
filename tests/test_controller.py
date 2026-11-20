from driftmix.controller import FillController


def test_zero_error_holds_unity():
    c = FillController(1000, 1e-5, 1e-6)
    assert c.update(1000) == 1.0


def test_full_buffer_raises_consume_ratio():
    c = FillController(1000, 1e-5, 1e-6)
    assert c.update(1500) > 1.0


def test_empty_buffer_lowers_consume_ratio():
    c = FillController(1000, 1e-5, 1e-6)
    assert c.update(500) < 1.0


def test_slew_rate_limits_step_size():
    c = FillController(1000, 1e-5, 1e-6, max_slew=1e-4)
    first = c.update(5000)
    assert abs(first - 1.0) <= 1e-4 + 1e-12


def test_clamp_bounds_are_respected():
    c = FillController(1000, 1.0, 1.0, ratio_min=0.99, ratio_max=1.01)
    for _ in range(100):
        c.update(10_000)
    assert c.ratio >= 0.99 - 1e-12
