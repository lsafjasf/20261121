import numpy as np

from gyro_stab.integrate import (
    integrate_absolute_quaternions,
    integrate_frame_rotvecs,
)
from gyro_stab.rotations import (
    matrix_to_rotvec,
    quat_to_matrix,
    rotvec_to_matrix,
)


def test_constant_rate_recovers_total_angle():
    rate = 2.0
    dt = 0.005
    stamps = np.arange(0.0, 1.0 + dt, dt)
    rates = np.zeros((len(stamps), 3))
    rates[:, 1] = rate
    frame_times = np.arange(0.0, 1.01, 0.1)
    rv = integrate_frame_rotvecs(stamps, rates, frame_times)
    # Each 0.1 s interval covers ~0.2 rad about Y; discrete sample boundaries
    # can cost at most one step, so allow a single-step tolerance.
    for i in range(1, len(frame_times)):
        assert abs(np.linalg.norm(rv[i]) - rate * 0.1) <= rate * dt * 1.5


def test_absolute_quaternions_match_direct_composition():
    stamps = np.arange(0.0, 0.31, 0.01)
    rates = np.zeros((len(stamps), 3))
    rates[:, 0] = 2.0
    frame_times = np.array([0.0, 0.1, 0.2, 0.3])
    quats = integrate_absolute_quaternions(stamps, rates, frame_times)
    expected = np.eye(3)
    for _ in range(3):
        expected = rotvec_to_matrix([2.0 * 0.1, 0.0, 0.0]) @ expected
    # Camera-to-world chain equals the composed body increments.
    assert np.allclose(
        matrix_to_rotvec(quat_to_matrix(quats[-1]) @ expected.T),
        np.zeros(3),
        atol=1e-6,
    )
