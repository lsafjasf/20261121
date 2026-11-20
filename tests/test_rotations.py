import numpy as np

from gyro_stab.rotations import (
    matrix_to_quat,
    matrix_to_rotvec,
    nearest_rotation,
    quat_to_matrix,
    rotvec_to_matrix,
    slerp,
)


def test_rotvec_roundtrip():
    rv = np.array([0.15, -0.22, 0.31])
    back = matrix_to_rotvec(rotvec_to_matrix(rv))
    assert np.allclose(back, rv, atol=1e-9)


def test_quat_matrix_roundtrip():
    matrix = rotvec_to_matrix(np.array([0.4, -0.1, 0.2]))
    quat = matrix_to_quat(matrix)
    assert np.allclose(quat_to_matrix(quat), matrix, atol=1e-9)


def test_nearest_rotation_is_orthonormal():
    rng = np.random.default_rng(0)
    noisy = rotvec_to_matrix([0.2, 0.1, -0.3]) + rng.normal(0, 0.05, (3, 3))
    rotation = nearest_rotation(noisy)
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(rotation), 1.0)


def test_slerp_endpoints_and_unit():
    q0 = matrix_to_quat(np.eye(3))
    q1 = matrix_to_quat(rotvec_to_matrix([0.0, 0.0, 0.6]))
    assert np.allclose(slerp(q0, q1, 0.0), q0)
    assert np.allclose(slerp(q0, q1, 1.0), q1)
    mid = slerp(q0, q1, 0.5)
    assert np.isclose(np.linalg.norm(mid), 1.0)


def test_body_increment_convention():
    # R_prev^T R_curr is the standard body-frame rotation increment.
    r_prev = rotvec_to_matrix([0.1, 0.0, 0.0])
    r_curr = rotvec_to_matrix([0.2, 0.0, 0.0])
    increment = matrix_to_rotvec(r_prev.T @ r_curr)
    assert np.allclose(increment, [0.1, 0.0, 0.0], atol=1e-9)
