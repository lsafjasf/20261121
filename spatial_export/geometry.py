"""Minimal 3D vector helpers used by the offline spatial exporter.

Everything is plain Python (standard library only) so the exporter runs on a
machine with no third-party packages and no network access.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

Vec3 = tuple[float, float, float]


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(a: Vec3, k: float) -> Vec3:
    return (a[0] * k, a[1] * k, a[2] * k)


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(a: Vec3) -> float:
    return math.sqrt(dot(a, a))


def normalize(a: Vec3) -> Vec3:
    n = norm(a)
    if n <= 1e-15:
        raise ValueError("cannot normalize a zero-length vector")
    return (a[0] / n, a[1] / n, a[2] / n)


def angle_between(a: Vec3, b: Vec3) -> float:
    """Unsigned angle in radians between two directions, range [0, pi]."""
    c = dot(a, b) / (norm(a) * norm(b))
    return math.acos(min(1.0, max(-1.0, c)))


def unit_from_az_el(azimuth_deg: float, elevation_deg: float) -> Vec3:
    """Build a unit direction from audio-style spherical coordinates.

    azimuth: degrees, 0 = +X (front), positive rotates toward +Y (left).
    elevation: degrees, positive toward +Z (up).
    """
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    ce = math.cos(el)
    return (ce * math.cos(az), ce * math.sin(az), math.sin(el))


def az_el_of_unit(a: Vec3) -> Vec3:
    """Inverse of unit_from_az_el; returns (azimuth_deg, elevation_deg, 0)."""
    u = normalize(a)
    el = math.degrees(math.asin(min(1.0, max(-1.0, u[2]))))
    az = math.degrees(math.atan2(u[1], u[0])) % 360.0
    return (az, el, 0.0)


def det3(rows: Sequence[Sequence[float]]) -> float:
    """Determinant of a 3x3 matrix given as three row vectors."""
    a, b, c = rows
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def solve3(
    matrix: Sequence[Sequence[float]], rhs: Sequence[float]
) -> tuple[float, float, float] | None:
    """Cramer's rule solver for a 3x3 system; None if (near) singular."""
    d = det3(matrix)
    if abs(d) < 1e-12:
        return None
    d0 = det3([rhs, matrix[1], matrix[2]])
    d1 = det3([matrix[0], rhs, matrix[2]])
    d2 = det3([matrix[0], matrix[1], rhs])
    return (d0 / d, d1 / d, d2 / d)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
