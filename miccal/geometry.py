from __future__ import annotations

import itertools
import math
from typing import Iterable, Sequence

from .config import Scene, Vec3


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale(a: Vec3, value: float) -> Vec3:
    return (a[0] * value, a[1] * value, a[2] * value)


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a: Vec3) -> float:
    return math.sqrt(dot(a, a))


def unit(a: Vec3) -> Vec3:
    length = norm(a)
    if length < 1e-12:
        raise ValueError("cannot normalize a zero vector")
    return scale(a, 1.0 / length)


def distance(a: Vec3, b: Vec3) -> float:
    return norm(sub(a, b))


def centroid(points: Sequence[Vec3]) -> Vec3:
    count = float(len(points))
    return (
        sum(p[0] for p in points) / count,
        sum(p[1] for p in points) / count,
        sum(p[2] for p in points) / count,
    )


def direct_distances(mics: Sequence[Vec3], source: Vec3) -> tuple[float, ...]:
    return tuple(distance(source, mic) for mic in mics)


def relative_delays(
    mics: Sequence[Vec3],
    source: Vec3,
    reference: int,
    sound_speed: float,
) -> tuple[float, ...]:
    distances = direct_distances(mics, source)
    base = distances[reference]
    return tuple((r - base) / sound_speed for r in distances)


def image_source(source: Vec3, point: Vec3, normal: Vec3) -> Vec3:
    n = unit(normal)
    signed = dot(sub(source, point), n)
    return sub(source, scale(n, 2.0 * signed))


def reflection_paths(scene: Scene, source: Vec3) -> tuple[float, ...] | None:
    if scene.wall is None:
        return None
    image = image_source(source, scene.wall.point, scene.wall.normal)
    return tuple(
        distance(image, mic) for mic in scene.installed_microphones()
    )


def permuted_mics(
    mics: Sequence[Vec3],
    order: Sequence[int],
) -> tuple[Vec3, ...]:
    return tuple(mics[index] for index in order)


def predicted_relative_delays(
    mics: Sequence[Vec3],
    sources: Iterable[Vec3],
    reference: int,
    sound_speed: float,
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        relative_delays(mics, source, reference, sound_speed)
        for source in sources
    )


def all_permutations() -> tuple[tuple[int, int, int, int], ...]:
    return tuple(itertools.permutations((0, 1, 2, 3)))  # type: ignore[return-value]


def delay_rank(mics: Sequence[Vec3], sources: Sequence[Vec3]) -> int:
    rows: list[list[float]] = []
    base = centroid(mics)
    for source in sources:
        direction = sub(source, base)
        if norm(direction) < 1e-12:
            continue
        u = unit(direction)
        rows.append([dot(u, sub(mic, base)) for mic in mics[1:]])
    return matrix_rank(rows)


def matrix_rank(rows: Sequence[Sequence[float]], tolerance: float = 1e-9) -> int:
    matrix = [list(map(float, row)) for row in rows if row]
    if not matrix:
        return 0
    rank = 0
    columns = len(matrix[0])
    for column in range(columns):
        pivot = max(range(rank, len(matrix)), key=lambda r: abs(matrix[r][column]))
        if abs(matrix[pivot][column]) <= tolerance:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        value = matrix[rank][column]
        matrix[rank] = [entry / value for entry in matrix[rank]]
        for row in range(len(matrix)):
            if row == rank:
                continue
            factor = matrix[row][column]
            if abs(factor) <= tolerance:
                continue
            matrix[row] = [
                entry - factor * pivot_entry
                for entry, pivot_entry in zip(matrix[row], matrix[rank])
            ]
        rank += 1
        if rank == len(matrix):
            break
    return rank


def matrix_distance(
    a: Sequence[Sequence[float]],
    b: Sequence[Sequence[float]],
) -> float:
    total = 0.0
    count = 0
    for row_a, row_b in zip(a, b):
        for value_a, value_b in zip(row_a, row_b):
            total += (value_a - value_b) ** 2
            count += 1
    return math.sqrt(total / count) if count else 0.0
