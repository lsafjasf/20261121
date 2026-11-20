"""Three-dimensional Vector Base Amplitude Panning (VBAP).

Reference: Pulkki, V. (1997), "Virtual Sound Source Positioning Using Vector
Base Amplitude Panning". The implementation is self-contained and uses only
the standard library, which keeps the exporter fully offline.

Why VBAP for asymmetric layouts:
    * It works with arbitrary, non-regular speaker positions instead of
      assuming a fixed symmetric ring (unlike pairwise sine-law panning).
    * At most three speakers carry energy for any direction, preserving
      locality of the image even with irregular angular spacing.
    * Directions outside the convex cone of speaker unit vectors are
      detectable explicitly, so uncovered directions can be reported and
      handled by a defined degradation strategy instead of silently failing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

from .geometry import (
    Vec3,
    angle_between,
    det3,
    dot,
    normalize,
    solve3,
)
from .layouts import Layout

GAIN_EPS = 1e-9


@dataclass(frozen=True)
class Triplet:
    indices: tuple[int, int, int]
    inv_rows: tuple[Vec3, Vec3, Vec3]


@dataclass
class PanResult:
    direction: Vec3
    gains: tuple[float, ...]
    active: tuple[int, ...]
    degraded: bool
    degradation_kind: str | None
    detail: str
    angular_error_deg: float


@dataclass
class DegradationEvent:
    frame: int
    time_s: float
    azimuth_deg: float
    elevation_deg: float
    nearest_speaker: str
    angular_gap_deg: float
    end_time_s: float | None = None
    end_azimuth_deg: float | None = None


def _inverse_rows(matrix_rows: tuple[Vec3, Vec3, Vec3]) -> tuple[Vec3, Vec3, Vec3]:
    """Rows of L^-1 for the speaker-direction matrix L (row vectors).

    Columns of L^-1 are computed via Cramer's rule (``L c_k = e_k``), then
    transposed into rows. This avoids hand-written cofactor index bookkeeping.
    """
    columns = []
    for k in range(3):
        rhs = tuple(1.0 if i == k else 0.0 for i in range(3))
        solution = solve3(matrix_rows, rhs)
        if solution is None:
            raise ValueError("singular speaker matrix cannot be inverted")
        columns.append(solution)
    return tuple(
        tuple(columns[col][row] for col in range(3))  # type: ignore[arg-type]
        for row in range(3)
    )


class VbapPanner:
    """Precomputes valid VBAP triangles and pans unit directions."""

    def __init__(self, layout: Layout, inv_min_dot: float = 0.0):
        self.layout = layout
        self.units = layout.units
        self.inv_min_dot = inv_min_dot
        self.triplets: list[Triplet] = []
        self._build_triplets()

    def _build_triplets(self) -> None:
        n = len(self.units)
        for combo in combinations(range(n), 3):
            matrix = tuple(self.units[i] for i in combo)
            if abs(det3(matrix)) < 1e-9:
                continue
            inv_rows = _inverse_rows(matrix)
            # Pulkki's validity test: each inverse row (per speaker) must have
            # a non-negative dot product with that speaker's unit direction.
            if all(
                dot(inv_rows[k], self.units[combo[k]]) >= self.inv_min_dot
                for k in range(3)
            ):
                self.triplets.append(Triplet(combo, inv_rows))

    def _nearest_speaker(self, direction: Vec3) -> tuple[int, float]:
        best_i, best_angle = 0, math.pi
        for i, unit in enumerate(self.units):
            gap = angle_between(direction, unit)
            if gap < best_angle:
                best_i, best_angle = i, gap
        return best_i, best_angle

    def pan(self, direction: Vec3) -> PanResult:
        u = normalize(direction)
        n = len(self.units)
        best: tuple[float, ...] | None = None
        best_active: tuple[int, ...] = ()

        for triplet in self.triplets:
            raw = tuple(dot(row, u) for row in triplet.inv_rows)
            if any(g < -GAIN_EPS for g in raw):
                continue
            raw = tuple(max(0.0, g) for g in raw)
            power = math.sqrt(sum(g * g for g in raw))
            if power <= GAIN_EPS:
                continue
            gains = tuple(g / power for g in raw)
            if best is None or sum(g * g for g in gains) > sum(
                g * g for g in best
            ):
                best, best_active = gains, triplet.indices

        if best is not None:
            gains = [0.0] * n
            for k, idx in enumerate(best_active):
                gains[idx] = best[k]
            reconstructed = self._reconstruct(tuple(gains))
            error = math.degrees(angle_between(u, reconstructed))
            return PanResult(
                direction=u,
                gains=tuple(gains),
                active=tuple(i for i in best_active if gains[i] > GAIN_EPS),
                degraded=False,
                degradation_kind=None,
                detail="vbap",
                angular_error_deg=error,
            )

        # Degradation path: no convex triangle contains the direction.
        nearest_i, gap = self._nearest_speaker(u)
        gains = [0.0] * n
        gains[nearest_i] = 1.0
        return PanResult(
            direction=u,
            gains=tuple(gains),
            active=(nearest_i,),
            degraded=True,
            degradation_kind="uncovered_nearest_speaker",
            detail=(
                "direction outside convex cone of speakers; routed fully to "
                f"nearest speaker {self.layout.speakers[nearest_i].name}"
            ),
            angular_error_deg=math.degrees(gap),
        )

    def _reconstruct(self, gains: tuple[float, ...]) -> Vec3:
        vec = (0.0, 0.0, 0.0)
        for gain, unit in zip(gains, self.units):
            vec = (
                vec[0] + gain * unit[0],
                vec[1] + gain * unit[1],
                vec[2] + gain * unit[2],
            )
        return normalize(vec)

    def coverage_sample(
        self, az_step_deg: float = 5.0, el_step_deg: float = 5.0
    ) -> dict:
        """Sample the whole sphere and report covered vs degraded share."""
        total = covered = 0
        max_error = 0.0
        el = -90.0
        while el <= 90.0 + 1e-9:
            az = 0.0
            while az < 360.0 - 1e-9:
                from .geometry import unit_from_az_el

                result = self.pan(unit_from_az_el(az, el))
                total += 1
                if not result.degraded:
                    covered += 1
                    max_error = max(max_error, result.angular_error_deg)
                az += az_step_deg
            el += el_step_deg
        return {
            "samples": total,
            "covered_samples": covered,
            "degraded_samples": total - covered,
            "coverage_ratio": covered / total,
            "max_pan_error_deg_when_covered": max_error,
            "valid_triplets": len(self.triplets),
        }
