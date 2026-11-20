"""Trajectory of a moving 3D sound object, sampled per audio block."""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import Vec3, unit_from_az_el


@dataclass(frozen=True)
class Keyframe:
    time_s: float
    azimuth_deg: float
    elevation_deg: float


@dataclass(frozen=True)
class Trajectory:
    name: str
    keyframes: tuple[Keyframe, ...]

    def _wrapped_lerp(self, a0: float, a1: float, t: float) -> float:
        """Interpolate azimuth along the shorter angular path."""
        delta = (a1 - a0 + 180.0) % 360.0 - 180.0
        return (a0 + delta * t) % 360.0

    def direction_at(self, time_s: float) -> Vec3:
        frames = self.keyframes
        if time_s <= frames[0].time_s:
            kf = frames[0]
            return unit_from_az_el(kf.azimuth_deg, kf.elevation_deg)
        if time_s >= frames[-1].time_s:
            kf = frames[-1]
            return unit_from_az_el(kf.azimuth_deg, kf.elevation_deg)
        for k0, k1 in zip(frames, frames[1:]):
            if k0.time_s <= time_s <= k1.time_s:
                span = k1.time_s - k0.time_s
                t = 0.0 if span == 0.0 else (time_s - k0.time_s) / span
                az = self._wrapped_lerp(k0.azimuth_deg, k1.azimuth_deg, t)
                el = k0.elevation_deg + (k1.elevation_deg - k0.elevation_deg) * t
                return unit_from_az_el(az, el)
        raise RuntimeError("unreachable trajectory segment")


def demo_trajectory(duration_s: float = 8.0) -> Trajectory:
    """A full 360-degree sweep that also climbs and dips in elevation.

    The rear and bottom portions intentionally leave the coverage region of
    the asymmetric layout so degradation can be exercised and reported.
    """
    keyframes = (
        Keyframe(0.0, 0.0, 0.0),
        Keyframe(0.25 * duration_s, 90.0, 25.0),
        Keyframe(0.5 * duration_s, 180.0, 0.0),
        Keyframe(0.75 * duration_s, 270.0, -25.0),
        Keyframe(duration_s, 360.0, 0.0),
    )
    return Trajectory(name="azimuth_sweep_with_elevation", keyframes=keyframes)
