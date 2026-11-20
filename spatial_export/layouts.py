"""Loudspeaker layouts, including one symmetric and one asymmetric set."""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import Vec3, normalize


@dataclass(frozen=True)
class Speaker:
    name: str
    position: Vec3

    @property
    def unit(self) -> Vec3:
        return normalize(self.position)


@dataclass(frozen=True)
class Layout:
    name: str
    speakers: tuple[Speaker, ...]
    description: str

    @property
    def channel_order(self) -> tuple[str, ...]:
        """Canonical export order: channel index follows this tuple."""
        return tuple(s.name for s in self.speakers)

    @property
    def units(self) -> tuple[Vec3, ...]:
        return tuple(s.unit for s in self.speakers)


def _cube_speakers() -> tuple[Speaker, ...]:
    """Eight speakers at cube corners: symmetric, full-sphere coverage."""
    specs = [
        ("FLU", (1.0, 1.0, 1.0)),
        ("FRU", (1.0, -1.0, 1.0)),
        ("BLU", (-1.0, 1.0, 1.0)),
        ("BRU", (-1.0, -1.0, 1.0)),
        ("FLD", (1.0, 1.0, -1.0)),
        ("FRD", (1.0, -1.0, -1.0)),
        ("BLD", (-1.0, 1.0, -1.0)),
        ("BRD", (-1.0, -1.0, -1.0)),
    ]
    return tuple(Speaker(name, pos) for name, pos in specs)


def _asym_speakers() -> tuple[Speaker, ...]:
    """Eight biased speakers covering the front hemisphere, rear uncovered.

    All speakers sit strictly in the front half-space (positive X), spanning
    nearly +/-90 degrees in azimuth and about +/-40 degrees in elevation.
    Front and side directions render with normal VBAP; the rear half-space
    cannot belong to the convex cone of unit vectors and triggers the
    nearest-speaker degradation path.
    """
    specs = [
        ("FL", (1.0, 1.15, 0.0)),
        ("FR", (1.0, -1.15, 0.0)),
        ("L90", (0.12, 1.6, 0.05)),
        ("R90", (0.12, -1.6, 0.05)),
        ("TFL", (0.9, 1.3, 1.3)),
        ("TFR", (0.9, -1.3, 1.3)),
        ("BFL", (0.9, 1.3, -1.0)),
        ("BFR", (0.9, -1.3, -1.0)),
    ]
    return tuple(Speaker(name, pos) for name, pos in specs)


LAYOUT_CUBE = Layout(
    name="cube_symmetric_8ch",
    speakers=_cube_speakers(),
    description=(
        "Symmetric 8-channel cube (4 upper / 4 lower). The unit direction "
        "vectors form a closed convex polyhedron around the origin, so every "
        "direction on the sphere is covered by at least one VBAP triangle."
    ),
)

LAYOUT_ASYM = Layout(
    name="front_top_cluster_8ch",
    speakers=_asym_speakers(),
    description=(
        "Asymmetric 8-channel layout clustered in the front/top hemisphere. "
        "Directions outside the convex cone of speaker unit vectors are not "
        "covered by any VBAP triangle and trigger nearest-speaker degradation."
    ),
)


LAYOUTS: dict[str, Layout] = {
    LAYOUT_CUBE.name: LAYOUT_CUBE,
    LAYOUT_ASYM.name: LAYOUT_ASYM,
}
