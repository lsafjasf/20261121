"""Synthetic scene: a large textured ground/wall plane plus foreground props.

The scene is rendered by full pinhole projection (not homography), so a
lateral camera translation creates real parallax between the background
plane and the foreground objects. That parallax is the cue that lets the
calibration stage reject translation-only segments.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )


def _smooth_random_grid(rng: np.random.Generator, size: int, cells: int) -> np.ndarray:
    """Low-frequency random field upsampled with cubic interpolation."""
    coarse = rng.random((cells, cells), dtype=np.float32)
    return cv2.resize(coarse, (size, size), interpolation=cv2.INTER_CUBIC)


def make_background_texture(size: int = 2048, seed: int = 7) -> np.ndarray:
    """A richly textured planar background (grayscale, 0-255)."""
    rng = np.random.default_rng(seed)
    texture = np.full((size, size), 200.0, dtype=np.float32)

    # Large soft color blobs for low-frequency structure.
    blobs = _smooth_random_grid(rng, size, 12)
    texture += (blobs - 0.5) * 70.0

    # Checker patches at two scales (strong corners for feature tracking).
    for block, shade in ((128, 25.0), (64, 18.0)):
        yy, xx = np.mgrid[0:size, 0:size]
        checker = (((xx // block) + (yy // block)) % 2).astype(np.float32)
        texture += (checker - 0.5) * shade

    # Random rectangles and circles with anti-aliased edges.
    canvas = np.zeros_like(texture)
    for idx in range(90):
        p0 = rng.integers(0, size, 2)
        if idx % 3 == 0:
            radius = int(rng.integers(12, 70))
            shade = float(rng.integers(60, 240))
            cv2.circle(canvas, tuple(p0), radius, shade, -1, cv2.LINE_AA)
        else:
            p1 = p0 + rng.integers(20, 180, 2)
            shade = float(rng.integers(60, 240))
            cv2.rectangle(canvas, tuple(p0), tuple(np.minimum(p1, size - 1)), shade, -1)
    texture = texture * 0.8 + canvas * 0.2

    # Fine grain so every pixel carries trackable texture.
    texture += rng.normal(0.0, 6.0, texture.shape)
    return np.clip(texture, 0, 255).astype(np.uint8)


@dataclass
class ForegroundProp:
    """A small textured quad floating in front of the background plane."""

    center: np.ndarray  # world coordinates, meters
    width: float
    height: float
    texture: np.ndarray


def make_foreground_props(
    base_texture: np.ndarray, rng: np.random.Generator
) -> list[ForegroundProp]:
    """Three props at distinct depths, spanning the field of view."""
    props = []
    spec = [
        (np.array([-1.4, -0.9, 5.0]), 0.9, 1.2),
        (np.array([0.2, 0.35, 3.6]), 0.7, 1.0),
        (np.array([1.5, -0.5, 5.8]), 1.0, 1.3),
    ]
    for idx, (center, width, height) in enumerate(spec):
        tile = base_texture[
            200 + idx * 350 : 200 + idx * 350 + 300,
            250 + idx * 280 : 250 + idx * 280 + 260,
        ].copy()
        cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), 30, 6)
        props.append(ForegroundProp(center, width, height, tile))
    return props


def build_default_scene(intr: CameraIntrinsics, seed: int = 7):
    """Return (background texture, world plane function, props).

    The background is the fronto-parallel plane ``Z = 8 m`` in world frame.
    """
    texture = make_background_texture(seed=seed)
    rng = np.random.default_rng(seed + 100)
    props = make_foreground_props(texture, rng)
    plane_z = 8.0
    return texture, plane_z, props


def render_frame(
    intr: CameraIntrinsics,
    r_cw: np.ndarray,
    t_cw: np.ndarray,
    texture: np.ndarray,
    plane_z: float,
    props: list[ForegroundProp],
    rng: np.random.Generator,
) -> np.ndarray:
    """Render one grayscale frame for camera pose (R_cw, t_cw).

    ``r_cw`` rotates world points into the camera frame; ``t_cw`` is the
    world-to-camera translation.
    """
    h, w = intr.height, intr.width
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    rays_cam = np.stack(
        [
            (u.ravel() - intr.cx) / intr.fx,
            (v.ravel() - intr.cy) / intr.fy,
            np.ones(h * w),
        ],
        axis=0,
    )

    # Camera rays expressed in world frame: d_w = R_cw^T d_c.
    rays_world = r_cw.T @ rays_cam
    # Ray from camera center C = -R_cw^T t_cw intersects plane z = plane_z.
    cam_center = -r_cw.T @ t_cw
    denom = rays_world[2]
    safe = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    depth = (plane_z - cam_center[2]) / safe
    points_world = cam_center[:, None] + depth * rays_world

    tex_scale = texture.shape[0] / 20.0  # plane spans 20 m x 20 m
    map_x = (points_world[0] + 10.0) * tex_scale
    map_y = (points_world[1] + 10.0) * tex_scale
    frame = cv2.remap(
        texture,
        map_x.reshape(h, w).astype(np.float32),
        map_y.reshape(h, w).astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    ).astype(np.float32)

    # Overlay foreground props via their own depth-tested projection.
    frame = _render_props(frame, intr, r_cw, t_cw, props)

    # Sensor noise and slight vignetting, matching a real handheld camera.
    frame += rng.normal(0.0, 2.5, frame.shape)
    vign_x = (u - intr.cx) / intr.width
    vign_y = (v - intr.cy) / intr.height
    vignette = 1.0 - 0.18 * (vign_x**2 + vign_y**2)
    frame *= vignette
    return np.clip(frame, 0, 255).astype(np.uint8)


def _render_props(
    frame: np.ndarray,
    intr: CameraIntrinsics,
    r_cw: np.ndarray,
    t_cw: np.ndarray,
    props: list[ForegroundProp],
) -> np.ndarray:
    h, w = intr.height, intr.width
    depth_buffer = np.full((h, w), np.inf, dtype=np.float32)
    for prop in props:
        hw, hh = prop.width / 2.0, prop.height / 2.0
        corners_world = np.array(
            [
                prop.center + [-hw, -hh, 0.0],
                prop.center + [hw, -hh, 0.0],
                prop.center + [hw, hh, 0.0],
                prop.center + [-hw, hh, 0.0],
            ]
        )
        cam_pts = (r_cw @ corners_world.T).T + t_cw
        if np.any(cam_pts[:, 2] <= 0.2):
            continue
        proj = (intr.matrix @ cam_pts.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        if np.any(np.abs(proj) > 4 * max(h, w)):
            continue
        dst = proj.astype(np.float32)
        th, tw = prop.texture.shape
        src = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32)
        warp = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(
            prop.texture, warp, (w, h), flags=cv2.INTER_LINEAR
        ).astype(np.float32)
        warped_depth = cv2.warpPerspective(
            np.full((th, tw), cam_pts[:, 2].mean(), np.float32),
            warp,
            (w, h),
        )
        mask = (warped > 0) & (warped_depth < depth_buffer) & (warped_depth > 0)
        frame[mask] = warped[mask]
        depth_buffer[mask] = warped_depth[mask]
    return frame
