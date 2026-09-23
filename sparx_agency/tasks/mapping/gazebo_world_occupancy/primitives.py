"""Tessellate SDF primitive shapes into triangle meshes.

The rasteriser in ``core/mapping/geometry_raster`` speaks only triangles, which
keeps it free of any mesh library. SDF worlds are mostly meshes but a link may
declare a ``<box>``, ``<cylinder>`` or ``<sphere>`` directly -- a shelf, a
bollard, a pillar -- and dropping those would leave holes in the map.

The tessellations are deliberately inscribed-free: a cylinder's facets are
pushed out to the circumscribed radius, so the polygon *contains* the true
circle rather than cutting inside it. A ground-truth obstacle map should never
be smaller than the obstacle.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

CYLINDER_SEGMENTS = 32
SPHERE_RINGS = 16
SPHERE_SEGMENTS = 32

_BOX_FACES = np.array(
    [
        [0, 2, 1], [0, 3, 2],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6],
        [1, 2, 6], [1, 6, 5],
        [3, 0, 4], [3, 4, 7],
    ],
    dtype=np.int32,
)


def box_mesh(size) -> Tuple[np.ndarray, np.ndarray]:
    """Tessellate an SDF ``<box>``, centred on its link origin.

    Args:
        size: ``(3,)`` full extents in metres.

    Returns:
        ``(vertices, faces)``.

    Raises:
        ValueError: If ``size`` is not three numbers.
    """
    extents = np.asarray(size, dtype=np.float64).reshape(-1)
    if extents.shape != (3,):
        raise ValueError("box size must have 3 components, got %r" % (size,))
    half = extents / 2.0
    signs = np.array(
        [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ],
        dtype=np.float64,
    )
    return signs * half, _BOX_FACES.copy()


def cylinder_mesh(
    radius: float, length: float, segments: int = CYLINDER_SEGMENTS
) -> Tuple[np.ndarray, np.ndarray]:
    """Tessellate an SDF ``<cylinder>``, axis along +z, centred on the origin.

    Args:
        radius: Cylinder radius in metres.
        length: Axial length in metres.
        segments: Facets around the circumference.

    Returns:
        ``(vertices, faces)``. The facets circumscribe the true circle.

    Raises:
        ValueError: If the radius or length is not positive.
    """
    if not radius > 0.0 or not length > 0.0:
        raise ValueError("cylinder needs positive radius and length")
    count = max(3, int(segments))
    outer = float(radius) / math.cos(math.pi / count)
    angles = np.arange(count) * (2.0 * math.pi / count)
    ring = np.stack([outer * np.cos(angles), outer * np.sin(angles)], axis=1)
    half = float(length) / 2.0

    lower = np.hstack([ring, np.full((count, 1), -half)])
    upper = np.hstack([ring, np.full((count, 1), half)])
    centres = np.array([[0.0, 0.0, -half], [0.0, 0.0, half]])
    vertices = np.vstack([lower, upper, centres])

    index = np.arange(count)
    following = (index + 1) % count
    side = np.concatenate(
        [
            np.stack([index, following, following + count], axis=1),
            np.stack([index, following + count, index + count], axis=1),
        ]
    )
    bottom = np.stack([np.full(count, 2 * count), following, index], axis=1)
    top = np.stack([np.full(count, 2 * count + 1), index + count, following + count],
                   axis=1)
    faces = np.vstack([side, bottom, top]).astype(np.int32)
    return vertices, faces


def sphere_mesh(
    radius: float, rings: int = SPHERE_RINGS, segments: int = SPHERE_SEGMENTS
) -> Tuple[np.ndarray, np.ndarray]:
    """Tessellate an SDF ``<sphere>`` centred on the origin.

    Args:
        radius: Sphere radius in metres.
        rings: Latitude bands.
        segments: Longitude facets.

    Returns:
        ``(vertices, faces)``. The hull circumscribes the true sphere.

    Raises:
        ValueError: If the radius is not positive.
    """
    if not radius > 0.0:
        raise ValueError("sphere needs a positive radius")
    ring_count = max(2, int(rings))
    segment_count = max(3, int(segments))
    # Push out by whichever direction cuts deepest. Scaling by the longitude
    # half-angle alone circumscribes the equator and leaves every latitude
    # band inscribed -- the two agree only when rings == segments / 2, which
    # the defaults happen to satisfy and a caller passing its own numbers does
    # not. A ground-truth obstacle is never allowed to come out smaller.
    half_angle = max(math.pi / segment_count, math.pi / (2 * ring_count))
    outer = float(radius) / math.cos(half_angle)

    latitude = np.linspace(0.0, math.pi, ring_count + 1)
    longitude = np.arange(segment_count) * (2.0 * math.pi / segment_count)
    lat_grid, lon_grid = np.meshgrid(latitude, longitude, indexing="ij")
    vertices = outer * np.stack(
        [
            np.sin(lat_grid) * np.cos(lon_grid),
            np.sin(lat_grid) * np.sin(lon_grid),
            np.cos(lat_grid),
        ],
        axis=-1,
    ).reshape(-1, 3)

    row = np.arange(ring_count)[:, None]
    column = np.arange(segment_count)[None, :]
    here = row * segment_count + column
    right = row * segment_count + (column + 1) % segment_count
    below = here + segment_count
    below_right = right + segment_count
    faces = np.concatenate(
        [
            np.stack([here, right, below_right], axis=-1).reshape(-1, 3),
            np.stack([here, below_right, below], axis=-1).reshape(-1, 3),
        ]
    ).astype(np.int32)
    return vertices, faces
