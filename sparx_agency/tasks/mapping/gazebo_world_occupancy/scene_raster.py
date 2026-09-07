"""Place a scene's geometry in the world and rasterise it into one grid.

The join between :mod:`sdf_scene` -- which says what shapes exist and where --
and ``core.mapping.geometry_raster``, which draws triangles. Everything here is
bookkeeping: fetch or tessellate each instance's mesh, apply its 4x4 transform,
and hand the triangles to the rasteriser.

The map's extent is measured from the geometry itself in a first pass rather
than being asked for on the command line. A hand-typed extent is one more thing
that can silently disagree with the world, and the world already knows how big
it is.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence, Tuple

import numpy as np

from sparx_agency.core.mapping.geometry_raster import GridSpec, rasterise_mesh_slab
from sparx_agency.tasks.mapping.gazebo_world_occupancy import primitives
from sparx_agency.tasks.mapping.gazebo_world_occupancy.mesh_cache import load_mesh
from sparx_agency.tasks.mapping.gazebo_world_occupancy.sdf_scene import (
    BOX_KIND,
    CYLINDER_KIND,
    GeometryInstance,
    MESH_KIND,
    SPHERE_KIND,
)


@dataclass(frozen=True)
class SceneExtent:
    """The XY bounding box of every instance that reaches the height band.

    Note what that is *not*: it is not the extent of the geometry inside the
    band. An instance qualifies as a whole, and then its whole XY box is taken,
    parts far above and below the band included -- so a roof overhang or a
    floor slab on a model that also has a wall in the band widens the map.

    Attributes:
        min_x: Minimum world x, metres, margin already applied.
        min_y: Minimum world y, metres.
        max_x: Maximum world x, metres.
        max_y: Maximum world y, metres.
        instance_count: Instances that reach the band.
        triangle_count: Triangles those instances hold.
    """

    min_x: float
    min_y: float
    max_x: float
    max_y: float
    instance_count: int
    triangle_count: int


def world_triangles(instance: GeometryInstance) -> Tuple[np.ndarray, np.ndarray]:
    """Return one instance's mesh, placed in world coordinates.

    Args:
        instance: The placed shape.

    Returns:
        ``(vertices, faces)`` with vertices in world metres.

    Raises:
        ValueError: If the instance's kind is not supported.
    """
    if instance.kind == MESH_KIND:
        vertices, faces = load_mesh(instance.mesh_path, tuple(instance.scale))
    elif instance.kind == BOX_KIND:
        vertices, faces = primitives.box_mesh(instance.size)
    elif instance.kind == CYLINDER_KIND:
        vertices, faces = primitives.cylinder_mesh(instance.radius, instance.length)
    elif instance.kind == SPHERE_KIND:
        vertices, faces = primitives.sphere_mesh(instance.radius)
    else:
        raise ValueError("unsupported geometry kind %r" % (instance.kind,))

    transform = np.asarray(instance.transform, dtype=np.float64)
    placed = vertices.dot(transform[:3, :3].T) + transform[:3, 3]
    return placed, faces


def iter_world_meshes(
    instances: Iterable[GeometryInstance],
) -> Iterator[Tuple[GeometryInstance, np.ndarray, np.ndarray]]:
    """Yield ``(instance, world_vertices, faces)`` for each instance."""
    for instance in instances:
        vertices, faces = world_triangles(instance)
        yield instance, vertices, faces


def measure_extent(
    instances: Sequence[GeometryInstance],
    z_min: float,
    z_max: float,
    margin: float = 0.0,
) -> SceneExtent:
    """Measure the XY bounding box of the instances that reach the height band.

    Instances entirely above or below the band are excluded, which is also how
    the world's deleted props -- the ones parked at ``z = -7.6e+08`` -- drop
    out without needing a special case. An instance that *does* reach the band
    then contributes all of its vertices, not only the ones inside it: this
    pass deliberately does not clip to the slab, which would cost a second
    clipping pass over every triangle and can only ever shrink the map. An
    over-large map is free space at the edge; a smaller one moves the origin,
    and the origin is what every recorded flight is indexed against.

    Args:
        instances: The scene's placed geometry.
        z_min: Lower band boundary, metres.
        z_max: Upper band boundary, metres.
        margin: Free space added on every side, metres.

    Returns:
        The extent.

    Raises:
        ValueError: If nothing reaches the band.
    """
    lower = np.array([np.inf, np.inf])
    upper = np.array([-np.inf, -np.inf])
    kept = 0
    triangles = 0
    for instance, vertices, faces in iter_world_meshes(instances):
        if vertices[:, 2].max() < z_min or vertices[:, 2].min() > z_max:
            continue
        kept += 1
        triangles += int(faces.shape[0])
        lower = np.minimum(lower, vertices[:, :2].min(axis=0))
        upper = np.maximum(upper, vertices[:, :2].max(axis=0))

    if kept == 0:
        raise ValueError(
            "no geometry between z=%.3f and z=%.3f; check the world, the search "
            "paths and the height band" % (z_min, z_max)
        )
    return SceneExtent(
        min_x=float(lower[0]) - margin,
        min_y=float(lower[1]) - margin,
        max_x=float(upper[0]) + margin,
        max_y=float(upper[1]) + margin,
        instance_count=kept,
        triangle_count=triangles,
    )


def grid_spec_for(extent: SceneExtent, resolution: float) -> GridSpec:
    """Build the raster geometry covering an extent.

    The origin is snapped down to a multiple of the resolution so that the same
    world always produces the same grid, whatever margin was asked for, and
    rounded so it writes as ``-13.6`` rather than ``-13.600000000000001``.

    Args:
        extent: The measured extent.
        resolution: Metres per cell.

    Returns:
        The :class:`GridSpec`.
    """
    origin_x = _snap_down(extent.min_x, resolution)
    origin_y = _snap_down(extent.min_y, resolution)
    # The far edge must be *inside* the grid, so count the cell the maximum
    # falls in and add it. With ceil(), an extent that divides exactly puts the
    # grid's right edge on max_x, and a wall standing there lands at column
    # `width` and is dropped -- a hole in the map exactly where the outermost
    # wall is. The default margin usually hides it; --margin 0 does not.
    width = int(np.floor((extent.max_x - origin_x) / resolution)) + 1
    height = int(np.floor((extent.max_y - origin_y) / resolution)) + 1
    return GridSpec(
        resolution=float(resolution),
        origin_x=origin_x,
        origin_y=origin_y,
        width=max(1, width),
        height=max(1, height),
    )


def rasterise_scene(
    instances: Sequence[GeometryInstance],
    spec: GridSpec,
    z_min: float,
    z_max: float,
) -> Tuple[np.ndarray, int]:
    """Draw every instance's slab slice into one shared occupancy grid.

    Args:
        instances: The scene's placed geometry.
        spec: The raster geometry, normally from :func:`grid_spec_for`.
        z_min: Lower band boundary, metres.
        z_max: Upper band boundary, metres.

    Returns:
        ``(grid, triangle_count)`` -- a boolean ``(height, width)`` array with
        row 0 at minimum y, and how many triangles were offered to the
        rasteriser.
    """
    grid = spec.empty()
    triangles = 0
    for _instance, vertices, faces in iter_world_meshes(instances):
        if vertices[:, 2].max() < z_min or vertices[:, 2].min() > z_max:
            continue
        triangles += int(faces.shape[0])
        rasterise_mesh_slab(
            vertices,
            faces,
            z_min=z_min,
            z_max=z_max,
            resolution=spec.resolution,
            origin_x=spec.origin_x,
            origin_y=spec.origin_y,
            width=spec.width,
            height=spec.height,
            out=grid,
        )
    return grid, triangles


def _snap_down(value: float, resolution: float) -> float:
    """Round ``value`` down to a clean multiple of ``resolution``."""
    return float(np.round(np.floor(value / resolution) * resolution, 9))
