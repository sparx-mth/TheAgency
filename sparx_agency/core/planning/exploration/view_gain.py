"""Unknown floor a camera would reveal from a pose: the gain of a glance, in plan view.

A route raises one question at every step of it: is it worth stopping here
to look left, right or all the way round before walking on? Turning in place
costs actions but no path length, so under SPL it is free; under the action
budget it is not, so the answer has to be a number -- how much unknown floor
the look would reveal that the walk itself will not -- against the actions
the look costs. This module computes the number.

**Optimistic about the unknown.** A ray from the camera runs through FREE
cells and through UNKNOWN cells alike and stops at an OCCUPIED one or at
the edge of the map: the unknown is what the glance is for, and a glance
toward a dark region should be valued by the whole region it could show,
not by its first cell. The count is therefore an upper bound on what will
be learned -- a wall standing in the unknown shortens every ray behind it
-- and that is the right side to err on when deciding whether to look at
all: a look that reveals less than hoped costs six actions; a doorway not
looked into costs a room.

**A sector, not a cone.** The camera sees a cone of ``2 * half_fov`` while
the body turns, so a glance to one side sweeps from the heading's cone
out to the side's: every bearing between is seen on the way. The gain of
"look left" is the unknown in the sector from ``heading - half_fov`` to
``heading + turn + half_fov``, not in the cone at ``heading + turn`` alone.

**Below the camera is blind.** The floor within ``min_range_m`` of the
body is under the camera's lower edge; unknown there is not a glance's to
reveal and is not counted.

Pure numpy, Python 3.8 syntax, the same vectorised fan as
:class:`~sparx_agency.core.planning.exploration.visibility_coverage.VisibilityCoverage`
-- one ``(rays, ranges)`` array per pose, the first blocking sample per ray
by one ``argmax``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from sparx_agency.core.planning.environment.occupancy_grid2d import OccupancyGrid2D


@dataclass(frozen=True)
class ViewCone:
    """What the camera sees in plan view.

    Attributes:
        half_fov_rad: Half the horizontal field of view.
        max_range_m: How far a ray is followed: the depth sensor's far clip,
            or less when a shorter horizon is wanted.
        min_range_m: The blind radius under the camera, not counted.
    """

    half_fov_rad: float
    max_range_m: float
    min_range_m: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 < self.half_fov_rad <= math.pi):
            raise ValueError("half_fov_rad must lie in (0, pi], got %r" % (self.half_fov_rad,))
        if not (math.isfinite(self.max_range_m) and self.max_range_m > 0.0):
            raise ValueError("max_range_m must be positive and finite, got %r" % (self.max_range_m,))
        if not (math.isfinite(self.min_range_m) and 0.0 <= self.min_range_m < self.max_range_m):
            raise ValueError("min_range_m must lie in [0, max_range_m), got %r" % (self.min_range_m,))


def cone_from_camera(width_px: int, fx_px: float, max_range_m: float, min_range_m: float = 0.0) -> ViewCone:
    """A :class:`ViewCone` from a pinhole's width and focal length: ``half_fov = atan(width / (2 fx))``."""
    return ViewCone(half_fov_rad=float(math.atan(0.5 * float(width_px) / float(fx_px))),
                    max_range_m=float(max_range_m), min_range_m=float(min_range_m))


class UnknownView:
    """Rays over one grid: which UNKNOWN cells a camera at a pose would reach.

    Built once per map update (the masks are read off the grid in the
    constructor) and asked many times -- once per candidate pose and
    sector -- so the per-call work is the fan alone.

    Args:
        world: The occupancy grid; its values name FREE, OCCUPIED and UNKNOWN.
        cone: The camera, for the range band and the ray density.
        resolved: Optional ``(H, W)`` bool of unknown cells the search has
            settled -- looked through without a depth return (a railing, a
            window), or enclosed pockets. They block a ray as a wall does
            and are not counted: a glance toward a window would otherwise
            be valued by the whole garden behind it.
    """

    def __init__(self, world: OccupancyGrid2D, cone: ViewCone, resolved: Optional[np.ndarray] = None) -> None:
        grid = world.grid
        self._blocking = grid == world.values.occupied
        self._unknown = grid == world.values.unknown
        if resolved is not None:
            settled = np.asarray(resolved, dtype=bool)
            if settled.shape != grid.shape:
                raise ValueError("resolved %s is not shaped like the grid %s" % (settled.shape, grid.shape))
            self._blocking = self._blocking | settled
            self._unknown = self._unknown & ~settled
        self._height, self._width = grid.shape
        self._resolution = float(world.resolution)
        self._origin_x = float(world.origin_x)
        self._origin_y = float(world.origin_y)
        self.cone = cone
        range_cells = cone.max_range_m / self._resolution
        # Neighbouring rays under half a cell apart at max range, so the far
        # end of a full circle is not combed into stripes.
        self._rays_per_rad = max(2.0, range_cells / 0.5)
        step = 0.5 * self._resolution
        self._ranges = np.arange(max(cone.min_range_m, step), cone.max_range_m + step, step, dtype=np.float64)

    @property
    def cell_area_m2(self) -> float:
        return self._resolution * self._resolution

    def unknown_in_sector(self, xy: Tuple[float, float], centre_rad: float, half_angle_rad: float) -> np.ndarray:
        """The UNKNOWN cells a camera at ``xy`` reaches anywhere in the sector ``centre +- half_angle``.

        ``half_angle_rad >= pi`` is the full circle. Returns an ``(H, W)``
        bool mask; empty off the map or for a non-finite pose.

        Args:
            xy: The camera's position, world metres.
            centre_rad: The sector's middle bearing, radians CCW from +x.
            half_angle_rad: Half the sector's angular width.
        """
        seen = np.zeros((self._height, self._width), dtype=bool)
        x, y = float(xy[0]), float(xy[1])
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(centre_rad)):
            return seen
        half = min(math.pi, max(0.0, float(half_angle_rad)))
        if half <= 0.0:
            return seen
        rays = int(math.ceil(2.0 * half * self._rays_per_rad)) + 1
        bearings = np.linspace(centre_rad - half, centre_rad + half, rays, endpoint=half < math.pi)
        xs = x + np.outer(np.cos(bearings), self._ranges)
        ys = y + np.outer(np.sin(bearings), self._ranges)
        gx = np.floor((xs - self._origin_x) / self._resolution).astype(np.int64)
        gy = np.floor((ys - self._origin_y) / self._resolution).astype(np.int64)
        inside = (gx >= 0) & (gx < self._width) & (gy >= 0) & (gy < self._height)
        gx_c = np.clip(gx, 0, self._width - 1)
        gy_c = np.clip(gy, 0, self._height - 1)
        blocked = self._blocking[gy_c, gx_c] | ~inside
        any_block = blocked.any(axis=1)
        first = np.where(any_block, blocked.argmax(axis=1), blocked.shape[1])
        visible = np.arange(blocked.shape[1])[None, :] < first[:, None]
        rows, cols = gy_c[visible], gx_c[visible]
        keep = self._unknown[rows, cols]
        seen[rows[keep], cols[keep]] = True
        return seen

    def unknown_ahead(self, xy: Tuple[float, float], yaw: float) -> np.ndarray:
        """The camera's own cone at ``yaw``: what a pose sees without turning."""
        return self.unknown_in_sector(xy, yaw, self.cone.half_fov_rad)


@dataclass(frozen=True)
class GlanceGain:
    """What one glance from one pose would reveal beyond what the walk reveals anyway.

    Attributes:
        left_m2: Unknown floor swept by turning ``side_rad`` to the left and back.
        right_m2: The same to the right.
        full_m2: Unknown floor swept by a full circle.
    """

    left_m2: float
    right_m2: float
    full_m2: float


def glance_gains(view: UnknownView, xy: Tuple[float, float], heading: float, side_rad: float,
                 already: Optional[np.ndarray] = None) -> GlanceGain:
    """The gains of a left, a right and a full glance from ``xy`` facing ``heading``.

    Args:
        view: The rays over the current map.
        xy: Where the glance would stand.
        heading: The body heading there (the route's direction at that point).
        side_rad: How far a side glance turns (``pi / 2`` for "look left").
        already: Optional ``(H, W)`` bool of unknown cells the walk reveals
            anyway (the forward cones along the route); subtracted from
            every gain so a glance is valued only by what is its own.

    Returns:
        The three gains in square metres.
    """
    half_fov = view.cone.half_fov_rad
    left = view.unknown_in_sector(xy, heading + 0.5 * side_rad, 0.5 * side_rad + half_fov)
    right = view.unknown_in_sector(xy, heading - 0.5 * side_rad, 0.5 * side_rad + half_fov)
    full = view.unknown_in_sector(xy, heading, math.pi)
    if already is not None:
        left &= ~already
        right &= ~already
        full &= ~already
    area = view.cell_area_m2
    return GlanceGain(left_m2=float(left.sum()) * area, right_m2=float(right.sum()) * area,
                      full_m2=float(full.sum()) * area)
