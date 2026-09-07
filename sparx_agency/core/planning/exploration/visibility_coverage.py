"""How much of a surveyed building the camera has actually looked at.

An exploration order -- *go into every room, see what is in it, come back out*
-- has no state at which it is satisfied, so a flight under one can only be
judged on how much of the building it saw. This turns that into a number: sweep
the camera's horizontal field of view across the ground-truth occupancy map from
every pose the aircraft holds, mark the free cells the rays reach before a wall
stops them, and divide by the building's floor.

**Seen, not visited.** The distinction is the whole point and it is why this is
not :mod:`sparx_agency.tools.campaign_monitor.coverage`, which asks a different
question for a different purpose. That one marks the cell the aircraft *is in*,
in one-metre cells, aggregated over a whole campaign -- "has this been flown".
Here a room looked into from its doorway counts and a corridor flown down blind
would not, because the instruction is about looking.

**The denominator is the building, not the map.** A map computed from world
geometry has free space on both sides of the outer wall and inside every closed
cupboard, and counting either puts a ceiling on the percentage that looks like a
plateau. :func:`largest_enclosed_region` picks out the floor.

**A ray is point-sampled, not swept.** Each bearing is tested every half cell,
so a barrier one cell thick that is only *corner*-connected -- a perfect
diagonal -- lets a ray through. Real walls in a rasterised building are
edge-connected and stop it; this is worth knowing before trusting the number on
a synthetic map made of diagonals.

**It is a plan-view proxy and nothing more.** The wedge is horizontal: the
camera's vertical field of view, the aircraft's altitude and the height of what
it is looking over are not modelled, and a map built over a height band counts a
desk as an occluder because a desk *is* one at 5 cm resolution in plan. Read the
number as "the share of the floor that has passed through the camera's bearing
and range window with nothing between", which is a good deal more than nothing
and a good deal less than a semantic guarantee that the room was understood.

Pure numpy, no scipy, no OpenCV, Python 3.8 syntax.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from sparx_agency.core.planning.environment.grid_regions import (
    largest_enclosed_region,
)
from sparx_agency.core.planning.environment.occupancy_grid2d import OccupancyGrid2D


@dataclass(frozen=True)
class SensorCone:
    """The horizontal wedge a forward camera can see, in plan view.

    Attributes:
        half_fov_rad: Half the horizontal field of view. For a symmetric
            pinhole this is ``atan(width / (2 fx))``.
        max_range_m: How far a return is trusted. Use the depth sensor's far
            clip -- past it the pixels carry no measurement, and treating them
            as clear marks free space straight through walls.
        forward_offset_m: How far ahead of the body origin the camera sits,
            body FLU. Not negligible: the camera carves space outward from
            itself, so the body origin is the one place it can never observe.
    """

    half_fov_rad: float
    max_range_m: float
    forward_offset_m: float = 0.0


class VisibilityCoverage:
    """Accumulate the floor a forward camera has looked at, over a flight.

    Stateful and cheap to update -- one :meth:`observe` is a couple of
    milliseconds on a 25 x 59 m building at 5 cm -- so it can be driven from a
    recorder's frame timer without going anywhere near the flight loop.

    Args:
        grid: The surveyed map. Its FREE cells are the candidate floor and its
            OCCUPIED cells are the occluders; UNKNOWN counts as an occluder,
            because an unsurveyed cell is not a cell anything has seen through.
        cone: What the camera can see.
        countable: Optional ``(H, W)`` boolean naming the cells that make up the
            denominator. Defaults to the map's largest enclosed free region --
            the building's floor.

    Raises:
        ValueError: ``countable`` does not match the grid, or the map has no
            enclosed free region for the default to find.
    """

    def __init__(self, grid, cone, countable=None):
        # type: (OccupancyGrid2D, SensorCone, Optional[np.ndarray]) -> None
        cells = grid.grid
        values = grid.values
        self._free = cells == values.free
        # Anything that is not known-free stops a ray. UNKNOWN included: a cell
        # nothing has surveyed is not a cell anything can see through.
        self._blocking = ~self._free
        if countable is None:
            countable = largest_enclosed_region(self._free, connectivity=4)
            if countable is None:
                raise ValueError(
                    "the map has no enclosed free region, so there is no building "
                    "to measure coverage against -- every free component runs off "
                    "the edge of the grid")
        countable = np.asarray(countable, dtype=bool)
        if countable.shape != self._free.shape:
            raise ValueError("countable %r does not match the grid %r"
                             % (countable.shape, self._free.shape))
        self._countable = countable & self._free
        self._seen = np.zeros(self._free.shape, dtype=bool)
        self._cells_seen = 0
        self._cells_total = int(self._countable.sum())

        self._resolution = float(grid.resolution)
        self._origin_x = float(grid.origin_x)
        self._origin_y = float(grid.origin_y)
        self._height, self._width = self._free.shape
        self.cone = cone

        # Angular sampling fine enough that two neighbouring rays are less than
        # half a cell apart AT MAX RANGE. Coarser than that and the far end of
        # the fan combs the floor into stripes, which reads as an honest-looking
        # coverage number that is simply an artefact of the ray count.
        span = 2.0 * float(cone.half_fov_rad)
        arc = max(span * float(cone.max_range_m), self._resolution)
        rays = int(np.ceil(arc / (0.5 * self._resolution))) + 1
        self._rays = int(min(max(rays, 3), 4096))
        self._bearings = np.linspace(-cone.half_fov_rad, cone.half_fov_rad,
                                     self._rays)
        step = 0.5 * self._resolution
        self._ranges = np.arange(
            0.0, float(cone.max_range_m) + step, step, dtype=np.float64)

    # -- geometry ---------------------------------------------------------

    @property
    def countable_mask(self):
        # type: () -> np.ndarray
        """``(H, W)`` boolean: the cells that make up the denominator."""
        return self._countable

    @property
    def seen_mask(self):
        # type: () -> np.ndarray
        """``(H, W)`` boolean: the countable cells looked at so far."""
        return self._seen

    def restore_seen(self, seen):
        # type: (np.ndarray) -> None
        """Adopt a seen-mask from an earlier flight over the same building.

        Intersected with the denominator on the way in, so a mask saved against
        a slightly different countable region cannot inflate the percentage --
        the number is always cells of THIS building's floor.

        Raises:
            ValueError: The mask is not this grid's shape.
        """
        seen = np.asarray(seen, dtype=bool)
        if seen.shape != self._seen.shape:
            raise ValueError("seen mask %r does not match the grid %r"
                             % (seen.shape, self._seen.shape))
        self._seen = seen & self._countable
        self._cells_seen = int(self._seen.sum())

    def contains(self, x, y):
        # type: (float, float) -> bool
        """Is a world point inside the region being measured?

        The one sanity check worth running at start-up: an aircraft outside the
        countable region will fly a whole recording and report 0 %, and that
        looks exactly like a broken tracker.
        """
        cell = self._cell(x, y)
        if cell is None:
            return False
        gx, gy = cell
        return bool(self._countable[gy, gx])

    def cell_of(self, x, y):
        # type: (float, float) -> Optional[Tuple[int, int]]
        """``(col, row)`` for a world point, or None off the grid.

        The public face of :meth:`_cell`, for a caller that has a world point
        and wants to ask the seen mask about it -- "has the camera reached this
        object yet?". Returning None rather than clamping is deliberate: a
        point outside the building is not a point on its edge.
        """
        return self._cell(x, y)

    def _cell(self, x, y):
        # type: (float, float) -> Optional[Tuple[int, int]]
        """World point -> cell, the way :meth:`observe` indexes.

        Deliberately not ``OccupancyGrid2D.world_to_grid``, though it is
        arithmetically the same today. The invariant this has to hold is that
        :meth:`contains` agrees with the cells :meth:`observe` actually credits,
        and those come from the vectorised floor below -- a guard that agreed
        with the planners' convention instead of with the measurement it
        protects would warn on correct poses. The constructor also keeps only
        the derived masks and scalars, not the grid, so there is nothing to
        delegate to.
        """
        gx = int(np.floor((x - self._origin_x) / self._resolution))
        gy = int(np.floor((y - self._origin_y) / self._resolution))
        if 0 <= gx < self._width and 0 <= gy < self._height:
            return gx, gy
        return None

    # -- accumulation -----------------------------------------------------

    def observe(self, x, y, yaw):
        # type: (float, float, float) -> int
        """Sweep the cone from one pose and mark what it reaches.

        Vectorised on purpose: the fan is sampled as one ``(rays, ranges)``
        array and the first occluded sample on each ray is found with a single
        ``argmax``. The Bresenham walk in
        ``planners/common/grid_geometry_2d.line_cells`` is the exact answer for
        one segment and the wrong tool for five hundred of them at 10 Hz -- it
        is a Python loop per cell, which is three orders of magnitude slower
        than this and buys an exactness the metric does not need.

        Args:
            x: World x of the body origin, metres.
            y: World y of the body origin, metres.
            yaw: Body heading, radians CCW from +x.

        Returns:
            How many countable cells this pose added. Zero for a
            non-finite pose, a pose off the map, or a pose that saw only
            ground already covered.
        """
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(yaw)):
            return 0
        eye_x = float(x) + self.cone.forward_offset_m * np.cos(yaw)
        eye_y = float(y) + self.cone.forward_offset_m * np.sin(yaw)

        bearings = self._bearings + float(yaw)
        xs = eye_x + np.outer(np.cos(bearings), self._ranges)
        ys = eye_y + np.outer(np.sin(bearings), self._ranges)
        gx = np.floor((xs - self._origin_x) / self._resolution).astype(np.int64)
        gy = np.floor((ys - self._origin_y) / self._resolution).astype(np.int64)
        inside = ((gx >= 0) & (gx < self._width)
                  & (gy >= 0) & (gy < self._height))
        gx_c = np.clip(gx, 0, self._width - 1)
        gy_c = np.clip(gy, 0, self._height - 1)

        # A sample outside the map blocks as surely as a wall does: past the
        # edge there is no evidence either way, and rays that run on regardless
        # would wrap coverage onto the far side of the building.
        blocked = self._blocking[gy_c, gx_c] | ~inside
        any_block = blocked.any(axis=1)
        first = np.where(any_block, blocked.argmax(axis=1), blocked.shape[1])
        visible = np.arange(blocked.shape[1])[None, :] < first[:, None]

        # Only countable cells are ever written, so `_seen` stays a subset of
        # the denominator by construction rather than by a mask-off afterwards
        # -- and the running count can then be one array sum per pose instead
        # of two.
        rows, cols = gy_c[visible], gx_c[visible]
        keep = self._countable[rows, cols]
        self._seen[rows[keep], cols[keep]] = True
        after = int(self._seen.sum())
        gained = after - self._cells_seen
        self._cells_seen = after
        return gained

    # -- the number -------------------------------------------------------

    @property
    def cells_total(self):
        # type: () -> int
        """Countable cells in the building."""
        return self._cells_total

    @property
    def cells_seen(self):
        # type: () -> int
        """Countable cells looked at so far."""
        return self._cells_seen

    @property
    def fraction_seen(self):
        # type: () -> float
        """Share of the building's floor looked at, 0..1."""
        total = self.cells_total
        return (self.cells_seen / float(total)) if total else 0.0

    @property
    def cell_area_m2(self):
        # type: () -> float
        """Area of one cell, square metres."""
        return self._resolution * self._resolution

    @property
    def area_total_m2(self):
        # type: () -> float
        """The building's floor, square metres."""
        return self.cells_total * self.cell_area_m2

    @property
    def area_seen_m2(self):
        # type: () -> float
        """Floor looked at so far, square metres."""
        return self.cells_seen * self.cell_area_m2

    def summary(self):
        # type: () -> str
        """One line for a log, e.g. ``23.4% seen (267 of 1140 m2)``."""
        return "%.1f%% seen (%.0f of %.0f m2)" % (
            100.0 * self.fraction_seen, self.area_seen_m2, self.area_total_m2)


def cone_from_intrinsics(width, fx, max_range_m, forward_offset_m=0.0):
    # type: (int, float, float, float) -> SensorCone
    """A :class:`SensorCone` from a pinhole's width and focal length.

    Args:
        width: Image width, pixels.
        fx: Horizontal focal length, pixels.
        max_range_m: The depth sensor's far clip, metres.
        forward_offset_m: Camera position ahead of the body origin, metres.

    Returns:
        The cone. ``half_fov_rad`` is ``atan(width / (2 fx))``.
    """
    return SensorCone(
        half_fov_rad=float(np.arctan(0.5 * float(width) / float(fx))),
        max_range_m=float(max_range_m),
        forward_offset_m=float(forward_offset_m))
