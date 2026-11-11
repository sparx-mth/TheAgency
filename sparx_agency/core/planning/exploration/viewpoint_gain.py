"""Expected useful area a camera would reveal from a viewpoint: the MapEx scorer on the observed map, with our sensor.

MapEx (Kim, Ho et al., arXiv 2409.15590) scores a candidate viewpoint by
raycasting from it, counting the unknown area the rays reach, and dividing
by the distance to get there; on the observed map alone (no learned map
prediction) that score was competitive in the paper's ablation. This module
is that scorer with three corrections for the ObjectNav camera:

* **A camera, not a LiDAR.** A frame is a cone of ``2 * half_fov`` (79
  degrees on the benchmark), and a look-around is a sweep of frames, so
  the gain of a viewpoint is a *profile over bearing*: one expected area
  per ray, from which the gain of any sector -- one frame at a heading, the
  sliver one more turn adds, the whole circle -- is a sum over bearings
  (:meth:`ExpectedView.profile`, :meth:`ExpectedView.sector_gain`,
  :meth:`ExpectedView.frame_gains`).
* **Useful range, not sensor range.** A ray is followed to
  :attr:`SensorModel.detection_range_m` -- the range at which an object
  can actually be detected, not the depth sensor's 5 m clip -- and nothing
  under :attr:`SensorModel.blind_range_m` (the depth sensor's near clip)
  counts. "Useful information" is area a detector could find the target
  in, not any area.
* **A prior on the unknown, not transparency.** MapEx's observed-map
  raycast lets a ray run through unknown cells to full range, which values
  a look at an unmapped wall like a look through a door (on this method's
  232 recorded glances the uncapped promise had r = -0.17 with what the
  map gained; see ``object_search/exploration/path_glances.py``). Here an
  unknown cell is credited with the probability that the ray reaches it: a
  ray that has crossed ``d`` metres of unknown survives with probability
  ``exp(-d / unknown_free_run_m)`` -- MapEx's own probabilistic raycast
  (occupancy accumulated along the ray until a threshold) with a constant
  prior occupancy per unknown cell, which integrates to this exponential.
  ``unknown_free_run_m`` is the mean free run into unseen indoor space
  (2 m by default: a room is 3-5 m across, furniture stands in it); None
  restores MapEx's transparent unknown.

Area is integrated in polar coordinates -- each sample on a ray stands for
``r * delta_theta * delta_r`` of floor -- so a sector's gain is a sum over
its rays and no cell is counted twice by neighbouring rays. Cells the
caller passes as ``resolved`` (unknown the search has settled: looked
through without a return, enclosed pockets) block a ray as a wall does and
are not counted; cells passed as ``ignore`` (unknown floor under the
camera's near horizon at a pose the agent stood at, where anything standing
would have been mapped) neither block nor count.

Pure numpy, Python 3.8 syntax; the same vectorised fan as
:mod:`~sparx_agency.core.planning.exploration.view_gain`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from sparx_agency.core.planning.environment.occupancy_grid2d import OccupancyGrid2D
from sparx_agency.core.planning.exploration.view_gain import ViewCone

#: A ray whose survival has fallen under this is finished: nothing it could still reveal is worth a sample.
MIN_SURVIVAL = 0.02


@dataclass(frozen=True)
class SensorModel:
    """What the camera can learn from a viewpoint, in plan view.

    Attributes:
        half_fov_rad: Half the horizontal field of view.
        detection_range_m: How far a ray is followed and unknown counted:
            the range at which the detector in force finds an object, not
            the depth sensor's far clip (which may be longer: a region
            mapped from 4.5 m is known floor on which nothing was detected).
        blind_range_m: The depth sensor's near clip; nothing under it counts.
        unknown_free_run_m: Mean free run of a ray into unknown space (the
            survival prior above); None lets a ray run through unknown to
            the detection range, MapEx's transparent unknown.

    Raises:
        ValueError: On a field outside its domain.
    """

    half_fov_rad: float
    detection_range_m: float
    blind_range_m: float = 0.0
    unknown_free_run_m: Optional[float] = 2.0

    def __post_init__(self) -> None:
        if not (0.0 < self.half_fov_rad <= math.pi):
            raise ValueError("half_fov_rad must lie in (0, pi], got %r" % (self.half_fov_rad,))
        if not (math.isfinite(self.detection_range_m) and self.detection_range_m > 0.0):
            raise ValueError("detection_range_m must be positive and finite, got %r" % (self.detection_range_m,))
        if not (math.isfinite(self.blind_range_m) and 0.0 <= self.blind_range_m < self.detection_range_m):
            raise ValueError("blind_range_m must lie in [0, detection_range_m), got %r" % (self.blind_range_m,))
        if self.unknown_free_run_m is not None and (
                isinstance(self.unknown_free_run_m, bool) or not math.isfinite(self.unknown_free_run_m)
                or self.unknown_free_run_m <= 0.0):
            raise ValueError("unknown_free_run_m must be None or positive and finite, got %r" % (self.unknown_free_run_m,))

    @property
    def cone(self) -> ViewCone:
        """The same camera as a :class:`~view_gain.ViewCone`, for the rotation arithmetic."""
        return ViewCone(half_fov_rad=self.half_fov_rad, max_range_m=self.detection_range_m,
                        min_range_m=self.blind_range_m)

    def rotation_turns(self, turn_rad: float) -> int:
        """In-place turns of ``turn_rad`` that show this camera the whole circle (:meth:`ViewCone.rotation_turns`)."""
        return self.cone.rotation_turns(turn_rad)


def sensor_from_camera(width_px: int, fx_px: float, detection_range_m: float, blind_range_m: float = 0.0,
                       unknown_free_run_m: Optional[float] = 2.0) -> SensorModel:
    """A :class:`SensorModel` from a pinhole's width and focal length: ``half_fov = atan(width / (2 fx))``."""
    return SensorModel(half_fov_rad=float(math.atan(0.5 * float(width_px) / float(fx_px))),
                       detection_range_m=float(detection_range_m), blind_range_m=float(blind_range_m),
                       unknown_free_run_m=unknown_free_run_m)


class ExpectedView:
    """Rays over one grid: the expected useful area a camera at a pose would reveal, by bearing.

    Built once per map update (the masks are read off the grid in the
    constructor) and asked once per candidate pose; the per-call work is
    one vectorised fan over the full circle.

    Args:
        world: The occupancy grid; its values name FREE, OCCUPIED and UNKNOWN.
        sensor: The camera and the prior on the unknown.
        resolved: Optional ``(H, W)`` bool of unknown cells the search has
            settled; they block a ray and are not counted.
        ignore: Optional ``(H, W)`` bool of unknown cells that are as good
            as seen (the floor under the camera's near horizon at a past
            pose); they neither block nor count.
    """

    def __init__(self, world: OccupancyGrid2D, sensor: SensorModel, resolved: Optional[np.ndarray] = None,
                 ignore: Optional[np.ndarray] = None) -> None:
        grid = world.grid
        self._blocking = grid == world.values.occupied
        self._unknown = grid == world.values.unknown
        for name, mask in (("resolved", resolved), ("ignore", ignore)):
            if mask is None:
                continue
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != grid.shape:
                raise ValueError("%s %s is not shaped like the grid %s" % (name, mask.shape, grid.shape))
            if name == "resolved":
                self._blocking = self._blocking | mask
            self._unknown = self._unknown & ~mask
        self._height, self._width = grid.shape
        self._resolution = float(world.resolution)
        self._origin_x, self._origin_y = float(world.origin_x), float(world.origin_y)
        self.sensor = sensor
        # Neighbouring rays under half a cell apart at the detection range, so the far end of the
        # circle is not combed into stripes; the ray count is what the polar integral divides by.
        self.rays = max(8, int(math.ceil(2.0 * math.pi * sensor.detection_range_m / (0.5 * self._resolution))))
        self._delta_theta = 2.0 * math.pi / self.rays
        self.bearings = -math.pi + self._delta_theta * np.arange(self.rays, dtype=np.float64)
        self._delta_r = 0.5 * self._resolution
        self._ranges = np.arange(max(sensor.blind_range_m, self._delta_r), sensor.detection_range_m + 1e-9,
                                 self._delta_r, dtype=np.float64)
        self._cos, self._sin = np.cos(self.bearings), np.sin(self.bearings)

    @property
    def resolution(self) -> float:
        return self._resolution

    def profile(self, xy: Tuple[float, float]) -> np.ndarray:
        """Expected useful area per ray, square metres, over :attr:`bearings` (all zero off the map).

        A ray runs through FREE cells, is finished by an OCCUPIED (or
        settled) cell or the edge of the map, and through UNKNOWN cells
        with the survival prior: each unknown sample is credited with
        ``survival * r * delta_theta * delta_r`` where ``survival`` is
        ``exp(-d / unknown_free_run_m)`` for the ``d`` metres of unknown
        the ray crossed before it (1 with the prior off), and the ray is
        finished once its survival falls under :data:`MIN_SURVIVAL`.
        """
        x, y = float(xy[0]), float(xy[1])
        out = np.zeros(self.rays, dtype=np.float64)
        if not (math.isfinite(x) and math.isfinite(y)) or not len(self._ranges):
            return out
        ranges = self._ranges
        gx = np.floor((x + np.outer(self._cos, ranges) - self._origin_x) / self._resolution).astype(np.int64)
        gy = np.floor((y + np.outer(self._sin, ranges) - self._origin_y) / self._resolution).astype(np.int64)
        inside = (gx >= 0) & (gx < self._width) & (gy >= 0) & (gy < self._height)
        gx_c, gy_c = np.clip(gx, 0, self._width - 1), np.clip(gy, 0, self._height - 1)
        blocked = self._blocking[gy_c, gx_c] | ~inside
        first = np.where(blocked.any(axis=1), blocked.argmax(axis=1), blocked.shape[1])
        visible = np.arange(blocked.shape[1])[None, :] < first[:, None]
        unknown = visible & self._unknown[gy_c, gx_c]
        if self.sensor.unknown_free_run_m is None:
            survival = np.ones(unknown.shape, dtype=np.float64)
        else:
            # Unknown crossed BEFORE each sample (exclusive cumulative sum): a cell's own passage does not
            # discount its own credit.
            crossed = (np.cumsum(unknown, axis=1) - unknown) * self._delta_r
            survival = np.exp(-crossed / float(self.sensor.unknown_free_run_m))
            survival[survival < MIN_SURVIVAL] = 0.0
        area = ranges[None, :] * self._delta_theta * self._delta_r
        out[:] = (unknown * survival * area).sum(axis=1)
        return out

    def circle_gain(self, xy: Tuple[float, float]) -> float:
        """Expected useful area a full look-around from ``xy`` reveals, square metres."""
        return float(self.profile(xy).sum())

    def sector_gain(self, xy: Tuple[float, float], centre_rad: float, half_angle_rad: float,
                    profile: Optional[np.ndarray] = None) -> float:
        """Expected useful area in the sector ``centre +- half_angle`` from ``xy`` (``half >= pi`` is the circle)."""
        if profile is None:
            profile = self.profile(xy)
        return float(profile[self.sector_mask(centre_rad, half_angle_rad)].sum())

    def sector_mask(self, centre_rad: float, half_angle_rad: float) -> np.ndarray:
        """``(rays,)`` bool: the bearings inside ``centre +- half_angle``."""
        half = float(half_angle_rad)
        if half >= math.pi - 1e-12:
            return np.ones(self.rays, dtype=bool)
        if half <= 0.0:
            return np.zeros(self.rays, dtype=bool)
        offset = (self.bearings - float(centre_rad) + math.pi) % (2.0 * math.pi) - math.pi
        return np.abs(offset) <= half + 1e-9

    def frame_gains(self, profile: np.ndarray, headings: Sequence[float]) -> np.ndarray:
        """Expected useful area of one frame at each of ``headings`` (the camera's cone), from a ray profile."""
        half = self.sensor.half_fov_rad
        return np.array([float(profile[self.sector_mask(float(h), half)].sum()) for h in headings], dtype=np.float64)


def arc_uncovered(covered: Sequence[Tuple[float, float]], centre_rad: float, half_angle_rad: float) -> float:
    """Radians of the sector ``centre +- half_angle`` not inside any of the ``covered`` arcs.

    An arc is ``(start, end)`` in radians, counter-clockwise from ``start``
    to ``end`` with ``0 <= end - start <= 2 pi``. Angles are compared on
    the circle; the sector may straddle ``+-pi``. Used by the look-around
    bookkeeping: which part of the next frame is new from this spot.
    """
    half = max(0.0, min(math.pi, float(half_angle_rad)))
    if half <= 0.0:
        return 0.0
    start, width = float(centre_rad) - half, 2.0 * half
    # Sample the sector finely and count what no arc covers: exact to a hundredth of a degree, and
    # free of the interval-splicing cases a closed form needs on the circle.
    samples = max(16, int(math.ceil(math.degrees(width) * 100)))
    thetas = start + (np.arange(samples) + 0.5) * (width / samples)
    free = np.ones(samples, dtype=bool)
    for a, b in covered:
        span = float(b) - float(a)
        if span >= 2.0 * math.pi - 1e-9:
            return 0.0
        rel = (thetas - float(a)) % (2.0 * math.pi)
        free &= ~(rel <= span + 1e-9)
    return float(free.sum()) * (width / samples)


def near_floor_mask(world: OccupancyGrid2D, poses: Sequence[Tuple[float, float, float]], radius_m: float,
                    half_fov_rad: float) -> np.ndarray:
    """``(H, W)`` bool: the floor under the camera's near horizon in every frame of ``poses``.

    A level camera 0.88 m up sees the floor from about 1.4 m out, so the
    cells inside ``radius_m`` of a pose, within the frame's cone about its
    yaw, stay UNKNOWN on a floor-sampled map even though the frame looked
    straight across them -- anything standing there was mapped as an
    obstacle. They are as good as seen, and a viewpoint planner passes them
    as the ``ignore`` mask of :class:`ExpectedView`: not counted, not
    blocking. Cells behind the camera are not included: nothing looked there.

    Args:
        world: The occupancy grid (for the geometry).
        poses: ``(x, y, yaw)`` of every frame, world metres and radians.
        radius_m: The near horizon.
        half_fov_rad: Half the horizontal field of view.
    """
    h, w = world.grid.shape
    mask = np.zeros((h, w), dtype=bool)
    n = int(math.ceil(float(radius_m) / float(world.resolution)))
    if n <= 0:
        return mask
    ys, xs = np.mgrid[-n:n + 1, -n:n + 1]
    disc = (xs * xs + ys * ys) <= n * n
    angles = np.arctan2(ys, xs)
    half = float(half_fov_rad)
    for x, y, yaw in poses:
        gx, gy = world.world_to_grid(float(x), float(y))
        if not (-n <= gx < w + n and -n <= gy < h + n):
            continue
        offset = (angles - float(yaw) + math.pi) % (2.0 * math.pi) - math.pi
        window = disc & (np.abs(offset) <= half)
        window[n, n] = True
        y0, y1, x0, x1 = max(0, gy - n), min(h, gy + n + 1), max(0, gx - n), min(w, gx + n + 1)
        mask[y0:y1, x0:x1] |= window[y0 - (gy - n):y1 - (gy - n), x0 - (gx - n):x1 - (gx - n)]
    return mask
