"""Where the camera has looked, and the unknown it looked at without resolving it.

The map is what the depth camera saw, and the frontier -- free beside
unknown -- is where exploration goes. Two kinds of unknown are not worth a
step, and a human never takes one toward them:

* **Looked-through unknown.** The agent stood within the camera's floor
  range of it, with nothing between, looked straight at it, and the map
  still says unknown: a balcony's railing with the garden below it, a
  window, a glass door, a hole in the mesh -- the depth returned nothing at
  floor height, so there is no floor there to walk onto and nothing behind
  it to walk to. The Hanson recording of 2026-10-05 walked to both ends of
  a balcony it had seen whole from its threshold (actions 25-51: two
  openings, two peeks) because the unknown beyond the railing read as an
  exit. Every frame, the camera's cone is cast over the map it has just
  updated; a ray runs through known free cells and ends at the first
  occupied or unknown one; an unknown cell it ends at inside the floor's
  visible band -- beyond the blind radius under the camera, within
  :attr:`SightSettings.far_m` -- is a cell the depth should have resolved
  and did not. Seen so from :attr:`SightSettings.min_looks` distinct poses
  it is **resolved**: not an exit, not an opening, not a glance's gain, not
  a room's frontier. A cell that later turns out to be floor (seen from
  another side) is known from then on, and the mark is void by itself.
* **Gaps** (since 2026-10-08, :mod:`frontier_gaps`). The shallow unknown
  behind a frontier -- a band that ends at known cells within a fraction of
  a metre in every direction, the slit between a sofa's back and the wall
  -- is settled by the gap prober through :meth:`SightLedger.settle_gaps`
  once the frontier has been judged a gap: the pocket rule misses it,
  because such a strip leaks through one-cell holes in the observed wall
  into the unknown outside the house and is never *enclosed*.
* **Pockets.** An unknown region enclosed by known cells -- the strip
  behind a bed against an observed wall, the island between the spawn
  point and the bed in front of it (the camera's blind radius leaves the
  floor under its own feet unseen) -- connects to nothing: whatever is in
  it is in it, and there is no room behind it. A connected component of
  unknown cells of at most :attr:`SightSettings.pocket_max_m2` that does
  not reach the edge of the map is a pocket, and resolved. The same
  recording spent actions 51-82 walking back to such an island (the O4
  "gap", valued at 0.01 and visited because it was near) and actions
  150-192 on the strip behind a bed.

:meth:`SightLedger.resolved` is the union, masked to what is still unknown;
:meth:`SightLedger.overlay` is the map with those cells written OCCUPIED
for the frontier logic alone -- the planner keeps the real map (unknown is
never inflated, occupied is; a resolved railing must not shrink the
balcony the agent may still stand on), and so does the display.

The ledger also keeps every pose the camera stood at on each storey
(:meth:`poses`), so the scan ledger can ask whether a room was stood in or
looked into (:func:`cone_seen`), independently of the room numbers the
watershed hands out and takes back.

The far limit is conservative on purpose. The depth image is sampled at
a stride, and the floor samples thin out with range: at ``depth_stride``
8 on a 640 x 480 frame from 0.88 m, successive floor rows are 0.09 m
apart at 2 m, 0.15 m at 2.5 m, 0.21 m at 3 m and 0.29 m at 3.5 m -- past
about 2.5 m an unknown cell between two seen rows is a sampling hole, not
a railing, and would be "looked through" wrongly. Up to about 2 m the
floor is sampled denser than the 0.1 m grid; between 2 and 2.5 m a hole
is at most one cell, which ``min_looks`` from two poses and the pocket
rule absorb.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import label as connected_components

from sparx_agency.core.planning.environment import OccupancyGrid2D

LOOKED_THROUGH, POCKET, GAP = "looked_through", "pocket", "gap"


@dataclass(frozen=True)
class SightSettings:
    """What counts as looked through, and what as a pocket. Distances in metres.

    Attributes:
        enabled: Keep the ledger and apply it to the frontier logic (the
            ablation is ``False``: every unknown cell is an exit again).
        min_looks: Distinct poses (binned by ``pose_bin_m`` and
            ``pose_bin_deg``) that must have looked through an unknown cell
            before it is resolved. One frame can be a flicker; two poses
            are a look.
        near_margin_m: Added to the geometric blind radius -- the floor
            under the camera's lower edge -- before a cell counts as inside
            the visible band.
        far_m: Farthest ground distance a looked-through cell is credited
            at (see the module docstring for why 2.5 m).
        depth_cells: How many consecutive unknown samples past the boundary a
            ray marks -- samples lie every half cell, so 3 is about a cell and
            a half of unknown behind the boundary: the free cell on the
            boundary is a frontier cell while ANY of its unknown neighbours is
            unresolved, so the mark is a band, not a line.
        pocket_max_m2: Largest enclosed unknown region that is a pocket. A
            toilet, a bed or a sofa does not fit in 3 m2 behind a wardrobe;
            a cup would, and this is the trade the setting makes.
        pose_bin_m: Positions within this of each other are one pose for
            ``min_looks``.
        pose_bin_deg: ... and headings within this.
    """

    enabled: bool = True
    min_looks: int = 2
    near_margin_m: float = 0.1
    far_m: float = 2.5
    depth_cells: int = 3
    pocket_max_m2: float = 3.0
    pose_bin_m: float = 0.2
    pose_bin_deg: float = 15.0

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("sight.enabled must be a bool")
        for name in ("min_looks", "depth_cells"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("sight.%s must be a positive integer" % name)
        for name in ("far_m", "pocket_max_m2", "pose_bin_m", "pose_bin_deg"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("sight.%s must be positive and finite" % name)
        if isinstance(self.near_margin_m, bool) or not math.isfinite(self.near_margin_m) or self.near_margin_m < 0:
            raise ValueError("sight.near_margin_m must be finite and non-negative")


@dataclass
class _FloorSight:
    """The ledger's state for one storey."""

    looked: np.ndarray             # uint8 -- distinct poses that looked through the cell while it was unknown
    last_key: np.ndarray           # int64 -- the pose bin that last counted, so one pose counts once
    poses: List[Tuple[int, float, float, float, float]]   # (step, x, y, yaw, pitch)
    cache: Dict[str, Tuple[int, object]]                  # name -> (step, value), per-step memo
    gaps: Optional[np.ndarray] = None                     # bool -- unknown behind a frontier judged a gap (frontier_gaps)


def floor_band(camera_height_m, pitch_rad, half_vfov_rad, min_depth_m, max_depth_m):
    """Ground distances ``(near, far)`` at which a camera pitched ``pitch_rad`` down sees the floor, or None.

    The image's bottom ray points ``pitch + half_vfov`` below the horizon
    and meets the floor at ``height / tan`` of that; the top ray points
    ``pitch - half_vfov`` below it (above, when negative, and then the far
    limit is open). Depth is measured along the optical axis, so the
    sensor's own near and far clips are converted to ground distance at
    this pitch before they are applied. None when the whole image is above
    the floor or the band is empty.
    """
    lower = float(pitch_rad) + float(half_vfov_rad)
    upper = float(pitch_rad) - float(half_vfov_rad)
    if lower <= 1e-6:
        return None
    near = camera_height_m / math.tan(lower)
    far = camera_height_m / math.tan(upper) if upper > 1e-6 else math.inf
    cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)
    if cos_p > 1e-6:
        # depth z of a floor point at ground distance d: z = d cos(p) + h sin(p)
        near = max(near, (float(min_depth_m) - camera_height_m * sin_p) / cos_p)
        far = min(far, (float(max_depth_m) - camera_height_m * sin_p) / cos_p)
    near = max(0.0, near)
    if not far > near:
        return None
    return near, far


def cone_seen(world, xy, yaw, half_fov_rad, max_range_m, min_range_m=0.0):
    """The FREE cells a camera at ``xy`` facing ``yaw`` reaches: rays through free cells, ended by anything else.

    Returns an ``(H, W)`` bool mask. The fan is vectorised as in
    :class:`~sparx_agency.core.planning.exploration.view_gain.UnknownView`:
    neighbouring rays under half a cell apart at ``max_range_m``, samples
    every half cell, the first non-free sample per ray by one ``argmax``.
    Unknown ends a ray as a wall does: a cell nothing has resolved is not
    a cell anything is seen through. Cells nearer than ``min_range_m`` are
    passed over but not credited.
    """
    grid = world.grid
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    x, y = float(xy[0]), float(xy[1])
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(yaw)):
        return seen
    res = float(world.resolution)
    half = min(math.pi, max(1e-3, float(half_fov_rad)))
    rays = int(math.ceil(2.0 * half * (float(max_range_m) / res) / 0.5)) + 1
    bearings = np.linspace(yaw - half, yaw + half, max(3, rays), endpoint=half < math.pi)
    step = 0.5 * res
    ranges = np.arange(step, float(max_range_m) + step, step, dtype=np.float64)
    xs = x + np.outer(np.cos(bearings), ranges)
    ys = y + np.outer(np.sin(bearings), ranges)
    gx = np.floor((xs - world.origin_x) / res).astype(np.int64)
    gy = np.floor((ys - world.origin_y) / res).astype(np.int64)
    inside = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
    gx_c, gy_c = np.clip(gx, 0, w - 1), np.clip(gy, 0, h - 1)
    blocking = (grid[gy_c, gx_c] != world.values.free) | ~inside
    any_block = blocking.any(axis=1)
    first = np.where(any_block, blocking.argmax(axis=1), blocking.shape[1])
    visible = (np.arange(blocking.shape[1])[None, :] < first[:, None]) & (ranges[None, :] >= float(min_range_m))
    seen[gy_c[visible], gx_c[visible]] = True
    return seen


class SightLedger:
    """Per-storey record of looked-through unknown, pockets and the camera's poses.

    Attributes:
        settings: :class:`SightSettings`.
        stats: Counters for the record: frames cast, cells marked, pockets found.
    """

    def __init__(self, policy, settings=None):
        self.policy = policy
        self.settings = settings or SightSettings()
        self._floors: Dict[int, _FloorSight] = {}
        self.stats = {"frames": 0, "cells_marked": 0, "frames_without_band": 0}

    # -- state ------------------------------------------------------------------
    def _floor(self, floor_id, shape) -> _FloorSight:
        state = self._floors.get(int(floor_id))
        if state is None or state.looked.shape != tuple(shape):
            state = _FloorSight(looked=np.zeros(shape, dtype=np.uint8), last_key=np.full(shape, -1, dtype=np.int64),
                                poses=[], cache={}, gaps=np.zeros(shape, dtype=bool))
            self._floors[int(floor_id)] = state
        return state

    def _floor_id(self, floor_id=None):
        if floor_id is not None:
            return int(floor_id)
        return int(getattr(getattr(self.policy, "mapping", None), "floor_id", 0))

    def poses(self, floor_id=None):
        """Every ``(step, x, y, yaw, pitch)`` the camera stood at on the storey, oldest first."""
        state = self._floors.get(self._floor_id(floor_id))
        return [] if state is None else list(state.poses)

    def walked_mask(self, world, floor_id=None, radius_m=0.0, include=None) -> np.ndarray:
        """``(H, W)`` bool: every cell within ``radius_m`` of a spot the agent stood at on the storey.

        The one fact about passability nothing on the map can outvote: the
        agent was physically there. Masks written over the world for
        planning (seen staircases, observed drops) are carved out here so
        the agent is never left standing inside "genuine" obstacles with no
        start for A* (Collierville 2026-10-08: 309 idle turns 0.6 m from the
        foot of a flight). ``include`` adds poses not yet in the ledger --
        the current one, which is recorded after the world is confined.
        """
        state = self._floors.get(self._floor_id(floor_id))
        points = [(float(x), float(y)) for _, x, y, _, _ in (state.poses if state is not None else [])]
        points += [(float(x), float(y)) for x, y in (include or ())]
        mask = np.zeros(world.grid.shape, dtype=bool)
        if not points:
            return mask
        cells = {world.world_to_grid(x, y) for x, y in points}
        n = int(math.ceil(float(radius_m) / world.resolution))
        h, w = mask.shape
        ys, xs = np.ogrid[-n:n + 1, -n:n + 1]
        disc = (xs * xs + ys * ys) <= n * n
        for gx, gy in cells:
            if not (-n <= gx < w + n and -n <= gy < h + n):
                continue
            y0, y1, x0, x1 = max(0, gy - n), min(h, gy + n + 1), max(0, gx - n), min(w, gx + n + 1)
            mask[y0:y1, x0:x1] |= disc[y0 - (gy - n):y1 - (gy - n), x0 - (gx - n):x1 - (gx - n)]
        return mask

    def _memo(self, state, name, step, compute):
        hit = state.cache.get(name)
        if hit is not None and hit[0] == step:
            return hit[1]
        value = compute()
        state.cache[name] = (step, value)
        return value

    # -- the look ----------------------------------------------------------------
    def observe(self, obs, world, floor_id=None):
        """Record the pose and cast this frame's cone over the map it has just updated. Returns cells marked."""
        s = self.settings
        floor = self._floor_id(floor_id)
        state = self._floor(floor, world.grid.shape)
        pose, camera = obs.pose, obs.camera
        pitch = float(getattr(pose, "camera_pitch", 0.0) or 0.0)
        state.poses.append((int(obs.step), float(pose.x), float(pose.y), float(pose.yaw), pitch))
        state.cache.clear()
        if not s.enabled:
            return 0
        self.stats["frames"] += 1
        k = camera.intrinsics
        half_vfov = math.atan(0.5 * float(k.height) / float(k.fy))
        band = floor_band(float(camera.height_m), pitch, half_vfov, float(camera.min_depth_m), float(camera.max_depth_m))
        if band is None:
            self.stats["frames_without_band"] += 1
            return 0
        near, far = band[0] + s.near_margin_m, min(band[1], s.far_m)
        if far <= near:
            self.stats["frames_without_band"] += 1
            return 0
        half_fov = math.atan(0.5 * float(k.width) / float(k.fx))
        marked = self._cast(state, world, (pose.x, pose.y), float(pose.yaw), half_fov, near, far,
                            key=self._pose_key(pose.x, pose.y, pose.yaw))
        self.stats["cells_marked"] += marked
        return marked

    def _pose_key(self, x, y, yaw):
        s = self.settings
        bx = int(math.floor(float(x) / s.pose_bin_m))
        by = int(math.floor(float(y) / s.pose_bin_m))
        bt = int(math.floor((float(yaw) % (2 * math.pi)) / math.radians(s.pose_bin_deg)))
        return (bx * 73856093) ^ (by * 19349663) ^ (bt * 83492791)

    def _cast(self, state, world, xy, yaw, half_fov, near, far, key):
        """Rays through free cells; the first unknown cell inside ``[near, far]`` -- and a few behind it -- is looked through."""
        s = self.settings
        grid = world.grid
        h, w = grid.shape
        res = float(world.resolution)
        rays = int(math.ceil(2.0 * half_fov * (far / res) / 0.5)) + 1
        bearings = np.linspace(yaw - half_fov, yaw + half_fov, max(3, rays))
        step = 0.5 * res
        ranges = np.arange(step, far + step, step, dtype=np.float64)
        xs = float(xy[0]) + np.outer(np.cos(bearings), ranges)
        ys = float(xy[1]) + np.outer(np.sin(bearings), ranges)
        gx = np.floor((xs - world.origin_x) / res).astype(np.int64)
        gy = np.floor((ys - world.origin_y) / res).astype(np.int64)
        inside = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
        gx_c, gy_c = np.clip(gx, 0, w - 1), np.clip(gy, 0, h - 1)
        cells = grid[gy_c, gx_c]
        unknown = (cells == world.values.unknown) & inside
        occupied = (cells == world.values.occupied) | ~inside
        in_band = ranges[None, :] >= near
        # Inside the blind radius an unknown cell is passed over (the camera looks
        # over it); from the band on, the first unknown cell is where the look ends.
        blocking = occupied | (unknown & in_band)
        n = blocking.shape[1]
        any_block = blocking.any(axis=1)
        first = np.where(any_block, blocking.argmax(axis=1), n)
        rows = np.arange(blocking.shape[0])
        hit = (first < n)
        hit[hit] &= unknown[rows[hit], first[hit]]
        if not hit.any():
            return 0
        flat = []
        run = hit.copy()
        for j in range(s.depth_cells):
            idx = first + j
            ok = run & (idx < n)
            ok[ok] &= unknown[rows[ok], idx[ok]]
            run = ok
            if not ok.any():
                break
            flat.append(gy_c[rows[ok], idx[ok]] * w + gx_c[rows[ok], idx[ok]])
        if not flat:
            return 0
        index = np.unique(np.concatenate(flat))
        looked, last_key = state.looked.reshape(-1), state.last_key.reshape(-1)
        fresh = last_key[index] != key
        index = index[fresh]
        if not len(index):
            return 0
        looked[index] = np.minimum(255, looked[index].astype(np.int64) + 1).astype(np.uint8)
        last_key[index] = key
        return int(len(index))

    # -- what is resolved ----------------------------------------------------------
    def looked_through(self, world, floor_id=None):
        """``(H, W)`` bool: unknown cells looked through from ``min_looks`` poses."""
        state = self._floor(self._floor_id(floor_id), world.grid.shape)
        if not self.settings.enabled:
            return np.zeros(world.grid.shape, dtype=bool)
        return (state.looked >= self.settings.min_looks) & (world.grid == world.values.unknown)

    def pockets(self, world, floor_id=None, step=None):
        """``(H, W)`` bool: unknown regions of at most ``pocket_max_m2`` enclosed by known cells."""
        state = self._floor(self._floor_id(floor_id), world.grid.shape)
        if not self.settings.enabled:
            return np.zeros(world.grid.shape, dtype=bool)
        return self._memo(state, "pockets", self._step(step), lambda: enclosed_pockets(world, self.settings.pocket_max_m2))

    def gaps(self, world, floor_id=None):
        """``(H, W)`` bool: unknown cells settled as the shallow unknown behind a gap frontier (``frontier_gaps``)."""
        state = self._floor(self._floor_id(floor_id), world.grid.shape)
        if not self.settings.enabled or state.gaps is None:
            return np.zeros(world.grid.shape, dtype=bool)
        return state.gaps & (world.grid == world.values.unknown)

    def settle_gaps(self, world, mask, floor_id=None) -> int:
        """Settle the unknown cells of ``mask`` as a gap's; returns how many were new. The per-step memo is dropped."""
        state = self._floor(self._floor_id(floor_id), world.grid.shape)
        cells = np.asarray(mask, dtype=bool) & (world.grid == world.values.unknown)
        fresh = int((cells & ~state.gaps).sum())
        if fresh:
            state.gaps |= cells
            for name in ("resolved", "overlay"):
                state.cache.pop(name, None)
            self.stats["gap_cells"] = self.stats.get("gap_cells", 0) + fresh
        return fresh

    def resolved(self, world, floor_id=None, step=None):
        """``(H, W)`` bool: the unknown cells the frontier logic should treat as settled."""
        if not self.settings.enabled:
            return None
        state = self._floor(self._floor_id(floor_id), world.grid.shape)
        return self._memo(state, "resolved", self._step(step),
                          lambda: (self.looked_through(world, floor_id) | self.pockets(world, floor_id, step)
                                   | self.gaps(world, floor_id)))

    def overlay(self, world, floor_id=None, step=None):
        """The map with every resolved cell written OCCUPIED -- for the frontier logic, never for the planner."""
        resolved = self.resolved(world, floor_id, step)
        if resolved is None or not resolved.any():
            return world
        state = self._floor(self._floor_id(floor_id), world.grid.shape)

        def build():
            data = world.grid.copy()
            data[resolved] = world.values.occupied
            return OccupancyGrid2D(data, world.params, values=world.values)
        return self._memo(state, "overlay", self._step(step), build)

    def _step(self, step):
        if step is not None:
            return int(step)
        state = self._floors.get(self._floor_id())
        return state.poses[-1][0] if state is not None and state.poses else -1

    # -- where the camera stood ----------------------------------------------------------
    def stood_in(self, world, mask, floor_id=None):
        """Whether any recorded pose on the storey lies inside ``mask``."""
        poses = self.poses(floor_id)
        if not poses:
            return False
        mask = np.asarray(mask, dtype=bool)
        xy = np.asarray([(p[1], p[2]) for p in poses], dtype=float)
        gx = np.floor((xy[:, 0] - world.origin_x) / world.resolution).astype(np.int64)
        gy = np.floor((xy[:, 1] - world.origin_y) / world.resolution).astype(np.int64)
        h, w = mask.shape
        inside = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
        return bool(mask[gy[inside], gx[inside]].any())

    def diagnostics(self):
        out = {"settings": asdict(self.settings), "stats": dict(self.stats), "floors": {}}
        for floor, state in self._floors.items():
            out["floors"][str(floor)] = {"poses": len(state.poses),
                                         "looked_cells": int((state.looked > 0).sum()),
                                         "looked_%d+" % self.settings.min_looks:
                                             int((state.looked >= self.settings.min_looks).sum()),
                                         "gap_cells": 0 if state.gaps is None else int(state.gaps.sum())}
        return out


def unknown_around(world, xy, radius_m):
    """The share of the cells within ``radius_m`` of ``xy`` that are UNKNOWN -- the blind disk under a camera that has not moved.

    The footing sweep's precondition: a circle at a shallow pitch maps the
    floor under the camera's blind radius, and is worth its actions only
    where that floor is still unknown; an agent that walked here already
    knows its footing, and a pathless lock there is not a mapping problem.
    """
    gx, gy = world.world_to_grid(float(xy[0]), float(xy[1]))
    h, w = world.grid.shape
    r = max(1, int(math.ceil(float(radius_m) / world.resolution)))
    x0, x1 = max(0, gx - r), min(w, gx + r + 1)
    y0, y1 = max(0, gy - r), min(h, gy + r + 1)
    if x0 >= x1 or y0 >= y1:
        return 0.0
    ys, xs = np.ogrid[y0:y1, x0:x1]
    disk = (xs - gx) ** 2 + (ys - gy) ** 2 <= r * r
    window = world.grid[y0:y1, x0:x1]
    total = int(disk.sum())
    if total == 0:
        return 0.0
    return float(((window == world.values.unknown) & disk).sum()) / total


def enclosed_pockets(world, max_area_m2):
    """``(H, W)`` bool: the unknown components of at most ``max_area_m2`` that do not touch the map's edge."""
    unknown = world.grid == world.values.unknown
    if not unknown.any():
        return np.zeros_like(unknown)
    labels, count = connected_components(unknown, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros_like(unknown)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    max_cells = int(math.floor(float(max_area_m2) / (world.resolution ** 2)))
    small = sizes <= max_cells
    small[0] = False
    border = np.unique(np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
    small[border] = False
    if not small.any():
        return np.zeros_like(unknown)
    return small[labels]
