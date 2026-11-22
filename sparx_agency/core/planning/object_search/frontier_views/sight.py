"""What one frame shows of a set of map cells: which of them it sees, how well, and whether it sees PAST them.

The question a frame selector asks of every logged frame, answered in plan
view over the map as it stands NOW (a wall mapped after the frame was taken
still stood when it was taken), with the frame's own capture pose:

1. **In range.** The cell lies within ``max_range_m`` of the camera.
2. **In the picture.** The band of heights that matters at the cell
   (``band_m``: floor to door handle) is cut to the part of it inside the
   image -- exactly, as five linear constraints along the vertical segment in
   the camera's optical frame (in front of the camera, inside the left and
   right borders less a margin, inside the top and bottom rows). The share
   left is the view's vertical worth, so the pitch of the camera needs no
   rule of its own: a camera tilted down shows less of the band and counts
   for less, one tilted further shows none of it and does not count.
3. **Seen.** The sight line from the camera to the cell's centre crosses no
   OCCUPIED cell, and no more DEEP unknown than a sampling hole
   (``unknown_slack_m``). Three things a grid would otherwise call an
   obstruction hide nothing and are passed over:

   * the floor inside the camera's own blind radius, unknown wherever
     nothing else looked at it;
   * the **fringe** -- unknown cells touching known floor;
   * a **corner** clipped by no more than a cell (``corner_slack_cells``): a
     line a cell to either side of the direct one may be the clear one.

   The last two are what let a boundary be seen at all from where it was
   made. The edge of what a camera saw through a doorway runs along that
   camera's own sight lines: the line to a cell of it is tangent to the
   jamb and skims the unknown behind the jamb for its whole length. Read
   strictly, every such cell is hidden from the very frame that mapped it.
4. **Seen through.** The sight line, continued past the cell, runs straight
   into unknown the search has not settled. The picture then shows what
   lies beyond the boundary at that cell. Otherwise it shows the boundary
   and what stands beside it: the line runs on over known floor (a view
   along the boundary) or ends at a wall.

Why step 4 cannot be the rule. The map is what the depth camera saw, so the
boundary of the map is the edge of what every frame's depth resolved. A
frame with a clear line across a boundary cell and the far side of it
inside its depth range would have mapped the far side: the cell would not
be on the boundary. A frame therefore sees THROUGH a boundary only from
beyond its depth range (the colour image carries farther than the depth
does) or, rarely, where its depth failed. Nearer than that, the best a
frame can do is see the boundary itself -- the doorway, the jamb the unseen
part of the room is hidden behind, and the part of the room that is not
hidden. Both are evidence; a consumer weighs them
(:attr:`~sparx_agency.core.planning.object_search.frontier_views.settings.ViewSettings.at_weight`).

**Two-dimensional on purpose, and what that costs.** The map marks a cell
OCCUPIED for anything between 0.15 and 0.93 m above the floor, and the
camera stands at 0.88 m: it sees over a sofa the sight line here stops at.
A view lost that way is reported as occluded, not as absent, so a replay
can count how often it happens.

A seventh fan beside the six this tree already holds would have been one too
many; this is not a fan. The existing casts (``exploration/view_gain.py``,
``world_model/sightlines.py``) sweep a cone and return the cells it reaches;
here the cells are given and each is asked about, which is exact per cell,
needs no ray density, and yields the pixel each cell lands on.

Python 3.8 syntax, numpy.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import numpy as np

from sparx_agency.core.planning.environment import OccupancyGrid2D
from sparx_agency.core.planning.object_search.frontier_views.settings import ViewSettings
from sparx_agency.core.planning.object_search.world_model.sightlines import floor_band
from sparx_agency.core.planning.objnav.camera_geometry import world_T_camera_optical
from sparx_agency.core.planning.objnav.types.camera import CameraSpec
from sparx_agency.core.planning.objnav.types.pose import AgentPose

#: How far a frame's sight of a cell got: the stages of the module docstring, in order.
OUT_OF_RANGE, OUT_OF_FRAME, OCCLUDED, SEEN = 0, 1, 2, 3
STAGE_NAMES = ("out of range", "out of the picture", "occluded", "seen")

#: What a map cell is to a sight line.
_FREE, _OPEN, _SETTLED, _OCCUPIED = 0, 1, 2, 3
#: Nearest optical depth a point may have and still be in front of the camera, metres.
_NEAR_PLANE_M = 0.05
#: Added to the floor's near edge before unknown floor counts against a sight line (the ledger's own margin).
_BLIND_MARGIN_M = 0.1
_EPS = 1e-9


@dataclass(frozen=True)
class FrameSight:
    """One frame's sight of every cell of a :class:`SightScene`; each array has one entry per cell, in the scene's order.

    Attributes:
        stage: ``int8`` -- :data:`OUT_OF_RANGE`, :data:`OUT_OF_FRAME`,
            :data:`OCCLUDED` or :data:`SEEN`.
        quality: What the frame's view of the cell is worth, ``[0, 1]``: the
            range falloff times the share of the height band in the picture
            times 1 for a cell seen through or ``at_weight`` for one only
            seen. 0 unless :data:`SEEN`.
        through: The frame sees past the cell into unsettled unknown.
        range_m: Ground distance from the camera to the cell.
        u: The image column the cell's band lands on (its visible part's
            middle); ``NaN`` unless the cell is in the picture.
        v_top: The image row of the top of the band's visible part.
        v_bottom: The image row of its bottom; ``v_bottom >= v_top``.
    """

    stage: np.ndarray
    quality: np.ndarray
    through: np.ndarray
    range_m: np.ndarray
    u: np.ndarray
    v_top: np.ndarray
    v_bottom: np.ndarray


class SightScene:
    """The map of one decision and the cells whose sight is asked: built once, asked of every frame.

    Args:
        world: The occupancy grid as it stands at the decision.
        rows: Row index of every cell asked about.
        cols: Column index of every cell asked about, same order.
        camera: The camera every frame was captured with.
        settings: The selector's geometry.
        settled: Optional ``(H, W)`` bool of unknown cells the search has
            settled (looked through without a depth return, enclosed
            pockets): a line that runs on into them does not see through.

    Raises:
        ValueError: On index arrays of different lengths, a cell off the
            grid, or a ``settled`` mask not shaped like the grid.
    """

    def __init__(self, world: OccupancyGrid2D, rows, cols, camera: CameraSpec,
                 settings: Optional[ViewSettings] = None, settled: Optional[np.ndarray] = None) -> None:
        self.settings = settings or ViewSettings()
        self.camera = camera
        grid = world.grid
        self._height, self._width = grid.shape
        self._res = float(world.resolution)
        self._origin = (float(world.origin_x), float(world.origin_y))
        self.rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        self.cols = np.asarray(cols, dtype=np.int64).reshape(-1)
        if self.rows.shape != self.cols.shape:
            raise ValueError("rows and cols must be the same length, got %d and %d" % (self.rows.size, self.cols.size))
        if self.rows.size and (self.rows.min() < 0 or self.rows.max() >= self._height
                               or self.cols.min() < 0 or self.cols.max() >= self._width):
            raise ValueError("a cell asked about lies off the %d x %d grid" % (self._height, self._width))
        code = np.full(grid.shape, _OCCUPIED, dtype=np.uint8)
        free = grid == world.values.free
        code[free] = _FREE
        unknown = grid == world.values.unknown
        code[unknown] = _OPEN
        self._deep = unknown & ~_grown(free)               # unknown that touches no known floor: it may hide a wall
        if settled is not None:
            mask = np.asarray(settled, dtype=bool)
            if mask.shape != grid.shape:
                raise ValueError("settled %s is not shaped like the grid %s" % (mask.shape, grid.shape))
            code[unknown & mask] = _SETTLED
        self._code = code
        self._x = (self.cols + 0.5) * self._res + self._origin[0]
        self._y = (self.rows + 0.5) * self._res + self._origin[1]
        # Samples every half cell, as every cast in this tree; the cell's own neighbourhood (three quarters of a
        # cell either side of its centre) is neither "before" nor "beyond" it.
        self._step = 0.5 * self._res
        self._at_m = 0.75 * self._res
        reach = self.settings.max_range_m + self._at_m + self.settings.through_run_m
        self._ranges = (np.arange(int(math.ceil(reach / self._step)) + 1, dtype=np.float64) + 0.5) * self._step
        k = camera.intrinsics
        self._half_vfov = math.atan(0.5 * float(k.height) / float(k.fy))

    def __len__(self) -> int:
        return int(self.rows.size)

    def sight(self, pose: AgentPose) -> FrameSight:
        """What the frame captured at ``pose`` shows of every cell of the scene."""
        s, n = self.settings, len(self)
        stage = np.full(n, OUT_OF_RANGE, dtype=np.int8)
        quality = np.zeros(n, dtype=np.float32)
        through = np.zeros(n, dtype=bool)
        u = np.full(n, np.nan, dtype=np.float32)
        v_top = np.full(n, np.nan, dtype=np.float32)
        v_bottom = np.full(n, np.nan, dtype=np.float32)
        dx, dy = self._x - float(pose.x), self._y - float(pose.y)
        distance = np.hypot(dx, dy)
        sight = FrameSight(stage, quality, through, distance.astype(np.float32), u, v_top, v_bottom)
        near = np.nonzero(distance <= s.max_range_m)[0]
        if not near.size:
            return sight
        stage[near] = OUT_OF_FRAME
        share, column, top, bottom = self._band_in_picture(pose, near)
        framed = share >= s.min_band_visible
        inside = near[framed]
        if not inside.size:
            return sight
        u[inside], v_top[inside], v_bottom[inside] = column[framed], top[framed], bottom[framed]
        clear, past = self._sight_lines(pose, dx[inside], dy[inside], distance[inside])
        stage[inside] = np.where(clear, SEEN, OCCLUDED)
        falloff = np.minimum(1.0, s.good_range_m / np.maximum(distance[inside], _EPS))
        worth = falloff * share[framed] * np.where(past, 1.0, s.at_weight)
        quality[inside] = np.where(clear, worth, 0.0)
        through[inside] = clear & past
        return sight

    def _band_in_picture(self, pose: AgentPose, index: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The share of the height band at each cell of ``index`` inside the image, and where its visible part lands.

        The band is the vertical segment over the cell's centre from
        ``band_m[0]`` to ``band_m[1]`` above the floor under the camera.
        In the optical frame a point is in the picture when it is in front of
        the near plane and its pixel is inside the borders, and each of those
        is linear in the point -- so along the segment each is a half-line,
        and their intersection with ``[0, 1]`` is the visible part.

        Returns:
            ``(share, u, v_top, v_bottom)`` per cell of ``index``; the pixel
            values are those of the visible part, meaningless where
            ``share`` is 0.
        """
        s, k = self.settings, self.camera.intrinsics
        transform = world_T_camera_optical(pose, self.camera)
        rotation, centre = transform[:3, :3], transform[:3, 3]
        x, y = self._x[index], self._y[index]
        foot = (np.stack([x, y, np.full(index.size, float(pose.z) + s.band_m[0])], axis=1) - centre).dot(rotation)
        head = (np.stack([x, y, np.full(index.size, float(pose.z) + s.band_m[1])], axis=1) - centre).dot(rotation)
        margin = s.border_fraction * float(k.width)

        def constraints(point):
            px, py, pz = point[:, 0], point[:, 1], point[:, 2]
            return np.stack([pz - _NEAR_PLANE_M,
                             k.fx * px + (k.cx - margin) * pz,                       # u >= margin
                             (k.width - 1.0 - margin - k.cx) * pz - k.fx * px,       # u <= width - 1 - margin
                             k.fy * py + k.cy * pz,                                  # v >= 0
                             (k.height - 1.0 - k.cy) * pz - k.fy * py])              # v <= height - 1

        at_foot = constraints(foot)
        slope = constraints(head) - at_foot
        rising, falling = slope > _EPS, slope < -_EPS
        with np.errstate(divide="ignore", invalid="ignore"):
            root = -at_foot / slope
        low = np.maximum(0.0, np.where(rising, root, -np.inf).max(axis=0))
        high = np.minimum(1.0, np.where(falling, root, np.inf).min(axis=0))
        feasible = ~((~rising & ~falling) & (at_foot < 0.0)).any(axis=0)
        share = np.where(feasible, np.clip(high - low, 0.0, 1.0), 0.0)

        def pixel(t):
            point = foot + t[:, None] * (head - foot)
            depth = np.maximum(point[:, 2], _NEAR_PLANE_M)
            return k.fx * point[:, 0] / depth + k.cx, k.fy * point[:, 1] / depth + k.cy

        column, _ = pixel(0.5 * (low + high))
        _, bottom = pixel(low)
        _, top = pixel(high)
        return share, column, top, bottom

    def _sight_lines(self, pose: AgentPose, dx: np.ndarray, dy: np.ndarray,
                     distance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """``(clear, past)`` per cell: a line from the camera reaches the cell; the direct one, continued, runs into unsettled unknown.

        Parallel lines are tried per cell -- the direct one and, with
        ``corner_slack_cells`` of 1, one a cell to either side of it -- and
        the cell is reached when any of them is clear. The edge of what a
        camera saw past a door jamb is the camera's own line TANGENT to the
        jamb, and on a grid a tangent clips the corner cell: without the
        two neighbours every cell of such a boundary reads as hidden from
        the very frame that mapped it. A wall is not got round by a cell's
        shift; a corner is, by no more than the map's own resolution.
        """
        s, ranges = self.settings, self._ranges
        safe = np.maximum(distance, _EPS)
        ux, uy = dx / safe, dy / safe
        before = ranges[None, :] < (distance[:, None] - self._at_m)
        beyond_blind = ranges[None, :] >= self._blind_radius(pose)
        clear = np.zeros(distance.shape, dtype=bool)
        direct = None
        shifts = [0.0] + [side * cells * self._res for cells in range(1, s.corner_slack_cells + 1) for side in (1.0, -1.0)]
        for shift in shifts:
            xs = float(pose.x) - shift * uy[:, None] + np.outer(ux, ranges)
            ys = float(pose.y) + shift * ux[:, None] + np.outer(uy, ranges)
            gx = np.floor((xs - self._origin[0]) / self._res).astype(np.int64)
            gy = np.floor((ys - self._origin[1]) / self._res).astype(np.int64)
            on_map = (gx >= 0) & (gx < self._width) & (gy >= 0) & (gy < self._height)
            gy_on, gx_on = np.clip(gy, 0, self._height - 1), np.clip(gx, 0, self._width - 1)
            code = np.where(on_map, self._code[gy_on, gx_on], _OCCUPIED)
            if direct is None:
                direct = code
            walled = ((code == _OCCUPIED) & before).any(axis=1)
            dark = self._deep[gy_on, gx_on] & on_map & before & beyond_blind
            clear |= ~walled & (dark.sum(axis=1) * self._step <= s.unknown_slack_m + _EPS)
        start = distance[:, None] + self._at_m
        beyond = (ranges[None, :] > start) & (ranges[None, :] <= start + s.through_run_m)
        past = beyond.any(axis=1) & ~((direct != _OPEN) & beyond).any(axis=1)
        return clear, clear & past

    def _blind_radius(self, pose: AgentPose) -> float:
        """Ground distance within which this frame's camera could not see the floor: unknown there hides nothing."""
        band = floor_band(float(self.camera.height_m), float(pose.camera_pitch), self._half_vfov, 0.0, math.inf)
        if band is None:
            return float(self.settings.max_range_m)        # the floor is in no row of this frame
        return min(float(self.settings.max_range_m), float(band[0]) + _BLIND_MARGIN_M)


def _grown(mask: np.ndarray) -> np.ndarray:
    """``mask`` and every cell touching it, corners included (a 3 x 3 dilation by eight shifts)."""
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out
