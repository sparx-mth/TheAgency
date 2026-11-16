"""Where the agent may stand: floor it has observed free, with room for its body, joined to where it stands now.

The closing has to put the agent somewhere -- a standoff beside the target, a
spot to back off to, the nearest floor when no standoff has a route. Until
2026-11-15 it asked the simulator: a candidate point was projected onto the
navigation mesh, which knows where walkable floor lies whether or not the
agent has ever looked at it. That is privileged geometry (no published
zero-shot ObjectNav baseline is given it), and it is gone. This module is the
whole of what replaces it, and it reads one thing: the occupancy grid the
agent built from its own depth frames.

A cell is **standing floor** at a clearance ``c`` when three things hold on
that grid, and nothing else is consulted:

1. **Observed free.** The cell reads FREE: a depth ray crossed it or a floor
   sample fell on it, and no obstacle evidence outvotes that. UNKNOWN is not
   floor -- it is unobserved, and what the camera has not seen may be a wall,
   a table leg or a stairwell.
2. **Room for the body, all of it observed.** Every cell whose centre lies
   within ``c`` of this cell's centre reads FREE as well. An OCCUPIED cell
   there is a collision; an UNKNOWN one is part of the body standing on
   floor nobody has seen. Both disqualify the cell, and so does the edge of
   the grid, beyond which nothing is observed. ``c`` is never under the body
   radius (0.18 m); the disc is the one the observed map itself clears under
   the agent's feet and the planner inflates obstacles by (``<=`` the radius,
   centre to centre), so the three agree on what "the body fits" means.
3. **Reachable on the same map.** The cell is joined to the agent by an
   8-connected chain of cells that satisfy 1 and 2. The chain starts from the
   standing cells inside the disc the agent's own body covers -- the agent is
   physically there, so stepping out of an obstacle's inflation skirt is
   allowed exactly as the planner's start relaxation allows it -- and it
   never crosses an OCCUPIED or UNKNOWN cell, nor one too tight for the body.
   A patch of free floor seen through a doorway whose threshold is still
   unknown is not reachable: nothing on the map says the two are joined.

Unobserved space is therefore never a standing point, never part of the
body's clearance and never a link in the chain that reaches one. What the
agent cannot yet stand on it has to go and look at.

**One floor, and the planner's own standoff.** The three rules are asked at
the body radius and make ONE floor. A planner that keeps a wider standoff
from obstacles than the body radius (the default planner's 0.25 m) cannot
end a route nearer an obstacle than that, so a floor may be asked for with
a ``standoff_m``: the same floor without the standing cells nearer than the
standoff to an OCCUPIED cell. Those cells are standing floor still -- the
squeeze planner's -- only not where that planner can stop. Unknown cells
do not count for the standoff: beyond the body's own disc nothing has to
have been observed, and a rule that asked for it kept the default planner
from every standoff on a map of a few frames (in the first paired runs of
2026-11-15 the squeeze planner took a standoff in 32 of 85 episodes where
the navigation-mesh runs had taken none; the squeeze planner prefers a
tight short way to a roomy long one). Whether the wider
planner can WALK to such a cell is its own to answer when the route is
planned; the chain of rule 3 says only that a body-wide observed way exists.

Pure functions of the grid: no pose history, no simulator, no ROS. numpy and
scipy (``ndimage``), Python 3.8 syntax.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from sparx_agency.core.planning.environment import OccupancyGrid2D
from sparx_agency.core.planning.exploration.room_costs import passable_graph
from sparx_agency.core.planning.planners.common.grid_geometry_2d import line_cells

Xy = Tuple[float, float]

#: Slack on the centre-to-centre distance test, as the planner's lethal test has (``cost_grid_2d._EPS``): a cell
#: exactly the clearance away, an integer number of cells, is within it.
_EPS = 1e-9
#: A diagonal step between two standing cells is a step the body can take: the cells either side of it lie inside
#: both cells' own free discs.
_EIGHT = np.ones((3, 3), dtype=bool)


def nearest_cell_in_sight(ids: np.ndarray, occupied: np.ndarray, gx: int, gy: int,
                          max_radius: int, through_own_obstacle: bool = False) -> Optional[Tuple[int, int]]:
    """The nearest cell of ``ids`` to ``(gx, gy)`` that is not on the far side of an obstacle from it.

    :func:`~sparx_agency.core.planning.exploration.room_costs.snap_cell`
    takes the Euclidean-nearest cell, and beside a wall the nearest cell is
    as likely to lie in the room BEHIND the wall as in front of it -- a
    stair head's approach point did, by one centimetre, in the first
    recorded multi-storey campaign, and a standoff ring point on unseen
    floor before a wall is moved to the floor beyond the wall the same way.
    A candidate here must also have a clear Bresenham line to the query
    cell through the observed map: UNKNOWN cells on the line are allowed
    (the query point usually lies in space not yet seen; the line is not
    walked, it only says which side of the wall the answer is on), OCCUPIED
    ones are not. What is exempt is the query's own end of the line:

    * by default the query cell alone -- a stair head's own cell may be
      marked occupied by the drop beyond it, and that is where the flight
      begins (``stair_approach.approach_points``);
    * with ``through_own_obstacle`` the whole run of occupied cells the
      line starts in -- a point inside a cabinet or a wall's thickness is
      moved to the floor in front of it (:meth:`StandingFloor.nearest`) --
      but once the line has left that run it may not enter another.

    The window grows as ``snap_cell``'s does; within a window the candidates
    are tried nearest first.

    Args:
        ids: ``(H, W)`` integer image, ``>= 0`` on the candidate cells and
            ``-1`` elsewhere (``passable_graph``'s index image, or a mask
            written that way).
        occupied: ``(H, W)`` boolean, True where the observed map is occupied.
        gx: Query cell column.
        gy: Query cell row.
        max_radius: Largest window half-width to try, in cells.
        through_own_obstacle: Exempt the obstacle the query lies in, not
            only the query's own cell.

    Returns:
        ``(gx, gy)`` of a candidate cell with a clear line to the query, or None.
    """
    h, w = ids.shape
    if not (0 <= gx < w and 0 <= gy < h):
        return None
    if ids[gy, gx] >= 0:
        return int(gx), int(gy)
    for r in range(1, int(max_radius) + 1):
        y0, y1 = max(0, gy - r), min(h, gy + r + 1)
        x0, x1 = max(0, gx - r), min(w, gx + r + 1)
        ys, xs = np.nonzero(ids[y0:y1, x0:x1] >= 0)
        if ys.size == 0:
            continue
        cy, cx = ys + y0, xs + x0
        for k in np.argsort((cx - gx) ** 2 + (cy - gy) ** 2, kind="stable"):
            cells = line_cells(int(gx), int(gy), int(cx[k]), int(cy[k]))
            if through_own_obstacle:
                clear = _line_leaves_one_obstacle_at_most(occupied, cells)
            else:
                clear = not any(occupied[y, x] for x, y in cells[1:])
            if clear:
                return int(cx[k]), int(cy[k])
    return None


def _line_leaves_one_obstacle_at_most(occupied: np.ndarray, cells) -> bool:
    """Whether a line of cells, read from its start, is clear once it has left the obstacle it starts in (if any)."""
    outside = False
    for x, y in cells:
        if occupied[y, x]:
            if outside:
                return False
        else:
            outside = True
    return True


def body_disc(clearance_m: float, resolution_m: float) -> np.ndarray:
    """The cells whose centres lie within ``clearance_m`` of a cell's centre, as a boolean structuring element.

    Args:
        clearance_m: The radius the body needs clear, metres.
        resolution_m: Metres per cell.

    Returns:
        ``(2n + 1, 2n + 1)`` boolean, True within the radius (the centre cell
        always). At 0.18 m on a 0.1 m grid this is the 3 x 3 block; at 0.25 m
        the 5 x 5 block without its corners.

    Raises:
        ValueError: A clearance that is negative or not finite, or a
            resolution that is not positive.
    """
    if not math.isfinite(clearance_m) or clearance_m < 0.0:
        raise ValueError("clearance_m must be finite and non-negative, got %r" % (clearance_m,))
    if not math.isfinite(resolution_m) or resolution_m <= 0.0:
        raise ValueError("resolution_m must be positive and finite, got %r" % (resolution_m,))
    n = int(math.floor(clearance_m / resolution_m + _EPS))
    ys, xs = np.mgrid[-n:n + 1, -n:n + 1]
    return (xs * xs + ys * ys) * resolution_m * resolution_m <= clearance_m * clearance_m + _EPS


def standable_cells(world: OccupancyGrid2D, clearance_m: float) -> np.ndarray:
    """Cells observed free whose whole body disc is observed free too (rules 1 and 2 of the module docstring).

    Args:
        world: The agent's own occupancy grid.
        clearance_m: The radius the body needs clear, metres.

    Returns:
        ``(H, W)`` boolean, indexed ``[gy, gx]`` like the grid. Nothing
        within ``clearance_m`` of the grid's edge is set: what lies beyond
        the grid is unobserved.
    """
    free = np.asarray(world.grid) == world.values.free
    if not free.any():
        return free
    return ndimage.binary_erosion(free, structure=body_disc(clearance_m, world.resolution), border_value=0)


@dataclass
class StandingFloor:
    """The standing floor of one map, for one agent position and one clearance.

    Build it with :func:`standing_floor`. The masks are read-only by
    convention: one instance is shared by every question the closing asks on
    an action.

    Attributes:
        world: The occupancy grid it was read from.
        clearance_m: The radius kept clear of everything not observed free.
        standable: ``(H, W)`` boolean -- rules 1 and 2.
        reachable: ``(H, W)`` boolean -- the cells the agent may be sent to
            stand on: the standable cells it is joined to (rule 3) that also
            keep ``standoff_m`` from every observed obstacle. A subset of
            ``joined``, and the same array when the standoff is the
            clearance; empty when the agent stands where no standable cell
            is within its own body.
        seeds: ``(H, W)`` boolean -- the standable cells under the agent's
            body, where the chain starts.
        origin_xy: Where the agent stands, world metres.
        standoff_m: The distance ``reachable`` keeps from OCCUPIED cells --
            the standoff of the planner that is to end a route there; never
            under ``clearance_m``.
        joined: ``(H, W)`` boolean -- rule 3 alone: every standable cell
            joined to the agent, whatever its distance from an obstacle. The
            chain walks over these; None reads as ``reachable``.
    """

    world: OccupancyGrid2D
    clearance_m: float
    standable: np.ndarray
    reachable: np.ndarray
    seeds: np.ndarray
    origin_xy: Xy = (0.0, 0.0)
    standoff_m: Optional[float] = None
    joined: Optional[np.ndarray] = None
    _ids: Optional[np.ndarray] = field(default=None, repr=False)
    _occupied: Optional[np.ndarray] = field(default=None, repr=False)
    _geodesic: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        if self.standoff_m is None:
            self.standoff_m = self.clearance_m
        if self.joined is None:
            self.joined = self.reachable

    def contains(self, xy) -> bool:
        """Whether ``xy`` lies in a cell the agent may stand on and can walk to (rules 1 to 3)."""
        gx, gy = self.world.world_to_grid(float(xy[0]), float(xy[1]))
        return bool(self.world.in_bounds(gx, gy) and self.reachable[gy, gx])

    def standable_at(self, xy) -> bool:
        """Whether ``xy`` lies in a cell observed free with the body's disc observed free (rules 1 and 2 alone).

        For a standing point already chosen and being walked to: the map
        must still show it as a place to stand, while whether the agent is
        joined to it is the committed route's to answer -- a body squeezing
        through a doorway the map draws tight has no standing cell under it
        for a step or two, and its goal is no worse for that.
        """
        gx, gy = self.world.world_to_grid(float(xy[0]), float(xy[1]))
        return bool(self.world.in_bounds(gx, gy) and self.standable[gy, gx])

    def nearest(self, xy, within_m: float) -> Optional[Xy]:
        """The centre of the reachable standing cell nearest ``xy``, if one lies within ``within_m`` of it.

        What the navigation-mesh projection used to answer, asked of the
        agent's own map: a candidate point that is itself standing floor
        comes back as its cell's centre; one in a wall, on unseen floor or in
        a gap too tight for the body is moved to the nearest floor the agent
        can stand on, or refused when there is none near. The point is never
        moved across an obstacle the map shows (:func:`nearest_cell_in_sight`):
        the floor behind the wall a target stands against is near, and it is
        not where the point was.

        Args:
            xy: The candidate point, world metres.
            within_m: How far the answer may lie from ``xy`` (cell centre to
                the point itself).

        Returns:
            ``(x, y)`` of a cell centre, or None.
        """
        x, y = float(xy[0]), float(xy[1])
        gx, gy = self.world.world_to_grid(x, y)
        if self._ids is None:
            self._ids = np.where(self.reachable, 0, -1)
            # Whatever was observed and is not floor stands in the way; only what was never observed does not.
            cells = np.asarray(self.world.grid)
            self._occupied = (cells != self.world.values.free) & (cells != self.world.values.unknown)
        # One cell more than the radius: the point may sit at its own cell's edge, half a cell from the centre.
        cell = nearest_cell_in_sight(self._ids, self._occupied, gx, gy,
                                     int(math.ceil(float(within_m) / self.world.resolution)) + 1,
                                     through_own_obstacle=True)
        if cell is None:
            return None
        sx, sy = self.world.grid_to_world(cell[0], cell[1])
        if math.hypot(sx - x, sy - y) > float(within_m) + _EPS:
            return None
        return float(sx), float(sy)

    def cells_xy(self) -> Tuple[np.ndarray, np.ndarray]:
        """World ``(xs, ys)`` of every reachable standing cell's centre, in row-major order."""
        gys, gxs = np.nonzero(self.reachable)
        return (self.world.origin_x + (gxs + 0.5) * self.world.resolution,
                self.world.origin_y + (gys + 0.5) * self.world.resolution)

    def geodesic_m(self) -> np.ndarray:
        """Metres walked over standing floor from the agent to every cell: ``(H, W)`` float, ``inf`` where unreachable.

        Measured along the 8-connected chain of rule 3 (``joined``: the
        walk may pass a standing cell nearer an obstacle than
        ``standoff_m``, where this floor's planner would not stop),
        starting with the straight step from where the agent stands onto a
        cell under its body. Computed on first use and kept: only the
        questions that weigh a walk's length against the straight-line
        range ask.
        """
        if self._geodesic is None:
            out = np.full(self.joined.shape, np.inf)
            if self.seeds.any():
                ids, graph = passable_graph(np.where(self.joined, 1.0, np.inf))
                n = graph.shape[0]
                # One more node for the agent itself, joined to each cell under its body by the straight step
                # there (in cells, like the graph's own weights; never zero, which a sparse matrix reads as no edge).
                gys, gxs = np.nonzero(self.seeds)
                ax = (self.origin_xy[0] - self.world.origin_x) / self.world.resolution
                ay = (self.origin_xy[1] - self.world.origin_y) / self.world.resolution
                steps = np.maximum(np.hypot(gxs + 0.5 - ax, gys + 0.5 - ay), 1e-9)
                graph = graph.tocoo()
                graph = coo_matrix((np.concatenate([graph.data, steps]),
                                    (np.concatenate([graph.row, np.full(len(steps), n)]),
                                     np.concatenate([graph.col, ids[gys, gxs]]))), shape=(n + 1, n + 1)).tocsr()
                distances = dijkstra(graph, directed=False, indices=n)
                out[self.joined] = distances[ids[self.joined]] * self.world.resolution
            self._geodesic = out
        return self._geodesic


def obstacle_skirt(world: OccupancyGrid2D, standoff_m: float) -> np.ndarray:
    """Cells within ``standoff_m`` of a cell observed and not floor: where a planner keeping that standoff cannot stop.

    Centre to centre and inclusive, as the planner's own lethal test is
    (``cost_grid_2d.assemble_cost_grid``). UNKNOWN cells cast no skirt: the
    standoff is from what was seen to be in the way.

    Args:
        world: The agent's occupancy grid.
        standoff_m: The planner's standoff from obstacles, metres.

    Returns:
        ``(H, W)`` boolean, indexed ``[gy, gx]`` like the grid.
    """
    cells = np.asarray(world.grid)
    observed_obstacle = (cells != world.values.free) & (cells != world.values.unknown)
    if not observed_obstacle.any():
        return observed_obstacle
    return ndimage.binary_dilation(observed_obstacle, structure=body_disc(standoff_m, world.resolution))


def standing_floor(world: OccupancyGrid2D, xy, clearance_m: float,
                   standoff_m: Optional[float] = None) -> StandingFloor:
    """The floor an agent at ``xy`` may stand on and can walk to, on its own map.

    Args:
        world: The agent's occupancy grid (FREE / OCCUPIED / UNKNOWN).
        xy: Where the agent stands, world metres.
        clearance_m: The radius the body needs clear of everything that is
            not observed free -- the body radius, or more.
        standoff_m: The standoff from observed obstacles of the planner
            that is to end a route on this floor, when wider than
            ``clearance_m`` (the module docstring's last paragraph): the
            standing cells nearer than this to an OCCUPIED cell are left
            out of ``reachable``. None, or anything not above
            ``clearance_m``, leaves the floor whole.

    Returns:
        The :class:`StandingFloor`; its ``reachable`` mask is empty when no
        standable cell lies under the agent's own body.

    Raises:
        ValueError: A ``standoff_m`` that is negative or not finite.
    """
    if standoff_m is not None and (not math.isfinite(standoff_m) or standoff_m < 0.0):
        raise ValueError("standoff_m must be finite and non-negative, got %r" % (standoff_m,))
    standable = standable_cells(world, clearance_m)
    seeds = np.zeros(standable.shape, dtype=bool)
    reachable = np.zeros(standable.shape, dtype=bool)
    gx, gy = world.world_to_grid(float(xy[0]), float(xy[1]))
    h, w = standable.shape
    # The disc the planner relaxes round a start it finds blocked (``_clear_start_footprint``): the body is there.
    n = max(1, int(round(float(clearance_m) / world.resolution)))
    x0, x1 = max(0, gx - n), min(w, gx + n + 1)
    y0, y1 = max(0, gy - n), min(h, gy + n + 1)
    if x0 < x1 and y0 < y1:
        ys, xs = np.ogrid[y0:y1, x0:x1]
        under_body = (xs - gx) ** 2 + (ys - gy) ** 2 <= n * n
        seeds[y0:y1, x0:x1] = standable[y0:y1, x0:x1] & under_body
    if seeds.any():
        labels, _ = ndimage.label(standable, structure=_EIGHT)
        reachable = np.isin(labels, np.unique(labels[seeds]))
    joined = reachable
    standoff = float(clearance_m)
    if standoff_m is not None and float(standoff_m) > float(clearance_m) + _EPS:
        standoff = float(standoff_m)
        reachable = joined & ~obstacle_skirt(world, standoff)
    return StandingFloor(world, float(clearance_m), standable, reachable, seeds, (float(xy[0]), float(xy[1])),
                         standoff_m=standoff, joined=joined)
