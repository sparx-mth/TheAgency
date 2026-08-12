from __future__ import annotations
from dataclasses import dataclass
from math import radians


@dataclass(frozen=True)
class AStarParams:
    """
    Parameters for A* planning on grids.

    Attributes:
        allow_unknown: If True, planner may traverse UNKNOWN cells.
                       Default False (classic exploration planning behavior).
        connectivity: 4 or 8 for 2D. (3D uses 6/18/26 separately in AStar3DParams)
        max_expansions: Safety cap; None means unlimited.
    """
    allow_unknown: bool = False
    connectivity: int = 4
    max_expansions: int | None = 200_000


@dataclass(frozen=True)
class WeightedAStarParams:
    """
    Parameters for the weighted 2D A* planner on an OccupancyGrid2D.

    Geometry-aware extension of :class:`AStarParams` that builds a float cost
    map (inflation + UNKNOWN weighting), restricts the search to a bounding
    box, and post-processes the path (line-of-sight smoothing + corner-
    preserving resample + start-prefix trim). Distances are in meters.

    Clearance shaping (the wall-avoidance + corridor-centering layer): obstacle
    distance is computed with an exact Euclidean transform. Cells within
    ``inflate_radius_m`` of an obstacle are lethal (blocked); beyond that, a soft
    cost that fades to zero over ``clearance_margin_m`` makes routes that hug
    walls more expensive than routes that stay clear. Because the middle of a
    corridor is the farthest point from both walls, this soft cost pulls the
    route to the centre. Set ``clearance_weight = 0`` to recover the old binary
    (free / blocked) behaviour.

    Confidence-weighted lethality (the depth-noise layer): a monocular-depth BEV
    paints occasional single-frame speckle as OCCUPIED, and a lone speckle on the
    route must not make the map infeasible. When the caller supplies a per-cell
    confidence grid, only cells whose evidence reaches ``lethal_confidence``
    block the search; an unconfirmed cell costs ``soft_obstacle_cost`` -- high
    enough that the route bends around it wherever there is room, finite so that
    a speckled corridor still yields a plan. ``lethal_confidence = 0`` (the
    default) disables the split: every OCCUPIED cell blocks, as before.

    Relaxable standoff (the "stop or squeeze" layer): a confirmed obstacle set
    can pinch a corridor below ``2 * inflate_radius_m`` -- a stably mis-detected
    cell, or a genuinely narrow spot -- and treating the preferred radius as an
    ultimatum means the robot simply stops. So a failed search is retried at
    progressively smaller standoffs, ``relax_step_m`` at a time, down to
    ``inflate_floor_m`` (the robot's true inscribed radius, never crossed). The
    first success wins, which is by construction the *safest* feasible route:
    clearance only ever decreases down the ladder, so walking it top-down and
    stopping early beats evaluating every rung. The soft cost stays shaped
    around the *preferred* radius throughout, so a squeezed route still rides as
    centred as it physically can.

    Every ``plan`` call restarts at ``inflate_radius_m``: the relaxation is never
    sticky, so a route squeezed past a transient pinch is re-planned at the full
    standoff as soon as the map allows, and a robot can never be left creeping
    along at a relaxed clearance because of an obstruction that has since gone.

    Phantom probe: if even ``inflate_floor_m`` fails, one further search at
    ``probe_radius_m`` asks a *diagnostic* question -- "is anything at all
    getting through?". A path that exists only at a sub-airframe clearance means
    the blockage is a thin obstruction, most likely a mis-detected voxel rather
    than a wall. That is reported (``artifacts["phantom_suspected"]``), never
    flown: the honest response is to stop and re-observe until the map corrects
    itself, not to delete map cells because a path was wanted.

    Attributes:
        connectivity: 4 or 8 (8 = diagonals at sqrt(2) cost, octile heuristic).
        inflate_radius_m: Preferred lethal obstacle radius (the standoff you want
            to fly). Cells this close to an obstacle are impassable; gaps
            narrower than twice this are never threaded unless the standoff is
            relaxed. 0 = only obstacle cells block.
        inflate_floor_m: Hard lower bound on a *flyable* standoff -- the robot's
            physical inscribed radius. A route is never planned below this, so
            whatever comes back can always be flown without hitting anything.
            Set >= ``inflate_radius_m`` to disable relaxation entirely.
        relax_step_m: Standoff decrement per retry between the preferred radius
            and the floor. Smaller = more attempts, but the surviving route keeps
            more clearance, so prefer a fine step over a coarse one.
        probe_radius_m: Diagnostic-only standoff, below ``inflate_floor_m``, used
            once when every flyable rung has failed. Its path is never returned
            (it is not flyable); success merely sets
            ``artifacts["phantom_suspected"]`` so the caller can distinguish "a
            thin, probably-spurious obstruction" from "genuinely walled in" and
            react by re-observing rather than giving up. 0 disables the probe.
        unknown_blocked: If True, UNKNOWN ("gray") cells are impassable.
        unknown_cost: Flat traversal cost for UNKNOWN cells when not blocked.
            Keep it above ``1 + clearance_weight`` (the most expensive known-free
            cell) so the planner prefers any known-free route over driving blind
            through gray, and only enters gray when there is no alternative.
        clearance_weight: Peak extra cost added at the lethal boundary, decaying
            to 0 at ``inflate_radius_m + clearance_margin_m``. Larger = stronger
            wall avoidance / corridor centering (and slightly more A* work).
            0 disables the soft layer.
        clearance_margin_m: Width of the soft cost band beyond the lethal
            radius. Routes are centred within corridors up to roughly
            ``2*(inflate_radius_m + clearance_margin_m)`` wide.
        lethal_confidence: Per-cell confidence in ``[0, 1]`` an OCCUPIED cell
            needs before it blocks the search. Requires the caller to supply a
            confidence grid (see :meth:`WeightedAStarPlanner2D.set_confidence`);
            without one the split is inactive. 0 = every OCCUPIED cell blocks.
        soft_obstacle_cost: Traversal cost of an OCCUPIED cell that did not reach
            ``lethal_confidence``. Keep it well above ``1 + clearance_weight`` so
            any known-free detour wins, but finite so a speckled corridor is
            still solvable.
        path_simplify_m: Douglas–Peucker tolerance used to thin the A* path
            without changing its shape (so the centred route keeps its
            clearance — string-pulling is deliberately *not* used, as it would
            make the path taut and undo the centring at corners). Larger = fewer
            waypoints but coarser corners. <= 0 = auto (~1 cell).
        search_margin_m: Bounding-box margin around start/goal. Smaller =
            faster but may miss paths that detour far outside the box.
        turn_penalty: Extra cost on a direction change (suppresses staircasing
            before simplification). 0 disables.
        los_smoothing: Simplify the raw A* cell path (Douglas–Peucker) before
            emitting waypoints. (Name kept for the rosparam; it no longer
            string-pulls.)
        waypoint_spacing_m: Target output segment length. Corners are preserved;
            each long leg is divided into the nearest whole number of equal
            sub-segments, so legs land close to this value (a leg may run up to
            ~1.5x before it splits, rather than always coming out below it).
            <= 0 disables resampling.
        goal_snap_radius_m: If the goal cell is blocked, snap to the nearest
            free cell within this radius. 0 disables.
        start_skip_m: Drop leading waypoints within this distance of the start
            so a follower is never pointed at a point it already occupies.
        max_expansions: Safety cap on A* node pops; None means unlimited.
        corner_round: Run corner rounding (merge near-collinear + chamfer
            moderate corners) so a stop-and-turn follower needs fewer/smaller
            in-place yaws. Applied after LOS smoothing, before segment splitting.
        corner_merge_rad: Interior vertices that turn less than this are dropped
            as near-collinear (rad).
        corner_max_turn_rad: Corners gentler than this are already glide-able and
            left untouched (rad). Pair with the follower's ``skip_yaw_thresh``.
        corner_chamfer_max_rad: Corners between ``corner_max_turn_rad`` and this
            are chamfered into two half-angle turns; sharper corners are kept as
            genuine turns (rad). Keep near ``2 * corner_max_turn_rad`` so each
            half is glide-able.
        corner_chamfer_dist_m: How far back from a corner to cut it (m); clamped
            to half the shorter adjacent leg.
        corner_min_runup_m: Minimum leg length on both sides for a corner to be
            chamfered (m); shorter corners are left sharp.
    """
    connectivity: int = 8
    inflate_radius_m: float = 0.4
    inflate_floor_m: float = 0.4      # == inflate_radius_m: relaxation off by default
    relax_step_m: float = 0.05
    probe_radius_m: float = 0.0       # 0 = no phantom probe
    unknown_blocked: bool = False
    unknown_cost: float = 5.0
    clearance_weight: float = 3.0
    clearance_margin_m: float = 0.8
    lethal_confidence: float = 0.0
    soft_obstacle_cost: float = 25.0
    path_simplify_m: float = 0.0
    search_margin_m: float = 3.0
    turn_penalty: float = 0.3
    # Heading awareness: cost (in metres of extra path) charged for a route that
    # turns the drone AROUND at the start, scaled by how backward the first move is
    # (0 for flying forward or turning 90 deg, full for a 180 deg reversal). Makes
    # A* fly the way the drone already looks -- e.g. straight down a hallway --
    # rather than spinning in place because the shortest path runs backward. Needs
    # the start pose's ``yaw`` set (the planner reads ``request.start.yaw``). 0 = off.
    heading_penalty_m: float = 0.0
    # The rotation-time counterpart: metres of extra path charged per RADIAN the
    # route turns the drone away from its current heading at the start, linear
    # from zero rather than free below 90 deg. Use this when the follower stops
    # and yaws on the spot before a corner it cannot glide, so a 90 deg turn
    # really does cost half of what a reversal costs; set it to cruise speed over
    # yaw rate and the trade is in seconds. Adds to ``heading_penalty_m``, needs
    # the same start ``yaw``. 0 = off.
    start_turn_cost_m_per_rad: float = 0.0
    # How far from the start, in metres, that cost is allowed to reach. It is
    # charged once, against the bearing on which the route leaves a disc of this
    # radius -- so set it to roughly the distance the aircraft covers while it
    # turns, and the route has to commit to leaving the area the way the robot is
    # pointing or pay for the rotation. 0 charges the first cell step instead,
    # which is measurably almost no bias at all: see astar_cost_grid_2d, which
    # also explains why a per-cell angular turn cost is a trap.
    start_turn_radius_m: float = 0.0
    los_smoothing: bool = True
    waypoint_spacing_m: float = 3.0
    goal_snap_radius_m: float = 2.0
    start_skip_m: float = 0.4
    max_expansions: int | None = 200_000

    # Corner rounding (gentler paths for a stop-and-turn follower).
    corner_round: bool = True
    corner_merge_rad: float = radians(8.0)
    corner_max_turn_rad: float = radians(14.0)
    corner_chamfer_max_rad: float = radians(28.0)
    corner_chamfer_dist_m: float = 0.5
    corner_min_runup_m: float = 0.6


@dataclass(frozen=True)
class AStar3DParams:
    """
    Parameters for voxel-grid A* (3D).

    Attributes:
        allow_unknown: If True, planner may traverse UNKNOWN voxels (if map supports it).
        connectivity: 6, 18, or 26.
        max_expansions: Safety cap; None means unlimited.
    """
    allow_unknown: bool = False
    connectivity: int = 6
    max_expansions: int | None = 800_000
