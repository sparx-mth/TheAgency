"""Turn a surveyed map into one autonomous A-to-B mission: sample, then plan.

This is the "expert" in *expert demonstration*. Every episode of a collection
campaign starts here: pick two clear, reachable, far-apart points in the
building, then plan a wall-avoiding route between them with the same weighted A*
the real drones fly. What comes out is a waypoint list the autopilot can track.

Deliberately simulator-free -- it takes an :class:`OccupancyGrid2D` and a seeded
generator and returns waypoints -- so a campaign's whole route plan can be
generated, inspected and unit-tested without booting Isaac Sim.

Two responsibilities are kept apart on purpose. *Where to fly* is
:mod:`sparx_agency.core.planning.mission.free_space_sampler`, which only
guarantees both ends are clear and connected. *How to get there* is
:class:`~sparx_agency.core.planning.planners.astar.weighted_planner_2d.WeightedAStarPlanner2D`,
which is the thing that actually keeps the route off the walls -- it inflates
every obstacle by the airframe's standoff and then prefers the middle of a
corridor to its edge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.environment import OccupancyGrid2D
from sparx_agency.core.planning.interfaces.planner import PlanRequest
from sparx_agency.core.planning.mission import (
    largest_region, sample_goal_from, sample_start_goal, snap_to_region,
)
from sparx_agency.core.planning.planners.astar.params import WeightedAStarParams
from sparx_agency.core.planning.planners.astar.weighted_planner_2d import (
    WeightedAStarPlanner2D,
)


@dataclass(frozen=True)
class EpisodeSpec:
    """What every episode of a campaign is drawn to look like.

    Attributes:
        altitude_m: Cruise height above the floor, metres. The map is surveyed
            at this height, so changing it invalidates the map.
        clearance_m: Obstacle clearance the start and goal must have. Larger
            than ``inflate_radius_m`` on purpose: the aircraft has to *hold
            position* at both ends (through takeoff and landing, when it is
            least precise), not merely pass through.
        inflate_radius_m: Standoff the route is planned at -- the airframe's
            radius plus however far the position controller is expected to
            wander. The single most important number here: too small and the
            route shaves furniture, too large and no route exists at all.
        inflate_floor_m: Hard lower bound the planner may relax to when a
            corridor is genuinely narrower than the preferred standoff. Set to
            the airframe's true inscribed radius; a route is never planned
            below it.
        min_separation_m: Shortest mission worth recording, metres.
        max_separation_m: Longest, metres. None = no cap.
        waypoint_spacing_m: Target distance between emitted waypoints. Short
            spacing keeps the autopilot on the planned line rather than cutting
            its own corners between distant setpoints.
        start_yaw_jitter_rad: How far the aircraft may start off pointing away
            from its goal. Default is a full half-turn either way, i.e.
            uniformly random: a policy trained only on flights that begin
            already facing the target never learns to turn around.
        max_attempts: How many samples to try before admitting the map cannot
            produce an episode to this spec.
    """

    altitude_m: float = 1.5
    clearance_m: float = 0.8
    inflate_radius_m: float = 0.6
    inflate_floor_m: float = 0.45
    min_separation_m: float = 5.0
    max_separation_m: Optional[float] = None
    waypoint_spacing_m: float = 2.0
    start_yaw_jitter_rad: float = math.pi
    max_attempts: int = 20


@dataclass(frozen=True)
class EpisodePlan:
    """One planned mission, ready to fly.

    Attributes:
        start: Where the aircraft takes off from (``yaw`` = its initial heading).
        goal: Where it lands.
        waypoints: ``(x, y, z, yaw)`` in the world frame, in order, ending at
            the goal. ``yaw`` is radians CCW from +X (FLU), facing along the leg
            that reaches each point.
        path_length_m: Total length of the planned route.
        straight_line_m: Start-to-goal distance, for a detour ratio.
        inflate_used_m: Standoff the route was actually achieved at -- below
            ``spec.inflate_radius_m`` if the planner had to relax through a
            pinch.
    """

    start: Pose2D
    goal: Pose2D
    waypoints: Tuple[Tuple[float, float, float, float], ...]
    path_length_m: float
    straight_line_m: float
    inflate_used_m: float

    @property
    def detour_ratio(self) -> float:
        """Planned length over straight-line distance. 1.0 = a clear run."""
        return self.path_length_m / max(self.straight_line_m, 1e-6)


def make_planner(spec: EpisodeSpec) -> WeightedAStarPlanner2D:
    """Build the weighted A* planner an :class:`EpisodeSpec` describes.

    ``heading_penalty_m`` is deliberately left at 0: an episode is free to start
    with a turn (that is what ``start_yaw_jitter_rad`` is for), so a route
    should not be biased toward whichever way the aircraft happens to be
    pointing when it spawns.
    """
    return WeightedAStarPlanner2D(WeightedAStarParams(
        inflate_radius_m=spec.inflate_radius_m,
        inflate_floor_m=spec.inflate_floor_m,
        waypoint_spacing_m=spec.waypoint_spacing_m,
        unknown_blocked=True,   # unsurveyed space is not free space
        heading_penalty_m=0.0,
    ))


def waypoints_with_heading(
    start: Pose2D, points: List[Pose2D], altitude_m: float,
) -> Tuple[Tuple[float, float, float, float], ...]:
    """Attach a cruise altitude and a facing to a planned 2D path.

    Each waypoint faces along the leg that arrives at it, so the onboard camera
    looks where the aircraft is going -- which is the difference between a
    recording that teaches navigation and a sequence of sideways drifts. A leg
    too short to define a bearing inherits the previous one rather than
    snapping the camera to an arbitrary angle.

    Args:
        start: Pose the path is flown from.
        points: Planned world-frame path, excluding the start.
        altitude_m: Cruise height for every waypoint.

    Returns:
        ``(x, y, z, yaw)`` tuples.
    """
    out = []
    previous = start
    yaw = start.yaw
    for point in points:
        dx, dy = point.x - previous.x, point.y - previous.y
        if math.hypot(dx, dy) > 1e-3:
            yaw = math.atan2(dy, dx)
        out.append((point.x, point.y, altitude_m, yaw))
        previous = point
    return tuple(out)


def _path_length(start: Pose2D, points: List[Pose2D]) -> float:
    total = 0.0
    previous = start
    for point in points:
        total += math.hypot(point.x - previous.x, point.y - previous.y)
        previous = point
    return total


def plan_between(
    grid: OccupancyGrid2D, start: Pose2D, goal: Pose2D, spec: EpisodeSpec,
    planner: Optional[WeightedAStarPlanner2D] = None,
) -> Optional[EpisodePlan]:
    """Plan a wall-avoiding route between two poses.

    Args:
        grid: The surveyed map.
        start: Take-off pose; its ``yaw`` becomes the episode's initial heading.
        goal: Landing point.
        spec: Episode parameters.
        planner: Reuse an existing planner (it caches its cost map per grid, so
            passing the same one across a campaign saves rebuilding the
            distance transform every episode).

    Returns:
        The plan, or None if no route exists between the two.
    """
    planner = planner or make_planner(spec)
    result = planner.plan(PlanRequest(start=start, goal=goal, frame_id=grid.frame_id), grid)
    if not result.ok or result.path is None:
        return None

    points = list(result.path.points)
    # The planner emits cell centres, so its last waypoint is up to half a cell
    # short of the requested goal; finish the trip exactly. But it also *snaps* a
    # blocked goal to a nearby free cell, and overriding that would fly the
    # aircraft into whatever blocked it -- so only close the gap when it is small
    # enough to be the discretisation and not a snap.
    snap_tolerance = grid.resolution
    if points and math.hypot(points[-1].x - goal.x, points[-1].y - goal.y) <= snap_tolerance:
        points[-1] = Pose2D(goal.x, goal.y, 0.0)
    elif not points:
        points = [Pose2D(goal.x, goal.y, 0.0)]

    # Re-check what will actually be flown. The planner validates its own output,
    # but the waypoints above are not quite that output (the goal was pulled back
    # onto the exact requested point), and an unflyable route is far cheaper to
    # reject here than to discover with an aircraft against a wall.
    if planner.path_collides(grid, [start] + points, passable_start=start):
        return None

    return EpisodePlan(
        start=start,
        goal=Pose2D(points[-1].x, points[-1].y, 0.0),
        waypoints=waypoints_with_heading(start, points, spec.altitude_m),
        path_length_m=_path_length(start, points),
        straight_line_m=math.hypot(goal.x - start.x, goal.y - start.y),
        inflate_used_m=float(result.artifacts.get("inflate_used_m", spec.inflate_radius_m)),
    )


def sample_episode(
    grid: OccupancyGrid2D,
    rng: np.random.Generator,
    spec: EpisodeSpec,
    region: Optional[np.ndarray] = None,
    goal_region: Optional[np.ndarray] = None,
    start_from: Optional[Pose2D] = None,
    planner: Optional[WeightedAStarPlanner2D] = None,
) -> EpisodePlan:
    """Draw and plan one episode, retrying until a route is found.

    Sampling from a single connected component makes an unreachable goal
    structurally impossible, but the planner's standoff is stricter than the
    sampler's clearance test -- a doorway can be wide enough to *stand* in and
    too narrow to *fly through* at the preferred inflation. So a sample can
    still fail to plan, and this retries rather than propagating that.

    Args:
        grid: The surveyed map.
        rng: Seeded generator. The only source of randomness.
        spec: Episode parameters.
        region: Precomputed traversable region (:func:`largest_region`) -- where
            the aircraft may fly, and what a real position is snapped onto. Pass
            it across a campaign; recomputing it per episode is pure waste.
        goal_region: Where an *end* may be drawn, if that is narrower than where
            the aircraft may fly. Defaults to ``region``. A campaign passes the
            landable subset here: a goal has to be somewhere the aircraft can be
            put down, not merely somewhere it can hover.
        start_from: Fly on from where the aircraft already is instead of drawing
            a fresh start. This is the chained mode a campaign uses between
            episodes -- it avoids teleporting the vehicle, which would force the
            autopilot's estimator to reinitialise.
        planner: Reuse an existing planner across episodes, see
            :func:`plan_between`.

    Returns:
        The planned episode.

    Raises:
        RuntimeError: If ``spec.max_attempts`` samples all failed to plan.
    """
    if region is None:
        region = largest_region(grid, spec.clearance_m)
    if goal_region is None:
        goal_region = region
    planner = planner or make_planner(spec)

    failures = []
    for _ in range(spec.max_attempts):
        try:
            if start_from is None:
                mission = sample_start_goal(
                    grid, rng, clearance_m=spec.clearance_m,
                    min_separation_m=spec.min_separation_m,
                    max_separation_m=spec.max_separation_m,
                    start_yaw_jitter_rad=spec.start_yaw_jitter_rad,
                    region=goal_region,
                )
                start, goal = mission.start, mission.goal
            else:
                start = snap_to_region(grid, region, start_from.x, start_from.y)
                start = Pose2D(start.x, start.y, start_from.yaw)
                goal, _ = sample_goal_from(
                    grid, rng, start, clearance_m=spec.clearance_m,
                    min_separation_m=spec.min_separation_m,
                    max_separation_m=spec.max_separation_m,
                    region=goal_region,
                )
        except ValueError as error:
            failures.append(str(error))
            continue

        plan = plan_between(grid, start, goal, spec, planner=planner)
        if plan is not None:
            return plan
        failures.append(
            "no route from (%.1f, %.1f) to (%.1f, %.1f) at %.2f m standoff"
            % (start.x, start.y, goal.x, goal.y, spec.inflate_radius_m)
        )

    raise RuntimeError(
        "could not plan an episode in %d attempts; last failures: %s"
        % (spec.max_attempts, "; ".join(failures[-3:]))
    )
