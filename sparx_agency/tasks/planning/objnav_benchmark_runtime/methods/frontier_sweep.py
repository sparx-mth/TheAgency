"""Frontier goals for one room or the whole floor, their persistence, and the look-around.

The goal generator the room-search loop
(:mod:`~sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.room_search_loop`)
sweeps a room with. The loop owns the seven steps -- which room, when a room's
turn ends, the transit to the next one; this module answers one question per
action: *where in this mask is the next thing worth looking at, and is the goal
already committed to still worth reaching?* What the Ranchester recording
(``e7c4f2ad5402``, 472 actions) taught it is still in force:

* **Largest cluster first, wherever it was.** Goals come from
  :func:`~sparx_agency.core.planning.exploration.frontier_ranking.ranked_frontier_goals`,
  geodesic-distance and facing aware.
* **One goal, one A*, one idle action when it failed.** Up to
  :attr:`SweepSettings.plan_attempts` ranked goals are tried per action.
* **Walking to a boundary that no longer exists, or dropping one that does.** A
  committed frontier goal is kept while unknown space remains near it, it is
  still passable and it still lies near the room being swept. A re-segmented
  room mask alone no longer drops it (that replaced routes every ten actions),
  and a boundary the camera has already resolved no longer keeps it.
* **The look-around is optional and off by default.** The loop's termination
  rule is *N local steps or no frontier left*, so a swept room is released on
  the action its last goal resolves. :attr:`SweepSettings.room_scan_turns`
  keeps the bounded rotation available for a detector that wants one more
  look before leaving.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

import numpy as np

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.exploration.frontier_ranking import (
    FrontierRankingParams, accessible_frontiers, ranked_frontier_goals)
from sparx_agency.core.planning.objnav.types.command import NavigationCommand


@dataclass(frozen=True)
class SweepSettings:
    """Bounds on the sweep. Distances in metres, counts in actions.

    Attributes:
        plan_attempts: Ranked goals the planner is asked about per action
            before the sweep gives up for this action.
        room_scan_turns: In-place turns a room may spend looking around once
            nothing reachable is left in it, per visit. ``0`` (the default)
            leaves at once, which is the loop's termination rule; ``-1``
            means one full rotation at the benchmark's turn angle.
        informative_radius_m: A committed goal stays committed while unknown
            cells remain within this radius of it.
        informative_cells: ... at least this many of them.
        mask_slack_m: A committed goal stays inside a room while room-mask
            cells remain within this distance of it -- boundaries sit at the
            room's edge, and the watershed moves that edge every update.
        visited_radius_m: A new goal this close to a retired one is skipped.
        visited_memory: How many retired goals are remembered for that.
        ranking: Utility tuning for the frontier order.
    """

    plan_attempts: int = 3
    room_scan_turns: int = 0
    informative_radius_m: float = 1.0
    informative_cells: int = 4
    mask_slack_m: float = 1.0
    visited_radius_m: float = 0.4
    visited_memory: int = 100
    ranking: FrontierRankingParams = field(default_factory=FrontierRankingParams)

    def __post_init__(self):
        for name in ("plan_attempts", "informative_cells", "visited_memory"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("%s must be a positive integer" % name)
        if type(self.room_scan_turns) is not int or self.room_scan_turns < -1:
            raise ValueError("room_scan_turns must be -1 (one rotation) or a non-negative integer")
        for name in ("informative_radius_m", "mask_slack_m", "visited_radius_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be positive and finite" % name)
        if isinstance(self.ranking, dict):
            object.__setattr__(self, "ranking", FrontierRankingParams(**self.ranking))
        if not isinstance(self.ranking, FrontierRankingParams):
            raise ValueError("ranking must be FrontierRankingParams or parameter overrides")


class FrontierSweep:
    """Ranked frontier goals inside a mask, a committed goal's lifetime, and the look-around."""

    def __init__(self, policy, settings=None):
        self.policy = policy
        self.settings = settings or SweepSettings()
        self.scans = {}
        self.stats = {"scan_turns": 0, "plan_failures": 0, "goals_resolved": 0}

    # -- frontier goals -----------------------------------------------------
    def explore(self, obs, world, cost, mask, kind="frontier", planning_world=None, labels=None, label=None):
        """Navigate to the first ranked goal the planner accepts, or None.

        Frontier goals are read off ``world`` -- the map as observed -- and
        their reachability off ``cost``. The route itself is planned on
        ``planning_world`` when given: the loop passes a room-confined copy
        of the map (other rooms written unknown) together with that copy's
        cost, so a goal the confinement makes unreachable is never offered
        and a route never leaves the room. Frontiers are never extracted from
        the copy, whose artificial unknown cells would read as boundaries.

        With ``labels`` and ``label`` the goals are the clusters the room
        label image credits to that room by majority vote -- the population
        the scene graph's ``frontier_clusters`` counts -- rather than the
        clusters whose cells lie inside ``mask``. The two differ at exactly
        the cells that matter: the watershed erodes its masks by the minimum
        clearance, so a room's boundary cells sit just outside its mask and
        a mask-only sweep finds nothing while the count still says three.
        ``mask`` still governs the committed goal's persistence.
        """
        for goal in self._goals(obs, world, cost, mask, labels, label):
            command = self.policy._navigate(obs, planning_world if planning_world is not None else world,
                                            goal, kind)
            if command is not None:
                return command
            self.stats["plan_failures"] += 1
        return None

    def nearest(self, obs, world, cost, mask, labels=None, label=None):
        """The admissible frontier goal of the room nearest along the floor, or None."""
        goals = self.admissible(obs, world, self._ranked(obs, world, cost, mask, labels, label))
        return min(goals, key=lambda g: g.geodesic_m) if goals else None

    def _ranked(self, obs, world, cost, mask, labels=None, label=None):
        """The room's ranked frontier goals, by label vote when labels are given, else by mask."""
        xy, yaw, ranking = (obs.pose.x, obs.pose.y), obs.pose.yaw, self.settings.ranking
        inventory = self.policy.graph.frontier_inventory
        if labels is None and inventory is not None and np.asarray(mask).all():
            return list(inventory.goals)
        if labels is not None and label is not None:
            return accessible_frontiers(world, cost, labels, xy, yaw, ranking).by_room.get(int(label), [])
        return ranked_frontier_goals(world, cost, mask, xy, yaw, ranking)

    def admissible(self, obs, world, goals):
        """The ranked goals worth proposing: not under the agent, not beside a retired one.

        The ONE filter between a frontier cluster and a goal. In-room goals,
        a room's entry point and the transit re-aim all pass through it, so
        the room-search loop cannot be told by one path that a room has a
        frontier left and by another that it has none -- which is exactly
        the disagreement that once released and re-entered the same room on
        every action.
        """
        p, s = self.policy, self.settings
        xy = (obs.pose.x, obs.pose.y)
        arrival = p.converter_params.goal_tolerance_m + world.resolution
        visited = p._visited_frontiers[-s.visited_memory:]
        return [g for g in goals
                if math.dist(xy, g.xy) > arrival
                and not any(math.dist(g.xy, used) < s.visited_radius_m for used in visited)]

    # -- the look-around ----------------------------------------------------
    def begin_scan(self, key):
        """A fresh visit: the look-around allowance for ``key`` starts again."""
        self.scans[key] = 0

    def scan_turn(self, obs, key):
        """One in-place turn of the look-around for ``key``, or None once it is spent."""
        turn = self.policy.episode.action_spec.turn_angle_rad
        budget = self.settings.room_scan_turns
        if budget < 0:
            budget = int(math.ceil(2 * math.pi / turn - 1e-9))
        spent = self.scans.get(key, 0)
        if spent >= budget:
            return None
        self.scans[key] = spent + 1
        self.stats["scan_turns"] += 1
        return NavigationCommand.hold(
            final_yaw=normalize_angle(obs.pose.yaw + turn),
            info={"kind": "room_scan", "reason": "room swept; one look around before leaving",
                  "scan_turn": spent + 1, "scan_budget": budget})

    # -- the committed goal -------------------------------------------------
    def _goals(self, obs, world, cost, mask, labels=None, label=None):
        """The committed goal while it is still worth reaching, else fresh ranked ones."""
        p, s = self.policy, self.settings
        arrival = p.converter_params.goal_tolerance_m + world.resolution
        current = self._current_goal(obs, world, cost, mask, arrival)
        if current is not None:
            return [current]
        ranked = self._ranked(obs, world, cost, mask, labels, label)
        return [g.xy for g in self.admissible(obs, world, ranked)][:s.plan_attempts]

    def _current_goal(self, obs, world, cost, mask, arrival):
        p = self.policy
        if p._goal is None or p.route_memory.kind != "frontier":
            return None
        xy = (obs.pose.x, obs.pose.y)
        if math.dist(xy, p._goal) > arrival and not p.route_memory.arrived(obs):
            gx, gy = world.world_to_grid(*p._goal)
            if (world.in_bounds(gx, gy) and np.isfinite(cost[gy, gx])
                    and self._near(mask, gx, gy, world.resolution, self.settings.mask_slack_m)
                    and self.informative(world, gx, gy)):
                return p._goal
        p._visited_frontiers.append(p._goal)
        p._goal = p._route = None
        p.route_memory.clear("frontier_completed_or_invalid")
        self.stats["goals_resolved"] += 1
        return None

    def informative(self, world, gx, gy):
        """Does unknown space still sit within the informative radius of this cell?"""
        r = int(math.ceil(self.settings.informative_radius_m / world.resolution))
        patch = world.grid[max(0, gy - r):gy + r + 1, max(0, gx - r):gx + r + 1]
        return int(np.count_nonzero(patch == world.values.unknown)) >= self.settings.informative_cells

    @staticmethod
    def _near(mask, gx, gy, resolution, slack_m):
        r = int(math.ceil(slack_m / resolution))
        return bool(np.asarray(mask, dtype=bool)[max(0, gy - r):gy + r + 1, max(0, gx - r):gx + r + 1].any())

    def diagnostics(self):
        return {"settings": asdict(self.settings), "stats": dict(self.stats),
                "room_scans": {"f%d/r%d" % key: turns for key, turns in self.scans.items()}}

