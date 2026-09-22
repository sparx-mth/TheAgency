"""Room-by-room frontier sweep under the object-search supervisor, bounded in place.

What the Ranchester recording (``e7c4f2ad5402``, 472 actions) showed the
previous inline version of this doing, and what each part here answers:

* **114 idle turns with no route (24 %).** The room's goal generator was empty
  but the supervisor kept the room in SEARCH until its 30 s stall clock ran
  out, and every empty action was spent on the agent's idle TURN_LEFT -- four
  full rotations in one room. Here a room with nothing reachable left gets ONE
  bounded look-around (:attr:`SweepSettings.room_scan_turns`, a full rotation
  by default, once per visit) and is then released through the supervisor's
  ``frontier_exhausted`` exit on the same action.
* **A throwaway route on every room release.** The release action fell through
  to a floor-wide frontier, planned a 7-9 m route, and the next action's SELECT
  replaced it with a transit. Here a release re-runs the supervisor on the same
  action, so the transit is planned at once.
* **One goal, one A*, one idle action when it failed.** Up to
  :attr:`SweepSettings.plan_attempts` ranked goals are tried per action, and a
  transit goal the planner refuses ends the room through ``route_failed``
  instead of idling through the supervisor's plan grace.
* **Largest cluster first, wherever it was.** Goals come from
  :func:`~sparx_agency.core.planning.exploration.frontier_ranking.ranked_frontier_goals`,
  geodesic-distance and facing aware.
* **Walking to a boundary that no longer exists, or dropping one that does.** A
  committed frontier goal is kept while unknown space remains near it, it is
  still passable and it still lies near the room being swept. A re-segmented
  room mask alone no longer drops it (that replaced routes every ten actions),
  and a boundary the camera has already resolved no longer keeps it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

import numpy as np

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.exploration.frontier_ranking import FrontierRankingParams, ranked_frontier_goals
from sparx_agency.core.planning.exploration.object_search_supervisor import SEARCH, TRANSIT
from sparx_agency.core.planning.exploration.room_costs import build_instance
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.planners.astar.cost_grid_2d import assemble_cost_grid


@dataclass(frozen=True)
class SweepSettings:
    """Bounds on the sweep. Distances in metres, counts in actions.

    Attributes:
        plan_attempts: Ranked goals the planner is asked about per action
            before the sweep gives up for this action.
        room_scan_turns: In-place turns a room may spend looking around once
            nothing reachable is left in it, per visit. ``-1`` means one full
            rotation at the benchmark's turn angle; ``0`` leaves at once.
        informative_radius_m: A committed goal stays committed while unknown
            cells remain within this radius of it.
        informative_cells: ... at least this many of them.
        mask_slack_m: A committed goal stays inside a room while room-mask
            cells remain within this distance of it -- boundaries sit at the
            room's edge, and the watershed moves that edge every update.
        visited_radius_m: A new goal this close to a retired one is skipped.
        visited_memory: How many retired goals are remembered for that.
        supervisor_rounds: Release-and-reselect rounds allowed per action
            before falling back to the floor-wide frontier.
        ranking: Utility tuning for the frontier order.
    """

    plan_attempts: int = 3
    room_scan_turns: int = -1
    informative_radius_m: float = 1.0
    informative_cells: int = 4
    mask_slack_m: float = 1.0
    visited_radius_m: float = 0.4
    visited_memory: int = 100
    supervisor_rounds: int = 3
    ranking: FrontierRankingParams = field(default_factory=FrontierRankingParams)

    def __post_init__(self):
        for name in ("plan_attempts", "informative_cells", "visited_memory", "supervisor_rounds"):
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
    """Drives the policy's supervisor, room sweep and floor-wide frontier."""

    def __init__(self, policy, settings=None):
        self.policy = policy
        self.settings = settings or SweepSettings()
        self.scans = {}
        self.stats = {"scan_turns": 0, "rooms_released": 0, "plan_failures": 0,
                      "goals_resolved": 0, "supervisor_rounds": 0}

    # -- one action ---------------------------------------------------------
    def plan(self, obs, world):
        """The command for this action: transit, in-room sweep, or floor frontier."""
        p = self.policy
        cost = assemble_cost_grid(p.planner.fields_for(world), p.planner_params, p.settings.body_radius_m)[0]
        flags = {}
        for _ in range(self.settings.supervisor_rounds):
            state = self._settled(obs, world, cost, **flags)
            if state.state == TRANSIT and state.goal_xy is not None:
                command = p._navigate(obs, world, state.goal_xy, "transit/%s" % state.room_id)
                if command is not None:
                    return command
                self.stats["plan_failures"] += 1
                flags = {"route_failed": True}
            elif state.state == SEARCH:
                command = self._room(obs, world, cost, state)
                if command is not None:
                    return command
                flags = {"frontier_exhausted": True}
            else:
                break
            self.stats["supervisor_rounds"] += 1
        command = self._explore(obs, world, cost, np.ones(world.grid.shape, bool))
        if command is not None:
            return command
        if p.building:
            command = p.building.plan(obs, world, exhausted=True)
            if command is not None:
                return command
        return NavigationCommand.hold(info={"reason": "no safe observed frontier; acquire another view"})

    # -- the supervisor -----------------------------------------------------
    def _settled(self, obs, world, cost, **flags):
        """Advance the supervisor; a released room is replaced on this same action."""
        p = self.policy
        state = self._update(obs, world, cost, **flags)
        if state.completed is not None:
            p._route = p._goal = None
            p.route_memory.clear("room_completed")
            self.stats["rooms_released"] += 1
            state = self._update(obs, world, cost)
        if state.changed and state.state == SEARCH:
            self.scans[(p.mapping.floor_id, state.room_id)] = 0
        return state

    def _update(self, obs, world, cost, frontier_exhausted=False, route_failed=False):
        p, s, pose = self.policy, self.policy.settings, obs.pose
        instance = None
        if p.graph.options and p.supervisor.state == "select":
            instance, _ = build_instance(world, cost, {r.room_id: r.xy for r in p.graph.options}, p.graph.probs,
                                         depot_xy=(pose.x, pose.y),
                                         cruise_speed_mps=p.episode.action_spec.forward_step_m / s.action_time_s)
        calls = p.solver.calls
        state = p.supervisor.update(p.graph.options, p.graph.facts, (pose.x, pose.y), p._floor_time,
                                    last_plan_s=p._last_plan_s, instance=instance, blocked_since=p._blocked_since,
                                    frontier_exhausted=frontier_exhausted, route_failed=route_failed)
        if p.solver.calls != calls:
            data = asdict(p.solver.last)
            p._solver_records.append({k: None if isinstance(v, float) and not math.isfinite(v) else v
                                      for k, v in data.items()})
        return state

    # -- inside the room ----------------------------------------------------
    def _room(self, obs, world, cost, state):
        """Sweep the room in force; None once it is swept and looked around."""
        room = self.policy.graph.registry.rooms.get(state.room_id)
        if room is None:
            return None
        command = self._explore(obs, world, cost, room.mask)
        if command is not None:
            return command
        return self._scan_turn(obs, (self.policy.mapping.floor_id, state.room_id))

    def _scan_turn(self, obs, key):
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

    # -- frontier goals -----------------------------------------------------
    def _explore(self, obs, world, cost, mask, kind="frontier"):
        """Navigate to the first ranked goal the planner accepts, or None."""
        for goal in self._goals(obs, world, cost, mask):
            command = self.policy._navigate(obs, world, goal, kind)
            if command is not None:
                return command
            self.stats["plan_failures"] += 1
        return None

    def _goals(self, obs, world, cost, mask):
        """The committed goal while it is still worth reaching, else fresh ranked ones."""
        p, s = self.policy, self.settings
        xy = (obs.pose.x, obs.pose.y)
        arrival = p.converter_params.goal_tolerance_m + world.resolution
        current = self._current_goal(obs, world, cost, mask, arrival)
        if current is not None:
            return [current]
        visited = p._visited_frontiers[-s.visited_memory:]
        ranked = ranked_frontier_goals(world, cost, mask, xy, obs.pose.yaw, s.ranking)
        goals = [g.xy for g in ranked
                 if math.dist(xy, g.xy) > arrival
                 and not any(math.dist(g.xy, used) < s.visited_radius_m for used in visited)]
        return goals[:s.plan_attempts]

    def _current_goal(self, obs, world, cost, mask, arrival):
        p = self.policy
        if p._goal is None or p.route_memory.kind != "frontier":
            return None
        xy = (obs.pose.x, obs.pose.y)
        if math.dist(xy, p._goal) > arrival and not p.route_memory.arrived(obs):
            gx, gy = world.world_to_grid(*p._goal)
            if (world.in_bounds(gx, gy) and np.isfinite(cost[gy, gx])
                    and self._near(mask, gx, gy, world.resolution, self.settings.mask_slack_m)
                    and self._informative(world, gx, gy)):
                return p._goal
        p._visited_frontiers.append(p._goal)
        p._goal = p._route = None
        p.route_memory.clear("frontier_completed_or_invalid")
        self.stats["goals_resolved"] += 1
        return None

    def _informative(self, world, gx, gy):
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

