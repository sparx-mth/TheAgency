"""The room-search loop: bounded local exploration, then re-classify, re-estimate, re-order, transit.

Seven steps, as specified on 2026-09-28, each named at the line where it
happens so a recording can be read against this docstring:

* **Background, every action** -- ``rpt_policy._search`` refreshes the scene
  graph without an LLM call: room geometry, object->room association, doors,
  cumulative search time and remaining frontier clusters per room. Stairs and
  floor transitions belong to the multi-floor policy, which this loop yields
  to before it does anything of its own.
* **1. Local exploration, bounded to the room in force** (:meth:`_local`).
  Goals come from inside the room's mask, and the route to each is planned on
  a COPY of the map in which every other room's cells are blocked, so no route
  crosses a door or takes the stairs. It ends after :attr:`LoopSettings
  .local_steps` actions or when nothing reachable is left in the room --
  whichever comes first -- and the supervisor is told so on that same action.
* **2. Global re-classification** (:meth:`_reason`) -- ``graph.reason``
  re-labels every known room from every confirmed object observed so far.
* **3. Target probability per room** -- the same call: the oracle's
  P(target is in the room AND the search ends there), from the room's
  remaining frontiers, cumulative search time and type. Distance is an input
  to the method but is consumed by the RPT* objective in step 4 rather than
  multiplied into the probability -- doing both would charge travel twice.
  The per-room record in :attr:`RoomSearchLoop.estimates` carries it.
* **4. Visit order** -- the supervisor's SELECT hands the surviving rooms and
  the arc-weight instance to RPT*; the instance charges the local budget as
  per-room service time, and ``resolve_on_release`` makes every loop point a
  fresh solve rather than the next room of a stale order.
* **5. A\\*** -- the weighted A* behind ``policy._navigate``.
* **6. Direct transit to the closest frontier INSIDE the chosen room**
  (:meth:`_transit`). One extraction and one Dijkstra give every room its
  nearest reachable frontier; a room with none keeps its centroid so the
  detector can still get a close look.
* **7. Reset** -- arrival is the aircraft's own cell inside the room's mask
  (the supervisor only sees a point); it resets the local counter and the
  loop returns to step 1.

Steps 2-6 run on the SAME action a room's turn ends: a released room is
replaced by a transit at once, never by a throwaway floor-wide route. The LLM
is called at loop points only -- and once at the start, before any room has a
probability -- which makes ``local_steps`` its cadence too.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import time

import numpy as np

from sparx_agency.core.planning.environment import OccupancyGrid2D
from sparx_agency.core.planning.exploration.frontier_ranking import frontier_goals_by_room
from sparx_agency.core.planning.exploration.object_search_supervisor import (
    ALL_VERDICTS, BUDGET_SPENT, EXHAUSTED, MAPPED, SEARCH, SELECT, TRANSIT, ObjectSearchParams)
from sparx_agency.core.planning.exploration.room_costs import build_instance
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.planners.astar.cost_grid_2d import assemble_cost_grid


@dataclass(frozen=True)
class LoopSettings:
    """The loop's own knobs. Counts are actions.

    Attributes:
        local_steps: N -- how many actions a room may be explored for before
            its turn ends and the loop point runs. Also the LLM cadence: one
            reasoning round per local burst, plus one per room released for
            any other reason.
        supervisor_rounds: Supervisor rounds allowed on one action before the
            loop falls back to the floor-wide frontier. A guard against a
            transition loop nobody has found yet, not a budget: the longest
            legitimate chain is five -- a budget release and reselect, an
            arrival into the room the aircraft already stands in, that room
            found swept and released, a reselect whose transit route the
            planner refuses, and the reselect after that. A run that hits
            the guard is counted and logged (``rounds_exhausted``).
        confine_routes: Plan in-room routes on a copy of the map with every
            other room blocked. Off, the room mask still bounds the GOALS but
            a route to one of them may cut through a neighbouring room.
        entry_frontier: Transit to the nearest reachable frontier inside the
            chosen room. Off, the room's centroid -- what the loop flew
            before, and what a room with no frontier left still gets.
    """

    local_steps: int = 10
    supervisor_rounds: int = 5
    confine_routes: bool = True
    entry_frontier: bool = True

    def __post_init__(self):
        for name in ("local_steps", "supervisor_rounds"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("%s must be a positive integer" % name)
        for name in ("confine_routes", "entry_frontier"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError("%s must be a bool" % name)

    def supervisor_params(self, seed):
        """The supervisor configuration the loop needs.

        A room whose local budget ran out is NOT put on the visit cooldown --
        the fresh estimate may rightly send the aircraft straight back into it
        -- and the room order is re-solved at every release, because the
        estimate it was built on has just been replaced. A room the live map
        says is finished (exhausted or mapped) is never repeated by the
        every-room-cooling escape hatch: the loop falls back to the floor-wide
        frontier instead, so the search goes on without re-entering it.
        """
        return ObjectSearchParams(
            seed=seed, resolve_on_release=True,
            cooldown_verdicts=tuple(v for v in ALL_VERDICTS if v != BUDGET_SPENT),
            repeat_verdicts=tuple(v for v in ALL_VERDICTS if v not in (EXHAUSTED, MAPPED)))


class RoomSearchLoop:
    """Runs the seven steps over the policy's supervisor, scene graph and sweep.

    Attributes:
        local_steps: Actions spent in the room in force since arrival.
        room_id: The room being explored, or None outside step 1.
        estimates: Step 3's record from the latest loop point, per room:
            label, probability, remaining frontiers, cumulative search time,
            distance the order was charged, and how the room is entered.
        events: One entry per step transition, for the recording.
    """

    def __init__(self, policy, settings=None):
        self.policy = policy
        self.settings = settings or LoopSettings()
        self.local_steps = 0
        self.room_id = None
        self._entry = None            # (xy, from_frontier) of the transit in force
        self._entries = {}            # room_id -> (xy, from_frontier) at the last SELECT
        self._needs_reason = True
        self.estimates = {}
        self.events = []
        self.stats = {"loop_points": 0, "llm_reasonings": 0, "rooms_released": 0,
                      "budget_releases": 0, "exhausted_releases": 0, "skipped_in_transit": 0,
                      "arrivals": 0, "entry_frontier": 0, "entry_centroid": 0, "entry_reaimed": 0, "entry_lost": 0,
                      "confined_actions": 0, "unconfined_actions": 0, "plan_failures": 0,
                      "supervisor_rounds": 0, "rounds_exhausted": 0}

    # -- the action ledger --------------------------------------------------
    def charge(self):
        """One action was emitted while a room was being explored: count it toward N."""
        if self.policy.supervisor.state == SEARCH and self.room_id is not None:
            self.local_steps += 1

    # -- one action ---------------------------------------------------------
    def plan(self, obs, world):
        """The command for this action: in-room exploration, transit, or the floor-wide frontier."""
        p = self.policy
        cost = assemble_cost_grid(p.planner.fields_for(world), p.planner_params, p.settings.body_radius_m)[0]
        flags = self._budget_flags()
        for _ in range(self.settings.supervisor_rounds):
            state = self._tick(obs, world, cost, **flags)
            if state.state == TRANSIT and state.room_id is not None:
                command, flags = self._transit(obs, world, cost, state)
            elif state.state == SEARCH and state.room_id is not None:
                command, flags = self._local(obs, world, cost, state)
            else:
                flags = {}
                break
            if command is not None:
                return command
            self.stats["supervisor_rounds"] += 1
        if flags:
            # The chain of same-action transitions outran the guard with a verdict
            # still pending. Visible, because the next action starts afresh and
            # the verdict is re-derived rather than carried over.
            self.stats["rounds_exhausted"] += 1
            self._log(obs, "rounds_exhausted", pending=sorted(flags))
        return self._fallback(obs, world, cost)

    def _budget_flags(self):
        """Step 1's N-step rule, judged before the first tick so the release costs no round."""
        p = self.policy
        if (p.supervisor.state == SEARCH and self.room_id is not None
                and self.local_steps >= self.settings.local_steps):
            self.stats["budget_releases"] += 1
            return {"budget_spent": True}
        return {}

    # -- the supervisor and the loop point --------------------------------
    def _tick(self, obs, world, cost, **flags):
        """Advance the supervisor; a released room is replaced on this same action."""
        p = self.policy
        state = self._update(obs, world, cost, **flags)
        if state.completed is not None:
            room_id, verdict = state.completed
            p._route = p._goal = None
            p.route_memory.clear("room_completed")
            self._entry = None
            self.stats["rooms_released"] += 1
            self.stats["loop_points"] += 1
            self._log(obs, "release", room=room_id, verdict=verdict, local_steps=self.local_steps)
            self.room_id = None
            self._needs_reason = True                       # step 7 -> steps 2-4 on this action
            state = self._update(obs, world, cost)
        if state.changed and state.state == TRANSIT:
            self._entry = self._entries.get(state.room_id, (state.goal_xy, False))
            self.stats["entry_frontier" if self._entry[1] else "entry_centroid"] += 1
            self._log(obs, "transit", room=state.room_id, goal=list(state.goal_xy),
                      entry="frontier" if self._entry[1] else "centroid", order=list(state.order))
        if state.changed and state.state == SEARCH:
            self.local_steps = 0                            # step 7
            self.room_id = state.room_id
            p.sweep.begin_scan((p.mapping.floor_id, state.room_id))
            self.stats["arrivals"] += 1
            self._log(obs, "arrive", room=state.room_id)
        return state

    def _update(self, obs, world, cost, frontier_exhausted=False, route_failed=False,
                budget_spent=False, arrived=False):
        p, s, pose = self.policy, self.policy.settings, obs.pose
        rooms = p.graph.registry.rooms
        options, instance = p.graph.options, None
        if p.supervisor.state == SELECT and rooms:
            # Steps 2 and 3 run at a loop point, and otherwise only when a room
            # the estimate has never seen appears while nothing is in force --
            # a room without a probability can never be chosen, and no release
            # is coming to give it one.
            if self._needs_reason or any(pid not in p.graph.probs for pid in rooms):
                self._reason(obs, world)
            options = self._options(obs, world, cost)       # step 6's entry point per room
            if options:
                started = time.monotonic()
                instance, dropped = build_instance(
                    world, cost, {o.room_id: o.xy for o in options}, p.graph.probs,
                    depot_xy=(pose.x, pose.y),
                    cruise_speed_mps=p.episode.action_spec.forward_step_m / s.action_time_s,
                    search_time_s=self.settings.local_steps * s.action_time_s)
                self._record(instance, dropped, options)
                p.telemetry.latencies["room_selection"].append((time.monotonic() - started) * 1000)
        calls = p.solver.calls
        state = p.supervisor.update(                        # step 4 in SELECT
            options, p.graph.facts, (pose.x, pose.y), p._floor_time,
            last_plan_s=p._last_plan_s, instance=instance, blocked_since=p._blocked_since,
            frontier_exhausted=frontier_exhausted, route_failed=route_failed,
            budget_spent=budget_spent, arrived=arrived)
        if p.solver.calls != calls:
            data = asdict(p.solver.last)
            p._solver_records.append({k: None if isinstance(v, float) and not math.isfinite(v) else v
                                      for k, v in data.items()})
        return state

    def _reason(self, obs, world):
        """Steps 2 and 3: every room re-labelled from every object so far, then P(target) per room.

        Over geometry refreshed on THIS action: the background refresh runs
        every ``graph_period_steps`` actions, and a loop point that fell
        between two of them would otherwise estimate rooms whose frontier
        counts and areas are up to a period old.
        """
        p = self.policy
        if p._last_graph_step != obs.step:
            started = time.monotonic()
            p.graph.update(world, p.landmarks.confirmed(), p.target, doors=p.doors.confirmed(),
                           step=obs.step, reason=False)
            p.telemetry.latencies["scene_graph"].append((time.monotonic() - started) * 1000)
            p._last_graph_step, p._last_door_revision = obs.step, p.doors.revision
        started = time.monotonic()
        try:
            p.graph.reason(world, p.target, obs.step)
        finally:
            p.telemetry.latencies["room_reasoning"].append((time.monotonic() - started) * 1000)
        self._needs_reason = False
        self.stats["llm_reasonings"] += 1

    def _options(self, obs, world, cost):
        """Every room with its entry point: the nearest admissible frontier inside it, else its centroid.

        Admissible by the sweep's own rule -- not under the agent, not beside a
        retired goal -- so a room the sweep would find nothing left in is
        entered at its centroid, never sent back to a frontier the sweep will
        refuse on arrival.
        """
        p = self.policy
        by_room = {}
        if self.settings.entry_frontier and p.graph.labels is not None:
            by_room = frontier_goals_by_room(world, cost, p.graph.labels, (obs.pose.x, obs.pose.y),
                                             obs.pose.yaw, p.sweep.settings.ranking)
        self._entries, options = {}, []
        for option in p.graph.options:
            goals = p.sweep.admissible(obs, world, by_room.get(option.room_id + 1, ()))
            if goals:
                nearest = min(goals, key=lambda g: g.geodesic_m)
                xy, from_frontier = nearest.xy, True
            else:
                xy, from_frontier = option.xy, False
            self._entries[option.room_id] = (xy, from_frontier)
            options.append(replace(option, xy=xy))
        return options

    def _record(self, instance, dropped, options):
        """Step 3's record, with the distance step 4 charges each room."""
        p = self.policy
        labels = {o.room_id: o.label for o in options}
        cruise = p.episode.action_spec.forward_step_m / p.settings.action_time_s
        service = self.settings.local_steps * p.settings.action_time_s
        estimates = {}
        for i, pid in enumerate(instance.index_to_pid):
            if pid < 0:
                continue
            leg = float(instance.C[instance.depot, i])
            if instance.units == "seconds":
                leg = max(0.0, leg - service) * cruise
            estimates[pid] = self._estimate(pid, labels, distance_m=leg)
        for pid in dropped:
            estimates[pid] = self._estimate(pid, labels, distance_m=None)
        self.estimates = estimates

    def _estimate(self, pid, labels, distance_m):
        p = self.policy
        fact = p.graph.facts.get(pid)
        entry = self._entries.get(pid)
        return {"label": labels.get(pid, "?"), "prob": float(p.graph.probs.get(pid, 0.0)),
                "frontier_clusters": None if fact is None else int(fact.frontier_clusters),
                "searched_s": None if fact is None else float(fact.time_in_room_s),
                "distance_m": distance_m,
                "entry": "unreachable" if distance_m is None else
                ("frontier" if entry is not None and entry[1] else "centroid")}

    # -- step 6: transit ----------------------------------------------------
    def _transit(self, obs, world, cost, state):
        """Fly to the chosen room's entry point; report arrival, or that nothing is left to reach.

        Arrival is the agent's cell inside the room's mask -- or, when the route
        has ended, within the sweep's mask slack of it: the entry point is a
        frontier on the room's boundary, and the watershed erodes its masks by
        the minimum clearance, so an agent standing exactly on its entry point
        routinely reads as just outside the room it has plainly reached.

        A route that ends somewhere the room is NOT is the map having
        re-segmented under the entry point while the agent walked to it. The
        loop re-aims once at the room's frontier nearest now; if the room has
        none left the transit is released as unreachable. Neither waits: the
        first Ranchester run spent 31 actions turning in place at such a route
        end until the route memory's no-progress clock released the room.
        """
        p = self.policy
        room = p.graph.registry.rooms.get(state.room_id)
        if room is None:
            return None, {"route_failed": True}
        entry = self._entry if self._entry is not None else (state.goal_xy, False)
        xy_goal, from_frontier = entry
        arrival = p.converter_params.goal_tolerance_m + world.resolution
        at_goal = (p.route_memory.arrived(obs)
                   or math.dist((obs.pose.x, obs.pose.y), xy_goal) <= arrival)
        inside = self._inside(obs, world, room)
        if inside and (not from_frontier or at_goal or not self._informative(world, xy_goal)):
            return None, {"arrived": True}          # step 7
        if at_goal and not inside:
            if self._near_room(obs, world, room):
                return None, {"arrived": True}      # step 7, on the boundary the mask stops short of
            p._visited_frontiers.append(tuple(xy_goal))
            p.route_memory.clear("transit_entry_lost")
            p._route = p._goal = None
            nearest = p.sweep.nearest(obs, world, cost, room.mask, labels=p.graph.labels, label=room.id + 1)
            self.stats["entry_lost"] += 1
            self._log(obs, "entry_lost", room=state.room_id, goal=[round(v, 2) for v in xy_goal],
                      reaimed=nearest is not None)
            if nearest is None:
                return None, {"route_failed": True}
            self._entry = (nearest.xy, True)
            self.stats["entry_reaimed"] += 1
            xy_goal, from_frontier = self._entry
        goal = self._entry_goal(obs, world, cost, room, xy_goal, from_frontier)
        if goal is None:
            self.stats["skipped_in_transit"] += 1
            return None, {"frontier_exhausted": True}
        command = p._navigate(obs, world, goal, "transit/%s" % state.room_id)     # step 5
        if command is not None:
            return command, {}
        self.stats["plan_failures"] += 1
        return None, {"route_failed": True}

    def _entry_goal(self, obs, world, cost, room, xy_goal, from_frontier):
        """The committed entry point while it is worth reaching; else the room's nearest frontier; else None."""
        if not from_frontier:
            return xy_goal
        gx, gy = world.world_to_grid(*xy_goal)
        if world.in_bounds(gx, gy) and np.isfinite(cost[gy, gx]) and self.policy.sweep.informative(world, gx, gy):
            return xy_goal
        nearest = self.policy.sweep.nearest(obs, world, cost, room.mask,
                                            labels=self.policy.graph.labels, label=room.id + 1)
        if nearest is None:
            return None
        self._entry = (nearest.xy, True)
        self.stats["entry_reaimed"] += 1
        return nearest.xy

    # -- step 1: local exploration --------------------------------------------
    def _local(self, obs, world, cost, state):
        """Explore the room in force, and only it, until nothing reachable is left.

        The other half of the termination rule -- N local steps -- is judged
        in :meth:`_budget_flags` before the first tick of the action.
        """
        p = self.policy
        room = p.graph.registry.rooms.get(state.room_id)
        if room is None:
            return None, {"frontier_exhausted": True}
        planning_world, planning_cost = self._confined(obs, world, cost, room, state.room_id)
        command = p.sweep.explore(obs, world, planning_cost, room.mask, planning_world=planning_world,
                                  labels=p.graph.labels, label=state.room_id + 1)
        if command is not None:
            return command, {}
        command = p.sweep.scan_turn(obs, (p.mapping.floor_id, state.room_id))
        if command is not None:
            return command, {}
        self.stats["exhausted_releases"] += 1
        return None, {"frontier_exhausted": True}

    def _confined(self, obs, world, cost, room, pid):
        """Step 1's constraint: a planning copy of the map with every other room blocked.

        Other rooms' cells are written UNKNOWN, which the planner treats as
        impassable without inflating them into the room in force. Applied
        only while the aircraft's own cell is in the room: standing in a
        doorway, or on a cell the last re-segmentation gave to a neighbour, a
        confined map would wall the aircraft out of the room it is meant to
        sweep and every in-room goal would read unreachable.
        """
        p = self.policy
        labels = p.graph.labels
        if not self.settings.confine_routes or labels is None:
            return world, cost
        if not self._inside(obs, world, room):
            self.stats["unconfined_actions"] += 1
            return world, cost
        others = (labels > 0) & (labels != pid + 1)
        if not others.any():
            return world, cost
        started = time.monotonic()
        data = world.grid.copy()
        data[others] = world.values.unknown
        confined = OccupancyGrid2D(data, world.params, values=world.values)
        confined_cost = assemble_cost_grid(p.planner.fields_for(confined), p.planner_params,
                                           p.settings.body_radius_m)[0]
        p.telemetry.latencies["room_confinement"].append((time.monotonic() - started) * 1000)
        self.stats["confined_actions"] += 1
        return confined, confined_cost

    # -- nothing to work in -------------------------------------------------
    def _fallback(self, obs, world, cost):
        """No room in force: the floor-wide frontier, then the stairs, then hold for another view."""
        p = self.policy
        command = p.sweep.explore(obs, world, cost, np.ones(world.grid.shape, bool))
        if command is not None:
            return command
        if p.building:
            command = p.building.plan(obs, world, exhausted=True)
            if command is not None:
                return command
        return NavigationCommand.hold(info={"reason": "no safe observed frontier; acquire another view"})

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _inside(obs, world, room):
        gx, gy = world.world_to_grid(obs.pose.x, obs.pose.y)
        return bool(world.in_bounds(gx, gy) and room.mask[gy, gx])

    def _near_room(self, obs, world, room):
        """Within the sweep's mask slack of the room -- on the boundary its eroded mask stops short of."""
        gx, gy = world.world_to_grid(obs.pose.x, obs.pose.y)
        return world.in_bounds(gx, gy) and self.policy.sweep._near(
            room.mask, gx, gy, world.resolution, self.policy.sweep.settings.mask_slack_m)

    def _informative(self, world, xy):
        gx, gy = world.world_to_grid(*xy)
        return world.in_bounds(gx, gy) and self.policy.sweep.informative(world, gx, gy)

    def _log(self, obs, event, **fields):
        self.events.append(dict(step=int(obs.step), floor_id=self.policy.mapping.floor_id,
                                event=event, **fields))

    def diagnostics(self):
        return {"settings": asdict(self.settings), "stats": dict(self.stats),
                "local_steps": self.local_steps, "room_id": self.room_id,
                "estimates": {str(pid): dict(item) for pid, item in self.estimates.items()},
                "events": list(self.events)}



