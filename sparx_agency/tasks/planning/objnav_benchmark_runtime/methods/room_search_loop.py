"""The room-search loop: bounded local exploration, then re-classify, re-estimate, re-order, transit.

Seven steps, as specified on 2026-09-28, each named at the line where it
happens so a recording can be read against this docstring:

* **Background, every action** -- ``rpt_policy._search`` refreshes the scene
  graph without an LLM call: room geometry, object->room association, doors,
  cumulative search time and remaining frontier clusters per room. A committed
  stair climb belongs to the multi-floor policy, which this loop yields to
  before it does anything of its own.
* **1. Local exploration, bounded to the room in force** (:meth:`_local`).
  Goals come from inside the room's mask, and the route to each is planned on
  a COPY of the map in which every other room's cells are blocked, so no route
  crosses a door or takes the stairs. It ends after :attr:`LoopSettings
  .local_steps` actions, when nothing reachable is left in the room, or when
  the room is RE-IDENTIFIED -- whichever comes first -- and the supervisor is
  told so on that same action. The re-identification is the clue rule: one
  confirmed object names the room (weakly; two distinct classes strongly),
  the classifier is re-asked for that one room the action a new KIND of
  object lands in it, and a changed name ends the room's turn
  (``reclassified``) so that steps 2-4 run over the new fact. Whether a
  bathroom is worth sweeping for a frying pan is then the oracle's call,
  and it says so with a probability of about zero.
* **2. Global re-classification** (:meth:`_reason`) -- ``graph.reason``
  re-labels every known room from every confirmed object observed so far.
* **3. Probability per NODE** -- the same call, to
  :class:`~sparx_agency.core.mapping.topology.search_node_oracle.SearchNodeOracle`:
  for every room on the floor -- its type or ``unknown``, size, frontier
  left, time searched and how long ago, objects seen -- and for every
  STAIRCASE off the floor -- up or down, whether the other storey was
  visited and what was found there -- the probability that going there NEXT
  finds the target, plus the mass in none of them. Everything the user
  listed is the model's to weigh: an irrelevant type and a fully observed
  room read as zero, a room searched long and recently as low, a large
  unknown room with frontier as an exploration node worth a look, a
  storey not yet stood on as the whole set of rooms it may hold. Distance
  is NOT the model's concern: it is charged by RPT* in step 4.
* **4. Visit order** -- the supervisor's SELECT hands the surviving nodes and
  the arc-weight instance to RPT*: rooms at their entry points, staircases at
  the foot of the flight with a LEAF as long as the flight plus the fixed
  cost of a storey change on every arc into and out of them (going upstairs
  puts every room down here that much further away). The instance charges
  the local budget as per-room service time, and ``resolve_on_release``
  makes every loop point a fresh solve rather than the next node of a stale
  order. Nothing else decides a floor change: no allowance, no clock.
* **5. A\\*** -- the weighted A* behind ``policy._navigate``.
* **6. Direct transit** (:meth:`_transit`) -- to the closest frontier INSIDE
  the chosen room (one extraction and one Dijkstra give every room its
  nearest reachable frontier; a room with none keeps its centroid so the
  detector can still get a close look), or to the foot of the chosen stairs,
  where the building coordinator takes over and climbs; the node's turn ends
  there (``traversed``) and the floor's search resumes clean when the
  aircraft comes back down.
* **7. Reset** -- arrival is the aircraft's own cell inside the room's mask
  (the supervisor only sees a point); it resets the local counter and the
  loop returns to step 1.

Steps 2-6 run on the SAME action a room's turn ends: a released room is
replaced by a transit at once, never by a throwaway floor-wide route. The LLM
is called at loop points only -- and once at the start, before any node has a
probability -- which makes ``local_steps`` its cadence too. That one call per
loop point is the judgement the search cannot afford to get wrong, so the
runtime routes it to the LLM client's REASONING model (``LLM_REASONING_MODEL``)
and waits for it.

When the loop has nothing to work in -- no node in force, every transit
refused, or the room LLM failed and is backing off -- the action goes to the
policy's :class:`~sparx_agency.tasks.planning.objnav_benchmark_runtime.methods
.exploration_fallback.ExplorationFallback`: the nearest reachable frontier
anywhere on the floor, the stairs by the explicit fallback rule, a retired
frontier, a relocation. Never an idle hold while a move exists.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import time

import numpy as np

from sparx_agency.core.planning.environment import OccupancyGrid2D
from sparx_agency.core.planning.exploration.frontier_ranking import frontier_goals_by_room
from sparx_agency.core.planning.exploration.object_search_supervisor import (
    ALL_VERDICTS, BUDGET_SPENT, EXHAUSTED, MAPPED, NEUTRAL, SEARCH, SELECT, TRANSIT, TRAVERSED, ObjectSearchParams)
from sparx_agency.core.planning.exploration.room_costs import build_instance
from sparx_agency.core.planning.planners.astar.cost_grid_2d import assemble_cost_grid
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.exploration_fallback import ROOM_LLM
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.stair_nodes import (
    is_stair_node, portal_id_of, search_context, stair_node_id, stair_options)


class RoomReasoningUnavailable(Exception):
    """The room LLM failed or is backing off: this action belongs to the exploration fallback."""


@dataclass(frozen=True)
class LoopSettings:
    """The loop's own knobs. Counts are actions.

    Attributes:
        local_steps: N -- how many actions a room may be explored for before
            its turn ends and the loop point runs. Also the LLM cadence: one
            reasoning round per local burst, plus one per room released for
            any other reason. The one action count in the loop, and the
            user's own step 1; it is also the service time RPT* charges a
            room for.
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
        stairs_as_nodes: Offer the floor's staircases to the oracle and the
            solver as nodes beside the rooms (the default). Off, the floor is
            searched alone and the stairs are only ever the exploration
            fallback's last resort -- the ablation, not a mode to fly.
    """

    local_steps: int = 10
    supervisor_rounds: int = 5
    confine_routes: bool = True
    entry_frontier: bool = True
    stairs_as_nodes: bool = True

    def __post_init__(self):
        for name in ("local_steps", "supervisor_rounds"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("%s must be a positive integer" % name)
        for name in ("confine_routes", "entry_frontier", "stairs_as_nodes"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError("%s must be a bool" % name)

    def supervisor_params(self, seed):
        """The supervisor configuration the loop needs.

        A room whose local budget ran out is NOT put on the visit cooldown --
        the fresh estimate may rightly send the aircraft straight back into it
        -- and the room order is re-solved at every release, because the
        estimate it was built on has just been replaced. Neither is a
        staircase whose climb began (``traversed``): on the way back down the
        oracle is told the aircraft arrived by it, and the way back up is its
        to value. A neutral release (``reclassified``) cools nothing either.
        A room the live map says is finished (exhausted or mapped) is never
        repeated by the every-room-cooling escape hatch: the loop falls back
        to the floor-wide frontier instead, so the search goes on without
        re-entering it.
        """
        return ObjectSearchParams(
            seed=seed, resolve_on_release=True,
            cooldown_verdicts=tuple(v for v in ALL_VERDICTS if v not in (BUDGET_SPENT, TRAVERSED) + NEUTRAL),
            repeat_verdicts=tuple(v for v in ALL_VERDICTS if v not in (EXHAUSTED, MAPPED)))


class RoomSearchLoop:
    """Runs the seven steps over the policy's supervisor, scene graph and sweep.

    Attributes:
        local_steps: Actions spent in the room in force since arrival.
        room_id: The room being explored, or None outside step 1.
        order: The node ids of the solver's latest order, in visit order --
            room pids and stair node ids together.
        order_index: How far into that order the supervisor is.
        next_room: The node the loop is heading to or standing in, or None.
        estimates: Step 3's record from the latest loop point, per node:
            kind, label, probability, remaining frontiers, cumulative search
            time, distance the order was charged (a staircase's includes the
            climb), and how the node is entered.
        events: One entry per step transition, for the recording.
    """

    def __init__(self, policy, settings=None):
        self.policy = policy
        self.settings = settings or LoopSettings()
        self.local_steps = 0
        self.room_id = None
        self.order, self.order_index, self.next_room = (), 0, None
        self._entry = None            # (xy, from_frontier) of the transit in force
        self._entries = {}            # node_id -> (xy, from_frontier) at the last SELECT
        self._needs_reason = True
        self._stairs = {}             # node_id -> StairOption offered at the last SELECT
        self.estimates = {}
        self.events = []
        self.estimate_events = []
        self.stats = {"loop_points": 0, "llm_reasonings": 0, "rooms_released": 0,
                      "budget_releases": 0, "exhausted_releases": 0, "reclassified_releases": 0,
                      "reclassified_in_transit": 0, "relabels": 0,
                      "stairs_offered": 0, "stairs_chosen": 0, "stairs_taken": 0, "stairs_refused": 0,
                      "skipped_in_transit": 0,
                      "arrivals": 0, "entry_frontier": 0, "entry_centroid": 0, "entry_reaimed": 0, "entry_lost": 0,
                      "confined_actions": 0, "unconfined_actions": 0, "plan_failures": 0,
                      "supervisor_rounds": 0, "rounds_exhausted": 0, "llm_fallbacks": 0}

    # -- the action ledger --------------------------------------------------
    def charge(self):
        """One action was emitted while a room was being explored: count it toward N."""
        if self.policy.supervisor.state == SEARCH and self.room_id is not None:
            self.local_steps += 1

    def reconsider(self, obs, world):
        """A completed peek or new off-route semantic clue warrants a fresh decision.

        Reusing the selected room preserves its local action counter. Switching
        releases it neutrally; no visit cooldown punishes a useful new clue.
        """
        p, pose = self.policy, obs.pose
        cost = assemble_cost_grid(p.planner.fields_for(world), p.planner_params, p.settings.body_radius_m)[0]
        stairs = self._stair_options(obs, world, cost)
        self._reason(obs, world, stairs)
        options = self._options(obs, world, cost, stairs)
        instance = None
        if options:
            values = dict(p.graph.probs)
            values.update(p.graph.stair_probs)
            instance, dropped = build_instance(
                world, cost, {o.room_id: o.xy for o in options}, values, depot_xy=(pose.x, pose.y),
                cruise_speed_mps=p.episode.action_spec.forward_step_m / p.settings.action_time_s,
                search_time_s=self.settings.local_steps * p.settings.action_time_s,
                leaves={o.node_id: o.leaf_m for o in stairs})
            self._record(instance, dropped, options)
            options = [o for o in options if o.room_id in instance.index_to_pid]
        calls = p.solver.calls
        state = p.supervisor.reconsider(options, p._floor_time, instance)
        if p.solver.calls != calls:
            data = asdict(p.solver.last)
            p._solver_records.append({k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in data.items()})
        if state.completed is not None:
            self._log(obs, "semantic_switch", room=state.completed[0], local_steps=self.local_steps)
            self.room_id = None
            self._entry = None
            p.route_memory.clear("semantic_switch")
            p._route = p._goal = None
        self.order, self.order_index = tuple(state.order), 0
        self._log(obs, "semantic_replan", order=list(self.order), retained=state.completed is None)

    # -- one action ---------------------------------------------------------
    def plan(self, obs, world):
        """The command for this action: in-room exploration, transit, or the exploration fallback."""
        p = self.policy
        cost = assemble_cost_grid(p.planner.fields_for(world), p.planner_params, p.settings.body_radius_m)[0]
        flags = self._budget_flags()
        try:
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
        except RoomReasoningUnavailable as exc:
            # The loop point cannot run without the room LLM; the search does not
            # wait for it. The fallback carries every action until the back-off ends.
            self.stats["llm_fallbacks"] += 1
            return self._fallback(obs, world, cost, reason="room LLM unavailable: %s" % exc)
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
            if is_stair_node(state.room_id):
                entry = "stairs"
                self.stats["stairs_chosen"] += 1
            else:
                entry = "frontier" if self._entry[1] else "centroid"
                self.stats["entry_" + entry] += 1
            self._log(obs, "transit", room=state.room_id, goal=list(state.goal_xy), entry=entry, order=list(state.order))
        if state.changed and state.state == SEARCH:
            self.local_steps = 0                            # step 7
            self.room_id = state.room_id
            p.sweep.begin_scan((p.mapping.floor_id, state.room_id))
            self.stats["arrivals"] += 1
            self._log(obs, "arrive", room=state.room_id)
        self.order, self.order_index = tuple(state.order), int(state.order_index)
        self.next_room = state.room_id if state.state in (TRANSIT, SEARCH) else None
        return state

    def _update(self, obs, world, cost, frontier_exhausted=False, route_failed=False,
                budget_spent=False, arrived=False, room_reclassified=False):
        p, s, pose = self.policy, self.policy.settings, obs.pose
        rooms = p.graph.registry.rooms
        options, instance = p.graph.options, None
        if p.supervisor.state == SELECT and rooms:
            # Steps 2 and 3 run at a loop point, and otherwise only when a node
            # the estimate has never seen appears while nothing is in force --
            # a node without a probability can never be chosen, and no release
            # is coming to give it one.
            stairs = self._stair_options(obs, world, cost)
            unvalued = (any(pid not in p.graph.probs for pid in rooms)
                        or any(o.node_id not in p.graph.stair_probs for o in stairs))
            if self._needs_reason or unvalued:
                self._reason(obs, world, stairs)
            options = self._options(obs, world, cost, stairs)      # step 6's entry point per node
            if options:
                started = time.monotonic()
                values = dict(p.graph.probs)
                values.update(p.graph.stair_probs)
                instance, dropped = build_instance(
                    world, cost, {o.room_id: o.xy for o in options}, values,
                    depot_xy=(pose.x, pose.y),
                    cruise_speed_mps=p.episode.action_spec.forward_step_m / s.action_time_s,
                    search_time_s=self.settings.local_steps * s.action_time_s,
                    leaves={o.node_id: o.leaf_m for o in stairs})      # the climb, on every arc touching the stairs
                self._record(instance, dropped, options)
                options = [o for o in options if o.room_id in instance.index_to_pid]
                p.telemetry.latencies["room_selection"].append((time.monotonic() - started) * 1000)
        calls = p.solver.calls
        state = p.supervisor.update(                        # step 4 in SELECT
            options, p.graph.facts, (pose.x, pose.y), p._floor_time,
            last_plan_s=p._last_plan_s, instance=instance, blocked_since=p._blocked_since,
            frontier_exhausted=frontier_exhausted, route_failed=route_failed,
            budget_spent=budget_spent, arrived=arrived, room_reclassified=room_reclassified)
        if p.solver.calls != calls:
            data = asdict(p.solver.last)
            p._solver_records.append({k: None if isinstance(v, float) and not math.isfinite(v) else v
                                      for k, v in data.items()})
        return state

    def _stair_options(self, obs, world, cost):
        """The floor's staircases as nodes: eligible, and entered from a point the map has reached.

        Nothing here is a clock. A staircase whose foot the observed map has
        not reached is not a node THIS action and comes back when the map
        grows to it; one cooling after a failed approach comes back when the
        cooling ends. The stairs the aircraft just came down are offered like
        any other -- the oracle is told so, and values the way back.
        """
        p = self.policy
        building = getattr(p, "building", None)
        if not self.settings.stairs_as_nodes or building is None or building.ground_truth is None or building.committed:
            self._stairs = {}
            return []
        options = stair_options(building, obs, world, cost, p.floors.save(), p.settings.action_time_s)
        offered = {o.node_id: o for o in options}
        if set(offered) != set(self._stairs):
            self.stats["stairs_offered"] += len(set(offered) - set(self._stairs))
        self._stairs = offered
        return options

    def _reason(self, obs, world, stairs=()):
        """Steps 2 and 3: every room re-labelled from every object so far, then P(find) per node.

        Over geometry refreshed on THIS action: the background refresh runs
        every ``graph_period_steps`` actions, and a loop point that fell
        between two of them would otherwise estimate rooms whose frontier
        counts and areas are up to a period old. The staircases the loop
        offers are valued in the same call, beside the rooms.

        A model that fails -- timeout, malformed answer, a refused uniform
        prior -- does not end the episode and does not get asked again on the
        next action: the failure is recorded, a doubling back-off is armed on
        the policy's exploration fallback, and :class:`RoomReasoningUnavailable`
        hands this action to it.

        Raises:
            RoomReasoningUnavailable: When the room LLM is backing off or failed.
        """
        p = self.policy
        if p.fallback.service_unavailable(ROOM_LLM, obs.step):
            raise RoomReasoningUnavailable("back-off until action %d" % p.fallback.retry_step[ROOM_LLM])
        if p._last_graph_step != obs.step:
            started = time.monotonic()
            p.graph.update(world, p.landmarks.confirmed(), p.target, doors=p.doors.confirmed(),
                           step=obs.step, reason=False, cost=p.navigation_cost(world),
                           here_xy=(obs.pose.x, obs.pose.y), yaw=obs.pose.yaw, ranking=p.sweep.settings.ranking)
            p.telemetry.latencies["scene_graph"].append((time.monotonic() - started) * 1000)
            p._last_graph_step, p._last_door_revision = obs.step, p.doors.revision
        started = time.monotonic()
        try:
            p.graph.reason(world, p.target, obs.step,
                           extra_nodes=[o.node for o in stairs],
                           context=search_context(getattr(p, "building", None), p, p.floors.save()),
                           here_xy=(obs.pose.x, obs.pose.y), action_time_s=p.settings.action_time_s)
        except Exception as exc:  # the room LLM: timeout, bad JSON, refused uniform prior
            retry = p.fallback.note_service_failure(obs, ROOM_LLM, exc)
            self._log(obs, "llm_failure", error="%s: %s" % (type(exc).__name__, exc), retry_step=retry)
            raise RoomReasoningUnavailable(str(exc)) from exc
        finally:
            p.telemetry.latencies["room_reasoning"].append((time.monotonic() - started) * 1000)
        p.fallback.note_service_success(ROOM_LLM)
        self._needs_reason = False
        self.stats["llm_reasonings"] += 1
        oracle = p.graph.last_reasoning.get("oracle", {})
        # Kept apart from ``events``, which is the record of step transitions;
        # one entry per reasoning round, with what the oracle was shown and said.
        self.estimate_events.append(dict(
            step=int(obs.step), floor_id=p.mapping.floor_id, nodes=len(oracle.get("probs", {})),
            stairs=[o.node_id for o in stairs], p_present=round(float(oracle.get("p_present", 0.0)), 3),
            elsewhere=round(float(oracle.get("elsewhere", 0.0)), 3), reused=bool(oracle.get("reused")),
            omitted=list(oracle.get("omitted", ())), reading=dict(oracle.get("reading", {}))))

    # -- the clue rule: a new kind of object names the room; the oracle says what that is worth ----
    def _reclassify(self, obs, pid, where):
        """Re-ask the classifier for room ``pid`` if a new kind of object appeared in it; True when its name changed.

        A changed name is a new fact the oracle has not valued: the caller
        ends the room's turn (``reclassified``) so that steps 2-4 run over it
        on this same action -- the bathroom the sink was taken for is
        re-valued the action the toilet shows, and the order goes elsewhere
        if the model says so. One bounded classifier call for the one room
        it matters to; a set of kinds already judged is not bought again.
        """
        p = self.policy
        if is_stair_node(pid) or not p.graph.label_tracker.pending(pid) or p.fallback.service_unavailable(ROOM_LLM, obs.step):
            return False
        started = time.monotonic()
        try:
            changed = p.graph.relabel(pid, obs.step)
        except Exception as exc:  # the classifier is the room LLM too: same back-off, held label stands
            retry = p.fallback.note_service_failure(obs, ROOM_LLM, exc)
            self._log(obs, "llm_failure", error="%s: %s" % (type(exc).__name__, exc), retry_step=retry,
                      where="relabel")
            return False
        finally:
            p.telemetry.latencies["room_relabel"].append((time.monotonic() - started) * 1000)
        self.stats["relabels"] += 1
        if not changed:
            return False
        info = p.graph.label_info(pid) or {}
        self._log(obs, "relabel", room=pid, where=where, label=info.get("label"), strength=info.get("strength"),
                  objects=p.graph.objects_in(pid))
        self.stats["reclassified_releases"] += 1
        if where == "transit":
            self.stats["reclassified_in_transit"] += 1
        self._needs_reason = True
        return True

    def _options(self, obs, world, cost, stairs=()):
        """Every node with its entry point: a room's nearest admissible frontier, else its centroid; a staircase's foot.

        Admissible by the sweep's own rule -- not under the agent, not beside a
        retired goal -- so a room the sweep would find nothing left in is
        entered at its centroid, never sent back to a frontier the sweep will
        refuse on arrival.
        """
        p = self.policy
        by_room = {}
        if self.settings.entry_frontier and p.graph.labels is not None:
            inventory = p.graph.frontier_inventory
            by_room = (inventory.by_room if inventory is not None else
                       frontier_goals_by_room(world, cost, p.graph.labels, (obs.pose.x, obs.pose.y),
                                              obs.pose.yaw, p.sweep.settings.ranking))
        self._entries, options = {}, []
        for option in p.graph.options:
            goals = p.sweep.admissible(obs, world, by_room.get(option.room_id + 1, ()))
            if goals:
                nearest = min(goals, key=lambda g: g.geodesic_m)
                xy, from_frontier = nearest.xy, True
            else:
                xy, from_frontier = option.xy, False
            self._entries[option.room_id] = (xy, from_frontier)
            # The probability the supervisor filters on is the oracle's latest, whatever
            # the option carried: the two are one number, read from one place.
            options.append(replace(option, xy=xy, prob=p.graph.probs.get(option.room_id, option.prob)))
        for stair in stairs:
            self._entries[stair.node_id] = (stair.approach.xy, False)
            options.append(stair.option(p.graph.stair_probs.get(stair.node_id, 0.0)))
        return options

    def _record(self, instance, dropped, options):
        """Step 3's record, with the distance step 4 charges each node -- a staircase's includes the climb."""
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
        reasons = p.graph.last_reasoning.get("oracle", {}).get("reasons", {})
        why = reasons.get(pid, reasons.get(str(pid), ""))
        stair = self._stairs.get(pid)
        if stair is not None:
            return {"kind": "stairs", "label": stair.node.label, "direction": stair.node.direction,
                    "prob": float(p.graph.stair_probs.get(pid, 0.0)), "why": why,
                    "destination_visited": stair.node.destination_visited, "destination": stair.node.destination,
                    "arrived_by": stair.node.arrived_by, "leaf_m": round(stair.leaf_m, 2),
                    "approach_m": round(stair.approach.distance_m, 2), "portal_id": stair.portal["id"],
                    "distance_m": distance_m, "entry": "unreachable" if distance_m is None else "stairs"}
        fact = p.graph.facts.get(pid)
        entry = self._entries.get(pid)
        info = p.graph.label_info(pid) or {}
        return {"kind": "room", "label": labels.get(pid) or info.get("label", "?"), "strength": info.get("strength"),
                "prob": float(p.graph.probs.get(pid, 0.0)), "why": why,
                "objects": p.graph.objects_in(pid),
                "frontier_clusters": None if fact is None else int(fact.frontier_clusters),
                "searched_s": None if fact is None else float(fact.time_in_room_s),
                "last_inside_step": p.graph.last_inside.get(pid),
                "distance_m": distance_m,
                "entry": "unreachable" if distance_m is None else
                ("frontier" if entry is not None and entry[1] else "centroid")}

    # -- step 6: transit ----------------------------------------------------
    def _transit(self, obs, world, cost, state):
        """Fly to the chosen node's entry point; report arrival, or that nothing is left to reach.

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

        A room re-identified on the way in -- a toilet seen through the
        doorway of what a sink had made a kitchen -- ends its turn before it
        is entered, so the order is re-solved over the new fact.

        A STAIRCASE is handed to the building coordinator, which approaches
        the foot of the flight and climbs; see :meth:`_transit_stairs`.
        """
        p = self.policy
        if is_stair_node(state.room_id):
            return self._transit_stairs(obs, world, state)
        room = p.graph.registry.rooms.get(state.room_id)
        if room is None:
            return None, {"route_failed": True}
        if self._reclassify(obs, state.room_id, "transit"):
            return None, {"room_reclassified": True}
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

    # -- step 6 for a staircase: the building coordinator takes it from here ------
    def _transit_stairs(self, obs, world, state):
        """The order put a staircase first: commit it to the building coordinator and let it climb.

        The coordinator plans the approach to the foot of the flight and,
        on reaching it, starts the traversal -- at which point it calls
        :meth:`stairs_taken` and this node's turn ends. Until then every
        action is the coordinator's (``building.committed``), and the loop is
        not ticked. A staircase the coordinator has since deferred (a failed
        approach), or that the map no longer reaches, is released as
        unreachable and the order re-solved without it.
        """
        p = self.policy
        building = getattr(p, "building", None)
        portal = building.portal_by_id(portal_id_of(state.room_id)) if building is not None else None
        if portal is None or obs.step < portal.get("cooldown_until", 0):
            self.stats["stairs_refused"] += 1
            return None, {"route_failed": True}
        entry = self._entry if self._entry is not None else (state.goal_xy, False)
        building.commit(obs, portal, entry[0])
        command = building.plan(obs, world)
        if command is not None:
            return command, {}
        self.stats["stairs_refused"] += 1
        return None, {"route_failed": True}

    def stairs_taken(self, obs, portal):
        """The climb of a staircase the order chose has begun: the node's turn on this floor is over.

        Called by the building coordinator the action the traversal starts.
        The supervisor's turn ends ``traversed`` -- productive, not cooled --
        and the floor's loop is left in SELECT with a reasoning round due, so
        that when the aircraft comes back down the search here resumes from a
        fresh estimate, in which the way back up is a node like any other.
        """
        p = self.policy
        state = p.supervisor.finish(TRAVERSED, "stairs %d taken -- the search continues on the other storey"
                                    % portal["id"], p._floor_time)
        p._route = p._goal = None
        p.route_memory.clear("stairs_taken")
        self._entry = None
        self.room_id = None
        self.next_room = None
        self._needs_reason = True
        self.stats["stairs_taken"] += 1
        if state.completed is not None:
            self.stats["rooms_released"] += 1
            self.stats["loop_points"] += 1
        self._log(obs, "stairs_taken", node=stair_node_id(portal["id"]), portal=portal["id"],
                  direction="up" if portal["direction"] > 0 else "down",
                  released=state.completed is not None)

    # -- step 1: local exploration --------------------------------------------
    def _local(self, obs, world, cost, state):
        """Explore the room in force, and only it, until nothing reachable is left.

        The other half of the termination rule -- N local steps -- is judged
        in :meth:`_budget_flags` before the first tick of the action. The
        third exit is the clue rule: a room newly named by a new kind of
        object ends its turn so that the estimate and the order are redone
        over the new fact -- and a room the oracle then values at nothing is
        left for one it values.
        """
        p = self.policy
        room = p.graph.registry.rooms.get(state.room_id)
        if room is None:
            return None, {"frontier_exhausted": True}
        if self._reclassify(obs, state.room_id, "search"):
            return None, {"room_reclassified": True}
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
    def _fallback(self, obs, world, cost, reason="no room in force"):
        """No room in force: the policy's exploration fallback -- nearest floor-wide frontier,
        the stairs, a retired frontier, a relocation -- never an idle hold while a move exists."""
        return self.policy.fallback.plan(obs, world, cost, reason=reason)

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
                "order": list(self.order), "order_index": self.order_index, "next_room": self.next_room,
                "estimates": {str(pid): dict(item) for pid, item in self.estimates.items()},
                "stairs_offered": sorted(self._stairs),
                "events": list(self.events), "estimate_events": list(self.estimate_events)}



