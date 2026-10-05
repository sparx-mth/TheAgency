"""Where the search goes when its plan fails: the nearest reachable frontier, then further.

Every failure of the decision pipeline -- an A* that finds no path, an RPT*
instance that cannot be built, a room LLM that times out or answers badly,
or a bug in the loop -- used to end in one of two ways: an exception that
ended the episode as an agent error, or an empty hold that the headless
agent spends as an idle turn, on every action, until the step budget ran
out. The recording of the second looks like a camera spinning in place.

This module is the one answer to all of them. Called by the room-search
loop when no room is in force and by the policy when the decision itself
raised, it produces a command that MOVES, in this order:

1. the best reachable EXIT anywhere on the floor -- not confined to the room
   in force, so the search leaves a room whose routes all failed. A frontier
   at a finished room's edge is still the way to rooms not yet seen, so the
   scan ledger does not demote it here: whether the stairs beat the rest of
   this storey is the node layer's judgement (the oracle and RPT*), not this
   last resort's. Two kinds of frontier are NOT exits, and wait for rung 3:

   * an **object's shadow** -- the camera's ray ends at the nearest object,
     so behind every bed and cabinet a scan leaves a strip of unknown whose
     frontier sits beside the object and is no longer than the object could
     cast. Nothing is behind it but the wall. The 2026-10-04 Ranchester
     recording spent actions 61-102 walking from one such strip between the
     beds to the next -- ten goal changes, a 150-degree spin, back within
     6 cm of where it started -- while the passage to the stairs stood two
     metres away as a frontier the whole time, outranked because the strips
     were nearer;
   * a frontier of a room the **target cannot be in** (the loop's
     ``type:`` exclusion, :mod:`room_priors`): even a large unexplored
     bathroom is not where a couch is;
2. the building coordinator, when the floor is exhausted: the ground-truth
   stairs by the explicit fallback rule (up or down, ``floor_decision``).
   The NORMAL way to the stairs is not this: the room-search loop offers
   every staircase to the oracle and to RPT* as a node beside the rooms,
   and climbs when the order says so. This rung is for the floor where that
   machinery has nothing left -- the room LLM is away, or no node is worth
   anything and the frontier is gone;
3. the demoted frontiers of rung 1 -- a shadow is still unknown space once
   every exit is spent, and so is the bathroom;
4. frontiers already retired -- a goal retired for a transient plan failure
   is still unknown space;
5. a relocation: the farthest reachable known cell, for a new vantage point
   from which the map may show a frontier it does not show from here;
6. a **footing sweep** (since 2026-10-05): when nothing on the observed map
   is reachable, the likeliest reason is the floor under the camera's blind
   radius -- an agent that has not moved stands on a disk of unknown, and
   unknown is impassable. One LOOK_DOWN, a circle and a LOOK_UP map it
   (``camera_control.begin_inspection(reason="footing")``), once per spot;
7. only when even that has been done here, one hold -- the idle turn is
   then the only action that changes anything.

A failure is never silent: each one is counted, its type, message and origin
recorded in the episode diagnostics, and logged once per type per episode.
A failed model service (the room LLM, the detector) also arms a doubling
back-off during which it is not asked and this module carries the search, so
a dead service costs a few bounded attempts rather than a timeout per action.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import logging
import math
import traceback

import numpy as np
from scipy.sparse.csgraph import dijkstra

from sparx_agency.core.planning.exploration.frontier_ranking import ranked_frontier_goals
from sparx_agency.core.planning.exploration.room_costs import passable_graph, snap_cell
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.planners.astar.cost_grid_2d import assemble_cost_grid
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.sightlines import unknown_around

LOG = logging.getLogger(__name__)

#: The model services whose failures arm a back-off.
ROOM_LLM = "room_llm"
DETECTOR = "detector"


@dataclass(frozen=True)
class FallbackSettings:
    """Bounds of the fallback. Distances in metres, counts in actions.

    Attributes:
        frontier_attempts: Ranked floor-wide goals the planner is asked about
            per action, for the admissible pass and again for the retired one.
        relocation_min_m: A relocation target must be at least this far along
            the floor, or it is not a new vantage point.
        relocation_candidates: How many of the farthest cells are offered to
            the planner before relocation gives up for this action.
        service_backoff_actions: Actions a failed model service (room LLM,
            detector) is left alone after its first failure; doubles per
            consecutive failure.
        service_backoff_max_actions: Cap on that wait.
        exits_first: Rank exits before object shadows and the frontiers of
            rooms the target cannot be in (rung 1 vs rung 3). False restores
            the plain utility order.
        shadow_margin_m: A frontier goal within a confirmed object's footprint
            radius plus this margin is beside the object.
        shadow_default_radius_m: Footprint radius assumed for a landmark the
            observer never measured one for.
        shadow_length_factor: A frontier beside an object is its shadow only
            if it is no longer than this many halo radii -- a five-metre
            frontier that happens to pass a plant is a wall of unknown, not
            the plant's shadow.
        near_blind_m: A frontier nearer than this along the floor is inside
            the camera's blind radius (a level camera 0.88 m up sees the floor
            from about 1.2 m out): walking to it resolves nothing, and it is
            demoted with the shadows. Such frontiers appear under the agent
            wherever it stops.
        goal_switch_gain: The goal in force is kept, action after action,
            until it is gone or refused -- unless another goal of the same
            rung is worth at least this many times its current utility. The
            greedy order re-ranked every action and the agent's own turning
            re-ranked it back: a 180-degree turn toward one goal followed by a
            210-degree turn toward the goal it had just left (Ranchester
            2026-10-04, actions 42-55).
        goal_match_m: The goal in force is "still there" when a current goal
            of its rung lies within this distance -- clusters re-snap as the
            frontier shrinks.
        footing_radius_m: The footing sweep is taken only where at least
            ``footing_unknown_fraction`` of the cells within this of the
            agent are unknown -- the blind disk under a camera that has not
            moved; an agent that walked here knows its footing.
        footing_unknown_fraction: See ``footing_radius_m``.
    """

    frontier_attempts: int = 6
    relocation_min_m: float = 1.5
    relocation_candidates: int = 5
    service_backoff_actions: int = 25
    service_backoff_max_actions: int = 200
    exits_first: bool = True
    shadow_margin_m: float = 0.30
    shadow_default_radius_m: float = 0.50
    shadow_length_factor: float = 3.0
    near_blind_m: float = 1.2
    goal_switch_gain: float = 2.0
    goal_match_m: float = 0.6
    footing_radius_m: float = 1.2
    footing_unknown_fraction: float = 0.25

    def __post_init__(self):
        for name in ("frontier_attempts", "relocation_candidates", "service_backoff_actions",
                     "service_backoff_max_actions"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("%s must be a positive integer" % name)
        if isinstance(self.relocation_min_m, bool) or not math.isfinite(self.relocation_min_m) or self.relocation_min_m <= 0:
            raise ValueError("relocation_min_m must be positive and finite")
        if self.service_backoff_max_actions < self.service_backoff_actions:
            raise ValueError("service_backoff_max_actions cannot be shorter than the first back-off")
        if type(self.exits_first) is not bool:
            raise ValueError("exits_first must be a bool")
        for name in ("shadow_margin_m", "shadow_default_radius_m", "shadow_length_factor", "near_blind_m",
                     "goal_match_m", "footing_radius_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be positive and finite" % name)
        if (isinstance(self.footing_unknown_fraction, bool) or not math.isfinite(self.footing_unknown_fraction)
                or not 0 <= self.footing_unknown_fraction <= 1):
            raise ValueError("footing_unknown_fraction must lie in [0, 1]")
        if isinstance(self.goal_switch_gain, bool) or not math.isfinite(self.goal_switch_gain) or self.goal_switch_gain < 1.0:
            raise ValueError("goal_switch_gain must be finite and at least 1 (1 = no commitment)")


def object_shadow(goal, landmarks, resolution_m, margin_m, default_radius_m, length_factor):
    """The landmark whose shadow the frontier ``goal`` is, or None.

    The goal lies within the landmark's footprint radius plus ``margin_m``,
    and the frontier is no longer than ``length_factor`` halo radii -- the
    strip of unknown the camera's ray left behind the object, not an opening
    the object happens to stand beside.
    """
    length_m = float(goal.size_cells) * float(resolution_m)
    for landmark in landmarks:
        radius = landmark.radius_m if landmark.radius_m is not None else default_radius_m
        halo = float(radius) + float(margin_m)
        if math.dist(goal.xy, landmark.xy) <= halo and length_m <= length_factor * halo:
            return landmark
    return None


def split_exits(goals, landmarks, resolution_m, settings, excluded=None, room_of=None, type_rule=True):
    """Split ranked frontier goals into exits and demoted frontiers, order kept.

    Demoted, with the reason: inside the camera's blind radius
    (``settings.near_blind_m``); credited to a room the loop withheld as
    ``type:`` (when ``type_rule``; ``room_of`` maps a goal to its pid and
    ``excluded`` a pid to the reason); or an object's shadow
    (:func:`object_shadow`). Returns ``(exits, [(goal, why), ...])``.
    """
    excluded = excluded or {}
    room_of = room_of or {}
    exits, demoted = [], []
    for goal in goals:
        why = None
        pid = room_of.get(goal)
        if goal.geodesic_m < settings.near_blind_m:
            why = "inside the camera's blind radius (%.2f m)" % goal.geodesic_m
        elif type_rule and pid is not None and str(excluded.get(pid, "")).startswith("type:"):
            why = "room %d is %s" % (pid, excluded[pid])
        else:
            landmark = object_shadow(goal, landmarks, resolution_m, settings.shadow_margin_m,
                                     settings.shadow_default_radius_m, settings.shadow_length_factor)
            if landmark is not None:
                why = "shadow of %s %d" % (landmark.class_name, landmark.id)
        if why is None:
            exits.append(goal)
        else:
            demoted.append((goal, why))
    return exits, demoted


def goal_rooms(inventory):
    """``{goal: pid}`` from a frontier inventory, whose rooms are keyed by watershed label = pid + 1."""
    room_of = {}
    if inventory is not None:
        for label, members in inventory.by_room.items():
            for goal in members:
                room_of[goal] = int(label) - 1
    return room_of


class ExplorationFallback:
    """Keeps the agent exploring when the decision pipeline cannot."""

    def __init__(self, policy, settings=None):
        self.policy = policy
        self.settings = settings or FallbackSettings()
        self.failures = []
        self.stats = {"invocations": 0, "frontier": 0, "stairs": 0, "frontier_demoted": 0, "frontier_retired": 0,
                      "room_peek": 0, "relocation": 0, "footing": 0, "hold": 0, "failures": 0,
                      "shadows_demoted": 0, "type_demoted": 0, "blind_demoted": 0,
                      "goal_kept": 0, "goal_switched": 0, "goal_outranked": 0,
                      ROOM_LLM + "_failures": 0, DETECTOR + "_failures": 0}
        self._logged = set()
        self._consecutive = {}
        self.retry_step = {}
        self._last_relocation = None
        self.last_demoted = []            # [{"xy", "why"}] of the last action's demoted frontiers, for the record
        self.goal = None                  # the frontier goal in force: xy, its rung, the action it was last driven
        self.goal_stage = None
        self.goal_step = -1
        self.last_stage = None

    # -- failures ------------------------------------------------------------
    def record_failure(self, obs, kind, exc):
        """Count a pipeline failure and keep enough of it to debug from the recording."""
        frames = traceback.extract_tb(exc.__traceback__)
        origin = "%s:%s" % (frames[-1].filename.rsplit("/", 1)[-1], frames[-1].lineno) if frames else "?"
        entry = {"step": int(obs.step), "kind": kind, "error": "%s: %s" % (type(exc).__name__, exc), "origin": origin}
        self.failures.append(entry)
        self.stats["failures"] += 1
        key = (kind, type(exc).__name__)
        if key not in self._logged:
            self._logged.add(key)
            LOG.warning("ObjectNav %s failure at action %d (%s); exploring the nearest frontier instead: %s",
                        kind, obs.step, origin, entry["error"])

    def note_service_failure(self, obs, service, exc):
        """A model service failed: record it and arm the doubling back-off. Returns the retry action."""
        self.record_failure(obs, service, exc)
        self.stats[service + "_failures"] = self.stats.get(service + "_failures", 0) + 1
        self._consecutive[service] = self._consecutive.get(service, 0) + 1
        wait = min(self.settings.service_backoff_actions * 2 ** (self._consecutive[service] - 1),
                   self.settings.service_backoff_max_actions)
        self.retry_step[service] = int(obs.step) + wait
        return self.retry_step[service]

    def note_service_success(self, service):
        self._consecutive.pop(service, None)
        self.retry_step.pop(service, None)

    def service_unavailable(self, service, step):
        """Is the service inside its back-off window?"""
        retry = self.retry_step.get(service)
        return retry is not None and step < retry

    # -- the command -----------------------------------------------------------
    def plan(self, obs, world, cost=None, reason="no room in force"):
        """A command that moves the agent, by the order in the module docstring."""
        p = self.policy
        self.stats["invocations"] += 1
        if cost is None:
            cost = assemble_cost_grid(p.planner.fields_for(world), p.planner_params, p.settings.body_radius_m)[0]
        ids, graph = passable_graph(cost)                    # one graph for the ranking and the relocation
        inventory = p.graph.frontier_inventory
        ranked = (list(inventory.goals) if inventory is not None else
                  ranked_frontier_goals(world, cost, np.ones(world.grid.shape, bool),
                                        (obs.pose.x, obs.pose.y), obs.pose.yaw, p.sweep.settings.ranking,
                                        ids=ids, graph=graph))
        admissible = p.sweep.admissible(obs, world, ranked)
        if self.goal is not None and obs.step - self.goal_step > 1:
            self.goal = self.goal_stage = None                # the loop had the action in between: no stale pull
        exits, demoted = self._exits(world, admissible, inventory)
        command = self._frontiers(obs, world, exits, "frontier")
        if command is not None:
            return self._tag(command, "frontier", reason)
        if p.building is not None:
            if not p.building.can_leave_floor(obs):
                # No frontier does not prove a room was entered. Complete a
                # reachable outstanding peek, even outside the local trigger
                # radius or when opportunistic inspections were disabled.
                command = p.peek.plan(obs, world, force=True)
                if command is not None:
                    p._action_owner = "doorway_peek"
                    return self._tag(command, "room_peek", reason)
            command = p.building.plan(obs, world, exhausted=True)
            if command is not None:
                return self._tag(command, "stairs", reason)
        command = self._frontiers(obs, world, demoted, "frontier_demoted")
        if command is not None:
            return self._tag(command, "frontier_demoted", reason)
        # A goal under the agent's feet is not retired, it is unreachable by construction:
        # the converter has no action for a waypoint inside its arrival tolerance.
        arrival = p.converter_params.goal_tolerance_m + world.resolution
        retired = [g.xy for g in ranked if g not in admissible and math.dist((obs.pose.x, obs.pose.y), g.xy) > arrival]
        command, _ = self._try_goals(obs, world, retired)
        if command is not None:
            return self._tag(command, "frontier_retired", reason)
        command = self._relocate(obs, world, ids, graph)
        if command is not None:
            return self._tag(command, "relocation", reason)
        # Nothing reachable on the observed map: the floor under the camera's blind
        # radius is the likeliest reason (an agent that has not moved yet stands on
        # a disk of unknown). One footing sweep here maps it; the idle turn is for
        # when even that has been done.
        camera = getattr(p, "camera_control", None)
        blind = (unknown_around(world, (obs.pose.x, obs.pose.y), self.settings.footing_radius_m)
                 >= self.settings.footing_unknown_fraction)
        if blind and camera is not None and camera.begin_inspection(obs, p.mapping.floor_id, reason="footing"):
            command = camera.inspection_command(obs)
            if command is not None:
                return self._tag(command, "footing", reason)
        self.stats["hold"] += 1
        return NavigationCommand.hold(info={"kind": "fallback_hold", "fallback": reason,
                                            "reason": "off the observed passable map; the idle turn is the only move"})

    def _try_goals(self, obs, world, goals):
        """The first goal the planner accepts, as ``(command, goal_xy)``; ``(None, None)`` when none is."""
        p = self.policy
        for goal in goals[:self.settings.frontier_attempts]:
            command = p._navigate(obs, world, tuple(goal), "frontier")
            if command is not None:
                return command, tuple(goal)
        return None, None

    def _frontiers(self, obs, world, goals, stage):
        """Drive the goal in force of this rung, or the best goal of the rung, and remember which."""
        ordered = self._committed(goals, stage)
        command, taken = self._try_goals(obs, world, [g.xy for g in ordered])
        if command is None:
            return None
        if self.goal is None or self.goal_stage != stage or math.dist(taken, self.goal) > self.settings.goal_match_m:
            if self.goal is not None:
                self.stats["goal_switched"] += 1
        else:
            self.stats["goal_kept"] += 1
        self.goal, self.goal_stage, self.goal_step = taken, stage, int(obs.step)
        return command

    def _committed(self, goals, stage):
        """The rung's goals with the goal in force first, unless another is worth ``goal_switch_gain`` times more.

        The goal in force is the one within ``goal_match_m`` of where it was
        last driven; absent (resolved, demoted, or never of this rung) the
        utility order stands. Ranking is greedy per action, and the agent's
        own turning re-ranks it: without this, a goal behind the agent won on
        one action, the turn toward it put the former goal behind, and that
        one won back (Ranchester 2026-10-04, actions 42-55: 13 turns, no move).
        """
        if self.goal is None or self.goal_stage != stage or not goals:
            return list(goals)
        in_force = min(goals, key=lambda g: math.dist(g.xy, self.goal))
        if math.dist(in_force.xy, self.goal) > self.settings.goal_match_m:
            return list(goals)
        best = goals[0]
        if best is not in_force and best.utility >= self.settings.goal_switch_gain * in_force.utility:
            self.stats["goal_outranked"] += 1
            return list(goals)
        return [in_force] + [g for g in goals if g is not in_force]

    def _exits(self, world, goals, inventory):
        """Split the admissible goals, utility order kept, into exits and demoted frontiers.

        Demoted: an object's shadow (:func:`object_shadow`, against this
        storey's confirmed landmarks), a frontier inside the camera's blind
        radius, or a frontier the inventory credits to a room the loop
        withheld as ``type:`` -- one the target cannot be in. ``scanned:``
        exclusions do not demote: a finished room's opening is how the next
        room is found.
        """
        p, s = self.policy, self.settings
        self.last_demoted = []
        if not s.exits_first or not goals:
            return list(goals), []
        landmarks = list(p.landmarks.confirmed()) if getattr(p, "landmarks", None) is not None else []
        excluded = getattr(getattr(p, "loop", None), "excluded", None) or {}
        exits, demoted = split_exits(goals, landmarks, world.resolution, s, excluded=excluded,
                                     room_of=goal_rooms(inventory), type_rule=True)
        for goal, why in demoted:
            if why.startswith("inside the camera"):
                self.stats["blind_demoted"] += 1
            elif why.startswith("room "):
                self.stats["type_demoted"] += 1
            else:
                self.stats["shadows_demoted"] += 1
            self.last_demoted.append({"xy": [round(float(v), 2) for v in goal.xy],
                                      "size_cells": int(goal.size_cells), "why": why})
        return exits, [goal for goal, _ in demoted]


    def _relocate(self, obs, world, ids, graph):
        """Walk to the farthest reachable known cell: a new vantage point, never the last one."""
        gx, gy = world.world_to_grid(obs.pose.x, obs.pose.y)
        snap = max(1, int(round(self.policy.sweep.settings.ranking.snap_radius_m / world.resolution)))
        source = snap_cell(ids, gx, gy, snap)
        if source is None or ids.max() < 0:
            return None
        distances = dijkstra(graph, directed=False, indices=int(ids[source[1], source[0]]))
        reachable = np.isfinite(distances) & (distances * world.resolution >= self.settings.relocation_min_m)
        if not reachable.any():
            return None
        order = np.argsort(-np.where(reachable, distances, -np.inf), kind="stable")
        ys, xs = np.nonzero(ids >= 0)
        node_cells = np.empty((ids.max() + 1, 2), dtype=np.int64)
        node_cells[ids[ys, xs]] = np.stack([xs, ys], axis=1)
        tried = 0
        for node in order:
            if not reachable[node]:
                break
            x, y = (float(v) for v in world.grid_to_world(int(node_cells[node][0]), int(node_cells[node][1])))
            if self._last_relocation is not None and math.dist((x, y), self._last_relocation) < 1.0:
                continue
            command = self.policy._navigate(obs, world, (x, y), "relocate")
            tried += 1
            if command is not None:
                self._last_relocation = (x, y)
                return command
            if tried >= self.settings.relocation_candidates:
                break
        return None

    def _tag(self, command, stage, reason):
        self.stats[stage] += 1
        self.last_stage = stage
        return replace(command, info=dict(command.info, fallback=reason, fallback_stage=stage))

    def snapshot(self):
        """What the fallback is doing this action, for the per-step record and the HUD."""
        return {"stage": self.last_stage,
                "goal": None if self.goal is None else [round(float(v), 2) for v in self.goal],
                "goal_stage": self.goal_stage, "goal_step": self.goal_step,
                "demoted": list(self.last_demoted),
                "stats": {k: self.stats[k] for k in ("frontier", "frontier_demoted", "frontier_retired", "relocation",
                                                     "shadows_demoted", "type_demoted", "blind_demoted",
                                                     "goal_kept", "goal_switched", "goal_outranked")}}

    def diagnostics(self):
        return {"settings": asdict(self.settings), "stats": dict(self.stats),
                "failures": list(self.failures), "retry_step": dict(self.retry_step),
                "last_demoted": list(self.last_demoted)}



