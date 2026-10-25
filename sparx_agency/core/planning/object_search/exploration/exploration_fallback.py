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
   * a **sliver** (since 2026-10-14): a frontier from whose cell a look
     round would reveal less than ``min_gain_m2`` of unknown floor (the
     loop's ``explore_min_gain_m2`` by default) -- the strip beside a wall,
     the gap behind a chair. The exits that remain are ranked by that gain
     over the walk to them, as the explore visit ranks a room's frontiers,
     not by the length of their boundary;
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
4b. a **squeeze** (since 2026-10-25, :meth:`ExplorationFallback._squeeze`):
   the frontier goals reachable at the body radius alone, routed by the
   policy's ``squeeze_planner`` -- everything above is measured at the
   planner's floor rung (``RPTSettings.min_clearance_m``), and a doorway
   the map draws 0.4 m wide closes a whole wing at that rung;
5. a relocation: the farthest reachable known cell, for a new vantage point
   from which the map may show a frontier it does not show from here;
6. a **footing sweep** (since 2026-10-05): when nothing on the observed map
   is reachable, the likeliest reason is the floor under the camera's blind
   radius -- an agent that has not moved stands on a disk of unknown, and
   unknown is impassable. One LOOK_DOWN, a circle and a LOOK_UP map it
   (``camera_control.begin_inspection(reason="footing")``), once per spot;
7. an **escape probe** (since 2026-10-13, :mod:`escape_probe`): boxed in by
   KNOWN cells -- the strip behind a sofa, the ring a counter's top draws
   round the agent -- the inflated map has no way out of the pocket while
   the body may well have one. The grid is read without inflation along
   every turn heading; the heading with the thinnest occupied band and the
   most floor beyond it gets one forward step, and the simulator decides.
   A step that moves re-maps the pocket; a blocked one is remembered for
   that cell and heading. Collierville/000036 spent 488 actions on the
   hold below, twice, with the toilet 5.6 m away;
8. only when even that has been done here, one hold -- the idle turn is
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
from typing import Optional

import numpy as np
from scipy.sparse.csgraph import dijkstra

from sparx_agency.core.planning.exploration.frontier_ranking import rank_by_view_gain, ranked_frontier_goals
from sparx_agency.core.planning.exploration.room_costs import passable_graph, snap_cell
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.planners.astar.cost_grid_2d import assemble_cost_grid
from sparx_agency.core.planning.object_search.world_model.peek_stairs import peek_planning_world
from sparx_agency.core.planning.object_search.world_model.sightlines import unknown_around

LOG = logging.getLogger(__name__)

#: The model services whose failures arm a back-off.
ROOM_LLM = "room_llm"
DETECTOR = "detector"
SEGMENTER = "segmenter"


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
        relocation_revisit_m: A cell within this of a vantage point already
            walked to during the current run of fallback actions is not a
            new vantage point (since 2026-10-25). Only the LAST point was
            excluded before, so a pocket with two ends was relocated between
            for ever -- 00835/000064 made 458 relocations across a 2 m strip
            beside a bed and never reached the footing and escape rungs that
            would have probed the strip's one exit. The memory is cleared
            when the floor's search takes an action of its own between two
            fallback actions: the map has changed meanwhile.
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
        shadow_min_radius_m: A landmark whose measured footprint radius is
            under this casts no shadow (since 2026-10-14): a clock, a vase, a
            bottle, a cup stand on a wall or a surface and hide no floor.
            Collierville/000023 lost its doorway `O2` as "the shadow of clock
            24" 0.5 m from it. Classes in :data:`SHADOWLESS_CLASSES` cast none
            whatever their radius.
        shadow_exempt_m2: A frontier with at least this much unknown within
            ``shadow_exempt_range_m`` of it, in every direction and through
            the walls' blocking, is no object's shadow whatever stands
            beside it (:func:`shadow_exemption`, since 2026-10-14): the
            strip behind a bed has the wall a few cells behind it; a doorway
            has the room's floor from the threshold on. Collierville/000023
            lost the 2.9 m frontier onto its toilet's room as "the shadow of
            refrigerator 0". 0 disables.
        shadow_exempt_range_m: See ``shadow_exempt_m2``.
        near_blind_m: A frontier nearer than this along the floor is inside
            the camera's blind radius (a level camera 0.88 m up sees the floor
            from about 1.4 m out; 1.2 m leaves a step's worth of margin for
            the geodesic's snap): walking to it resolves nothing, and it is
            demoted with the shadows -- and retired (since 2026-10-05): a
            goal demoted on the step that brought it under the radius and
            promoted again by the step toward the next one made the fallback
            turn between two exits for ever. Such frontiers appear under the
            agent wherever it stops.
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
        goal_memory_actions: The goal in force is forgotten when the fallback
            did not drive it for more than this many actions: the loop had
            the action in between and walked the agent elsewhere, so the
            old goal would be a stale pull. A glance (ten turns at most) or
            a peek's look in between is not that: until 2026-10-14 the goal
            was forgotten after ONE action without it, every glance re-ranked
            the frontier greedily, and the agent's own turning re-ranked it
            back (Collierville/000036, actions 99-113: six turns left to one
            sliver, six right to another, 180 degrees and back).
        min_gain_m2: A frontier reveals at least this much unknown floor
            from its cell (rays through the unknown, as the loop's
            ``explore_min_gain_m2``) to be an EXIT of rung 1; a sliver that
            reveals less -- the strip beside a wall, the gap behind a chair
            -- waits with the shadows for rung 3 (since 2026-10-14: the
            fallback walked from one such sliver to the next, each resolved
            by the step toward it, the next one behind the agent). 0
            disables; by default the loop's own threshold is read.
        footing_radius_m: The footing sweep is taken only where at least
            ``footing_unknown_fraction`` of the cells within this of the
            agent are unknown -- the blind disk under a camera that has not
            moved; an agent that walked here knows its footing.
        footing_unknown_fraction: See ``footing_radius_m``.
    """

    frontier_attempts: int = 6
    relocation_min_m: float = 1.5
    relocation_candidates: int = 5
    relocation_revisit_m: float = 1.0
    service_backoff_actions: int = 25
    service_backoff_max_actions: int = 200
    exits_first: bool = True
    shadow_margin_m: float = 0.30
    shadow_default_radius_m: float = 0.50
    shadow_length_factor: float = 3.0
    shadow_min_radius_m: float = 0.25
    shadow_exempt_m2: float = 2.0
    shadow_exempt_range_m: float = 1.5
    near_blind_m: float = 1.2
    goal_switch_gain: float = 2.0
    goal_match_m: float = 0.6
    goal_memory_actions: int = 12
    min_gain_m2: Optional[float] = None
    footing_radius_m: float = 1.2
    footing_unknown_fraction: float = 0.25

    def __post_init__(self):
        for name in ("frontier_attempts", "relocation_candidates", "service_backoff_actions",
                     "service_backoff_max_actions", "goal_memory_actions"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("%s must be a positive integer" % name)
        if isinstance(self.shadow_exempt_m2, bool) or not math.isfinite(self.shadow_exempt_m2) or self.shadow_exempt_m2 < 0:
            raise ValueError("shadow_exempt_m2 must be finite and non-negative (0 disables)")
        if self.min_gain_m2 is not None and (isinstance(self.min_gain_m2, bool) or not math.isfinite(self.min_gain_m2)
                                             or self.min_gain_m2 < 0):
            raise ValueError("min_gain_m2 must be None (the loop's threshold), 0 (off) or a positive area")
        if isinstance(self.relocation_min_m, bool) or not math.isfinite(self.relocation_min_m) or self.relocation_min_m <= 0:
            raise ValueError("relocation_min_m must be positive and finite")
        if (isinstance(self.relocation_revisit_m, bool) or not math.isfinite(self.relocation_revisit_m)
                or self.relocation_revisit_m <= 0):
            raise ValueError("relocation_revisit_m must be positive and finite")
        if self.service_backoff_max_actions < self.service_backoff_actions:
            raise ValueError("service_backoff_max_actions cannot be shorter than the first back-off")
        if type(self.exits_first) is not bool:
            raise ValueError("exits_first must be a bool")
        for name in ("shadow_margin_m", "shadow_default_radius_m", "shadow_length_factor", "near_blind_m",
                     "goal_match_m", "footing_radius_m", "shadow_min_radius_m", "shadow_exempt_range_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be positive and finite" % name)
        if (isinstance(self.footing_unknown_fraction, bool) or not math.isfinite(self.footing_unknown_fraction)
                or not 0 <= self.footing_unknown_fraction <= 1):
            raise ValueError("footing_unknown_fraction must lie in [0, 1]")
        if isinstance(self.goal_switch_gain, bool) or not math.isfinite(self.goal_switch_gain) or self.goal_switch_gain < 1.0:
            raise ValueError("goal_switch_gain must be finite and at least 1 (1 = no commitment)")


#: Classes that cast no floor shadow whatever the frontier beside them: on a wall or a surface, not on the floor.
SHADOWLESS_CLASSES = frozenset({"clock", "vase", "book", "bottle", "cup", "toy", "stuffed animal", "television", "tv",
                                "monitor"})


def object_shadow(goal, landmarks, resolution_m, margin_m, default_radius_m, length_factor, min_radius_m=0.0):
    """The landmark whose shadow the frontier ``goal`` is, or None.

    The goal lies within the landmark's footprint radius plus ``margin_m``,
    and the frontier is no longer than ``length_factor`` halo radii -- the
    strip of unknown the camera's ray left behind the object, not an opening
    the object happens to stand beside. A landmark of a
    :data:`SHADOWLESS_CLASSES` class, or with a measured radius under
    ``min_radius_m``, casts none.
    """
    length_m = float(goal.size_cells) * float(resolution_m)
    for landmark in landmarks:
        if str(landmark.class_name).lower() in SHADOWLESS_CLASSES:
            continue
        if landmark.radius_m is not None and float(landmark.radius_m) < float(min_radius_m):
            continue
        radius = landmark.radius_m if landmark.radius_m is not None else default_radius_m
        halo = float(radius) + float(margin_m)
        if math.dist(goal.xy, landmark.xy) <= halo and length_m <= length_factor * halo:
            return landmark
    return None


def shadow_exemption(view, settings):
    """``goal -> bool``: whether the unknown RIGHT BEHIND a frontier is too much for an object's shadow.

    The unknown cells within ``shadow_exempt_range_m`` of the goal, in every
    direction, through the ray caster's blocking (walls, the settled
    unknown): a shadow strip has the wall a few cells behind it and
    reaches a fraction of a square metre; a doorway onto a room has the
    room's floor from the threshold on. ``shadow_exempt_m2`` is the bar;
    0 disables the exemption (None is returned).
    """
    if view is None or settings.shadow_exempt_m2 <= 0.0:
        return None
    cells = int(round(float(settings.shadow_exempt_m2) / view.cell_area_m2))
    reach = float(settings.shadow_exempt_range_m)

    def exempt(goal):
        return int(view.unknown_in_sector(goal.xy, 0.0, math.pi, max_range_m=reach).sum()) >= cells
    return exempt


def split_exits(goals, landmarks, resolution_m, settings, excluded=None, room_of=None, type_rule=True, exempt=None):
    """Split ranked frontier goals into exits and demoted frontiers, order kept.

    Demoted, with the reason: inside the camera's blind radius
    (``settings.near_blind_m``); credited to a room the loop withheld as
    ``type:`` (when ``type_rule``; ``room_of`` maps a goal to its pid and
    ``excluded`` a pid to the reason); or an object's shadow
    (:func:`object_shadow`) -- unless ``exempt(goal)`` says too much unknown
    lies right behind it to be one (:func:`shadow_exemption`, since
    2026-10-14: a strip behind a bed has the wall behind it; a 2.9 m
    frontier with a room's floor through it is a room, whatever stands
    beside it -- Collierville/000023 lost the frontier onto its toilet's
    room as "the shadow of refrigerator 0").
    Returns ``(exits, [(goal, why), ...])``.
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
                                     settings.shadow_default_radius_m, settings.shadow_length_factor,
                                     min_radius_m=getattr(settings, "shadow_min_radius_m", 0.0))
            if landmark is not None and not (exempt is not None and exempt(goal)):
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
                      "frontier_squeeze": 0,
                      "room_peek": 0, "relocation": 0, "footing": 0, "escape": 0, "hold": 0, "failures": 0,
                      "shadows_demoted": 0, "type_demoted": 0, "blind_demoted": 0, "slivers_demoted": 0,
                      "goal_kept": 0, "goal_switched": 0, "goal_outranked": 0,
                      ROOM_LLM + "_failures": 0, DETECTOR + "_failures": 0, SEGMENTER + "_failures": 0}
        self._logged = set()
        self._consecutive = {}
        self.retry_step = {}
        self._last_relocation = None
        self._relocation_step = None      # the action the relocation target was last driven on
        self._vantage_points = []         # every relocation target of the current run of fallback actions
        self._invocation_step = None      # the last action the fallback planned; a gap clears the vantage memory
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
        self.last_stage = None                                 # this action's rung, not the last invocation's
        if self._invocation_step is not None and int(obs.step) - self._invocation_step > 1:
            self._vantage_points = []                          # the floor's search had the actions between: a new map
        self._invocation_step = int(obs.step)
        if cost is None:
            cost = assemble_cost_grid(p.planner.fields_for(world), p.planner_params, p.planner_params.inflate_floor_m)[0]
        ids, graph = passable_graph(cost)                    # one graph for the ranking and the relocation
        inventory = p.graph.frontier_inventory
        ranked = (list(inventory.goals) if inventory is not None else
                  ranked_frontier_goals(world, cost, np.ones(world.grid.shape, bool),
                                        (obs.pose.x, obs.pose.y), obs.pose.yaw, p.sweep.settings.ranking,
                                        ids=ids, graph=graph))
        admissible = p.sweep.admissible(obs, world, ranked)
        if self.goal is not None and obs.step - self.goal_step > self.settings.goal_memory_actions:
            self.goal = self.goal_stage = None                # the loop walked the agent elsewhere: no stale pull
        # Routes are planned with every seen staircase written occupied: a floor-wide
        # frontier beyond a stairwell walked the Pomaria agent down a flight it had not
        # chosen (2026-10-04, actions 262-316), and the first treads into the storey's map.
        # Taking the stairs is the coordinator's rung below, on the plain world.
        planning = peek_planning_world(p, world) if p.building is not None else world
        exits, demoted = self._exits(world, admissible, inventory, obs=obs)
        command = self._frontiers(obs, planning, exits, "frontier")
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
        command = self._frontiers(obs, planning, demoted, "frontier_demoted")
        if command is not None:
            return self._tag(command, "frontier_demoted", reason)
        # A goal under the agent's feet is not retired, it is unreachable by construction:
        # the converter has no action for a waypoint inside its arrival tolerance.
        arrival = p.converter_params.goal_tolerance_m + world.resolution
        retired = [g.xy for g in ranked if g not in admissible and math.dist((obs.pose.x, obs.pose.y), g.xy) > arrival]
        command, _ = self._try_goals(obs, planning, retired)
        if command is not None:
            return self._tag(command, "frontier_retired", reason)
        command = self._squeeze(obs, planning, world)
        if command is not None:
            return self._tag(command, "frontier_squeeze", reason)
        command = self._relocate(obs, planning, ids, graph)
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
        # Boxed in by KNOWN cells: the inflated map has no way out of the pocket, the body
        # may well have one. One probing step through the thinnest barrier; the simulator
        # decides, a step that moves re-maps the pocket, a blocked one is remembered.
        escape = getattr(p, "escape", None)
        command = escape.plan(obs, world) if escape is not None else None
        if command is not None:
            return self._tag(command, "escape", reason)
        self.stats["hold"] += 1
        self.last_stage = "hold"
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

    def _squeeze(self, obs, planning, world):
        """Rung 4b (since 2026-10-25): the frontier goals reachable at the BODY radius only, through the squeeze planner.

        The passable map everything above is measured on relaxes no further
        than ``RPTSettings.min_clearance_m`` (0.25 m): a non-sliding body
        cannot walk a tighter gap without a refused touch. But a doorway the
        depth map draws 0.4 m wide is a doorway the body walked through on
        the way in (00835/000064: the strip beside the bed, 458 relocations
        and 211 probes with five frontier goals a squeeze away), so when
        NOTHING is reachable at the rung, the goals reachable at the body
        radius are offered through ``RPTSearchPolicy.squeeze_planner`` --
        the same planner with the body radius as its floor -- before the
        relocation and the probes. A refused step on the way retires the
        goal as any blocked frontier's, and the sticky bars of a second
        refusal close the gap on the map for good.
        """
        p = self.policy
        planner = getattr(p, "squeeze_planner", None)
        if planner is None or planner.params.inflate_floor_m >= p.planner_params.inflate_floor_m:
            return None
        cost = assemble_cost_grid(p.planner.fields_for(world), planner.params, planner.params.inflate_floor_m)[0]
        ids, graph = passable_graph(cost)
        ranked = ranked_frontier_goals(world, cost, np.ones(world.grid.shape, bool),
                                       (obs.pose.x, obs.pose.y), obs.pose.yaw, p.sweep.settings.ranking,
                                       ids=ids, graph=graph)
        goals = [g.xy for g in p.sweep.admissible(obs, world, ranked)]
        for goal in goals[:self.settings.frontier_attempts]:
            command = p._navigate(obs, planning, tuple(goal), "frontier", planner=planner)
            if command is not None:
                self.stats["frontier_squeeze"] += 1
                return command
        return None

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

    def _exits(self, world, goals, inventory, obs=None):
        """Split the admissible goals, utility order kept, into exits and demoted frontiers.

        Demoted: an object's shadow (:func:`object_shadow`, against this
        storey's confirmed landmarks), a frontier inside the camera's blind
        radius, a frontier the inventory credits to a room the loop
        withheld as ``type:`` -- one the target cannot be in -- or, with
        the pose given, a frontier that reveals less than ``min_gain_m2``
        of unknown floor from its cell (the loop's ``explore_min_gain_m2``
        by default); the exits that remain are then ordered by that gain
        over the walk to them, as the explore visit orders a room's
        frontiers (:meth:`_by_view_gain`). ``scanned:`` exclusions do not
        demote: a finished room's opening is how the next room is found.
        """
        p, s = self.policy, self.settings
        self.last_demoted = []
        if not s.exits_first or not goals:
            return list(goals), []
        landmarks = list(p.landmarks.confirmed()) if getattr(p, "landmarks", None) is not None else []
        excluded = getattr(getattr(p, "loop", None), "excluded", None) or {}
        exits, demoted = split_exits(goals, landmarks, world.resolution, s, excluded=excluded,
                                     room_of=goal_rooms(inventory), type_rule=True,
                                     exempt=self._reveals_a_room(obs, world))
        if obs is not None and exits:
            ranked, slivers = self._by_view_gain(obs, world, exits)
            if ranked is not None:
                exits = ranked
                demoted = demoted + [(g, "reveals under %.1f m2" % self._min_gain_m2()) for g in slivers]
        for goal, why in demoted:
            if why.startswith("inside the camera"):
                self.stats["blind_demoted"] += 1
                # Sticky: a blind-radius demotion retires the goal, so the step toward
                # the next exit cannot promote it back (the retired rung still tries it).
                visited = getattr(p, "_visited_frontiers", None)
                if visited is not None and not any(math.dist(goal.xy, xy) <= s.goal_match_m for xy in visited):
                    visited.append(tuple(float(v) for v in goal.xy))
                if self.goal is not None and math.dist(goal.xy, self.goal) <= s.goal_match_m:
                    self.goal = self.goal_stage = None
            elif why.startswith("room "):
                self.stats["type_demoted"] += 1
            elif why.startswith("reveals under"):
                self.stats["slivers_demoted"] += 1
            else:
                self.stats["shadows_demoted"] += 1
            self.last_demoted.append({"xy": [round(float(v), 2) for v in goal.xy],
                                      "size_cells": int(goal.size_cells), "why": why})
        return exits, [goal for goal, _ in demoted]

    def _reveals_a_room(self, obs, world):
        """The shadow exemption: ``goal -> bool`` (:func:`shadow_exemption`), or None without the pose or the ray caster."""
        loop = getattr(self.policy, "loop", None)
        if obs is None or loop is None or not hasattr(loop, "_unknown_view"):
            return None
        return shadow_exemption(loop._unknown_view(obs, world), self.settings)

    def _min_gain_m2(self):
        """The significance threshold in force: the setting, else the loop's ``explore_min_gain_m2``."""
        if self.settings.min_gain_m2 is not None:
            return float(self.settings.min_gain_m2)
        loop = getattr(self.policy, "loop", None)
        return float(getattr(getattr(loop, "settings", None), "explore_min_gain_m2", 0.0) or 0.0)

    def _by_view_gain(self, obs, world, goals):
        """``(ranked, slivers)``: the exits re-scored by the unknown floor a look from each would reveal, best first.

        The loop's own ray caster (``RoomSearchLoop._unknown_view``, cached
        per action) scores each goal as :func:`rank_by_view_gain` does for
        the explore visit -- the unknown within the camera's range from the
        goal's cell over the walk to it, not the length of the boundary,
        discounted by the actions already spent near it (the loop's trail
        visits, since 2026-10-22) -- and a goal under the threshold is a
        sliver, returned apart.
        ``(None, [])`` when the threshold is off or the loop has no caster:
        the inventory's order stands.
        """
        min_gain = self._min_gain_m2()
        loop = getattr(self.policy, "loop", None)
        if min_gain <= 0.0 or loop is None or not hasattr(loop, "_unknown_view"):
            return None, []
        view = loop._unknown_view(obs, world)
        cells = int(round(min_gain / view.cell_area_m2))
        visits = loop._visits(obs) if hasattr(loop, "_visits") else None
        ranked = rank_by_view_gain(list(goals), view, self.policy.sweep.settings.ranking, min_gain_cells=cells,
                                   visits=visits)
        kept = {tuple(g.cell) for g in ranked}
        return ranked, [g for g in goals if tuple(g.cell) not in kept]


    def _relocate(self, obs, world, ids, graph):
        """Walk to the farthest reachable known cell: a new vantage point, never the last one again.

        The target is kept while it is being walked to (the fallback had the
        previous action too and the agent is not there yet): the farthest
        cell from a moving agent flips between the two ends of a hall, and
        re-choosing it every action shuffled the agent a quarter metre back
        and forth between them. Once reached, the next target is another.
        """
        here = (obs.pose.x, obs.pose.y)
        last, last_step = self._last_relocation, self._relocation_step
        revisit = self.settings.relocation_revisit_m
        if last is not None and last_step is not None and obs.step - last_step <= 1 and math.dist(here, last) >= revisit:
            command = self.policy._navigate(obs, world, last, "relocate")
            if command is not None:
                self._relocation_step = int(obs.step)
                return command
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
            if any(math.dist((x, y), seen) < revisit for seen in self._vantage_points):
                continue                                   # never a vantage point of this run again
            command = self.policy._navigate(obs, world, (x, y), "relocate")
            tried += 1
            if command is not None:
                self._last_relocation, self._relocation_step = (x, y), int(obs.step)
                self._vantage_points.append((x, y))
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



