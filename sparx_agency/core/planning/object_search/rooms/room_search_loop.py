"""The room-search loop: bounded local exploration, then re-classify, re-estimate, re-order, transit.

Seven steps, as specified on 2026-09-28, each named at the line where it
happens so a recording can be read against this docstring:

* **Background, every action** -- ``RPTSearchPolicy._search`` refreshes the scene
  graph without an LLM call: room geometry, object->room association, doors,
  cumulative search time and remaining frontier clusters per room. A committed
  stair climb belongs to the multi-floor policy, which this loop yields to
  before it does anything of its own.
* **1. One visit, bounded to the room in force** (:meth:`_local`). Under
  ``visit="explore"`` (the default since 2026-10-13) a visit IS a frontier
  walk: enter the room at the frontier that would reveal the most unknown
  floor per metre walked (:meth:`_explore_goals`, the inventory's clusters
  re-scored by :func:`~frontier_ranking.rank_by_view_gain`), and when the
  camera has resolved it walk to the next, until the room has no live
  frontier left; the room is then FINISHED for the first pass
  (:class:`~room_scans.RoomScanLedger`, ``explored``: no live frontier and
  the agent stood in it), and a full rotation completed inside it on the
  way -- a glance -- finishes it at once. No vantage point, no fixed
  circle. Under ``visit="scan"`` (the default from 2026-10-04, kept as a
  mode, and the shape of a second-pass revisit) a visit is a look-around:
  walk to the room's vantage point -- the interior cell of greatest
  clearance -- turn the camera's circle (as many turns as its field of view
  needs, ten at 30 degrees), and leave; the room is finished whatever
  frontier its mask still shows. Under both, a finished
  room is still SHOWN to the oracle -- ``status=scanned(<how>, <ago>)`` --
  but not offered as a node, and -- since 2026-10-14 -- not shown to the
  oracle as a node either (:attr:`LoopSettings.scanned_to_oracle`; the HOUSE
  line still names its type): one look at a room is the look it gets. With
  :attr:`LoopSettings.second_pass` on (off by default since 2026-10-14) the
  finished rooms become nodes again once the storey is covered, and a
  revisit is a scan from a different spot. A room whose identified
  type cannot hold the target is likewise shown and valued by the model;
  :attr:`LoopSettings.type_prior` (off) is the hard version of that rule
  (:mod:`room_priors`). Under ``visit="sweep"`` (the former behaviour, kept as an
  ablation) goals come from inside the room's mask for ``local_steps``
  actions, routes planned on a COPY of the map with every other room
  blocked, and the same room may be chosen straight back. In every mode the
  re-identification clue rule holds: one confirmed object names the room
  (weakly; two distinct classes strongly), the classifier is re-asked for
  that one room the action a new KIND of object lands in it, and a changed
  name ends the room's turn (``reclassified``) so that steps 2-4 run over
  the new fact.
* **2. Global re-classification** (:meth:`_reason`) -- ``graph.reason``
  re-labels every known room from every confirmed object observed so far.
* **3. Probability per NODE** -- the same call, to
  :class:`~sparx_agency.core.mapping.topology.search_node_oracle.SearchNodeOracle`:
  for every room on the floor -- its type or ``unknown``, size, frontier
  left, time searched and how long ago, objects seen -- for every
  STAIRCASE off the floor -- up or down, whether the other storey was
  visited and what was found there -- and for every OPENING of the floor
  (:mod:`opening_nodes`, since 2026-10-04) -- a doorway or gap at the edge
  of the mapped floor leading to space not seen yet, named by the room it
  opens from and the objects glimpsed through it -- the probability that
  going there NEXT finds the target, plus the mass in none of them. The
  model is given the HOUSE line (:mod:`house_context`): every room type
  found so far with its status, the rooms still unidentified, the openings
  not looked into, and the action budget spent -- and asked to reason as a
  person who knows homes does: where the target lives (an en-suite toilet
  is reached through a bedroom), what the rooms found say about the rest (a
  living room found makes a sofa unlikely in the unidentified rooms), and
  how the budget weighs (early: the unexplored first; late: the detector
  may have missed the target in a scanned room of the right type). It
  writes ``home``, ``house``, ``stage`` and the ``pass`` verdict before the
  numbers. Distance is NOT the model's concern: it is charged by RPT* in
  step 4.
* **4. Visit order** -- the supervisor's SELECT hands the surviving nodes and
  the arc-weight instance to RPT*: rooms at their entry points, staircases at
  the foot of the flight with a LEAF as long as the flight plus the fixed
  cost of a storey change on every arc into and out of them (going upstairs
  puts every room down here that much further away). The instance charges
  the local budget as per-room service time, and ``resolve_on_release``
  makes every loop point a fresh solve rather than the next node of a stale
  order. Nothing else decides a floor change: no allowance, no clock.
* **5. A\\*** -- the weighted A* behind ``policy._navigate``.
* **6. Direct transit** (:meth:`_transit`) -- under ``explore`` to the
  chosen room's most revealing frontier (its DOORSTEP -- the passable cell
  nearest the agent along the floor -- when it has none left, since
  2026-10-14; its vantage point for a second-pass revisit); under ``scan`` to the chosen
  room's vantage point; under ``sweep`` to the closest frontier INSIDE it
  (one extraction and one Dijkstra give every room its nearest reachable
  frontier; a room with none keeps its centroid); to the foot of the
  chosen stairs, where the building coordinator takes over and climbs; the
  node's turn ends there (``traversed``) and the floor's search resumes
  clean when the aircraft comes back down; or to the threshold of the
  chosen opening, where the visit is a **peek** (:meth:`_peek_visit`): face
  the unknown, one look to each side, and the opening is retired for the
  storey -- what the peek saw is floor in the partition now, and the room
  behind is a node of its own, named by its objects, or ruled out by them.
  Openings are read off the live map every action and, under ``explore``,
  an opening off a room that is itself a node is folded into that room's
  walk (:meth:`_fold_openings`). An exit the oracle has not valued -- one
  the reply itself unfolded, by dropping its room -- is handed to RPT* at
  the oracle's last number for it or the unseen doorway's prior
  (:meth:`_exit_prob`, :attr:`OpeningSettings.unvalued_prob`), never 0.0.
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
policy's :class:`~sparx_agency.core.planning.object_search.exploration.exploration_fallback.ExplorationFallback`:
the nearest reachable frontier
anywhere on the floor, the stairs by the explicit fallback rule, a retired
frontier, a relocation. Never an idle hold while a move exists.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
import time

import numpy as np

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.mapping.topology.search_node_oracle import (
    BUDGET_STEP, HOME_FLOOR, UNEXPLORED_ELSEWHERE, UNEXPLORED_FLOOR)
from sparx_agency.core.planning.environment import OccupancyGrid2D
from sparx_agency.core.planning.exploration.frontier_ranking import (
    accessible_frontiers, frontier_goals_by_room, rank_by_view_gain)
from sparx_agency.core.planning.exploration.object_search_supervisor import (
    ALL_VERDICTS, BLOCKED, BUDGET_SPENT, EXHAUSTED, MAPPED, NEUTRAL, SEARCH, SELECT, TRANSIT, TRANSIT_TIMEOUT,
    TRAVERSED, UNREACHABLE, ObjectSearchParams)
from sparx_agency.core.planning.exploration.room_costs import build_instance
from sparx_agency.core.planning.exploration.view_gain import UnknownView, cone_from_camera
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.planners.astar.cost_grid_2d import assemble_cost_grid
from sparx_agency.core.planning.object_search.falcon.room_falcon import room_scope
from sparx_agency.core.planning.object_search.exploration.exploration_fallback import ROOM_LLM
from sparx_agency.core.planning.object_search.rooms.house_context import house_line, scan_words
from sparx_agency.core.mapping.topology.room_priors import home_object, implausible_room, ruled_out
from sparx_agency.core.planning.object_search.rooms.room_scans import SCAN_POINT_INSIDE
from sparx_agency.core.planning.object_search.rooms.opening_nodes import (
    LANDMARK, OpeningSettings, detect_openings, is_opening_node, landmark_openings, near_stair_mask,
    opening_options, opening_value_factor)
from sparx_agency.core.planning.object_search.rooms.room_vantage import vantage_point
from sparx_agency.core.planning.object_search.rooms.stair_nodes import (
    is_stair_node, portal_id_of, search_context, stair_node_id, stair_options)


class RoomReasoningUnavailable(Exception):
    """The room LLM failed or is backing off: this action belongs to the exploration fallback."""


@dataclass(frozen=True)
class LoopSettings:
    """The loop's own knobs. Counts are actions.

    Attributes:
        visit: What a room visit IS. ``explore`` (the default since
            2026-10-13): walk to the room's frontier that reveals the most
            unknown floor per metre of walking, and when the camera has
            resolved it to the next such frontier, until the room has no
            live frontier left; the room is then finished for the episode
            (:class:`~room_scans.RoomScanLedger`, ``explored``). No vantage
            point, no fixed circle: the Darden recording ``ec8d17aeaaaa``
            walked to the middle of a bedroom it had already walked across
            and turned twelve times (actions 255-279), and before that
            turned a full glance on the way into another, walked on to its
            vantage point and turned the same circle again (61-106).
            ``scan``: walk to the room's vantage point -- the interior cell
            of greatest clearance -- turn a full circle, and leave; the
            room is finished for the episode whatever frontier its mask
            still shows, because the camera's ray ends at the nearest
            object and a second viewpoint adds little a scan from the open
            floor did not. ``sweep``: the former bounded frontier sweep --
            ``local_steps`` actions chasing the room's frontier clusters,
            then a loop point, and the same room may be chosen straight
            back; rooms are never finished.
        local_steps: ``sweep`` -- how many actions a room may be explored
            for before its turn ends and the loop point runs. Also the LLM
            cadence there, and the service time RPT* charges a room for.
        explore_visit_steps: ``explore`` -- the hard bound on one visit,
            from arrival: the walks to the room's frontiers together. A
            visit that overruns it is released ``budget_spent`` (not cooled:
            the fresh estimate may send the agent straight back to finish).
        explore_service_steps: ``explore`` -- what RPT* charges a room for
            as service time when its live frontier count is unknown: the
            expected visit, a few frontiers at a few actions each, not its
            bound. With the count known the charge is
            ``openings.service_steps + explore_frontier_steps`` per frontier
            (:meth:`RoomSearchLoop._service_steps_for`), so a one-frontier
            sliver of a room glimpsed through its door costs little more
            than the peek it replaces, and a room with five frontiers costs
            a walk.
        explore_frontier_steps: ``explore`` -- actions one frontier of a
            room is expected to take: the walk to it and the frame at it.
        explore_min_gain_m2: ``explore`` -- a frontier of a room is worth
            walking to when the camera's full circle from it would reveal
            at least this much unknown floor (rays through the unknown,
            blocked by walls and by what the sight ledger has settled). The
            rest -- the strip behind a bed, the sliver under a window, a
            long boundary onto a hand's breadth of unknown -- is not walked
            to, does not keep the room open, and is not counted for the
            oracle's ``frontier=``: the search is for a large object, and a
            room whose doorways were looked through and whose large
            unknowns were walked is finished (since 2026-10-13; the
            Wiconisco recording ``de02bc7475e7`` walked four frontiers of
            one living room for 61 actions). A doorway onto an unseen room
            clears the bar by the room behind it; a gap onto a pocket the
            walls were seen round does not, and that pocket is given up
            (the toilet behind the jamb that only a step inside would show
            is the price). 0 restores every admissible frontier.
        room_missing_grace: Actions a transit keeps walking to its committed
            entry point while the room it was chosen for is missing from the
            registry (since 2026-10-14): the watershed re-segments every
            action and a room can be gone for one of them and back the next
            (Collierville/000023, action 170: room 4 missing for one action,
            the transit released ``unreachable``, the order re-solved without
            it). A room missing for longer is given up as before.
        failure_cooldown_steps: Actions a room is withheld after a turn the
            planner or the follower failed (``unreachable``, ``blocked``,
            ``transit_timeout``), instead of the supervisor's 120-action
            visit cooldown: a room unreachable on the map as it stood is
            routinely reachable once the agent has moved and the map has
            grown, and the 120 kept the toilet's room out of the order for
            the rest of Darden/000034 (released on one refused plan at
            action 112, cooling at 217 when the agent stood 4 m from it).
        entered_discount: The factor a room the agent has STOOD IN but not
            finished keeps of the oracle's probability. The prompt tells the
            model an entered room is worth less than an unexplored one of
            the same type; this is the deterministic backstop for a model
            that does not listen (``qwen2.5:3b`` valued a bathroom walked
            through at 0.6, "known to contain a toilet", over an unknown
            bathroom-sized room never entered at 0.2, and the agent walked
            11 m back to it -- Darden/000034, 2026-10-13). 1 disables.
        lingered_discount: The factor such a room keeps instead when the
            agent LINGERED in it -- searched it for at least
            ``lingered_steps`` actions -- and at most one significant
            frontier is left in it (since 2026-10-14; the user's rule: a
            room spent a long time in with almost nothing left to see is
            as good as scanned, whatever the model writes). Applied in
            place of ``entered_discount``, never on top of it; 1 disables.
        lingered_steps: See ``lingered_discount``.
        scan_visit_steps: ``scan`` -- the hard bound on one visit: the walk
            to the vantage point plus the rotation. A visit that overruns it
            is released ``budget_spent``. Also the bound of a second-pass
            revisit under ``explore``, which is a scan from a new spot.
        scan_approach_steps: ``scan`` -- actions the walk to the vantage
            point may take before the agent scans from where it stands (if
            inside the room) or gives the room up as unreachable.
        vantage_arrival_m: ``scan`` -- within this distance of the vantage
            point the rotation begins.
        vantage_min_clearance_m: ``scan`` -- a room whose best interior cell
            has less clearance than this has no vantage worth walking to;
            the agent scans from where it stands if it is inside, and the
            transit aims at the centroid.
        scan_seen_fraction: ``scan`` -- the share of a room a completed scan
            must have had in clear view (within depth range, through
            observed free space) to finish it WITHOUT standing in it: the
            half of a split room the scan stood in, the small room fully
            visible from the corridor.
        type_prior: Drop a room whose identified type cannot hold the
            target (:mod:`room_priors`) from the nodes altogether, whatever
            the oracle said -- when the label is STRONG (two or more kinds
            of object agree) and neither an object of the target's own
            class nor a home object of the target (a sink, for a toilet)
            has been confirmed inside it. **Off by default since 2026-10-07
            (LLM-first):** the oracle sees every room with its type and
            values it, and the table contradicted the model's own knowledge
            -- a toilet in an en-suite IS reached through a bedroom, which
            the table ruled out. On, it is the hard version of the rule:
            a sofa is not searched for in a bedroom. A weak label (one kind
            of object) never excludes either way.
        min_prob: A node the oracle values below this is not offered to the
            solver. The oracle's "1" for a bedroom in a search for a couch is
            not worth the walk; the exploration fallback carries the search
            when nothing clears the bar.
        second_pass: Offer the rooms the scan ledger has FINISHED as nodes
            again once the home has been COVERED: no room of the storey is
            left unfinished and no exit is left to look into (since
            2026-10-13 the geometry decides; the oracle's own ``pass``
            verdict is recorded and shown, but a ``second`` from the model
            while openings stand unlooked-into no longer re-offers the
            scanned rooms -- ``qwen2.5:3b`` wrote "no unexplored places or
            frontier left" with six openings on the HOUSE line, and the
            agent walked back across the storey to re-scan living rooms
            while the bathroom's door stood unpeeked, Corozal/000076 action
            159). **Off by default since 2026-10-14:** one look at a room
            is the look it gets. The re-scans the covered storey bought
            (Collierville/000023: room 0 scanned at the warm-up, again at
            237-263 and again at 403-420, each from its vantage point,
            while the toilet's WC stood unentered 3 m away) found nothing
            the first scan had not, and the user's rule is that a room
            scanned once is finished. With the storey covered the
            exploration fallback carries the search to the unknown that is
            left -- the slivers and shadows the first pass skipped -- which
            is where an unseen target is. On, a revisit is a scan from a
            different spot (``revisit_standoff_m``); the room is never
            released as finished on the way in.
        scanned_to_oracle: Show the FINISHED rooms to the oracle as nodes
            with their ``status=scanned(...)`` (the LLM-first convention of
            2026-10-07). **Off by default since 2026-10-14:** a finished
            room is not a node, so its line buys the model nothing but a
            longer prompt and -- for ``qwen2.5:3b`` -- the ``pass: second``
            confusion; the HOUSE line still names every room type found,
            scanned ones included, so the big picture stands. The room in
            force is shown regardless.
        revisit_standoff_m: ``second_pass`` -- a second scan stands at
            least this far from every earlier scan point in the room, so it
            sees what the first one grazed; a room too small for that is
            re-scanned from its one good spot.
        unexplored_floor: The least probability an UNEXPLORED node is read
            at, whatever the oracle wrote (``search_node_oracle.
            SearchNodeOracle(unexplored_floor=...)``): a room never entered
            and not yet identified, an opening nothing was glimpsed
            through. **0 (off) by default since 2026-10-07:** the oracle's
            prompt states the rule ("never below 10 while the home type is
            missing") and the 14B model applies it; the floor was the 3B
            era's arithmetic (Hanson 2026-10-04: two never-entered rooms
            and six doorways written off at 0-1%). Set it to restore.
        unexplored_elsewhere: What an unexplored node is read at instead
            when the multi-storey supplement's ``home_here`` verdict says
            the target's home type lives on ANOTHER storey. **0 (off) by
            default**; only the stair-traversal development protocol ever
            sees that verdict.
        home_floor: The least a room holding a confirmed HOME OBJECT of the
            target (``room_priors.HOME_OBJECTS``: a bathtub or a sink for a
            toilet) is read at, whatever the oracle wrote. **0 (off) by
            default since 2026-10-07:** the room's ``seen:`` line shows the
            bathtub and the model values it. The finishing guard stands
            regardless: such a room is never finished by sight from
            OUTSIDE it (``seen_from_scan`` / ``seen_through``) -- the
            Allensville toilet run's warm-up spin saw more than half of
            the bathroom's floor through its door and finished it with the
            toilet behind the jamb.
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
        entry_frontier: ``sweep`` -- transit to the nearest reachable
            frontier inside the chosen room. Off, the room's centroid --
            what the loop flew before, and what a room with no frontier
            left still gets. ``scan`` enters at the vantage point regardless.
        stairs_as_nodes: Offer the floor's staircases to the oracle and the
            solver as nodes beside the rooms (the default). Off, the floor is
            searched alone and the stairs are only ever the exploration
            fallback's last resort -- the ablation, not a mode to fly.
        openings: The floor's openings -- doorways and gaps to space not
            yet seen -- as nodes beside the rooms and the stairs, visited by
            a peek (:mod:`opening_nodes`). ``openings.enabled`` is the
            switch.
        weak_type_max_openings: Retained for configuration compatibility
            and the record. Until 2026-10-05 a weak label ruled a room out
            unless the room had more than this many openings (Ranchester
            2026-10-04, step 75: a toilet glimpsed through a door named the
            hallway a bathroom and every door off it was demoted with it);
            since then a weak label never rules a room out, whatever its
            openings, so this bound is not consulted.
        fragment_max_m2: ``scan`` -- a room under this area with no live
            frontier and no confirmed door is a fragment of another room
            (the strip behind a bed), finished without a visit
            (:class:`~room_scans.RoomScanLedger`); 0 disables.
        walkthrough_clearance_m: ``scan`` -- a room whose widest point has
            no more clearance than this (a corridor, a balcony) is finished
            once the agent has stood in it and no live frontier is left:
            the camera's cone spans it as the agent walks it; 0 disables.
    """

    visit: str = "explore"
    scan_local: str = "rotate"
    local_explorer: str = "frontier"
    local_steps: int = 10
    explore_visit_steps: int = 60
    explore_service_steps: int = 12
    explore_frontier_steps: int = 4
    explore_min_gain_m2: float = 1.0
    failure_cooldown_steps: int = 30
    room_missing_grace: int = 3
    entered_discount: float = 0.5
    lingered_discount: float = 0.2
    lingered_steps: int = 20
    scan_visit_steps: int = 36
    scan_approach_steps: int = 20
    vantage_arrival_m: float = 0.6
    vantage_min_clearance_m: float = 0.4
    scan_seen_fraction: float = 0.5
    type_prior: bool = False
    min_prob: float = 0.05
    second_pass: bool = False
    scanned_to_oracle: bool = False
    revisit_standoff_m: float = 1.5
    unexplored_floor: float = UNEXPLORED_FLOOR
    unexplored_elsewhere: float = UNEXPLORED_ELSEWHERE
    home_floor: float = HOME_FLOOR
    supervisor_rounds: int = 5
    confine_routes: bool = True
    entry_frontier: bool = True
    stairs_as_nodes: bool = True
    openings: OpeningSettings = field(default_factory=OpeningSettings)
    weak_type_max_openings: int = 1
    fragment_max_m2: float = 3.0
    walkthrough_clearance_m: float = 0.9

    def __post_init__(self):
        if self.visit not in ("explore", "scan", "sweep"):
            raise ValueError("visit must be 'explore', 'scan' or 'sweep'; no silent fallback")
        if self.scan_local not in ("rotate", "nearest_frontier"):
            raise ValueError("scan_local must be 'rotate' or 'nearest_frontier'; no silent fallback")
        if self.local_explorer not in ("frontier", "falcon"):
            raise ValueError("local_explorer must be 'frontier' or 'falcon'; no silent fallback")
        for name in ("local_steps", "supervisor_rounds", "scan_visit_steps", "scan_approach_steps",
                     "explore_visit_steps", "explore_service_steps", "explore_frontier_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("%s must be a positive integer" % name)
        if type(self.failure_cooldown_steps) is not int or self.failure_cooldown_steps < 0:
            raise ValueError("failure_cooldown_steps must be a non-negative integer")
        if type(self.room_missing_grace) is not int or self.room_missing_grace < 0:
            raise ValueError("room_missing_grace must be a non-negative integer")
        if self.scan_approach_steps >= self.scan_visit_steps:
            raise ValueError("scan_approach_steps must leave room for the rotation inside scan_visit_steps")
        for name in ("confine_routes", "entry_frontier", "stairs_as_nodes", "type_prior", "second_pass",
                     "scanned_to_oracle"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError("%s must be a bool" % name)
        for name in ("vantage_arrival_m", "vantage_min_clearance_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be positive and finite" % name)
        if isinstance(self.revisit_standoff_m, bool) or not math.isfinite(self.revisit_standoff_m) or self.revisit_standoff_m < 0:
            raise ValueError("revisit_standoff_m must be finite and non-negative")
        if isinstance(self.min_prob, bool) or not math.isfinite(self.min_prob) or not 0 <= self.min_prob < 1:
            raise ValueError("min_prob must lie in [0, 1)")
        for name in ("unexplored_floor", "unexplored_elsewhere", "home_floor"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError("%s must lie in [0, 1)" % name)
        if isinstance(self.scan_seen_fraction, bool) or not math.isfinite(self.scan_seen_fraction) or not 0 < self.scan_seen_fraction <= 1:
            raise ValueError("scan_seen_fraction must lie in (0, 1]")
        if isinstance(self.openings, dict):
            object.__setattr__(self, "openings", OpeningSettings(**self.openings))
        if not isinstance(self.openings, OpeningSettings):
            raise ValueError("openings must be an OpeningSettings (or its dict)")
        if type(self.weak_type_max_openings) is not int or self.weak_type_max_openings < 0:
            raise ValueError("weak_type_max_openings must be a non-negative integer")
        for name in ("fragment_max_m2", "walkthrough_clearance_m", "explore_min_gain_m2"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError("%s must be finite and non-negative" % name)
        if (isinstance(self.entered_discount, bool) or not math.isfinite(self.entered_discount)
                or not 0.0 < self.entered_discount <= 1.0):
            raise ValueError("entered_discount must lie in (0, 1]")
        if (isinstance(self.lingered_discount, bool) or not math.isfinite(self.lingered_discount)
                or not 0.0 < self.lingered_discount <= 1.0):
            raise ValueError("lingered_discount must lie in (0, 1]")
        if type(self.lingered_steps) is not int or self.lingered_steps < 1:
            raise ValueError("lingered_steps must be a positive integer")

    @property
    def scanning(self):
        """Whether a visit is a scan (vantage point, full rotation, finished for good)."""
        return self.visit == "scan"

    @property
    def falcon_visit(self):
        """Whether the explore visit is driven by FALCON's own viewpoints inside the room instead of the view-gain walk."""
        return self.exploring and self.local_explorer == "falcon"

    @property
    def nearest_frontier_scan(self):
        """Whether a scan visit enters at the room's nearest frontier and walks nearest frontiers, no middle, no circle."""
        return self.scanning and self.scan_local == "nearest_frontier"

    @property
    def exploring(self):
        """Whether a visit is the frontier walk (the room's frontiers by view gain, finished when none is left)."""
        return self.visit == "explore"

    @property
    def finishing(self):
        """Whether rooms are FINISHED by the scan ledger -- under ``scan`` and ``explore``; ``sweep`` never finishes one."""
        return self.visit in ("scan", "explore")

    def visit_budget(self):
        """Actions one visit may take before it is released ``budget_spent``."""
        if self.exploring or self.nearest_frontier_scan:
            return self.explore_visit_steps
        return self.scan_visit_steps if self.scanning else self.local_steps

    def service_steps(self):
        """Actions RPT* charges a room for as service time: the expected visit, not its bound."""
        if self.exploring or self.nearest_frontier_scan:
            return self.explore_service_steps
        return (self.scan_approach_steps // 2 + 12) if self.scanning else self.local_steps

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
        re-entering it. A turn the planner or the follower failed
        (``unreachable``, ``blocked``, ``transit_timeout``) cools the room
        for :attr:`failure_cooldown_steps` only: the map it was unreachable
        on is not the map a few actions on.

        Under ``scan`` and ``explore`` the supervisor's own done-tests stand
        down: the ledger finishes a room, not a frontier count that happens
        to read zero three ticks running (``min_frontier_clusters=-1`` never
        holds), and the stall and budget clocks are set beyond the visit
        bound so only the loop's own rules end a visit.
        """
        extra = {}
        if self.finishing:
            bound = self.visit_budget()
            extra = dict(min_frontier_clusters=-1, frontier_stall_s=3.0 * bound, search_timeout_s=3.0 * bound)
        return ObjectSearchParams(
            seed=seed, resolve_on_release=True, min_prob=self.min_prob,
            failure_cooldown_s=float(self.failure_cooldown_steps),
            cooldown_verdicts=tuple(v for v in ALL_VERDICTS if v not in (BUDGET_SPENT, TRAVERSED) + NEUTRAL),
            repeat_verdicts=tuple(v for v in ALL_VERDICTS if v not in (EXHAUSTED, MAPPED)), **extra)


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
        self._openings = {}           # node_id -> OpeningOption offered at the last SELECT
        self._peek = None             # the peek in force: node, opening, approach actions, look targets, index
        self._inspected = set()       # landmark ids looked at from close on this storey (target landmark nodes)
        self._weak_kept = set()       # pids whose weak type label was not allowed to rule them out (logged once)
        self._home_kept_scans = set()  # pids a sight-from-outside verdict was not allowed to finish (logged once)
        self._last_reason_step = -10 ** 9   # the action of the last oracle call, for the openings' revalue throttle
        self._excluded = {}           # pid -> why the room was not offered at the last SELECT
        self._finished = {}           # pid -> (how, step) for every room the scan ledger has finished, offered or not
        self._revisits = set()        # finished rooms offered as second-pass nodes at the last SELECT
        self._second_pass_logged = False   # the second_pass event is logged once per stretch of the verdict
        self._exhausted = False       # the floor has no unfinished room, no exit and no reachable frontier left
        self._covered = False         # ... no unfinished room and no exit (the second pass's gate; frontier may remain)
        self._view_cache = None       # (step, UnknownView) -- the openings' and the explore goals' ray caster this action
        self._value_factors = {}      # node id -> the factor the oracle's probability was scaled by at the last SELECT
        self._peek_missing = 0        # consecutive actions the peek in force's opening was absent from the inventory
        self._scan = None             # the visit in force under ``scan``: phase, room, approach actions, swept yaw
        self._explore = None          # the visit in force under ``explore``: room, started, goals walked, frontier left
        self._opening_cache = None    # (step, options) -- the openings read this action, shared by the HUD and SELECT
        self._folded = set()          # opening node ids folded into an unfinished room's own exploration (logged once)
        self._folded_now = set()      # ... and the ones folded this action, for the HUD
        self._openings_unfolded = []  # every exit and landmark read at the last SELECT, before the fold
        self._service_last = {}       # node id -> seconds the last RPT* instance charged it as service
        self._exit_memory = {}        # exit node id -> the oracle's last number for it, across calls (see _exit_prob)
        self._provisional = set()     # exit node ids read at ``openings.unvalued_prob`` (logged once each)
        self._way_back_held_logged = None    # (rooms left, openings left) the last way_back_held event was logged for
        self._room_missing, self._room_missing_pid = 0, None   # actions the room in transit has been gone from the registry
        self.estimates = {}
        self.events = []
        self.estimate_events = []
        self.stats = {"loop_points": 0, "llm_reasonings": 0, "rooms_released": 0,
                      "budget_releases": 0, "exhausted_releases": 0, "reclassified_releases": 0,
                      "reclassified_in_transit": 0, "relabels": 0, "relabels_kept_visit": 0,
                      "stairs_offered": 0, "stairs_chosen": 0, "stairs_taken": 0, "stairs_refused": 0,
                      "openings_offered": 0, "openings_chosen": 0, "peeks_completed": 0, "peeks_abandoned": 0,
                      "peeks_resolved_en_route": 0, "peeks_stepped_through": 0,
                      "peek_approach_extended": 0,
                      "landmarks_offered": 0, "landmarks_chosen": 0, "landmarks_inspected": 0,
                      "weak_type_kept": 0, "home_object_kept": 0,
                      "way_back_held_for_rooms": 0,
                      "skipped_in_transit": 0,
                      "arrivals": 0, "entry_frontier": 0, "entry_centroid": 0, "entry_vantage": 0, "entry_doorstep": 0,
                      "entry_peek": 0,
                      "entry_reaimed": 0, "entry_lost": 0, "route_reaimed": 0, "room_missing_held": 0,
                      "scans_completed": 0, "scans_in_place": 0, "scan_approach_actions": 0,
                      "scan_unreachable": 0, "excluded_scanned": 0, "excluded_type": 0, "finished_in_transit": 0,
                      "finished_by_rotation": 0, "explore_goals": 0, "explore_visits": 0, "explore_exhausted": 0,
                      "openings_folded": 0, "openings_provisional": 0,
                      "second_pass_rounds": 0, "second_pass_forced": 0, "revisits_offered": 0, "revisits_chosen": 0,
                      "revisit_scans": 0, "oracle_calls_skipped": 0,
                      "reconsiders_deferred": 0,
                      "confined_actions": 0, "unconfined_actions": 0, "plan_failures": 0,
                      "supervisor_rounds": 0, "rounds_exhausted": 0, "llm_fallbacks": 0}

    @property
    def excluded(self):
        """``{pid: why}`` -- the rooms withheld from the nodes at the last SELECT (``scanned:...`` / ``type:...``)."""
        return dict(self._excluded)

    def visit_budget(self):
        """Actions the visit in force may take; the HUD's denominator. A peek has its own, shorter bound."""
        if self._peek is not None and is_opening_node(self.room_id):
            return self.settings.openings.visit_steps
        return self.settings.visit_budget()

    # -- the action ledger --------------------------------------------------
    def charge(self):
        """One action was emitted while a node was in force: count it toward the visit's bound.

        A room's visit counts from arrival; a peek's walk to the threshold
        counts too (``approach_actions``), because the opening's bound is
        the whole peek, walk and look together.
        """
        p = self.policy
        if p.supervisor.state == SEARCH and self.room_id is not None:
            self.local_steps += 1
            if self._scan is not None and self._scan["phase"] == "approach":
                self._scan["approach_actions"] += 1
                self.stats["scan_approach_actions"] += 1
        elif (p.supervisor.state == TRANSIT and self._peek is not None
              and is_opening_node(p.supervisor.room_id) and self._peek["node"] == p.supervisor.room_id):
            self._peek["approach_actions"] += 1

    def reconsider(self, obs, world):
        """A completed peek or new off-route semantic clue warrants a fresh decision.

        Reusing the selected room preserves its local action counter. Switching
        releases it neutrally; no visit cooldown punishes a useful new clue.

        Under ``scan`` a visit in force is finished first: a room half
        turned-through is not left for another on a clue -- the Ranchester
        recording switched rooms mid-visit at 57, 73 and 121 and came back to
        each -- so the clue is kept for the loop point the visit ends on,
        where every node is re-valued anyway.
        """
        p, pose = self.policy, obs.pose
        if self.settings.finishing and p.supervisor.state in (TRANSIT, SEARCH) and p.supervisor.room_id is not None:
            self._needs_reason = True
            self.stats["reconsiders_deferred"] += 1
            self._log(obs, "reconsider_deferred", room=p.supervisor.room_id, state=p.supervisor.state)
            return
        cost = assemble_cost_grid(p.planner.fields_for(world), p.planner_params, p.settings.body_radius_m)[0]
        stairs, openings = self._collect_nodes(obs, world, cost)
        if not self._worth_asking(stairs, openings):
            # Nothing the answer could send the search to (the spawn room finished, no
            # doorway seen yet): the clue is kept for the loop point a node appears at.
            self._needs_reason = True
            self.stats["oracle_calls_skipped"] += 1
            self._log(obs, "oracle_skipped", where="reconsider", rooms=len(p.graph.registry.rooms),
                      finished=sorted(int(pid) for pid in self._finished), frontier_left=self._frontier_left())
            return
        self._reason(obs, world, stairs, openings)
        options = self._options(obs, world, cost, stairs, openings)
        instance = None
        if options:
            instance, dropped = self._instance(obs, world, cost, options, stairs, openings)
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
            self._peek = None
            p.route_memory.clear("semantic_switch")
            p._route = p._goal = None
        self.order, self.order_index = tuple(state.order), 0
        self._log(obs, "semantic_replan", order=list(self.order), retained=state.completed is None)

    # -- one action ---------------------------------------------------------
    def plan(self, obs, world):
        """The command for this action: in-room exploration, transit, or the exploration fallback."""
        p = self.policy
        cost = assemble_cost_grid(p.planner.fields_for(world), p.planner_params, p.settings.body_radius_m)[0]
        # The openings are read off the live map every action -- for the HUD, and so that an
        # opening the camera has just resolved is gone the action it is resolved, not at the
        # next loop point (Darden 2026-10-13: O3 stood on the HUD for fifty actions after the
        # agent had looked straight through it). The fold here is the HUD's, read against the
        # last SELECT's exclusions; the SELECT folds again over fresh ones and logs.
        self._fold_openings(obs, self._opening_options(obs, world, cost), log=False)
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
        """Step 1's bound, judged before the first tick so the release costs no round.

        ``sweep``: the N-step rule. ``scan``: the hard bound on one visit --
        a vantage walk that overruns it plus the rotation is released
        ``budget_spent`` rather than carried on for ever.
        """
        p = self.policy
        if (p.supervisor.state == SEARCH and self.room_id is not None
                and self.local_steps >= self.visit_budget()):
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
            if is_opening_node(room_id) and verdict in (BLOCKED, UNREACHABLE, TRANSIT_TIMEOUT):
                # An opening the agent could not get to is not offered again: left a node, the
                # supervisor's escape hatch re-chose the only candidate the action after each
                # release, and a wedged agent bought one oracle call per seven actions.
                option = self._openings.get(room_id)
                opening = (self._peek or {}).get("opening") if self._peek and self._peek.get("node") == room_id else None
                opening = opening or (option.opening if option is not None else None)
                if opening is not None and not self._retired(opening):     # the transit may have retired it already
                    self._retire_peek(obs, opening, "released %s" % verdict)
            p._route = p._goal = None
            p.route_memory.clear("room_completed")
            # The blocked clock was the released node's: it started when the follower pushed into
            # something on THAT route, and the next node's transit has not been blocked yet. Left
            # running, it condemned every node selected after a ``blocked`` release on its first
            # action (Darden/000034, 2026-10-14, actions 119-124: six releases in six actions, four
            # openings retired for good); ``notify_blocked`` starts it again if the new route sticks.
            p._blocked_since = None
            self._entry = None
            self._scan = None
            self._explore = None
            self._peek = None
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
            elif is_opening_node(state.room_id):
                # The peek begins with the walk: the opening as it was offered is
                # carried along, because the inventory it came from is re-read
                # only at the next SELECT and the look needs its heading. The
                # walk's bound grows with the distance to the threshold.
                entry = "peek"
                option = self._openings.get(state.room_id)
                opening = None if option is None else option.opening
                bound = self.settings.openings.approach_bound(
                    0.0 if opening is None else opening.geodesic_m, p.episode.action_spec.forward_step_m)
                self._peek = dict(node=state.room_id, opening=opening, approach_actions=0, approach_bound=bound,
                                  goal=None if opening is None else opening.xy, reaimed=False,
                                  started=int(obs.step), targets=[], index=0, turns=0)
                self._peek_missing = 0
                if opening is not None and opening.kind == LANDMARK:
                    entry = "landmark"
                    self.stats["landmarks_chosen"] += 1
                else:
                    self.stats["openings_chosen"] += 1
                self.stats["entry_peek"] += 1
            else:
                if self.settings.nearest_frontier_scan and state.room_id not in self._revisits:
                    entry = "frontier" if self._entry[1] else "doorstep"
                elif self.settings.scanning:
                    entry = "vantage"
                elif self.settings.exploring:
                    entry = ("frontier" if self._entry[1] else
                             "vantage" if state.room_id in self._revisits else "doorstep")
                else:
                    entry = "frontier" if self._entry[1] else "centroid"
                self.stats["entry_" + entry] += 1
                if state.room_id in self._revisits:
                    self.stats["revisits_chosen"] += 1
            self._log(obs, "transit", room=state.room_id, goal=list(state.goal_xy), entry=entry, order=list(state.order),
                      revisit=state.room_id in self._revisits)
        if state.changed and state.state == SEARCH:
            self.local_steps = 0                            # step 7
            self.room_id = state.room_id
            self._scan = None
            self._explore = None
            if not is_opening_node(state.room_id):
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
            # is coming to give it one. Finished and ruled-out rooms are not
            # nodes: they are neither valued nor missed. The openings of the
            # floor and its staircases are nodes beside the rooms; a NEW
            # opening buys a call at most every ``revalue_actions`` (they
            # appear and re-snap with every step the fallback takes).
            stairs, openings = self._collect_nodes(obs, world, cost)
            unvalued = (any(pid not in p.graph.probs for pid in rooms if pid not in self._excluded)
                        or any(o.node_id not in p.graph.stair_probs for o in stairs))
            new_openings = any(o.node_id not in p.graph.stair_probs for o in openings if o.opening.kind != LANDMARK)
            if new_openings and obs.step - self._last_reason_step >= self.settings.openings.revalue_actions:
                unvalued = True
            if (self.second_pass() and self._revisits
                    and (int(obs.step) // BUDGET_STEP) != (max(0, self._last_reason_step) // BUDGET_STEP)):
                # The budget line moved: the model re-weighs the scanned rooms against the time left.
                unvalued = True
            if self._needs_reason or unvalued:
                if self._worth_asking(stairs, openings):
                    self._reason(obs, world, stairs, openings)
                    # The fold was read against the probabilities BEFORE this call; a room the
                    # model has just raised over the floor is a node now and its openings are its
                    # walk's, a room it has just dropped under it is not and they are nodes again
                    # (Corozal/000142, action 68: two openings in the order, folded off the HUD
                    # the next action).
                    openings = self._fold_openings(obs, self._openings_unfolded)
                elif self._needs_reason:
                    self.stats["oracle_calls_skipped"] += 1
                    self._log(obs, "oracle_skipped", rooms=len(rooms), finished=sorted(int(pid) for pid in self._finished),
                              frontier_left=self._frontier_left())
                    self._needs_reason = False
            # An exit unfolded by the reply was not shown to the model and has no number; it is
            # read at the model's last number for it or the unseen doorway's prior, not 0.0.
            self._provision_exits(obs, openings)
            options = self._options(obs, world, cost, stairs, openings)      # step 6's entry point per node
            if options:
                started = time.monotonic()
                instance, dropped = self._instance(obs, world, cost, options, stairs, openings)
                self._record(instance, dropped, options)
                options = [o for o in options if o.room_id in instance.index_to_pid]
                p.telemetry.latencies["room_selection"].append((time.monotonic() - started) * 1000)
            else:
                self._record(None, (), options)
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

    def _collect_nodes(self, obs, world, cost):
        """Everything beside the rooms this action: ``(stairs, openings)`` -- and the rooms withheld.

        The openings are read first because the exclusions need them: a
        weak type label rules a room out only if the room has few openings,
        and the floor is exhausted only when no exit is left. Under
        ``explore`` an opening off a room that is itself a node is then
        FOLDED into that room (:meth:`_fold_openings`): the room's visit
        walks its frontiers, the opening among them.
        """
        openings = self._opening_options(obs, world, cost)
        self._exclusions(obs, world, openings)
        self._openings_unfolded = list(openings)
        openings = self._fold_openings(obs, openings)
        stairs = self._stair_options(obs, world, cost)
        return stairs, openings

    def _fold_openings(self, obs, openings, log=True):
        """Under ``explore``, drop the openings whose room is an offered node; its frontier walk covers them.

        An opening is a frontier at the edge of the mapped floor, credited
        to the room it opens from; the explore visit of that room walks
        exactly those frontiers. Offered beside the room it is the same
        place twice -- the Darden recording's RPT* order read ``R3 -> O3``,
        a room and the gap at its far end -- and a peek of it is a visit
        the room's own visit makes anyway. Openings off FINISHED rooms (and
        rooms withheld for any reason) stay nodes: they are how the next
        room is found. Target landmarks are never folded. (A 2026-10-14
        experiment kept openings with a room's worth of unknown behind them
        unfolded too, after Corozal/000019 folded the toilet's door into a
        never-entered living room; it was reverted with the rest of that
        round -- the run the user chose, ``…143259Z``, folds as described
        here.) The folded ids
        are kept in :attr:`_folded_now` for the HUD; :attr:`_openings`
        keeps every opening read this action, folded or not, because the
        transit and the retirement look the chosen node up there.
        """
        self._folded_now = set()
        if not self.settings.exploring or not openings:
            return openings
        p = self.policy
        rooms = p.graph.registry.rooms
        probs = getattr(p.graph, "probs", {}) or {}
        kept = []
        for option in openings:
            pid = option.opening.room_pid
            # A room the oracle has valued below the solver's floor is no node, and its
            # openings must stay nodes of their own (the kitchen at 0.0 whose doorways led on).
            offered = (pid is not None and pid in rooms and pid not in self._excluded
                       and (pid not in probs or float(probs[pid]) >= self.settings.min_prob))
            if option.opening.kind != LANDMARK and offered:
                self._folded_now.add(option.node_id)
                if log and option.node_id not in self._folded:
                    self._folded.add(option.node_id)
                    self.stats["openings_folded"] += 1
                    self._log(obs, "opening_folded", node=option.node_id, room=pid)
                continue
            kept.append(option)
        return kept

    def _exit_prob(self, node_id):
        """The probability an exit node is handed to RPT* with, and where it came from: ``(prob, source)``.

        The oracle's number from THIS call when it was shown the exit
        (``"oracle"``); else its last number for the exit from an earlier
        call (``"remembered"``: the graph replaces its probabilities
        wholesale at every call, so an exit folded into its room at the
        call and unfolded by the reply loses the number it had); else
        :attr:`OpeningSettings.unvalued_prob` (``"provisional"``: the model
        has never seen it). The node without a number still buys the next
        call (:meth:`_update` tests ``stair_probs`` membership), so the
        provisional value lasts one SELECT at most.
        """
        probs = self.policy.graph.stair_probs
        if node_id in probs:
            return float(probs[node_id]), "oracle"
        if node_id in self._exit_memory:
            return float(self._exit_memory[node_id]), "remembered"
        return float(self.settings.openings.unvalued_prob), "provisional"

    def _provision_exits(self, obs, openings):
        """Remember the oracle's numbers for the exits and log each exit read at the provisional value (once).

        Called at every SELECT after the (possible) call and re-fold, over
        the exits that are nodes this action. Corozal/000142 (2026-10-14,
        action 446): the toilet's doorway, unfolded by the reply that
        dropped its room to 1 %, read 0.0 for the 54 actions left -- see
        :attr:`OpeningSettings.unvalued_prob`.
        """
        p = self.policy
        self._exit_memory.update({nid: float(v) for nid, v in p.graph.stair_probs.items()})
        live = {o.node_id for o in openings}
        self._provisional &= live
        if self.settings.openings.unvalued_prob <= 0.0:
            return
        for option in openings:
            nid = option.node_id
            if option.opening.kind == LANDMARK or nid in self._provisional:
                continue
            prob, source = self._exit_prob(nid)
            if source != "provisional":
                continue
            self._provisional.add(nid)
            self.stats["openings_provisional"] += 1
            self._log(obs, "opening_provisional", node=nid, room=option.opening.room_pid, prob=round(prob, 3),
                      gain_m2=round(float(option.opening.gain_m2), 2))

    def _instance(self, obs, world, cost, options, stairs, openings):
        """Step 3's RPT* instance: every node's probability, the climb on the stairs, a peek's price on an opening.

        Under ``explore`` a room's service time is its own
        (:meth:`_service_steps_for`): a few actions per live frontier, so a
        sliver of a room with one frontier is priced like the peek it
        replaces and a room with many like the walk it will take.
        """
        p, s = self.policy, self.policy.settings
        # The options carry the probabilities the supervisor filtered on: the oracle's numbers
        # with the geometry's factors applied (:meth:`_scaled_values`), one number per node.
        values = {o.room_id: float(o.prob) for o in options}
        peek = self.settings.openings.service_steps * s.action_time_s
        service = {o.node_id: peek for o in openings}             # a look from the threshold, not a room's scan
        if self.settings.exploring:
            service.update({o.room_id: self._service_steps_for(o.room_id, obs, world) * s.action_time_s
                            for o in options if o.room_id in p.graph.registry.rooms})
        self._service_last = dict(service)                        # what each node was charged, for the record
        return build_instance(
            world, cost, {o.room_id: o.xy for o in options}, values,
            depot_xy=(obs.pose.x, obs.pose.y),
            cruise_speed_mps=p.episode.action_spec.forward_step_m / s.action_time_s,
            search_time_s=self.settings.service_steps() * s.action_time_s,
            leaves={o.node_id: o.leaf_m for o in stairs},          # the climb, on every arc touching the stairs
            service_s=service,
            preferred_cost=p.preferred_cost(world) if hasattr(p, "preferred_cost") else None)

    def _scaled_values(self, options, openings, here=None, obs=None, world=None):
        """The options with the geometry's word on each node's probability; the factors kept in :attr:`_value_factors`.

        An exit keeps :func:`~opening_nodes.opening_value_factor` of the
        oracle's number -- the unseen floor the look from its threshold
        would show, a doorway in full. A room the agent has stood in and
        the ledger has not finished keeps :attr:`LoopSettings.entered_discount`
        of its number: what the model calls "entered" the agent has seen
        most of, and the unexplored room of the same type comes first --
        and :attr:`LoopSettings.lingered_discount` instead when the agent
        searched it for :attr:`LoopSettings.lingered_steps` or more with at
        most one significant frontier left in it (as good as scanned). The
        model is told the rules in its prompt; this is where they hold
        whatever it wrote. The supervisor filters on the scaled number and
        RPT* is handed it, so a node scaled under ``min_prob`` is no node,
        as the model's own low number would make it. ``here`` is the pid of
        the room the agent stands in, which is not discounted.
        """
        p = self.policy
        settings = self.settings
        factors = {}
        by_node = {o.node_id: o for o in openings}
        out = []
        for option in options:
            nid = option.room_id
            factor = 1.0
            if nid in by_node:
                factor = opening_value_factor(by_node[nid].opening, settings.openings)
            elif (settings.entered_discount < 1.0 and nid in p.graph.registry.rooms and nid not in self._finished
                  and p.graph.last_inside.get(nid) is not None and nid != here):
                # Entered and LEFT. The room the agent stands in keeps its number: finishing it
                # from here is the cheap move, and leaving it with frontier is what the rule is against.
                factor = settings.entered_discount
                if settings.lingered_discount < factor and self._lingered(nid, obs, world):
                    factor = settings.lingered_discount
            if factor < 1.0:
                factors[nid] = round(factor, 3)
                option = replace(option, prob=float(option.prob) * factor)
            out.append(option)
        self._value_factors = factors
        return out

    def _lingered(self, pid, obs=None, world=None):
        """Whether room ``pid`` was searched for ``lingered_steps`` or more with at most one significant frontier left."""
        p = self.policy
        fact = p.graph.facts.get(pid) if getattr(p.graph, "facts", None) else None
        if fact is None:
            return False
        if float(fact.time_in_room_s) < self.settings.lingered_steps * p.settings.action_time_s:
            return False
        frontier = self._live_frontier(pid, obs, world)
        return frontier is not None and int(frontier) <= 1

    def _service_steps_for(self, pid, obs=None, world=None):
        """Actions RPT* charges room ``pid`` for under ``explore``: a peek's worth plus a few per live frontier.

        A room with no frontier left is a walk in to confirm it (the peek's
        price); a second-pass revisit is a scan and is charged the scan's
        expected visit. Bounded by :attr:`LoopSettings.explore_visit_steps`.
        """
        settings = self.settings
        if not settings.exploring:
            return settings.service_steps()
        if pid in self._finished:
            return settings.scan_approach_steps // 2 + self.policy.rotation_turns()
        frontier = self._live_frontier(pid, obs, world)
        if frontier is None:
            return settings.explore_service_steps
        return min(settings.explore_visit_steps,
                   settings.openings.service_steps + settings.explore_frontier_steps * int(frontier))

    def _opening_options(self, obs, world, cost):
        """The floor's openings as nodes (:mod:`opening_nodes`): exits not yet peeked, then the target landmarks not yet looked at.

        Read once per action and cached: :meth:`plan` reads them for the
        HUD and the fold, SELECT for the nodes, and both see one inventory.
        """
        p = self.policy
        registry = getattr(p, "openings", None)
        if not self.settings.openings.enabled or registry is None:
            self._openings = {}
            return []
        if self._opening_cache is not None and self._opening_cache[0] == int(obs.step):
            return list(self._opening_cache[1])
        openings = detect_openings(p, obs, world, self.settings.openings, registry)
        landmarks = landmark_openings(p, obs, world, cost, self.settings.openings, self._inspected)
        options = opening_options(p, openings + landmarks)
        offered = {o.node_id: o for o in options}
        new = set(offered) - set(self._openings)
        self.stats["openings_offered"] += sum(1 for nid in new if offered[nid].opening.kind != LANDMARK)
        self.stats["landmarks_offered"] += sum(1 for nid in new if offered[nid].opening.kind == LANDMARK)
        self._openings = offered
        self._opening_cache = (int(obs.step), list(options))
        return options

    def _stair_options(self, obs, world, cost):
        """The floor's staircases as nodes: eligible, and entered from a point the map has reached.

        Nothing here is a clock. A staircase whose foot the observed map has
        not reached is not a node THIS action and comes back when the map
        grows to it; one cooling after a failed approach comes back when the
        cooling ends. The stairs the aircraft just came down are offered like
        any other -- the oracle is told so, and values the way back -- EXCEPT,
        under ``scan``, while this storey still has a room that is neither
        finished nor ruled out (:meth:`_exclusions` runs first) or an
        opening not yet looked into: the storey is looked at before the way
        back is a choice. Under ``sweep`` rooms are never finished, so the
        arrival grace is the only hold there. Any other staircase is offered
        at once.
        """
        p = self.policy
        building = getattr(p, "building", None)
        if (not self.settings.stairs_as_nodes or not p.settings.allow_stair_traversal or building is None
                or building.ground_truth is None or building.committed):
            self._stairs = {}
            return []
        rooms_left = self.settings.finishing and (any(pid not in self._excluded and pid not in self._finished
                                                      for pid in p.graph.registry.rooms)
                                                  or bool(self._openings))
        options = stair_options(building, obs, world, cost, p.floors.save(), p.settings.action_time_s,
                                rooms_left=rooms_left)
        offered = {o.node_id: o for o in options}
        if set(offered) != set(self._stairs):
            self.stats["stairs_offered"] += len(set(offered) - set(self._stairs))
        arrived_by = getattr(building, "arrived_by", None)
        if rooms_left and arrived_by is not None and not any(
                o.portal.get("connector_id") == arrived_by for o in options) and any(
                q.get("connector_id") == arrived_by and q["floor_id"] == building.floor_id for q in building.portals):
            left = sorted(pid for pid in p.graph.registry.rooms if pid not in self._excluded)
            held = (tuple(left), tuple(sorted(self._openings)))
            if self._way_back_held_logged != held:              # once per change of what holds it, not per action
                self.stats["way_back_held_for_rooms"] += 1
                self._log(obs, "way_back_held", connector=arrived_by, rooms_left=left,
                          openings_left=len(self._openings))
                self._way_back_held_logged = held
        self._stairs = offered
        return options

    def _reason(self, obs, world, stairs=(), openings=()):
        """Steps 2 and 3: every room re-labelled from every object so far, then P(find) per node.

        Over geometry refreshed on THIS action: the background refresh runs
        every ``graph_period_steps`` actions, and a loop point that fell
        between two of them would otherwise estimate rooms whose frontier
        counts and areas are up to a period old. The staircases and the
        openings the loop offers are valued in the same call, beside the
        rooms.

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
                           here_xy=(obs.pose.x, obs.pose.y), yaw=obs.pose.yaw, ranking=p.sweep.settings.ranking,
                           exclude=p.room_exclusion(world) if hasattr(p, "room_exclusion") else None)
            p.telemetry.latencies["scene_graph"].append((time.monotonic() - started) * 1000)
            p._last_graph_step, p._last_door_revision = obs.step, p.doors.revision
        # The labels are read from the evidence BEFORE the nodes are chosen, so a
        # room that became a strong bedroom on this action is excluded on this
        # action, not shown to the oracle and kept in the order for one more
        # loop point (Hanson 2026-10-05, action 150).
        if self.settings.type_prior and callable(getattr(p.graph, "refresh_labels", None)):
            try:
                p.graph.refresh_labels(obs.step)
            except Exception as exc:  # the classifier is the room LLM: same back-off, held labels stand
                retry = p.fallback.note_service_failure(obs, ROOM_LLM, exc)
                self._log(obs, "llm_failure", error="%s: %s" % (type(exc).__name__, exc), retry_step=retry,
                          where="refresh_labels")
                raise RoomReasoningUnavailable(str(exc)) from exc
            self._exclusions(obs, world, openings)
        started = time.monotonic()
        shown = [o for o in openings if o.opening.kind != LANDMARK]        # a target landmark's probability is fixed
        scanned = self._scanned_for_prompt(obs)
        house = house_line(p.graph, scanned, len(shown), int(obs.step), int(p.episode.max_steps),
                           frontier_left=self._frontier_left())
        # The finished rooms are not nodes (``scanned_to_oracle`` off): they are left off the
        # model's list and read at 0.0 downstream; the HOUSE line still names their types. The
        # room in force and a room the second pass has re-offered are shown either way.
        hidden = () if self.settings.scanned_to_oracle else tuple(
            pid for pid, why in self._excluded.items() if str(why).startswith("scanned") and pid != self.room_id)
        try:
            # Every unfinished room is shown, ruled-out ones included, with its status: the
            # model values the home (LLM-first, 2026-10-07).
            p.graph.reason(world, p.target, obs.step,
                           extra_nodes=[o.node for o in stairs] + [o.node for o in shown],
                           context=search_context(getattr(p, "building", None), p, p.floors.save(), house=house),
                           here_xy=(obs.pose.x, obs.pose.y), action_time_s=p.settings.action_time_s,
                           exclude=hidden, scanned=scanned)
        except Exception as exc:  # the room LLM: timeout, bad JSON, refused uniform prior
            retry = p.fallback.note_service_failure(obs, ROOM_LLM, exc)
            self._log(obs, "llm_failure", error="%s: %s" % (type(exc).__name__, exc), retry_step=retry)
            raise RoomReasoningUnavailable(str(exc)) from exc
        finally:
            p.telemetry.latencies["room_reasoning"].append((time.monotonic() - started) * 1000)
        p.fallback.note_service_success(ROOM_LLM)
        self._needs_reason = False
        self._last_reason_step = int(obs.step)
        self.stats["llm_reasonings"] += 1
        oracle = p.graph.last_reasoning.get("oracle", {})
        # The fresh verdict decides whether the finished rooms are offered on THIS
        # action (the exclusions above were read under the previous one).
        self._exclusions(obs, world, openings, after_reasoning=True)
        # Kept apart from ``events``, which is the record of step transitions;
        # one entry per reasoning round, with what the oracle was shown and said.
        self.estimate_events.append(dict(
            step=int(obs.step), floor_id=p.mapping.floor_id, nodes=len(oracle.get("probs", {})),
            stairs=[o.node_id for o in stairs], openings=[o.node_id for o in shown],
            landmarks=[o.node_id for o in openings if o.opening.kind == LANDMARK],
            p_present=round(float(oracle.get("p_present", 0.0)), 3),
            elsewhere=round(float(oracle.get("elsewhere", 0.0)), 3), reused=bool(oracle.get("reused")),
            omitted=list(oracle.get("omitted", ())), reading=dict(oracle.get("reading", {})),
            home_here=oracle.get("home_here"), pass_verdict=oracle.get("pass_verdict"),
            floored=list(oracle.get("floored", ())), capped=list(oracle.get("capped", ())),
            house=house, scanned=sorted(int(pid) for pid in scanned),
            excluded={str(pid): why for pid, why in sorted(self._excluded.items())}))

    def _scanned_for_prompt(self, obs):
        """``{pid: (how, ago_s)}`` for every finished room, in the prompt's words (from the last :meth:`_exclusions`)."""
        p = self.policy
        out = {}
        for pid, (how, step) in self._finished.items():
            ago = None if step is None else max(0.0, (int(obs.step) - int(step)) * p.settings.action_time_s)
            out[pid] = (scan_words(how), ago)
        return out

    def second_pass(self, exhausted=None, covered=None):
        """Whether finished rooms are nodes again: the storey is COVERED -- no unfinished room, no exit to look into.

        The geometry decides (since 2026-10-13). ``covered`` is the reading
        from the last :meth:`_exclusions` unless given: every room the
        registry knows is finished (or in force) and no opening of the
        storey is left to peek; ``exhausted`` adds "no reachable frontier
        anywhere" and is the forced form, recorded as such. The oracle's
        own ``pass`` verdict is shown and recorded but does not open the
        second pass on its own: a ``second`` with openings still standing
        sent the agent back across the storey to re-scan living rooms
        while the bathroom's door stood unpeeked (Corozal/000076, action
        159; Darden/000034, action 321).
        """
        if not self.settings.second_pass or not self.settings.finishing:
            return False
        if exhausted is None:
            exhausted = self._exhausted
        if covered is None:
            covered = self._covered
        return bool(exhausted or covered)

    def oracle_pass(self):
        """The oracle's latest ``pass`` verdict (``first`` / ``second`` / None), for the record and the HUD."""
        oracle = self.policy.graph.last_reasoning.get("oracle", {}) if getattr(self.policy, "graph", None) else {}
        return oracle.get("pass_verdict")

    # -- what is not a node: finished rooms and rooms the target cannot be in ----
    def _exclusions(self, obs, world, openings=(), after_reasoning=False):
        """Rooms withheld from the solver and the transit this action, with why.

        Since 2026-10-07 (LLM-first) every room is SHOWN to the oracle with
        its status; this decides only which rooms are OFFERED as nodes. Two
        tests, both read off the live map:

        * ``scanned:<how>`` -- the :class:`~room_scans.RoomScanLedger` says a
          completed look-around stood in the room or saw most of it
          (``scan_point_inside``, ``seen_from_scan``); or, with no live
          frontier left in it, that the camera looked into it or walked it
          through (``seen_through``), that it is a doorless fragment under
          3 m2 (``fragment``), or -- under ``explore`` -- that the agent
          stood in it (``explored``: its frontiers were walked until none
          was left). Withheld while the oracle's ``pass``
          verdict is ``first`` (or missing): the first pass over the home
          comes before any second look. Under ``pass="second"``
          (:attr:`LoopSettings.second_pass`) finished rooms are offered at
          the oracle's probability -- the model has said the home is covered
          and the detector may have missed the target -- and recorded in
          :attr:`_finished` either way for the prompt's ``status=``. Under
          ``sweep`` this test stands down: the sweep's own termination rule
          decides when a room is done.
        * ``type:<label>`` -- with :attr:`LoopSettings.type_prior` on, the
          room's identified type cannot hold the target
          (:func:`room_priors.ruled_out`): the label is STRONG, and
          neither an object of the target's own class nor a home object of
          the target (a sink, for a toilet) has been confirmed inside it. A
          WEAK label (one kind of object) never rules a room out: the one
          object may belong to a room behind one of its openings, or to a
          room the partition merged into it (``weak_type_kept`` events,
          logged once per room); a home object in a room of another type
          is the partition's merge showing (``home_object_kept``). Off by
          default: the oracle sees the type and values the room.

        The room in force is never excluded mid-visit: its turn ends by the
        visit's own rule, and the exclusion takes effect at the loop point.
        """
        p = self.policy
        excluded, finished, verdicts = {}, {}, {}
        ledger = getattr(p, "scans", None)
        exits = self._exits_by_room(openings)
        doored = set()
        for door in getattr(p.graph, "doors", ()) or ():
            doored.update(int(pid) for pid in door.get("rooms", ()))
        in_force = self.room_id if p.supervisor.state in (TRANSIT, SEARCH) else None
        for pid, room in p.graph.registry.rooms.items():
            if pid == in_force:
                if pid in self._finished:
                    finished[pid] = self._finished[pid]
                continue
            if not self.settings.finishing or ledger is None:
                continue
            how = ledger.status(world, pid, room, frontier=self._live_frontier(pid, obs, world), doored=pid in doored,
                                step=int(obs.step))
            if how is not None and how != SCAN_POINT_INSIDE and self._home_objects(pid):
                # Finished by sight from OUTSIDE -- half its floor seen from the
                # door -- with a home object of the target standing in it: the
                # toilet is behind the jamb the sightline never passed. A room
                # the agent stood in and turned a circle in is finished as before.
                if pid not in self._home_kept_scans:
                    self._home_kept_scans.add(pid)
                    self.stats["home_object_kept"] += 1
                    self._log(obs, "home_object_kept", room=pid, scanned=how, home_objects=self._home_objects(pid))
                how = None
            if how is not None:
                verdicts[pid] = how
                finished[pid] = (how, ledger.finished_step(pid))
        # The second pass: the storey covered -- no unfinished room, no exit to look
        # into -- in which case re-checking the scanned rooms is the move there is
        # (exhausted: no reachable frontier anywhere either, the forced form).
        unfinished = [pid for pid in p.graph.registry.rooms if pid != in_force and pid not in verdicts]
        has_exit = any(o.opening.kind != LANDMARK for o in openings)
        # With a storey change allowed the stairs are a move too: there the second
        # pass is the model's verdict alone (the development protocol).
        covered = (self.settings.finishing and bool(verdicts) and not unfinished and not has_exit
                   and not (p.settings.allow_stair_traversal and self.settings.stairs_as_nodes)
                   and not self._significant_frontier_left(obs, world))
        exhausted = covered and not self._frontier_left()
        revisiting = self.second_pass(exhausted=exhausted, covered=covered) and bool(verdicts)
        for pid in p.graph.registry.rooms:
            if pid == in_force:
                continue
            how = verdicts.get(pid)
            if how is not None and not revisiting:
                excluded[pid] = "scanned:%s" % how
                continue
            if self.settings.type_prior:
                why = self._ruled_out(obs, pid, exits.get(pid, 0))
                if why is not None:
                    excluded[pid] = why
        self._exhausted, self._covered = exhausted, covered
        if revisiting and not self._second_pass_logged:
            self._second_pass_logged = True
            self.stats["second_pass_rounds"] += 1
            if exhausted:
                self.stats["second_pass_forced"] += 1
            if not after_reasoning:
                self._needs_reason = True       # the finished rooms are nodes now: value them afresh
            self._log(obs, "second_pass", rooms=sorted(int(pid) for pid in verdicts), forced=exhausted,
                      oracle_pass=self.oracle_pass())
        elif not revisiting:
            self._second_pass_logged = False
        if set(excluded) != set(self._excluded):
            newly = sorted(set(excluded) - set(self._excluded))
            self.stats["excluded_scanned"] += sum(1 for pid in newly if excluded[pid].startswith("scanned"))
            self.stats["excluded_type"] += sum(1 for pid in newly if excluded[pid].startswith("type"))
            if newly:
                self._log(obs, "excluded", rooms={str(pid): excluded[pid] for pid in newly})
        self._excluded = excluded
        self._finished = finished
        return excluded

    def _frontier_left(self):
        """Whether the floor still has a reachable frontier anywhere (the exploration fallback's first rung)."""
        inventory = getattr(self.policy.graph, "frontier_inventory", None)
        goals = getattr(inventory, "goals", None) if inventory is not None else None
        if not goals:
            return False
        return any(math.isfinite(float(getattr(goal, "geodesic_m", float("inf")))) for goal in goals)

    def _significant_frontier_left(self, obs, world):
        """Whether a reachable frontier WORTH a walk is left anywhere: ``explore_min_gain_m2`` of unknown a look from it would reveal.

        The second pass's gate reads this, not :meth:`_frontier_left`: the
        strip behind a bed is frontier for ever and must not keep the
        scanned rooms off the table once every room and doorway is done --
        and a doorway's worth of unknown must, because walking into unseen
        space comes before re-scanning seen space. Under ``explore`` only;
        the other visits read the plain frontier.
        """
        if not self.settings.exploring or self.settings.explore_min_gain_m2 <= 0.0:
            return self._frontier_left()
        inventory = getattr(self.policy.graph, "frontier_inventory", None)
        goals = getattr(inventory, "goals", None) if inventory is not None else None
        if not goals:
            return False
        view = self._unknown_view(obs, world)
        min_gain = int(round(self.settings.explore_min_gain_m2 / view.cell_area_m2))
        kept = rank_by_view_gain([g for g in goals if math.isfinite(float(g.geodesic_m))], view,
                                 self.policy.sweep.settings.ranking, min_gain_cells=min_gain)
        return bool(kept)

    def _worth_asking(self, stairs, openings):
        """Whether this loop point has a node the oracle's answer could send the search to.

        An unfinished room, a staircase or an exit to look into is one. With
        nothing of the kind -- the spawn room finished by the warm-up and no
        doorway seen yet -- the reasoning model is not spent on a reply that
        cannot change the action: the exploration fallback carries the search
        to the nearest frontier, and the call is made when a node appears.
        The finished rooms count once the second pass has them offered.
        """
        p = self.policy
        if stairs or any(o.opening.kind != LANDMARK for o in openings):
            return True
        return any(pid not in self._excluded for pid in p.graph.registry.rooms)

    @staticmethod
    def _exits_by_room(openings):
        """``{pid: n}`` -- how many exits (not landmarks) the offered openings put on each room."""
        exits = {}
        for option in openings:
            pid = option.opening.room_pid
            if pid is not None and option.opening.kind != LANDMARK:
                exits[pid] = exits.get(pid, 0) + 1
        return exits

    def _live_frontier(self, pid, obs=None, world=None):
        """Accessible, unresolved frontier clusters the scene graph credits to room ``pid``, or None when unknown.

        The count behind the oracle's ``frontier=`` and the ledger's
        ``seen_through`` / ``explored`` tests: read off the frontier
        inventory, which is built on the map with the settled unknown
        written occupied (:class:`~sightlines.SightLedger`), so a railing
        looked through or a pocket behind a bed is not a boundary left to
        explore. Under ``explore``, with the pose given, the clusters are
        counted through the sweep's ONE admissibility filter
        (:meth:`~frontier_sweep.FrontierSweep.admissible`): a goal under
        the agent's feet or beside one already retired is not a frontier
        the visit can walk to, so it is not one that keeps the room open --
        the visit and the verdict read the same population, or a room is
        released with "nothing left" and re-offered with "one left" on the
        next action. Since 2026-10-13 that population is also the
        SIGNIFICANT one (:attr:`LoopSettings.explore_min_gain_m2`): a
        sliver the visit would not walk to does not keep the room open
        either.
        """
        p = self.policy
        fact = p.graph.facts.get(pid) if getattr(p.graph, "facts", None) else None
        if fact is None:
            return None
        if self.settings.exploring and obs is not None and world is not None:
            inventory = getattr(p.graph, "frontier_inventory", None)
            room = p.graph.registry.rooms.get(pid)
            if inventory is not None and room is not None:
                # The visit's own goals: admissible AND significant (``explore_min_gain_m2``), so a
                # room whose only frontier is a sliver reads as finished, as the visit would find it.
                return len(self._explore_goals(obs, world, None, room))
            if inventory is not None:
                goals = inventory.by_room.get(int(pid) + 1, ())
                return len(p.sweep.admissible(obs, world, list(goals)))
        return int(fact.frontier_clusters)

    def _home_objects(self, pid):
        """The confirmed home objects of the target in room ``pid`` (``room_priors.HOME_OBJECTS``), sorted."""
        p = self.policy
        return sorted({c for c in p.graph.objects_in(pid) if home_object(p.target, c)})

    def _ruled_out(self, obs, pid, exits=0):
        """``type:<label>`` when the type prior rules room ``pid`` out, else None -- logging what kept it a node.

        A weak label that would have ruled the room out is kept
        (``weak_type_kept``, once per room); so is a home object of the
        target in a room of another type (``home_object_kept``).
        """
        p = self.policy
        info = p.graph.label_info(pid) or {}
        label = info.get("label")
        objects = p.graph.objects_in(pid)
        if not implausible_room(p.target, label) or any(p.target.accepts(c) for c in objects):
            return None
        why = ruled_out(p.target, label, info.get("strength"), objects)
        if why is not None:
            return why
        if pid not in self._weak_kept:
            self._weak_kept.add(pid)
            homes = sorted({c for c in objects if home_object(p.target, c)})
            if homes:
                self.stats["home_object_kept"] += 1
                self._log(obs, "home_object_kept", room=pid, label=label, objects=objects, home_objects=homes)
            else:
                self.stats["weak_type_kept"] += 1
                self._log(obs, "weak_type_kept", room=pid, label=label, openings=exits, objects=objects)
        return None

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
        if (is_stair_node(pid) or is_opening_node(pid) or not p.graph.label_tracker.pending(pid)
                or p.fallback.service_unavailable(ROOM_LLM, obs.step)):
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
        self._needs_reason = True
        return True

    def _relabel_ends_turn(self, obs, pid):
        """Whether a room's new name ends its turn now, or waits for the loop point.

        Under ``sweep`` every changed name ends the turn (the historical
        rule). Under ``scan`` a visit is a few actions of rotation that
        finish the room for good, and a name that merely changes what the
        oracle will think -- a potted plant making a bedroom a "living room"
        -- is not worth abandoning them for: the Ranchester upstairs scan was
        cut at 180 degrees by exactly that, and the half-turn was wasted. The
        one name that ends the turn at once is one that rules the room out
        for the target (:mod:`room_priors`): a toilet in the room being
        scanned for a sofa means leave now, not after five more turns. The
        new name is valued at the loop point either way (``_needs_reason``).
        """
        p = self.policy
        where = "transit" if p.supervisor.state == TRANSIT else "search"
        if self.settings.finishing:
            label = (p.graph.label_info(pid) or {}).get("label")
            if not self.settings.type_prior or self._ruled_out(obs, pid) is None:
                # LLM-first: no name ends a scan or a frontier walk; the model values the new
                # fact at the loop point. With the type prior on, only a ruling-out name does.
                self.stats["relabels_kept_visit"] += 1
                self._log(obs, "relabel_kept_visit", room=pid, label=label, where=where)
                return False
        self.stats["reclassified_releases"] += 1
        if where == "transit":
            self.stats["reclassified_in_transit"] += 1
        return True

    def _options(self, obs, world, cost, stairs=(), openings=()):
        """Every node with its entry point; a staircase's is the foot of the flight, an opening's its threshold.

        A room's entry is its vantage point under ``scan`` -- the interior
        cell of greatest clearance the map can reach, else the centroid;
        under ``explore`` the frontier of the room that reveals the most
        unknown floor per metre walked (:meth:`_explore_goals`), else the
        vantage point for a room with no frontier left to confirm on foot
        (and for a second-pass revisit, which is a scan from a new spot);
        and under ``sweep`` its nearest admissible frontier, else the
        centroid. Admissible by the sweep's own rule -- not under the
        agent, not beside a retired goal -- so a room the visit would find
        nothing left in is entered at its vantage point or centroid, never
        sent back to a frontier the visit will refuse on arrival. Rooms in
        :attr:`_excluded` are not offered at all.
        """
        p = self.policy
        by_room = {}
        if not self.settings.finishing and self.settings.entry_frontier and p.graph.labels is not None:
            inventory = p.graph.frontier_inventory
            by_room = (inventory.by_room if inventory is not None else
                       frontier_goals_by_room(world, cost, p.graph.labels, (obs.pose.x, obs.pose.y),
                                              obs.pose.yaw, p.sweep.settings.ranking))
        self._entries, options = {}, []
        revisits = set()
        for option in p.graph.options:
            if option.room_id in self._excluded:
                continue
            xy, from_frontier = option.xy, False
            revisit = option.room_id in self._finished
            room = p.graph.registry.rooms.get(option.room_id)
            if self.settings.exploring and room is not None:
                xy, from_frontier = self._explore_entry(obs, world, cost, room, revisit=revisit)
            elif self.settings.nearest_frontier_scan and room is not None and not revisit:
                xy, from_frontier = self._nearest_frontier_entry(obs, world, cost, room)
            elif self.settings.scanning:
                found = self._vantage(world, cost, room, revisit=revisit) if room is not None else None
                if found is not None:
                    xy = found[0]
            else:
                goals = p.sweep.admissible(obs, world, by_room.get(option.room_id + 1, ()))
                if goals:
                    nearest = min(goals, key=lambda g: g.geodesic_m)
                    xy, from_frontier = nearest.xy, True
            self._entries[option.room_id] = (xy, from_frontier)
            if revisit:
                revisits.add(option.room_id)
            # The probability the supervisor filters on is the oracle's latest, whatever
            # the option carried: the two are one number, read from one place.
            options.append(replace(option, xy=xy, prob=p.graph.probs.get(option.room_id, option.prob)))
        self.stats["revisits_offered"] += len(revisits - self._revisits)
        self._revisits = revisits
        for stair in stairs:
            self._entries[stair.node_id] = (stair.approach.xy, False)
            options.append(stair.option(p.graph.stair_probs.get(stair.node_id, 0.0)))
        for opening in openings:
            self._entries[opening.node_id] = (opening.opening.xy, False)
            prob = (self.settings.openings.landmark_prob if opening.opening.kind == LANDMARK
                    else self._exit_prob(opening.node_id)[0])
            options.append(opening.option(prob))
        return self._scaled_values(options, openings, here=p.graph.room_at(world, (obs.pose.x, obs.pose.y)),
                                   obs=obs, world=world)

    def _vantage(self, world, cost, room, revisit=False):
        """Where to stand in ``room`` to see it: ``((x, y), clearance_m)`` or None.

        The interior cell of greatest clearance among those the observed
        passable map reaches from the agent (the frontier inventory's
        geodesic field when it is current, else every finite-cost cell).
        For a second-pass ``revisit`` the room's earlier scan points are
        avoided by :attr:`LoopSettings.revisit_standoff_m`, so the second
        look is taken from a different spot.
        """
        p = self.policy
        inventory = p.graph.frontier_inventory
        reachable = (np.isfinite(inventory.distance_m) if inventory is not None
                     and getattr(inventory, "distance_m", None) is not None
                     and inventory.distance_m.shape == room.mask.shape else np.isfinite(cost))
        avoid, standoff = (), 0.0
        ledger = getattr(p, "scans", None)
        if revisit and ledger is not None and self.settings.revisit_standoff_m > 0.0:
            avoid, standoff = ledger.scan_points_in(world, room), self.settings.revisit_standoff_m
        found = vantage_point(world, room.mask, reachable, self.settings.vantage_min_clearance_m,
                              avoid=avoid, avoid_radius_m=standoff)
        if found is None:
            found = vantage_point(world, room.mask, None, self.settings.vantage_min_clearance_m,
                                  avoid=avoid, avoid_radius_m=standoff)
        return found

    # -- the explore visit's goals: the room's frontiers by what a look from each would reveal ----
    def _explore_goals(self, obs, world, cost, room):
        """The room's admissible frontier goals, best first by unknown revealed per metre walked.

        The goals are the frontier inventory's clusters credited to the
        room (the population the oracle's ``frontier=`` counts), through
        the sweep's admissibility filter, re-scored by
        :func:`~frontier_ranking.rank_by_view_gain`: the unknown floor the
        camera's full circle would reach from the goal's cell -- rays
        through the unknown, blocked by walls and by what the sight ledger
        has settled -- over the geodesic distance. "The frontier that
        reveals the most unknown area" (the user's rule of 2026-10-13), not
        the longest boundary and not the nearest cell.
        """
        p = self.policy
        inventory = p.graph.frontier_inventory
        if inventory is None:
            if p.graph.labels is None or cost is None:
                return []
            inventory = accessible_frontiers(world, cost, p.graph.labels, (obs.pose.x, obs.pose.y), obs.pose.yaw,
                                             p.sweep.settings.ranking, frontier_world=p.graph.frontier_world(world))
        goals = p.sweep.admissible(obs, world, list(inventory.by_room.get(int(room.id) + 1, ())))
        if not goals:
            return []
        view = self._unknown_view(obs, world)
        # Significant frontiers only: at least ``explore_min_gain_m2`` of unknown a look from the
        # goal would reveal. A long boundary onto a strip is not significant; a short gap onto a room is.
        min_gain = int(round(self.settings.explore_min_gain_m2 / view.cell_area_m2))
        return rank_by_view_gain(goals, view, p.sweep.settings.ranking, min_gain_cells=min_gain)

    def _unknown_view(self, obs, world):
        """The ray caster over this action's map, built once per action for every goal and opening read on it."""
        cached = self._view_cache
        if cached is not None and cached[0] == int(obs.step) and cached[1] is world:
            return cached[2]
        p = self.policy
        camera = p.episode.camera
        k = camera.intrinsics
        sight = getattr(p, "sight", None)
        view = UnknownView(world, cone_from_camera(k.width, k.fx, float(camera.max_depth_m), p.glance_settings.blind_m
                                                   if hasattr(p, "glance_settings") else 0.0),
                           resolved=sight.resolved(world) if sight is not None else None)
        self._view_cache = (int(obs.step), world, view)
        return view

    def _explore_entry(self, obs, world, cost, room, revisit=False):
        """Where a room is entered under ``explore``: ``((x, y), from_frontier)``.

        The best frontier by view gain; a room with none left is entered at
        its DOORSTEP -- its passable cell nearest the agent along the floor
        (:meth:`_nearest_entry`), since 2026-10-14 -- else its vantage
        point, else its centroid. A second-pass revisit, whose visit is a
        scan from a new spot, is entered at its vantage point.
        """
        if not revisit:
            goals = self._explore_goals(obs, world, cost, room)
            if goals:
                return (tuple(float(v) for v in goals[0].xy), True)
            doorstep = self._nearest_entry(obs, world, cost, room)
            if doorstep is not None:
                return (doorstep, False)
        found = self._vantage(world, cost, room, revisit=revisit)
        return ((found[0] if found is not None else tuple(room.centroid)), False)

    def _nearest_frontier_entry(self, obs, world, cost, room):
        """``((x, y), from_frontier)``: the room's nearest admissible frontier, else its doorstep, else its centroid."""
        p = self.policy
        goal = p.sweep.nearest(obs, world, cost, room.mask, labels=p.graph.labels, label=room.id + 1)
        if goal is not None:
            return tuple(float(v) for v in goal.xy), True
        doorstep = self._nearest_entry(obs, world, cost, room)
        return (doorstep, False) if doorstep is not None else (tuple(room.centroid), False)

    def _nearest_entry(self, obs, world, cost, room):
        """The room's passable cell nearest the agent along the floor, or None: where a room with no frontier left is entered.

        Walking to a room's vantage point -- its interior cell of greatest
        clearance, the middle of the room -- to confirm a room whose
        frontiers are all walked costs the walk across it and shows
        nothing its doorway did not (the user's rule of 2026-10-14: no walk
        to the centre of a room without a reason; the explore visit that
        follows finds no goal and the ledger finishes the room as
        ``explored``, stood in). The frontier inventory's geodesic field
        says which cell is nearest along the floor; without one the
        Euclidean nearest passable cell stands in.
        """
        p = self.policy
        mask = room.mask & np.isfinite(cost) if cost is not None else room.mask
        if not mask.any():
            return None
        ys, xs = np.nonzero(mask)
        inventory = p.graph.frontier_inventory
        if (inventory is not None and getattr(inventory, "distance_m", None) is not None
                and inventory.distance_m.shape == mask.shape):
            dist = inventory.distance_m[ys, xs]
            finite = np.isfinite(dist)
            if finite.any():
                i = int(np.argmin(np.where(finite, dist, np.inf)))
                return tuple(float(v) for v in world.grid_to_world(int(xs[i]), int(ys[i])))
            return None
        here = np.array([obs.pose.x, obs.pose.y])
        points = np.array([world.grid_to_world(int(x), int(y)) for x, y in zip(xs, ys)], dtype=float)
        i = int(np.argmin(np.hypot(points[:, 0] - here[0], points[:, 1] - here[1])))
        return (float(points[i, 0]), float(points[i, 1]))

    def _record(self, instance, dropped, options):
        """Step 3's record, with the distance step 4 charges each node -- a staircase's includes the climb.

        Rooms withheld as finished or ruled out by type are recorded too,
        with ``entry="excluded"`` and the reason, so the HUD and the
        recording say why a room with frontier left was never entered.
        """
        p = self.policy
        labels = {o.room_id: o.label for o in options}
        cruise = p.episode.action_spec.forward_step_m / p.settings.action_time_s
        service = self.settings.service_steps() * p.settings.action_time_s
        peek = self.settings.openings.service_steps * p.settings.action_time_s
        estimates = {}
        if instance is not None:
            for i, pid in enumerate(instance.index_to_pid):
                if pid < 0:
                    continue
                leg = float(instance.C[instance.depot, i])
                if instance.units == "seconds":
                    if pid in self._openings:
                        charged = peek
                    elif self.settings.exploring and pid in self._service_last:
                        charged = self._service_last[pid]
                    else:
                        charged = service
                    leg = max(0.0, leg - charged) * cruise
                estimates[pid] = self._estimate(pid, labels, distance_m=leg)
        for pid in dropped:
            estimates[pid] = self._estimate(pid, labels, distance_m=None)
        for pid, why in self._excluded.items():
            estimates[pid] = dict(self._estimate(pid, labels, distance_m=None), entry="excluded", excluded=why)
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
        opening = self._openings.get(pid)
        if opening is not None:
            o = opening.opening
            prob, source = ((float(self.settings.openings.landmark_prob), "fixed") if o.kind == LANDMARK
                            else self._exit_prob(pid))
            return {"kind": "landmark" if o.kind == LANDMARK else "opening", "label": opening.node.label,
                    "prob": prob, "prob_source": source,
                    "why": "confirmed %s on the map: look at it from close" % (o.glimpsed[0] if o.glimpsed else "landmark")
                    if o.kind == LANDMARK else (why or ("not yet shown to the model" if source == "provisional" else why)),
                    "via": opening.node.via, "room_pid": o.room_pid, "door_id": o.door_id,
                    "glimpsed": list(o.glimpsed), "size_cells": int(o.size_cells), "landmark_id": o.landmark_id,
                    "landmark_xy": None if o.landmark_xy is None else [round(float(v), 2) for v in o.landmark_xy],
                    "heading_deg": round(math.degrees(o.heading), 1), "xy": [round(float(v), 2) for v in o.xy],
                    "gain_m2": round(float(o.gain_m2), 2), "factor": self._value_factors.get(pid, 1.0),
                    "distance_m": distance_m, "entry": "unreachable" if distance_m is None else "peek"}
        fact = p.graph.facts.get(pid)
        entry = self._entries.get(pid)
        info = p.graph.label_info(pid) or {}
        if distance_m is None:
            how = "unreachable"
        elif self.settings.scanning and not (self.settings.nearest_frontier_scan and pid not in self._revisits):
            how = "vantage"
        elif entry is not None and entry[1]:
            how = "frontier"
        elif self.settings.exploring:
            how = "vantage" if pid in self._revisits else "doorstep"
        else:
            how = "centroid"
        return {"kind": "room", "label": labels.get(pid) or info.get("label", "?"), "strength": info.get("strength"),
                "prob": float(p.graph.probs.get(pid, 0.0)), "why": why, "factor": self._value_factors.get(pid, 1.0),
                "objects": p.graph.objects_in(pid),
                "frontier_clusters": None if fact is None else int(fact.frontier_clusters),
                "searched_s": None if fact is None else float(fact.time_in_room_s),
                "last_inside_step": p.graph.last_inside.get(pid),
                "distance_m": distance_m,
                "entry": how}

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
        end until the route memory's no-progress clock released the room. A
        plan the planner REFUSES is re-aimed the same way -- the room's next
        frontiers, its vantage point, its centroid -- before the room is
        released unreachable (:meth:`_reaim_after_refusal`, since 2026-10-13).

        A room re-identified on the way in -- a toilet seen through the
        doorway of what a sink had made a kitchen -- ends its turn before it
        is entered, so the order is re-solved over the new fact.

        A STAIRCASE is handed to the building coordinator, which approaches
        the foot of the flight and climbs; see :meth:`_transit_stairs`. An
        OPENING is walked to its threshold; see :meth:`_transit_opening`.
        """
        p = self.policy
        if is_stair_node(state.room_id):
            return self._transit_stairs(obs, world, state)
        if is_opening_node(state.room_id):
            return self._transit_opening(obs, world, state)
        room = p.graph.registry.rooms.get(state.room_id)
        if room is None:
            # The watershed re-segments every action and a room can be gone for one of them and
            # back the next (Collierville/000023, 2026-10-14, action 170: room 4 missing for one
            # action, the transit released ``unreachable`` and the order re-solved without it).
            # The committed entry point is still a point on the floor: the walk goes on for
            # ``room_missing_grace`` actions before the room is given up.
            self._room_missing = self._room_missing + 1 if self._room_missing_pid == state.room_id else 1
            self._room_missing_pid = state.room_id
            entry = self._entry if self._entry is not None else (state.goal_xy, False)
            if self._room_missing <= self.settings.room_missing_grace and entry[0] is not None:
                command = p._navigate(obs, world, tuple(entry[0]), "transit/%s" % state.room_id)
                if command is not None:
                    if self._room_missing == 1:
                        self.stats["room_missing_held"] += 1
                        self._log(obs, "room_missing_held", room=state.room_id, grace=self.settings.room_missing_grace)
                    return command, {}
            return None, {"route_failed": True}
        self._room_missing, self._room_missing_pid = 0, None
        if self._reclassify(obs, state.room_id, "transit") and self._relabel_ends_turn(obs, state.room_id):
            return None, {"room_reclassified": True}
        if self._finished_on_the_way(obs, world, state.room_id, room):
            return None, {"frontier_exhausted": True}
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
            self.stats["entry_lost"] += 1
            if self.settings.exploring:
                # The entry frontier moved out from under the route (resolved, or the mask
                # re-segmented): re-aim once at the room's best frontier now, else its doorstep,
                # else its vantage point, else the centroid.
                revisit = state.room_id in self._revisits
                candidates = []
                if not revisit:
                    candidates = [(tuple(float(v) for v in g.xy), True) for g in self._explore_goals(obs, world, cost, room)[:1]]
                    doorstep = self._nearest_entry(obs, world, cost, room)
                    if doorstep is not None:
                        candidates.append((doorstep, False))
                found = self._vantage(world, cost, room, revisit=revisit)
                candidates += [(c, False) for c in ([found[0]] if found is not None else []) + [tuple(room.centroid)]]
                candidates = [c for c in candidates if math.dist(c[0], xy_goal) > world.resolution]
                self._log(obs, "entry_lost", room=state.room_id, goal=[round(v, 2) for v in xy_goal],
                          reaimed=bool(candidates))
                if not candidates:
                    return None, {"route_failed": True}
                self._entry = candidates[0]
                self.stats["entry_reaimed"] += 1
                xy_goal, from_frontier = self._entry
            elif self.settings.scanning:
                # The vantage point moved out from under the route (the mask re-segmented):
                # re-aim at where it is now, once; the centroid is the last resort.
                found = self._vantage(world, cost, room, revisit=state.room_id in self._revisits)
                aims = [found[0]] if found is not None else []
                if self.settings.nearest_frontier_scan and state.room_id not in self._revisits:
                    aims.insert(0, self._nearest_frontier_entry(obs, world, cost, room)[0])
                candidates = [c for c in aims + [tuple(room.centroid)] if math.dist(c, xy_goal) > world.resolution]
                self._log(obs, "entry_lost", room=state.room_id, goal=[round(v, 2) for v in xy_goal],
                          reaimed=bool(candidates))
                if not candidates:
                    return None, {"route_failed": True}
                self._entry = (candidates[0], False)
                self.stats["entry_reaimed"] += 1
                xy_goal, from_frontier = self._entry
            else:
                nearest = p.sweep.nearest(obs, world, cost, room.mask, labels=p.graph.labels, label=room.id + 1)
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
        if command is None and p.route_memory.reason != "no_progress":
            # The planner's refusal, not the follower's: a route the agent made no
            # progress on releases the room as before (the room may be re-chosen).
            command = self._reaim_after_refusal(obs, world, cost, state, room, goal)
        if command is not None:
            return command, {}
        self.stats["plan_failures"] += 1
        return None, {"route_failed": True}

    def _reaim_after_refusal(self, obs, world, cost, state, room, failed):
        """The transit command to the room's next entry after the planner refused ``failed``, or None when every entry is refused.

        One refused plan is not an unreachable room. The committed entry is a
        frontier cell the map re-snaps under as the agent closes in -- its
        jamb mapped, the cell itself swallowed by the inflation -- while the
        room's other frontiers, its vantage point and its centroid have not
        been asked about. Darden/000034 (``4ae716bacfab``, 2026-10-13): the
        bathroom was released ``unreachable`` twice on one refused plan each,
        the second time 2.4 m from it after a 65-action walk, and the toilet
        stood 1.3 m away at the step limit. The candidates, in order: under
        ``explore`` the room's next three frontiers by view gain (not for a
        second-pass revisit, which is finished already), then the vantage
        point, then the centroid; the first the planner accepts becomes the
        entry in force (``route_reaimed``). Not for the route memory's
        ``no_progress`` refusal: a route the follower is stuck on releases
        the room as before.
        """
        p = self.policy
        revisit = state.room_id in self._revisits
        candidates = []
        if self.settings.exploring and not revisit:
            candidates += [(tuple(float(v) for v in g.xy), True) for g in self._explore_goals(obs, world, cost, room)[:3]]
            doorstep = self._nearest_entry(obs, world, cost, room)
            if doorstep is not None:
                candidates.append((doorstep, False))
        found = self._vantage(world, cost, room, revisit=revisit)
        candidates += [(c, False) for c in ([found[0]] if found is not None else []) + [tuple(room.centroid)]]
        tried = set()
        for xy, from_frontier in candidates:
            key = world.world_to_grid(*xy)
            if key in tried or math.dist(xy, failed) <= world.resolution:
                continue
            tried.add(key)
            command = p._navigate(obs, world, xy, "transit/%s" % state.room_id)
            if command is None:
                continue
            self._entry = (xy, from_frontier)
            self.stats["route_reaimed"] += 1
            self._log(obs, "route_reaimed", room=state.room_id, failed=[round(v, 2) for v in failed],
                      goal=[round(v, 2) for v in xy], frontier=from_frontier)
            return command
        return None

    def _finished_on_the_way(self, obs, world, pid, room):
        """Whether the room in transit was finished by what the walk showed: nothing left in it to look at.

        Under ``scan`` and ``explore``. The ledger's ``seen_through``,
        ``fragment`` and ``explored`` verdicts need the room's live frontier
        count, read from the scene graph's facts (refreshed every action);
        a room finished so is released ``exhausted`` before it is entered,
        and the order is re-solved without it. The scan tests themselves (a
        rotation stood in it, or saw it) are judged here too, so a room
        another room's scan -- or a full glance on the way -- has since seen
        is not walked to either.
        """
        p = self.policy
        ledger = getattr(p, "scans", None)
        if not self.settings.finishing or ledger is None or pid in self._revisits:
            return False       # a second-pass revisit is finished already: that is the point of going
        frontier = self._live_frontier(pid, obs, world)
        doored = any(int(pid) in (door.get("rooms") or ()) for door in getattr(p.graph, "doors", ()) or ())
        how = ledger.status(world, pid, room, frontier=frontier, doored=doored, step=int(obs.step))
        if how is None:
            return False
        self.stats["finished_in_transit"] += 1
        self._log(obs, "finished_in_transit", room=pid, how=how, frontier=frontier)
        return True

    def _entry_goal(self, obs, world, cost, room, xy_goal, from_frontier):
        """The committed entry point while it is worth reaching; else the room's next frontier; else None.

        The next frontier is the best by view gain under ``explore``
        (:meth:`_explore_goals`) and the nearest along the floor under
        ``sweep``.
        """
        if not from_frontier:
            return xy_goal
        gx, gy = world.world_to_grid(*xy_goal)
        if world.in_bounds(gx, gy) and np.isfinite(cost[gy, gx]) and self.policy.sweep.informative(world, gx, gy):
            return xy_goal
        if self.settings.exploring:
            goals = self._explore_goals(obs, world, cost, room)
            nearest = goals[0] if goals else None
        else:
            nearest = self.policy.sweep.nearest(obs, world, cost, room.mask,
                                                labels=self.policy.graph.labels, label=room.id + 1)
        if nearest is None:
            return None
        self._entry = (tuple(float(v) for v in nearest.xy), True)
        self.stats["entry_reaimed"] += 1
        return self._entry[0]

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
        if not building.commit(obs, portal, entry[0]):
            self.stats["stairs_refused"] += 1
            return None, {"route_failed": True}
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
        self._peek = None
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

    # -- step 6 for an opening: the walk to the threshold ---------------------------
    def _transit_opening(self, obs, world, state):
        """The order put an opening first: walk to its threshold, where the peek begins.

        Arrival is the route's end or the agent within the converter's goal
        tolerance of the threshold -- the frontier's nearest passable cell,
        a point the map has reached, so no mask test applies. The walk is
        bounded by :meth:`OpeningSettings.approach_bound` (the distance's
        worth of steps plus turns, at least ``approach_steps``); a walk that
        runs out looks from within ``merge_m`` of the threshold (the look is
        the point, not the cell) and otherwise retires the opening as
        unreachable (``peek_abandoned``). A planner that refuses the
        threshold -- the cell went occupied as the map grew around it -- gets
        one re-aim at the nearest passable cell within ``merge_m`` on the
        agent's side; a second refusal retires the opening: an opening the
        agent cannot get to is not offered again.
        """
        p, s = self.policy, self.settings.openings
        peek = self._peek
        if peek is None or peek["node"] != state.room_id or peek["opening"] is None:
            self.stats["peeks_abandoned"] += 1
            self._log(obs, "peek_abandoned", node=state.room_id, reason="the opening offered is gone")
            return None, {"route_failed": True}
        opening = peek["opening"]
        goal = tuple(peek["goal"] or opening.xy)
        here = (obs.pose.x, obs.pose.y)
        arrival = p.converter_params.goal_tolerance_m + world.resolution
        distance = math.dist(here, goal)
        if near_stair_mask(p, world, opening.xy, s.stair_exclusion_m, heading=opening.heading):
            # A staircase was placed since the opening was offered (Corozal/000076, actions
            # 168-255: the stairs seen at 203, the gap at their foot walked round for 50 more).
            self._retire_peek(obs, opening, "at a staircase placed since")
            return None, {"route_failed": True}
        if distance > s.merge_m:
            # The inventory is re-read every action; an opening the camera has resolved on the
            # way -- gone from it for a few actions running while the threshold is still a
            # walk away -- has been looked into already (within ``merge_m`` the sweep's own
            # filter hides the goal under the agent's feet, so a vanishing there says nothing).
            self._peek_missing = self._peek_missing + 1 if state.room_id not in self._openings else 0
            if self._peek_missing >= s.vanish_actions:
                self.stats["peeks_resolved_en_route"] += 1
                self._log(obs, "peek_resolved", node=state.room_id, xy=[round(v, 2) for v in opening.xy],
                          distance_m=round(distance, 2))
                return None, {"frontier_exhausted": True}
        if distance <= arrival or (p.route_memory.kind == "peek" and p.route_memory.arrived(obs)):
            return None, {"arrived": True}
        if peek["approach_actions"] >= peek["approach_bound"]:
            if math.dist(here, opening.xy) <= s.merge_m:
                peek["from_here"] = True                 # the walk's budget is spent: no step on to the threshold
                self._log(obs, "peek_from_here", node=state.room_id, distance_m=round(distance, 2))
                return None, {"arrived": True}
            self._retire_peek(obs, opening, "approach budget of %d spent %.1f m short" % (peek["approach_bound"], distance))
            return None, {"route_failed": True}
        command = p._navigate(obs, world, goal, "peek")
        if command is not None:
            self._size_approach_to_route(obs, peek)
            return command, {}
        self.stats["plan_failures"] += 1
        if not peek["reaimed"]:
            reaimed = self._reaim_threshold(obs, world, opening, here)
            if reaimed is not None:
                peek["goal"], peek["reaimed"] = reaimed, True
                p.route_memory.clear("peek_reaimed")
                p._route = p._goal = None
                self._log(obs, "peek_reaimed", node=state.room_id, goal=[round(float(v), 2) for v in reaimed])
                command = p._navigate(obs, world, reaimed, "peek")
                if command is not None:
                    return command, {}
        self._retire_peek(obs, opening, "unreachable")
        return None, {"route_failed": True}

    def _size_approach_to_route(self, obs, peek):
        """Raise the peek's approach bound to what the route the planner actually adopted needs.

        The bound was sized at SELECT from the frontier inventory's
        geodesic, which is walked at the body radius; A* plans at the
        preferred clearance and relaxes only when it must, so a threshold
        3.5 m away along the floor can be a 12.5 m route around a squeeze
        (Hanson 2026-10-04, actions 166-196: a 30-action bound spent
        5.8 m short on a route still being followed). Each time the route
        in force changes, the bound is the larger of what it was and what
        the remaining route is worth, charged from the actions already
        spent; the loop logs the extension.
        """
        p = self.policy
        path = p.route_memory.path
        if path is None or p.route_memory.reason != "new_goal_or_invalid_route":
            return
        points = [(float(q.x), float(q.y)) if hasattr(q, "x") else (float(q[0]), float(q[1]))
                  for q in getattr(path, "points", path)]
        if len(points) < 2:
            return
        length = sum(math.dist(points[i], points[i + 1]) for i in range(len(points) - 1))
        needed = peek["approach_actions"] + self.settings.openings.approach_bound(
            length, p.episode.action_spec.forward_step_m)
        if needed > peek["approach_bound"]:
            self.stats["peek_approach_extended"] += 1
            self._log(obs, "peek_approach_extended", node=peek["node"], route_m=round(length, 2),
                      approach_bound=int(needed), was=int(peek["approach_bound"]))
            peek["approach_bound"] = int(needed)

    def _reaim_threshold(self, obs, world, opening, here):
        """The passable cell nearest the threshold within ``merge_m`` of it, on the agent's side; None when none."""
        p, s = self.policy, self.settings.openings
        cost = p.navigation_cost(world)
        gx, gy = world.world_to_grid(*opening.xy)
        radius = max(1, int(round(s.merge_m / world.resolution)))
        best = None
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                cx, cy = gx + dx, gy + dy
                if not world.in_bounds(cx, cy) or not np.isfinite(cost[cy, cx]):
                    continue
                xy = tuple(float(v) for v in world.grid_to_world(cx, cy))
                off = math.dist(xy, opening.xy)
                if off < world.resolution or off > s.merge_m:
                    continue
                if math.dist(xy, here) >= math.dist(opening.xy, here):
                    continue                                        # beyond the threshold is the unknown's side
                if best is None or off < best[0]:
                    best = (off, xy)
        return None if best is None else best[1]

    def _retired(self, opening):
        """Whether ``opening`` is already done with for the storey (peeked, abandoned, or inspected for a landmark)."""
        p = self.policy
        if opening.kind == LANDMARK:
            return opening.landmark_id in self._inspected
        return p.openings.peeked(p.mapping.floor_id, opening.xy)

    def _retire_peek(self, obs, opening, why):
        """The opening is done with for the storey: looked into, or not worth another try.

        An exit is retired at its threshold in the building-wide registry; a
        target landmark is marked inspected by id on this storey (its
        standoff point moves with where the agent approached from).
        """
        p = self.policy
        # The inventory read this action is stale the moment an opening is retired: the
        # SELECT that follows on this same action must not offer it again.
        self._opening_cache = None
        self._openings.pop(opening.node_id, None)
        if opening.kind == LANDMARK:
            self._inspected.add(opening.landmark_id)
            self.stats["landmarks_inspected"] += 1
        elif why == "looked":
            # The look swept the opening's heading plus one turn each side, and the
            # camera's half field of view beyond that: the frontier it reveals through
            # the same door is covered, not a new opening (opening_nodes.OpeningRegistry.peeked).
            k = p.episode.camera.intrinsics
            half_fov = math.atan(0.5 * float(k.width) / float(k.fx))
            cone = self.settings.openings.look_turns * p.episode.action_spec.turn_angle_rad + half_fov
            p.openings.mark_peeked(p.mapping.floor_id, opening.xy, obs.step, why, heading=float(opening.heading),
                                   cone_rad=float(cone), looked_m=float(self.settings.openings.glimpse_m))
        else:
            p.openings.mark_peeked(p.mapping.floor_id, opening.xy, obs.step, why)
        if why != "looked":
            self.stats["peeks_abandoned"] += 1
            self._log(obs, "peek_abandoned", node=opening.node_id, reason=why, kind=opening.kind,
                      xy=[round(float(v), 2) for v in opening.xy])

    # -- step 1 for an opening: the peek ------------------------------------------
    def _peek_visit(self, obs, world, state):
        """At the threshold: face the unknown and look one turn to each side, then move on.

        The look is ``2 * look_turns + 1`` headings a turn apart, centred
        on the opening's heading, swept in one direction starting from the
        side nearer the agent's yaw so no frame is a repeat. One in-place
        turn per action, progress read off the measured yaw as the scan's
        is; a heading within half a turn of its target has been looked at
        (perception ran on this frame before the loop did). When the last
        heading is done the opening is retired (``peeks_completed``) and the
        node's turn ends ``exhausted`` -- what the look saw is floor in the
        partition now, and the loop point values the room behind as a room.
        Nothing here records a scan: a peek finishes an opening, not a room.
        """
        p, s = self.policy, self.settings.openings
        peek = self._peek
        if peek is None or peek["node"] != state.room_id or peek["opening"] is None:
            self.stats["peeks_abandoned"] += 1
            self._log(obs, "peek_abandoned", node=state.room_id, reason="no opening in force at arrival")
            return None, {"frontier_exhausted": True}
        opening = peek["opening"]
        closing = getattr(p, "closing", None)
        if (opening.kind == LANDMARK and closing is not None and opening.landmark_xy is not None
                and closing.rejected_near(opening.landmark_xy, p.mapping.floor_id)):
            # The takeover ran on it from here and released it: the look is over.
            self._retire_peek(obs, opening, "rejected by the takeover")
            return None, {"frontier_exhausted": True}
        turn = p.episode.action_spec.turn_angle_rad
        yaw = float(obs.pose.yaw)
        here = (float(obs.pose.x), float(obs.pose.y))
        if not peek["targets"] and peek.get("phase") is None and self._begin_step_through(obs, world, state, peek, opening):
            # The supervisor's arrival is 0.6 m from the goal -- a room centre's tolerance; a peek's
            # look is taken from the threshold itself (Collierville/000023, 2026-10-14: three headings
            # from 0.58 m behind the jamb saw a wedge of the room). Finish the walk first.
            pass
        if peek.get("phase") == "through":
            command = self._step_through(obs, world, state, peek, opening)
            if command is not None:
                return command, {}
            if peek.get("phase") == "through":
                # The walk ran out or the planner refused: the look is taken from here.
                peek["phase"] = "look"
                if peek.get("looked_once"):
                    return self._peek_done(obs, state, peek, opening)
        if not peek["targets"]:
            span = s.look_turns * turn
            leftward = normalize_angle(opening.heading - yaw) >= 0.0
            start = opening.heading - span if leftward else opening.heading + span
            step = turn if leftward else -turn
            peek["targets"] = [float(normalize_angle(start + k * step)) for k in range(2 * s.look_turns + 1)]
            peek["index"] = 0
            peek["look_xy"] = here
            p.route_memory.clear("peek_look")
            p._route = p._goal = None
            self._log(obs, "peek_look", node=state.room_id, kind=opening.kind,
                      heading_deg=round(math.degrees(opening.heading), 1),
                      headings=len(peek["targets"]), approach_actions=peek["approach_actions"])
        while (peek["index"] < len(peek["targets"])
               and abs(normalize_angle(peek["targets"][peek["index"]] - yaw)) <= 0.5 * turn + 1e-6):
            peek["index"] += 1
        if peek["index"] >= len(peek["targets"]):
            peek["looked_once"] = True
            if self._begin_step_through(obs, world, state, peek, opening):
                # The threshold was not passable before this look (its cell bordered unknown) and is now:
                # the look mapped the floor beyond it. Walk on to it and look again.
                command = self._step_through(obs, world, state, peek, opening)
                if command is not None:
                    return command, {}
            return self._peek_done(obs, state, peek, opening)
        error = normalize_angle(peek["targets"][peek["index"]] - yaw)
        peek["turns"] += 1
        return NavigationCommand.hold(
            final_yaw=normalize_angle(yaw + max(-turn, min(turn, error))),
            info={"kind": "peek", "node": state.room_id, "look": peek["index"] + 1, "of": len(peek["targets"]),
                  "heading_deg": round(math.degrees(opening.heading), 1), "through": peek.get("through", 0)}), {}

    def _peek_done(self, obs, state, peek, opening):
        """The look is over: retire the opening for the storey and end the node's turn ``exhausted``."""
        self._retire_peek(obs, opening, "looked")
        self.stats["peeks_completed"] += 1
        self.stats["exhausted_releases"] += 1
        self._log(obs, "peek_complete", node=state.room_id, kind=opening.kind, turns=peek["turns"],
                  actions=int(obs.step) - peek["started"], approach_actions=peek["approach_actions"],
                  glimpsed=list(opening.glimpsed), through=peek.get("through", 0))
        return None, {"frontier_exhausted": True}

    def _begin_step_through(self, obs, world, state, peek, opening):
        """Whether the peek walks on to the threshold before (or after) its look (``OpeningSettings.step_through``).

        Only for an exit (a landmark's look is from its standoff by design),
        once per peek, when the agent stands more than the arrival tolerance
        short of the threshold -- the supervisor's 0.6 m room-centre
        tolerance declared the transit arrived -- and only if the threshold
        cell is passable (before the look when the route merely stopped
        short; after it when the cell bordered unknown the look has since
        mapped). The look from the threshold is the wide one
        (``through_look_turns``).
        """
        p, s = self.policy, self.settings.openings
        if not s.step_through or opening.kind == LANDMARK or peek.get("through", 0) >= 1 or peek.get("from_here"):
            return False
        here = (float(obs.pose.x), float(obs.pose.y))
        arrival = p.converter_params.goal_tolerance_m + world.resolution
        if math.dist(here, opening.xy) <= arrival:
            return False
        cost = p.navigation_cost(world)
        gx, gy = world.world_to_grid(*opening.xy)
        if not world.in_bounds(gx, gy) or not np.isfinite(cost[gy, gx]):
            return False                                        # still frontier: the look did not open it
        peek["phase"], peek["through"], peek["through_actions"] = "through", 1, 0
        peek["through_from"] = here
        peek["targets"], peek["index"] = [], 0
        p.route_memory.clear("peek_through")
        p._route = p._goal = None
        self.stats["peeks_stepped_through"] += 1
        self._log(obs, "peek_step_through", node=state.room_id, distance_m=round(math.dist(here, opening.xy), 2),
                  xy=[round(float(v), 2) for v in opening.xy])
        return True

    def _step_through(self, obs, world, state, peek, opening):
        """The walk to the threshold, then the second look there; None when the walk is over or refused."""
        p, s = self.policy, self.settings.openings
        here = (float(obs.pose.x), float(obs.pose.y))
        turn = p.episode.action_spec.turn_angle_rad
        arrival = p.converter_params.goal_tolerance_m + world.resolution
        if not peek["targets"]:
            arrived = (math.dist(here, opening.xy) <= arrival
                       or (p.route_memory.kind == "peek_through" and p.route_memory.arrived(obs)))
            if not arrived:
                if peek["through_actions"] >= s.through_steps:
                    return None
                command = p._navigate(obs, world, tuple(opening.xy), "peek_through")
                if command is None:
                    return None
                peek["through_actions"] += 1
                return replace(command, info=dict(command.info, kind="peek", node=state.room_id, through=1,
                                                  phase_note="stepping through to the threshold"))
            # In the doorway: the wide look, to each side of the way the agent came through.
            came = peek.get("through_from") or here
            heading = (math.atan2(opening.xy[1] - came[1], opening.xy[0] - came[0])
                       if math.dist(came, opening.xy) > 1e-6 else float(opening.heading))
            span = s.through_look_turns * turn
            yaw = float(obs.pose.yaw)
            leftward = normalize_angle(heading - yaw) >= 0.0
            start = heading - span if leftward else heading + span
            step = turn if leftward else -turn
            peek["targets"] = [float(normalize_angle(start + k * step)) for k in range(2 * s.through_look_turns + 1)]
            peek["index"] = 0
            peek["phase"] = "look"
            peek["look_xy"] = here
            p.route_memory.clear("peek_look")
            p._route = p._goal = None
            self._log(obs, "peek_look", node=state.room_id, kind=opening.kind, heading_deg=round(math.degrees(heading), 1),
                      headings=len(peek["targets"]), approach_actions=peek["approach_actions"], through=1,
                      walked_m=round(math.dist(came, here), 2))
        return None

    # -- step 1: the visit ----------------------------------------------------
    def _local(self, obs, world, cost, state):
        """The room in force's visit: the frontier walk, a scan (vantage point, full rotation) or the bounded sweep.

        The other half of the termination rule -- the visit's action bound --
        is judged in :meth:`_budget_flags` before the first tick of the
        action. The third exit is the clue rule: a room newly named by a new
        kind of object ends its turn so that the estimate and the order are
        redone over the new fact -- and a room the oracle then values at
        nothing is left for one it values. Under ``explore`` a second-pass
        revisit of a finished room is a scan from a new spot.
        """
        p = self.policy
        if is_opening_node(state.room_id):
            return self._peek_visit(obs, world, state)
        room = p.graph.registry.rooms.get(state.room_id)
        if room is None:
            return None, {"frontier_exhausted": True}
        if self._reclassify(obs, state.room_id, "search") and self._relabel_ends_turn(obs, state.room_id):
            return None, {"room_reclassified": True}
        if self.settings.exploring and state.room_id not in self._revisits:
            return self._explore_visit(obs, world, cost, state, room)
        if self.settings.finishing:
            return self._scan_visit(obs, world, cost, state, room)
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

    def _explore_visit(self, obs, world, cost, state, room):
        """Walk the room's frontiers by view gain, or (``local_explorer="falcon"``) FALCON's viewpoints, until none is left.

        With none left the visit ends ``exhausted`` and the scan ledger's ``explored`` verdict finishes the room.
        The visit is bounded by ``explore_visit_steps``; a full rotation completed in the room since it began ends it at once.
        """
        p = self.policy
        visit = self._explore
        if visit is None or visit["room"] != state.room_id:
            visit = self._explore = dict(room=state.room_id, started=int(obs.step), goals=0, goal=None,
                                         frontier_left=None)
            self.stats["explore_visits"] += 1
            self._log(obs, "explore_begin", room=state.room_id,
                      frontier=self._live_frontier(state.room_id, obs, world))
        if self._finished_by_rotation(obs, world, state.room_id, room, visit["started"]):
            return None, {"frontier_exhausted": True}
        if self.settings.falcon_visit:
            command = p.room_falcon.plan(obs, world, room_scope(world, room.mask))
            if command is not None:
                return command, {}
            p.room_falcon.end()
            return self._end_explore(obs, visit, 0)
        goals = self._explore_goals(obs, world, cost, room)
        visit["frontier_left"] = len(goals)
        if goals:
            planning_world, planning_cost = self._confined(obs, world, cost, room, state.room_id)
            command = p.sweep.explore(obs, world, planning_cost, room.mask, planning_world=planning_world,
                                      ranked=goals)
            if command is None and planning_world is not world:
                command = p.sweep.explore(obs, world, cost, room.mask, ranked=goals)   # the confinement walled it off
            if command is not None:
                goal = tuple(round(float(v), 2) for v in (p._goal or goals[0].xy))
                if goal != visit["goal"]:
                    visit["goal"], visit["goals"] = goal, visit["goals"] + 1
                    self.stats["explore_goals"] += 1
                    self._log(obs, "explore_goal", room=state.room_id, goal=list(goal), frontier_left=len(goals),
                              gain=round(float(goals[0].utility), 3))
                return command, {}
            self.stats["plan_failures"] += 1
        return self._end_explore(obs, visit, len(goals))

    def _end_explore(self, obs, visit, frontier_left):
        """The explore visit is over: count it, log it and release the room ``exhausted``."""
        self.stats["explore_exhausted"] += 1
        self.stats["exhausted_releases"] += 1
        self._log(obs, "explore_complete", room=visit["room"], goals=visit["goals"],
                  actions=int(obs.step) - visit["started"], frontier_left=frontier_left)
        self._explore = None
        return None, {"frontier_exhausted": True}

    def _finished_by_rotation(self, obs, world, pid, room, since_step):
        """Whether a full rotation completed inside ``room`` since ``since_step`` (a glance on the way) has finished it."""
        ledger = getattr(self.policy, "scans", None)
        if ledger is None or not hasattr(ledger, "rotation_since"):
            return False
        record = ledger.rotation_since(world, room, since_step)
        if record is None:
            return False
        self.stats["finished_by_rotation"] += 1
        self.stats["exhausted_releases"] += 1
        self._log(obs, "finished_by_rotation", room=pid, source=record.get("source"), rotation_step=record.get("step"))
        self._scan = None
        self._explore = None
        return True

    def _scan_visit(self, obs, world, cost, state, room):
        """Walk to the room's vantage point and turn a full circle, or (``nearest_frontier``) walk its nearest frontiers.

        An approach that overruns ``scan_approach_steps`` or is refused scans in place when inside the room,
        else releases it ``unreachable``. Either local phase ends by recording the scan and finishing the room.
        """
        p = self.policy
        scan = self._scan
        if scan is None or scan["room"] != state.room_id:
            revisit = state.room_id in self._revisits
            walks_frontiers = self.settings.nearest_frontier_scan and not revisit
            scan = self._scan = dict(room=state.room_id, phase="frontier" if walks_frontiers else "approach",
                                     approach_actions=0, turns=0, swept=0.0, last_yaw=None, vantage=None,
                                     started=int(obs.step), revisit=revisit)
            if revisit:
                self.stats["revisit_scans"] += 1
            self._log(obs, "scan_begin", room=state.room_id, revisit=revisit, phase=scan["phase"])
        inside = self._inside(obs, world, room)
        if scan["phase"] == "approach":
            if not scan.get("revisit") and self._finished_by_rotation(obs, world, state.room_id, room, scan["started"]):
                return None, {"frontier_exhausted": True}
            found = self._vantage(world, cost, room, revisit=scan.get("revisit", False))
            vantage = found[0] if found is not None else None
            scan["vantage"] = None if vantage is None else [round(v, 2) for v in vantage]
            here = (obs.pose.x, obs.pose.y)
            if vantage is None:
                if inside:
                    return self._begin_rotation(obs, world, scan, "no vantage: scanning in place")
                self.stats["scan_unreachable"] += 1
                return None, {"route_failed": True}
            arrived = (math.dist(here, vantage) <= self.settings.vantage_arrival_m
                       or (p.route_memory.kind == "vantage" and p.route_memory.arrived(obs)))
            if arrived and inside:
                return self._begin_rotation(obs, world, scan, "vantage reached")
            if scan["approach_actions"] >= self.settings.scan_approach_steps:
                if inside:
                    return self._begin_rotation(obs, world, scan, "approach budget spent: scanning in place")
                self.stats["scan_unreachable"] += 1
                self._log(obs, "scan_abandoned", room=state.room_id, reason="approach budget spent outside the room")
                return None, {"route_failed": True}
            planning_world, _ = self._confined(obs, world, cost, room, state.room_id)
            command = p._navigate(obs, planning_world, vantage, "vantage")
            if command is None and planning_world is not world:
                command = p._navigate(obs, world, vantage, "vantage")        # the confinement walled the point off
            if command is not None:
                return command, {}
            self.stats["plan_failures"] += 1
            if inside:
                return self._begin_rotation(obs, world, scan, "vantage unreachable: scanning in place")
            self.stats["scan_unreachable"] += 1
            self._log(obs, "scan_abandoned", room=state.room_id, reason="vantage unreachable from outside the room")
            return None, {"route_failed": True}
        if scan["phase"] == "frontier":
            return self._nearest_frontier_step(obs, world, cost, state, room, scan)
        if p.settings.scan_kind == "pattern":
            return self._pattern_turn(obs, world, scan)
        turn = p.episode.action_spec.turn_angle_rad
        if scan["last_yaw"] is not None:
            scan["swept"] += max(0.0, normalize_angle(obs.pose.yaw - scan["last_yaw"]))
        scan["last_yaw"] = float(obs.pose.yaw)
        full_circle = p.rotation_turns()
        if scan["swept"] >= (full_circle - 0.5) * turn - 1e-9 or scan["turns"] >= full_circle + 2:
            return self._finish_scan(obs, room, scan, "room_scan")
        scan["turns"] += 1
        return NavigationCommand.hold(
            final_yaw=normalize_angle(obs.pose.yaw + turn),
            info={"kind": "room_scan", "room": state.room_id, "scan_turn": scan["turns"], "scan_budget": full_circle,
                  "swept_degrees": round(math.degrees(scan["swept"]), 1)}), {}

    def _nearest_frontier_step(self, obs, world, cost, state, room, scan):
        """Drive the room's nearest frontier the planner accepts; with none left the room is scanned and finished."""
        p = self.policy
        goals = p.sweep.nearest_goals(obs, world, cost, room.mask, labels=p.graph.labels, label=room.id + 1)
        if goals:
            planning_world, planning_cost = self._confined(obs, world, cost, room, state.room_id)
            command = p.sweep.explore(obs, world, planning_cost, room.mask, planning_world=planning_world, ranked=goals)
            if command is None and planning_world is not world:
                command = p.sweep.explore(obs, world, cost, room.mask, ranked=goals)
            if command is not None:
                return command, {}
            self.stats["plan_failures"] += 1
            self._log(obs, "scan_abandoned", room=state.room_id, reason="no route to any of the nearest frontiers")
            self._scan = None
            return None, {"route_failed": True}
        p.scans.record(obs, room, source="room_frontier_walk")
        self.stats["scans_completed"] += 1
        self.stats["exhausted_releases"] += 1
        self._log(obs, "scan_complete", room=state.room_id, mode="nearest_frontier",
                  actions=int(obs.step) - scan["started"], approach_actions=scan["approach_actions"])
        self._scan = None
        return None, {"frontier_exhausted": True}

    def _finish_scan(self, obs, room, scan, source):
        """Record the scan in the ledger, finish the room for the episode and end its turn."""
        self.policy.scans.record(obs, room, source=source)
        self.stats["scans_completed"] += 1
        self.stats["exhausted_releases"] += 1
        self._log(obs, "scan_complete", room=scan["room"], turns=scan["turns"], source=source,
                  swept_degrees=round(math.degrees(scan["swept"]), 1), actions=int(obs.step) - scan["started"],
                  approach_actions=scan["approach_actions"], in_place=scan.get("in_place", False),
                  revisit=scan.get("revisit", False))
        self._scan = None
        return None, {"frontier_exhausted": True}

    def _pattern_turn(self, obs, world, scan):
        """One turn of the gain-ranked look-around; the scan is finished once little unknown floor is left."""
        command = self.policy.pattern_scan.step(obs, world)
        if command is None:
            return self._finish_scan(obs, self.policy.graph.registry.rooms.get(scan["room"]), scan, "pattern_scan")
        scan["turns"] += 1
        return replace(command, info=dict(command.info, kind="room_scan", room=scan["room"])), {}

    def _begin_rotation(self, obs, world, scan, why):
        """Switch the visit in force to its rotation phase and emit its first turn."""
        p = self.policy
        scan["phase"] = "rotate"
        scan["in_place"] = "in place" in why
        scan["last_yaw"] = float(obs.pose.yaw)
        if scan["in_place"]:
            self.stats["scans_in_place"] += 1
        p.route_memory.clear("scan_rotation")
        p._route = p._goal = None
        self._log(obs, "scan_rotate", room=scan["room"], why=why, approach_actions=scan["approach_actions"])
        if p.settings.scan_kind == "pattern":
            p.pattern_scan.begin()
            return self._pattern_turn(obs, world, scan)
        turn = p.episode.action_spec.turn_angle_rad
        scan["turns"] = 1
        return NavigationCommand.hold(
            final_yaw=normalize_angle(obs.pose.yaw + turn),
            info={"kind": "room_scan", "room": scan["room"], "scan_turn": 1,
                  "scan_budget": p.rotation_turns(), "swept_degrees": 0.0}), {}

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

    def peek_state(self):
        """The peek in force for the record and the HUD, or None: the node, its threshold, the walk and the look."""
        if self._peek is None:
            return None
        opening = self._peek["opening"]
        return {"node": self._peek["node"], "kind": None if opening is None else opening.kind,
                "approach_actions": self._peek["approach_actions"],
                "approach_bound": self._peek["approach_bound"], "reaimed": self._peek["reaimed"],
                "started": self._peek["started"], "look": self._peek["index"], "of": len(self._peek["targets"]),
                "turns": self._peek["turns"], "through": self._peek.get("through", 0), "phase": self._peek.get("phase"),
                "xy": None if opening is None else [round(float(v), 2) for v in opening.xy],
                "heading_deg": None if opening is None else round(math.degrees(opening.heading), 1),
                "glimpsed": [] if opening is None else list(opening.glimpsed)}

    def diagnostics(self):
        return {"settings": asdict(self.settings), "stats": dict(self.stats),
                "local_steps": self.local_steps, "visit_budget": self.visit_budget(), "room_id": self.room_id,
                "order": list(self.order), "order_index": self.order_index, "next_room": self.next_room,
                "estimates": {str(pid): dict(item) for pid, item in self.estimates.items()},
                "excluded": {str(pid): why for pid, why in sorted(self._excluded.items())},
                "finished": {str(pid): {"how": how, "step": step} for pid, (how, step) in sorted(self._finished.items())},
                "second_pass": self.second_pass(), "covered": self._covered, "exhausted": self._exhausted,
                "oracle_pass": self.oracle_pass(), "value_factors": {str(k): v for k, v in sorted(self._value_factors.items())},
                "revisits_offered_now": sorted(int(pid) for pid in self._revisits),
                "scan": None if self._scan is None else dict(self._scan),
                "explore": None if self._explore is None else dict(self._explore),
                "peek": self.peek_state(),
                "stairs_offered": sorted(self._stairs), "openings_offered": sorted(self._openings),
                "openings_folded": sorted(self._folded_now),
                "events": list(self.events), "estimate_events": list(self.estimate_events)}



