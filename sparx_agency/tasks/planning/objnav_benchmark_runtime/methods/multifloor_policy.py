"""Building scheduling around floor-local LLM/RPT* search: which stairs, when, up or down.

Two sources of stairs, chosen by ``MultiFloorParams.stair_source``:

* ``ground_truth`` (default) -- the connectors the evaluator read from the
  simulator's navmesh and attached to the episode metadata, **known to the
  policy only once seen**: a perfect detector (:mod:`stair_sightings`) hands
  the coordinator a bounding box for a staircase when, and only when, its
  surface is in the current frame. A staircase never seen is not a portal,
  not a node of the search and never the explanation of a change of height
  -- a raised bathroom floor stays a bathroom floor. A seen staircase is
  placed on the map from that sighting, becomes a portal of the storey it
  touches, and is climbed by :mod:`ground_truth_traversal` along the
  connector's centreline. No look-down inspections, no depth stair
  candidates, no terrain veto on the committed route.
* ``observed`` -- the former RGB-D support-surface discovery
  (:mod:`stair_terrain`, :mod:`stair_traversal`), kept for comparison. It
  started a "stair traversal" on a 13 cm raised bathroom floor, and it still
  decides floor changes by the ``floor_search_actions`` allowance.

The floor atlas, the per-floor contexts and the portal bookkeeping are the
same in both modes.

**In ground-truth mode nothing here decides WHEN to change floors.** Every
SEEN portal on the floor in force is a node of the room-search loop's RPT*
instance (:mod:`stair_nodes`): the LLM values it beside the rooms -- the
storey beyond it against the unknown room down the hall -- and the solver
charges it for the climb. When the order puts a staircase first, the loop
calls :meth:`MultiFloorSearch.commit` and this coordinator approaches the
foot of the flight, starts the traversal there, and tells the loop the node
was taken (:meth:`~room_search_loop.RoomSearchLoop.stairs_taken`), so the
floor's supervisor is left clean for the way back. The old explicit rule in
:mod:`floor_decision` survives as the exploration fallback's last resort --
the room LLM is away, or no node on the floor is worth anything -- and a
floor change it makes is recorded with ``rule: fallback``.
"""
from __future__ import annotations

import math
from sparx_agency.core.planning.exploration.floor_atlas import STAIR_SOURCE_GROUND_TRUTH
from sparx_agency.core.planning.exploration.room_costs import build_instance
from sparx_agency.core.planning.exploration.room_search_policy import RoomCandidate
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.planners.astar.cost_grid_2d import assemble_cost_grid
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.floor_decision import decide_floor_change
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.floor_departure import FloorDepartureGuard
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.stair_ground_truth import GroundTruthStairs
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.stair_sightings import (
    GroundTruthStairDetector, StairSightings)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.stair_terrain import StairTerrain
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.stair_traversal import StairTraversal

#: Who chose the active portal: the loop's RPT* order, or the fallback rule.
SELECTED_BY_LOOP = "rpt_star"
SELECTED_BY_FALLBACK = "floor_decision"


class MultiFloorSearch:
    def __init__(self, policy):
        self.policy, self.params = policy, policy.settings.multifloor
        s = policy.settings
        self.terrain = StairTerrain(self.params, s.map_resolution_m, s.body_radius_m, s.body_height_m, s.depth_stride)
        self.ground_truth = None
        self.detector = None
        #: Every staircase seen so far -- the only stairs the coordinator knows.
        self.sightings = StairSightings()
        if self.params.stair_source == STAIR_SOURCE_GROUND_TRUTH:
            self.ground_truth = GroundTruthStairs.from_metadata(getattr(policy.episode, "metadata", None))
            self.detector = GroundTruthStairDetector(self.ground_truth)
        else:
            # Observed mode confirms a storey by depth-measured flat support; with
            # the storey heights known, the atlas's separation rule is enough.
            policy.mapping.atlas.plateau_observer = self.terrain.plateau_area
        self.portals = []
        self.active = None
        self.entered_step = 0
        self.floor_id = 0
        self.events = []
        self.transition = None
        self.completed = []
        self._observed_step = None
        self.semantic_stair_hint = False
        #: A stair detection seen while a route was committed is kept until the
        #: next action without one, rather than interrupting the route.
        self.stair_hint_pending = False
        self.safety_vetoes = 0
        self.source_trail = []
        self._last_decision = None
        #: Connector id of the stairs the agent last came up or down, and the action it arrived on.
        self.arrived_by = None
        self.arrived_step = None
        self._departure_ignored = False  # one event per off-level excursion, not one per action
        self._approach_failures = 0      # consecutive actions the approach to the active portal failed to plan
        self.departure = FloorDepartureGuard(self)

    @property
    def phase(self):
        return self.transition.phase if self.transition else "APPROACH_STAIRS" if self.active is not None else "SEARCH"

    @property
    def path(self):
        return self.transition.path if self.transition else []

    @property
    def traversing(self):
        if self.ground_truth is not None:
            # A height departure away from every known staircase is a step or a
            # threshold, not a traversal; the atlas alone decides what it was.
            return self.transition is not None
        return self.transition is not None or self.policy.mapping.atlas.in_transition

    @property
    def committed(self):
        return self.active is not None or self.traversing

    def portal_by_id(self, portal_id):
        """The portal record with this id, or None."""
        return next((q for q in self.portals if q["id"] == portal_id), None)

    def can_leave_floor(self, obs=None):
        """Pending unknown rooms block departure; classified or attempted rooms do not."""
        return self.departure.ready(obs)

    # -- the loop's decision --------------------------------------------------
    def commit(self, obs, portal, approach_xy):
        """The room-search loop's order put this staircase first: approach it and climb.

        Idempotent for the portal already active. Clears the committed route
        so the approach is planned afresh, and records who chose the portal
        -- the recording can then tell an RPT* floor change from a fallback
        one, and :meth:`_start` knows to end the node's turn in the loop.
        """
        if not self.can_leave_floor(obs):
            return False
        if self.active is portal and portal.get("selected_by") == SELECTED_BY_LOOP:
            return True
        self.active = portal
        portal["approach_xy"] = list(approach_xy)
        portal["selected_by"] = SELECTED_BY_LOOP
        self._approach_failures = 0
        self.policy.route_memory.clear("stairs chosen by the room-search loop")
        self.events.append({"action": obs.step, "event": "portal_selected", "portal_id": portal["id"],
                            "direction": "up" if portal["direction"] > 0 else "down",
                            "solver_source": SELECTED_BY_LOOP})
        return True

    def prepare_observation(self, obs):
        """Current depth before atlas settlement; never previous-view support.

        Ground truth: the perfect stair detector runs on this frame first, so
        a staircase in view is known before anything this action decides --
        the departure test below, the portals :meth:`observe` lists, the
        nodes the loop offers.
        """
        if obs.step == self._observed_step:
            return
        self._observed_step = obs.step
        if self.detector is not None:
            self._see_stairs(obs)
        atlas = self.policy.mapping.atlas
        height = atlas.elevation_m if atlas.floors else obs.pose.z
        floor_world = self.policy.mapping.worlds.get(self.floor_id)
        self.terrain.update(obs, floor_world, height)
        if self.transition is None and abs(obs.pose.z - height) <= self.params.stable_height_m:
            self._departure_ignored = False
            xyz = (obs.pose.x, obs.pose.y, obs.pose.z)
            if not self.source_trail or math.dist(xyz[:2], self.source_trail[-1][:2]) > 0.04:
                self.source_trail.append(xyz)
                self.source_trail = self.source_trail[-24:]
        if self.transition is None and atlas.floors and abs(obs.pose.z - height) > self.params.stable_height_m:
            if self.ground_truth is not None:
                self._start_unplanned(obs, height)
            else:
                direction = 1 if obs.pose.z > height else -1
                if self.active is None:
                    self.active = self._portal(obs, direction, [(obs.pose.x, obs.pose.y, height)])
                self._start(obs)
        if self.transition is not None:
            self.transition.observe(obs)

    def _see_stairs(self, obs):
        """Run the perfect detector on this frame and record what it saw; the first sighting of a staircase is an event."""
        try:
            sightings = self.detector.detect(obs)
        except Exception as exc:  # a malformed frame must not end the search; the detector is a sensor
            self.events.append({"action": obs.step, "event": "stair_detector_failed", "error": "%s: %s" % (type(exc).__name__, exc)})
            self.sightings.latest = []
            return
        for sighting in self.sightings.observe(sightings):
            connector = self.ground_truth.by_id(sighting.connector_id)
            self.events.append({"action": obs.step, "event": "stairs_seen", "connector_id": sighting.connector_id,
                                "bbox": list(sighting.bbox), "distance_m": round(sighting.distance_m, 2),
                                "visible_points": sighting.visible, "height_m": round(obs.pose.z, 3),
                                "traversable": connector is None or connector.traversable})

    @property
    def last_sightings(self):
        """This frame's ground-truth stair detections, as labelled boxes for the overlay and the trace."""
        return [s.as_detection() for s in self.sightings.latest]

    def _start_unplanned(self, obs, height):
        """Ground truth: a height departure starts a traversal only ON a known staircase.

        Known means seen: the 13 cm bathroom step that the observed mode
        took for a first tread never reaches ``departure_m``; a real flight
        walked onto by the frontier sweep does, lies on a connector's
        polyline, and -- having been in front of the camera on the way -- is
        a seen staircase, so the climb is completed in the direction already
        taken instead of dithering. A departure on stairs never seen is left
        to the atlas: the agent does not assume stairs it has not looked at.
        """
        if abs(obs.pose.z - height) <= self.params.departure_m:
            return
        connector = self.ground_truth.nearest(obs.pose.x, obs.pose.y, height, self.params.floor_match_m,
                                              self.params.near_connector_m, among=self.sightings.seen)
        if connector is None:
            if not self._departure_ignored:
                self._departure_ignored = True
                unseen = self.ground_truth.nearest(obs.pose.x, obs.pose.y, height, self.params.floor_match_m,
                                                   self.params.near_connector_m)
                reason = ("no ground-truth connector within %.2f m" % self.params.near_connector_m if unseen is None
                          else "connector %d underfoot has never been seen" % unseen.id)
                self.events.append({"action": obs.step, "event": "height_departure_ignored",
                                    "height_m": round(obs.pose.z - height, 3), "reason": reason})
            return
        if self.active is None or self.active.get("connector_id") != connector.id:
            self.active = self._portal_for(obs, connector, height)
        self._start(obs)

    def observe(self, obs):
        atlas = self.policy.mapping.atlas
        if atlas.active_id != self.floor_id:
            old = self.floor_id
            self.floor_id, self.entered_step = atlas.active_id, obs.step
            reason = self.transition.completion_reason if self.transition else atlas.completion_reason
            self.events.append({"action": obs.step, "event": "floor_arrival", "source": old,
                                "destination": self.floor_id, "reason": reason})
            if self.active is not None:
                self.active["destination"] = self.floor_id
                self.active["cooldown_until"] = obs.step + self.params.portal_cooldown_actions
                self.active["edge_id"] = atlas.last_connection
                self.active.pop("selected_by", None)
                self.arrived_by, self.arrived_step = self.active.get("connector_id"), obs.step
                if self.arrived_by is not None:
                    # A staircase just walked is known from both ends, whatever the camera saw of it.
                    self.sightings.mark_seen(self.arrived_by, obs.step, self.active.get("path", ()))
            if self.transition:
                self.completed.append(dict(self.transition.diagnostics(), destination=self.floor_id, action=obs.step))
            self.active = None
            self.transition = None
            self.source_trail = []
            self._last_decision = None
            self.policy.route_memory.clear("completed floor transition")
        elif self.transition and self.transition.retreat_path is not None and not atlas.in_transition:
            self._abandon(obs, self.transition.completion_reason)
        if not self.traversing:
            if not atlas.floors or abs(obs.pose.z - atlas.elevation_m) <= self.params.stable_height_m:
                if self.ground_truth is not None:
                    self._discover_ground_truth(obs)
                else:
                    self._discover(obs)
        if atlas.in_transition and self.transition is None:
            if self.ground_truth is not None:
                self._start_unplanned(obs, atlas.elevation_m)
            else:
                direction = 1 if obs.pose.z > atlas.elevation_m else -1
                if self.active is None:
                    self.active = self._portal(obs, direction, [(obs.pose.x, obs.pose.y, atlas.elevation_m)])
                self._start(obs)

    def _portal(self, obs, direction, path):
        portal = {"id": len(self.portals), "floor_id": self.floor_id,
                  "entry": list(path[0]), "direction": direction, "path": list(path),
                  "destination": None, "cooldown_until": 0, "observations": 1, "last_step": obs.step}
        self.portals.append(portal)
        return portal

    def _portal_for(self, obs, connector, height):
        """The portal record of a ground-truth connector as seen from the floor in force.

        Carries where the stairs were SEEN (``seen_step``) and the visible
        surface the sightings accumulated (``footprint``), which is what the
        map draws: the staircase is placed from the sighting, not from a
        file. A connector never seen has no portal, and this is not called.
        """
        for portal in self.portals:
            if portal["floor_id"] == self.floor_id and portal.get("connector_id") == connector.id:
                return portal
        direction, entry, _, destination_z, path = connector.oriented(height, self.params.floor_match_m)
        portal = self._portal(obs, direction, [tuple(p) for p in path])
        record = self.sightings.records.get(connector.id, {})
        far_exit = connector.far_exit(height, self.params.floor_match_m)
        portal.update(entry=list(entry), connector_id=connector.id, destination_z=destination_z,
                      exit_xyz=None if far_exit is None else list(far_exit),
                      observations=2, stair_source="ground_truth",
                      seen_step=record.get("first_step", obs.step), seen_from_m=record.get("nearest_m"),
                      footprint=[list(p) for p in self.sightings.footprint(connector.id)])
        self.events.append({"action": obs.step, "event": "portal_placed", "portal_id": portal["id"],
                            "connector_id": connector.id, "direction": "up" if direction > 0 else "down",
                            "seen_step": portal["seen_step"], "footprint_points": len(portal["footprint"])})
        return portal

    def _discover_ground_truth(self, obs):
        """Every SEEN connector touching this storey is a portal here, once.

        Seen is the whole test: the connector list is the detector's
        instrument, and a staircase the camera has never had in frame is not
        on this floor's list, whatever the navmesh says. The stairs the agent
        just arrived by are a portal like any other: the LLM is told it
        arrived by them and how long ago (``arrived_by`` in the node's facts)
        and values the way back accordingly. No clock keeps the way back off
        the list -- a search that just came up finds the unexplored rooms up
        here worth more, or it was wrong to come.
        """
        atlas = self.policy.mapping.atlas
        height = atlas.elevation_m if atlas.floors else obs.pose.z
        for connector in self.ground_truth.touching(height, self.params.floor_match_m, among=self.sightings.seen):
            known = any(p["floor_id"] == self.floor_id and p.get("connector_id") == connector.id for p in self.portals)
            if not known:
                self._portal_for(obs, connector, height)
        # A staircase seen again from this floor has more of its surface on record: keep the map's footprint current.
        for portal in self.portals:
            cid = portal.get("connector_id")
            if portal["floor_id"] == self.floor_id and cid is not None and cid in self.sightings.records:
                footprint = self.sightings.footprint(cid)
                if len(footprint) != len(portal.get("footprint", ())):
                    portal["footprint"] = [list(p) for p in footprint]

    def _discover(self, obs):
        for candidate in self.terrain.candidates:
            path = candidate["path"]
            near = [i for i, xyz in enumerate(path) if abs(xyz[2] - obs.pose.z) < 0.20]
            entry_index = max(0, (near[-1] if near else 0) - 2)
            path = path[entry_index:]
            match = next((p for p in self.portals if p["floor_id"] == self.floor_id
                          and p["direction"] == candidate["direction"]
                          and math.dist(p["entry"][:2], path[0][:2]) < 1.25), None)
            if match is None:
                self._portal(obs, candidate["direction"], path)
            elif match["last_step"] != obs.step:
                match["observations"] += 1
                match["last_step"] = obs.step
                # Preserve commitment, not stale pre-approach geometry. An
                # executed edge is different: its measured trail is immutable.
                if "edge_id" not in match:
                    match["path"] = path
        for edge in self.policy.mapping.atlas.connections:
            if self.floor_id not in (edge.source, edge.destination):
                continue
            destination, path = edge.oriented(self.floor_id)
            if not any(p.get("edge_id") == edge.id and p["floor_id"] == self.floor_id for p in self.portals):
                portal = self._portal(obs, 1 if path[-1][2] > path[0][2] else -1, path)
                portal.update(destination=destination, edge_id=edge.id, observations=2,
                              cooldown_until=obs.step + self.params.floor_search_actions)

    def plan(self, obs, world, exhausted=False, trigger=None):
        """The building's command for this action, or None when the floor's search owns it.

        Ground truth: a committed portal is approached and climbed; nothing
        else is decided here. ``exhausted`` is the exploration fallback
        asking for a last-resort floor change by the explicit rule, when the
        loop has nothing to offer. Observed mode keeps its allowance-based
        selection and look-down inspections.
        """
        if self.traversing:
            return self.transition.plan(obs)
        if self.active is not None and not self.can_leave_floor(obs):
            # A room may be discovered while approaching a previously allowed
            # staircase. Recheck before taking even the first tread.
            self._abandon(obs, "new room needs a peek before floor departure")
            return None
        if self.active is None:
            if self.ground_truth is not None:
                if exhausted:
                    self._decide(obs, world, trigger or "floor_exhausted")
            elif exhausted or obs.step - self.entered_step >= self.params.floor_search_actions:
                self._select(obs, world)
        if self.active is not None:
            entry = tuple(self.active["entry"][:2])
            here = (obs.pose.x, obs.pose.y)
            kind = "portal/%d" % self.active["id"]
            arrived = self.policy.route_memory.kind == kind and self.policy.route_memory.arrived(obs)
            if math.dist(here, entry) <= 0.65:
                return self.transition.plan(obs) if self._start(obs) else None
            if arrived and self.ground_truth is not None:
                # The approach ended short of the entry. The map has grown on the way here: if it
                # now reaches nearer the entry, go on to that point; the flight is walked from as
                # close to its foot as the observed map allows, not from wherever the first snap fell.
                before = math.dist(here, entry)
                if self._reapproach(obs, world) and math.dist(tuple(self.active["approach_xy"]), entry) < before - world.resolution:
                    command = self.policy._navigate(obs, world, tuple(self.active["approach_xy"]), kind)
                    if command is not None:
                        self.events.append({"action": obs.step, "event": "approach_extended", "portal_id": self.active["id"],
                                            "entry_distance_m": round(before, 2),
                                            "approach_distance_m": round(math.dist(tuple(self.active["approach_xy"]), entry), 2)})
                        return command
                return self.transition.plan(obs) if self._start(obs) else None
            if arrived:
                return self.transition.plan(obs) if self._start(obs) else None
            approach = tuple(self.active.get("approach_xy", entry))
            command = self.policy._navigate(obs, world, approach, kind)
            if command is None and self.ground_truth is not None and self._reapproach(obs, world):
                # The map moved under the approach point: re-snap the entry onto
                # the passable map as it stands now and plan again, once.
                command = self.policy._navigate(obs, world, tuple(self.active["approach_xy"]), kind)
            if command is not None:
                self._approach_failures = 0
                return command
            self._approach_failures += 1
            if self._approach_failures >= self.params.approach_failures or self.ground_truth is None:
                self._approach_failures = 0
                self._abandon(obs, "portal approach unavailable")
        if self.ground_truth is not None:
            return None                          # no look-down sweeps: the stairs are known
        return self._inspect(obs, exhausted)

    def _reapproach(self, obs, world):
        """Re-snap the active connector's approach point onto the observed passable map. True if it landed."""
        p = self.policy
        cost = assemble_cost_grid(p.planner.fields_for(world), p.planner_params, p.settings.body_radius_m)[0]
        cooldown = self.active.get("cooldown_until", 0)
        choice, _ = decide_floor_change([self.active], obs, world, cost, p.mapping.atlas, p.floors.save(), obs.step,
                                        self.params.floor_match_m, self.params.portal_cooldown_actions,
                                        floor_change_cost_m=self.params.floor_change_cost_m)
        self.active["cooldown_until"] = cooldown          # a re-snap is not a new verdict on the connector
        if choice is None:
            return False
        moved = math.dist(tuple(self.active.get("approach_xy", self.active["entry"][:2])), choice.approach_xy) > 1e-6
        self.active["approach_xy"] = list(choice.approach_xy)
        return moved

    def way_back_held(self, step) -> bool:
        """Whether the staircase the agent arrived by is still withheld from the search's choices.

        True for :attr:`MultiFloorParams.arrival_grace_actions` after an
        arrival: the storey has to be looked at before "not here" means
        anything. Read by :func:`stair_nodes.stair_options` (the loop's
        node list) and by the fallback rule alike; an unplanned walk onto
        the stairs is not a choice and is not held.
        """
        return (self.arrived_by is not None and self.arrived_step is not None
                and step - self.arrived_step < self.params.arrival_grace_actions)

    def _decide(self, obs, world, trigger):
        """The fallback rule: an explicit up/down choice, recorded with every candidate's verdict.

        The staircase the agent has just arrived by is not a candidate while
        :meth:`way_back_held`: a floor whose map is two frames old has no
        frontier yet, the rule reads it as exhausted, and the agent stands at
        the very stair head -- Pomaria's first climb was undone two actions
        after it was confirmed.
        """
        if not self.can_leave_floor(obs):
            return
        p = self.policy
        cost = assemble_cost_grid(p.planner.fields_for(world), p.planner_params, p.settings.body_radius_m)[0]
        portals = [q for q in self.portals if q["floor_id"] == self.floor_id and q.get("connector_id") is not None]
        if self.way_back_held(obs.step):
            held = [q for q in portals if q.get("connector_id") == self.arrived_by]
            portals = [q for q in portals if q.get("connector_id") != self.arrived_by]
            if held and not any(e.get("event") == "way_back_held" and e.get("arrived_step") == self.arrived_step
                                for e in self.events):
                self.events.append({"action": obs.step, "event": "way_back_held", "connector_id": self.arrived_by,
                                    "arrived_step": self.arrived_step, "until": self.arrived_step + self.params.arrival_grace_actions,
                                    "reason": "the stairs just climbed are not the fallback's answer on a storey not yet searched"})
        choice, record = decide_floor_change(
            portals, obs, world, cost, p.mapping.atlas, p.floors.save(), obs.step,
            self.params.floor_match_m, self.params.portal_cooldown_actions,
            floor_change_cost_m=self.params.floor_change_cost_m)
        record["trigger"] = trigger
        summary = (record["chosen"], record["reason"], record["trigger"])
        if summary != self._last_decision:          # one record per distinct verdict, not one per action
            self._last_decision = summary
            self.events.append(record)
        if choice is None:
            return
        self.active = choice.portal
        self.active["approach_xy"] = list(choice.approach_xy)
        self.active["selected_by"] = SELECTED_BY_FALLBACK
        self._approach_failures = 0
        p.route_memory.clear("ground-truth floor decision")
        self.events.append({"action": obs.step, "event": "portal_selected", "portal_id": self.active["id"],
                            "direction": record["direction"], "solver_source": SELECTED_BY_FALLBACK,
                            "destination_visited": choice.destination_visited,
                            "distance_m": round(choice.distance_m, 2), "score": round(choice.score, 4),
                            "trigger": record["trigger"]})

    def _inspect(self, obs, exhausted):
        """Start a look-down sweep only at a natural pause, and say what asked for it.

        The three triggers, in precedence: the floor has no frontier left
        (``exhausted`` -- the robot has nothing else to do); a stair detection
        is pending; or the floor allowance is spent and the long periodic
        cadence has elapsed. The last two wait until no route is committed,
        because a sweep mid-route costs its own actions plus the turns to
        recover the heading it leaves behind. The former per-cooldown timer
        interrupted routes ten times in one 472-action recording.
        """
        camera = self.policy.camera_control
        if self.semantic_stair_hint:
            self.stair_hint_pending = True
        idle = self.policy.route_memory.path is None
        allowance_spent = obs.step - self.entered_step >= self.params.floor_search_actions
        if exhausted:
            reason = "floor_exhausted"
        elif idle and self.stair_hint_pending:
            reason = "stair_detection"
        elif idle and allowance_spent and camera.periodic_due(obs.step):
            reason = "periodic"
        else:
            reason = None
        if reason is not None and camera.begin_inspection(obs, self.floor_id):
            self.stair_hint_pending = False
            self.events.append({"action": obs.step, "event": "inspection_started", "reason": reason,
                                "floor_id": self.floor_id})
        return camera.inspection_command(obs)

    def _select(self, obs, world):
        if not self.can_leave_floor(obs):
            return
        p = self.policy
        eligible = [q for q in self.portals if q["floor_id"] == self.floor_id
                    and q["observations"] >= 2 and obs.step >= q["cooldown_until"]]
        if not eligible:
            return
        cost = assemble_cost_grid(p.planner.fields_for(world), p.planner_params, p.settings.body_radius_m)[0]
        goals = {q["id"]: tuple(q["entry"][:2]) for q in eligible}
        states = p.floors.save()
        probabilities = {}
        for q in eligible:
            state = states.get(q["destination"])
            probabilities[q["id"]] = 0.30 if state is None else max(0.02, min(0.30, sum(state["graph"].probs.values()) / (1 + state["_floor_time"] / 60)))
        instance, _ = build_instance(world, cost, goals, probabilities, depot_xy=(obs.pose.x, obs.pose.y),
                                     cruise_speed_mps=p.episode.action_spec.forward_step_m / p.settings.action_time_s)
        if instance is None:
            return
        candidates = [RoomCandidate(room_id=q["id"], label="observed stair portal", prob=probabilities[q["id"]],
                                    prob_renorm=probabilities[q["id"]], xy=goals[q["id"]]) for q in eligible]
        allowed = set(instance.index_to_pid)
        order = p.solver([q for q in candidates if q.room_id in allowed], instance)
        if order:
            self.active = next(q for q in eligible if q["id"] == order[0])
            p.route_memory.clear("RPT selected floor portal")
            self.events.append({"action": obs.step, "event": "portal_selected", "portal_id": self.active["id"],
                                "solver_source": p.solver.last.source, "order": list(order)})

    def _start(self, obs):
        if self.transition is not None:
            return True
        allowed = self.can_leave_floor(obs)
        height = self.policy.mapping.atlas.elevation_m
        if not allowed and abs(obs.pose.z - height) <= self.params.stable_height_m:
            self._abandon(obs, "rooms still need interior peeks")
            return False
        if self.ground_truth is not None:
            from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.ground_truth_traversal import GroundTruthTraversal
            self.transition = GroundTruthTraversal(self, obs)
        else:
            self.transition = StairTraversal(self, obs)
        if not allowed:
            self.departure.recover_unplanned(obs)
        self.policy.route_memory.clear("stair traversal")
        self.events.append({"action": obs.step, "event": "traversal_started", "portal_id": self.active["id"],
                            "direction": self.active["direction"],
                            "stair_source": self.active.get("stair_source", "observed"),
                            "selected_by": self.active.get("selected_by", "unplanned"),
                            "room_coverage_complete": allowed})
        if allowed and self.active.get("selected_by") == SELECTED_BY_LOOP:
            # The climb has begun: the stair node's turn in the room-search loop is
            # over, and the floor's supervisor is left in SELECT for the way back.
            loop = getattr(self.policy, "loop", None)
            if loop is not None:
                loop.stairs_taken(obs, self.active)
        h = self.policy.hierarchy
        if h is not None:
            if h.regions.mask is not None:
                h.routes.suspend(h.regions.mask, h.current, self.policy, h.machine.actions)
            h.machine.end("floor_departure")
            h.machine.transition("TRAVERSE", "observed stair traversal; global allowance only")
        return True

    def _abandon(self, obs, reason):
        if self.transition and not self.transition.arrival_allowed:
            raise RuntimeError("Cannot abandon a committed transition before verified source return")
        self.events.append({"action": obs.step, "event": "portal_deferred", "reason": reason,
                            "portal_id": self.active["id"] if self.active else None})
        if self.active:
            self.active["cooldown_until"] = obs.step + self.params.portal_cooldown_actions
            self.active.pop("selected_by", None)
        if self.transition:
            self.completed.append(dict(self.transition.diagnostics(), action=obs.step))
        self.active = None
        self.transition = None
        self.policy.route_memory.clear("portal deferred")
        if self.policy.hierarchy is not None:
            self.policy.hierarchy.machine.transition("room_reasoning", "stair proposal deferred")

    def blocked(self, obs):
        if self.transition:
            if self.ground_truth is not None:
                self.transition.blocked(obs)
            else:
                self.transition.failures += 1
                self.transition.path = []
                self.transition.surface_goal = None
        self.terrain.blocked(obs.pose, self.policy.episode.action_spec.forward_step_m)

    def filter_action(self, obs, action):
        guarded = self.departure.filter_action(obs, action)
        if guarded != action:
            return guarded
        if not self.traversing:
            h = self.policy.hierarchy
            if h is not None:
                h.motion.update(obs, self.policy.mapping.worlds[self.floor_id])
                if not h.motion.permits(obs, action):
                    return DiscreteAction.TURN_LEFT
            return action
        if self.ground_truth is not None:
            return action                        # the connector is navmesh ground truth; depth cannot veto it
        if action == DiscreteAction.MOVE_FORWARD and not self.terrain.permits(obs.pose, self.policy.episode.action_spec.forward_step_m):
            self.safety_vetoes += 1
            return DiscreteAction.TURN_LEFT
        return action

    def diagnostics(self):
        return {"phase": self.phase, "stair_source": self.params.stair_source,
                "ground_truth": self.ground_truth.diagnostics() if self.ground_truth is not None else None,
                "stairs_seen": self.sightings.diagnostics(),
                "atlas": self.policy.mapping.atlas.diagnostics(),
                "transition": self.transition.diagnostics() if self.transition else None,
                "completed_transitions": list(self.completed), "safety_vetoes": self.safety_vetoes,
                "room_coverage": self.policy.peek.diagnostics()["coverage"],
                "coverage_vetoes": self.departure.vetoes,
                "floor_contexts": self.policy.floors.diagnostics(), "events": list(self.events),
                "portals": [dict({k: v for k, v in q.items() if k not in ("path", "footprint")},
                                 footprint_points=len(q.get("footprint", ()))) for q in self.portals],
                "active_portal": self.active["id"] if self.active else None,
                "selected_by": self.active.get("selected_by") if self.active else None,
                "entered_step": self.entered_step, "arrived_by": self.arrived_by, "arrived_step": self.arrived_step,
                "floor_change_cost_m": self.params.floor_change_cost_m,
                "decides_floor_changes": "room-search loop (RPT* stair nodes, seen stairs only)" if self.ground_truth is not None
                else "allowance (%d actions) + RPT* over portals" % self.params.floor_search_actions}

