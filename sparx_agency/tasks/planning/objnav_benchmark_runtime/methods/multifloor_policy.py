"""Building scheduling around floor-local LLM/RPT* search: which stairs, when, up or down.

Two sources of stairs, chosen by ``MultiFloorParams.stair_source``:

* ``ground_truth`` (default) -- the connectors the evaluator read from the
  simulator's navmesh and attached to the episode metadata. A transition is
  started only at one of them (:mod:`stair_ground_truth`), chosen by the
  explicit rule in :mod:`floor_decision`, and walked by
  :mod:`ground_truth_traversal`. No look-down inspections, no depth stair
  candidates, no terrain veto on the committed route.
* ``observed`` -- the former RGB-D support-surface discovery
  (:mod:`stair_terrain`, :mod:`stair_traversal`), kept for comparison. It
  started a "stair traversal" on a 13 cm raised bathroom floor.

The floor atlas, the per-floor contexts and the portal bookkeeping are the
same in both modes.
"""
from __future__ import annotations

import math
from sparx_agency.core.planning.exploration.floor_atlas import STAIR_SOURCE_GROUND_TRUTH
from sparx_agency.core.planning.exploration.room_costs import build_instance
from sparx_agency.core.planning.exploration.room_search_policy import RoomCandidate
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.planners.astar.cost_grid_2d import assemble_cost_grid
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.floor_decision import decide_floor_change
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.stair_ground_truth import GroundTruthStairs
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.stair_terrain import StairTerrain
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.stair_traversal import StairTraversal


class MultiFloorSearch:
    def __init__(self, policy):
        self.policy, self.params = policy, policy.settings.multifloor
        s = policy.settings
        self.terrain = StairTerrain(self.params, s.map_resolution_m, s.body_radius_m, s.body_height_m, s.depth_stride)
        self.ground_truth = None
        if self.params.stair_source == STAIR_SOURCE_GROUND_TRUTH:
            self.ground_truth = GroundTruthStairs.from_metadata(getattr(policy.episode, "metadata", None))
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
        self._arrived_by = None          # connector id of the stairs the agent last came up or down
        self._departure_ignored = False  # one event per off-level excursion, not one per action

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

    def prepare_observation(self, obs):
        """Current depth before atlas settlement; never previous-view support."""
        if obs.step == self._observed_step:
            return
        self._observed_step = obs.step
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

    def _start_unplanned(self, obs, height):
        """Ground truth: a height departure starts a traversal only ON a known staircase.

        The 13 cm bathroom step that the observed mode took for a first tread
        never reaches ``departure_m``; a real flight walked onto by the
        frontier sweep does, and lies on a connector's polyline, so the climb
        is completed in the direction already taken instead of dithering.
        """
        if abs(obs.pose.z - height) <= self.params.departure_m:
            return
        connector = self.ground_truth.nearest(obs.pose.x, obs.pose.y, height, self.params.floor_match_m,
                                              self.params.near_connector_m)
        if connector is None:
            if not self._departure_ignored:
                self._departure_ignored = True
                self.events.append({"action": obs.step, "event": "height_departure_ignored",
                                    "height_m": round(obs.pose.z - height, 3),
                                    "reason": "no ground-truth connector within %.2f m" % self.params.near_connector_m})
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
                self._arrived_by = self.active.get("connector_id")
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
        """The portal record of a ground-truth connector as seen from the floor in force."""
        for portal in self.portals:
            if portal["floor_id"] == self.floor_id and portal.get("connector_id") == connector.id:
                return portal
        direction, entry, _, destination_z, path = connector.oriented(height, self.params.floor_match_m)
        portal = self._portal(obs, direction, [tuple(p) for p in path])
        portal.update(entry=list(entry), connector_id=connector.id, destination_z=destination_z,
                      observations=2, stair_source="ground_truth")
        return portal

    def _discover_ground_truth(self, obs):
        """Every connector touching this storey is a portal here, once.

        The stairs the agent just arrived by cool for the floor's search
        allowance, so the storey is searched before the way back is eligible.
        """
        atlas = self.policy.mapping.atlas
        height = atlas.elevation_m if atlas.floors else obs.pose.z
        for connector in self.ground_truth.touching(height, self.params.floor_match_m):
            known = any(p["floor_id"] == self.floor_id and p.get("connector_id") == connector.id for p in self.portals)
            if known:
                continue
            portal = self._portal_for(obs, connector, height)
            if connector.id == self._arrived_by:
                portal["cooldown_until"] = obs.step + self.params.floor_search_actions

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

    def plan(self, obs, world, exhausted=False):
        if self.traversing:
            return self.transition.plan(obs)
        if self.active is None and (exhausted or obs.step - self.entered_step >= self.params.floor_search_actions):
            if self.ground_truth is not None:
                self._decide(obs, world, exhausted)
            else:
                self._select(obs, world)
        if self.active is not None:
            entry = tuple(self.active["entry"][:2])
            approach = tuple(self.active.get("approach_xy", entry))
            kind = "portal/%d" % self.active["id"]
            arrived = self.policy.route_memory.kind == kind and self.policy.route_memory.arrived(obs)
            if math.dist((obs.pose.x, obs.pose.y), entry) <= 0.65 or arrived:
                self._start(obs)
                return self.transition.plan(obs)
            command = self.policy._navigate(obs, world, approach, kind)
            if command is not None:
                return command
            self._abandon(obs, "portal approach unavailable")
        if self.ground_truth is not None:
            return None                          # no look-down sweeps: the stairs are known
        return self._inspect(obs, exhausted)

    def _decide(self, obs, world, exhausted):
        """Ground truth: the explicit up/down choice, recorded with every candidate's verdict."""
        p = self.policy
        cost = assemble_cost_grid(p.planner.fields_for(world), p.planner_params, p.settings.body_radius_m)[0]
        portals = [q for q in self.portals if q["floor_id"] == self.floor_id and q.get("connector_id") is not None]
        choice, record = decide_floor_change(
            portals, obs, world, cost, p.mapping.atlas, p.floors.save(), obs.step,
            self.params.floor_match_m, self.params.portal_cooldown_actions)
        record["trigger"] = "floor_exhausted" if exhausted else "allowance_spent"
        summary = (record["chosen"], record["reason"], record["trigger"])
        if summary != self._last_decision:          # one record per distinct verdict, not one per action
            self._last_decision = summary
            self.events.append(record)
        if choice is None:
            return
        self.active = choice.portal
        self.active["approach_xy"] = list(choice.approach_xy)
        p.route_memory.clear("ground-truth floor decision")
        self.events.append({"action": obs.step, "event": "portal_selected", "portal_id": self.active["id"],
                            "direction": record["direction"], "solver_source": "floor_decision",
                            "destination_visited": choice.destination_visited,
                            "distance_m": round(choice.distance_m, 2)})

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
            return
        if self.ground_truth is not None:
            from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.ground_truth_traversal import GroundTruthTraversal
            self.transition = GroundTruthTraversal(self, obs)
        else:
            self.transition = StairTraversal(self, obs)
        self.policy.route_memory.clear("stair traversal")
        self.events.append({"action": obs.step, "event": "traversal_started", "portal_id": self.active["id"],
                            "direction": self.active["direction"],
                            "stair_source": self.active.get("stair_source", "observed")})
        h = self.policy.hierarchy
        if h is not None:
            if h.regions.mask is not None:
                h.routes.suspend(h.regions.mask, h.current, self.policy, h.machine.actions)
            h.machine.end("floor_departure")
            h.machine.transition("TRAVERSE", "observed stair traversal; global allowance only")

    def _abandon(self, obs, reason):
        if self.transition and not self.transition.arrival_allowed:
            raise RuntimeError("Cannot abandon a committed transition before verified source return")
        self.events.append({"action": obs.step, "event": "portal_deferred", "reason": reason,
                            "portal_id": self.active["id"] if self.active else None})
        if self.active:
            self.active["cooldown_until"] = obs.step + self.params.portal_cooldown_actions
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
                "atlas": self.policy.mapping.atlas.diagnostics(),
                "transition": self.transition.diagnostics() if self.transition else None,
                "completed_transitions": list(self.completed), "safety_vetoes": self.safety_vetoes,
                "floor_contexts": self.policy.floors.diagnostics(), "events": list(self.events),
                "portals": [{k: v for k, v in q.items() if k != "path"} for q in self.portals],
                "active_portal": self.active["id"] if self.active else None}

