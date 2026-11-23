"""The original full-vocabulary room/LLM RPT experiment, separate from the visual policy."""
from __future__ import annotations

from dataclasses import asdict
import math
import time

from sparx_agency.core.mapping.objects.target_evidence import TargetEvidenceSettings
from sparx_agency.core.mapping.topology.revisable_room_labels import RoomLabelSettings
from sparx_agency.core.mapping.topology.room_extent import ExtentParams
from sparx_agency.core.mapping.topology.search_node_prompts import prompt_version
from sparx_agency.core.planning.object_search.closing.target_pursuit import TargetPursuit
from sparx_agency.core.planning.object_search.exploration.doorway_peek import DoorwayPeek
from sparx_agency.core.planning.object_search.exploration.exploration_fallback import ExplorationFallback
from sparx_agency.core.planning.object_search.perception.cycle import PerceptionCycle
from sparx_agency.core.planning.object_search.perception.doors import DoorSettings
from sparx_agency.core.planning.object_search.policy import _ObservedSearchPolicy
from sparx_agency.core.planning.object_search.rooms.discovery import SUSPENDED_PHASES, discover
from sparx_agency.core.planning.object_search.rooms.floor_context import FloorContextBank
from sparx_agency.core.planning.object_search.rooms.opening_nodes import OpeningRegistry
from sparx_agency.core.planning.object_search.rooms.room_scans import RoomScanLedger
from sparx_agency.core.planning.object_search.rooms.room_search_loop import LoopSettings
from sparx_agency.core.planning.object_search.rooms.scene_graph import DEFAULT_SEGMENTATION
from sparx_agency.core.planning.object_search.world_model.peek_stairs import stair_peek_mask
from sparx_agency.core.planning.objnav.types.command import NavigationCommand


class _RoomDescription:
    """Frozen description for reproducing historical text-room runs."""

    def configuration(self):
        return {**super().configuration(),
                "doors": asdict(self.door_settings), "target_evidence": asdict(self.target_settings),
                "supervisor": asdict(self.supervisor_params), "room_search_loop": asdict(self.loop_settings),
                "room_labels": asdict(self.room_label_settings), "room_segmentation": asdict(DEFAULT_SEGMENTATION),
                "room_extents": dict(asdict(self.extent_params),
                                     rule="observed map only: every room's boundary faces on the agent's own occupancy "
                                          "grid classified by what lies beyond them (wall, another room of its own "
                                          "partition, settled unknown, unseen unknown) and grouped into four sides by "
                                          "outward normal in the storey's wall frame; a side is seen when at most "
                                          "open_tolerance of its faces open onto unseen space in runs wider than "
                                          "doorway_max_m; with four seen sides no part of the room lies outside the map "
                                          "('all sides mapped' -- not a final size: the partition may still join or "
                                          "re-cut rooms), else the size is a lower bound shown "
                                          "as 'at least'; coverage is the room's share of the observed cells; recomputed "
                                          "from the live map and partition on every update, keyed by cell, never by room "
                                          "id; shown to the node oracle as mapped= -- the observed part of the room's free "
                                          "floor out of that floor, in m2 and as a share: floor the robot cannot stand on "
                                          "(furniture, settled pockets) is not counted (2026-11-17); nothing but that map, its "
                                          "partition, the sight record and the pose enters the reading -- no simulator "
                                          "geometry, no room annotations, no size priors "
                                          "(core/mapping/topology/room_extent.py)"),
                "room_visit": self._room_visit_description(),
                "room_partition": "watershed of observed free space with seen stair footprints excluded; stairs are never a room; "
                                  "a detected door's cut is snapped to the nearest choke within %.1f m; furniture is credited to "
                                  "the nearest room floor; room numbers are unique across the building"
                                  % DEFAULT_SEGMENTATION.door_snap_reach_m,
                "openings": (("every exit of the storey (a doorway or gap at the edge of the mapped floor) is an attribute "
                              "of the room it opens from, never a node with a number of its own (kept on the room's node "
                              "for the record; not on its prompt line since 2026-11-17, where the frontier count stands "
                              "for what is left); the room's own frontier walk looks "
                              "through them, and a finished room with GAPs left to walk is offered again for them "
                              "(room-level GAPs, 2026-10-21)"
                              if self.loop_settings.room_level_gaps else
                              "doorways and gaps at the edge of the mapped floor are nodes beside the rooms "
                              "(DEPRECATED exits_as_nodes%s), visited by a peek (threshold, face the unknown, one look to each "
                              "side) priced at %d actions; valued by the oracle (uncertainty floor %s)"
                              % ("" if self.loop_settings.exploring else ", the %s visit's only way into an exit"
                                 % self.loop_settings.visit,
                                 self.loop_settings.openings.service_steps,
                                 "off" if not self.loop_settings.unexplored_floor else "p=%.2f" % self.loop_settings.unexplored_floor))
                             + ("; a confirmed landmark of the target's class is a node at p=%.2f without an oracle call, "
                                "visited by a peek from %.1f m" % (self.loop_settings.openings.landmark_prob,
                                                                    self.loop_settings.openings.landmark_standoff_m)
                                if self.loop_settings.openings.landmarks_enabled else "")
                             if self.loop_settings.openings.enabled else "none: openings are the exploration fallback's alone"),
                "floor_exit_gate": ("unknown non-stair rooms get at most one peek attempt per episode before a floor change; "
                                    "initially classified, scanned and previously peeked rooms are exempt"
                                    if self.settings.doorway_peek.gate_floor_departure else
                                    "none: a floor change is the building coordinator's to take at any time"),
                "node_oracle": {"nodes": "every room of the floor in force (finished ones shown with "
                                         "their status)" + (
                                    "" if not self.loop_settings.openings.enabled else
                                    "; each room's exits are the room's attributes, not nodes"
                                    if self.loop_settings.room_level_gaps else " + its openings"),
                                "asks": "independent P(target is in the part of the room not seen yet) per node, "
                                        "from the model's own "
                                        "knowledge of standard homes and buildings; the system prompt (the version named "
                                        "under 'prompt') states the task, what each field means, what the search so far "
                                        "means for a room (the more of it is mapped and searched, the less of it is left "
                                        "to hold the target; a room never entered has not been searched) and the reply "
                                        "format "
                                        "-- no room named for any object, no number or band, no list of "
                                        "goal categories, no worked example (2026-11-16; "
                                        "room_priors.py removed 2026-11-15); the user prompt gives the name of the "
                                        "episode's target, the HOUSE line (room types found, rooms unidentified, exits "
                                        "unlooked, the frontier state, actions used of the budget) and one line per room "
                                        "with four things (2026-11-17): its type (the objects seen only while it is "
                                        "unclassified), the floor mapped out of its free floor (m2 and share), the "
                                        "frontiers left, the actions spent inside ('never' for a room not entered); no "
                                        "distance, no neighbours and no word on where the robot stands -- travel is "
                                        "RPT*'s; the reply keeps home/house/stage and a first|second pass verdict before "
                                        "the numbers",
                                # What the code does to the reply before RPT* sees it (search_node_oracle.score), and
                                # then by exploration state (RoomSearchLoop._scaled_values, the settings above).
                                "postprocessing": "format: p or a drifted key, a number as text, a percent sign; ids nobody "
                                                  "listed dropped; scale: percentages, also when written as fractions of "
                                                  "one, clamped to 0-100, never renormalised; meaning: an omitted node "
                                                  "1%, a room never entered never 0 (read as 1%, then the loop's floor "
                                                  "for unvisited rooms), an entered room's 0 stands and keeps it out of "
                                                  "the order (min_prob); no knowledge of any room or object",
                                # Which wording the model is asked under: the version's name, the model it was
                                # tuned on and the digest of its text (``search_node_prompts.py``). The model that
                                # answers is the LLM client's reasoning route, recorded by whoever builds the client
                                # (the benchmark factory: ``model`` here, ``llm`` / ``llm_identity`` beside this record).
                                "prompt": prompt_version(self.loop_settings.oracle_prompt).describe(),
                                "probability_model": "independent_search_success",
                                "flat_p": self.loop_settings.flat_p,
                                "code_floors": {"unexplored_floor": self.loop_settings.unexplored_floor},
                                "second_pass": self.loop_settings.second_pass,
                                "route": "LLM_REASONING_MODEL",
                                "cadence": "loop points and semantic discovery -- a room on or off the offer, a room's type "
                                           "(discovery.semantic_signature, 2026-11-09); geometry alone "
                                           "within reconsider_min_actions of the last call waits for the loop point; unchanged "
                                           "prompts reused"},
                "reasoning_cadence": "room LLM at loop points and semantic discovery events; geometry every %d action(s)"
                                     "%s; synchronous frame processing, one full rotation (warm-up scan) before room "
                                     "selection on every storey first entered"
                                     % (self.settings.graph_period_steps,
                                        (", every %d while a committed transit route is walked (rebuilt on the action "
                                         "the transit ends)" % self.settings.transit_graph_period_steps)
                                        if (self.settings.transit_graph_period_steps > self.settings.graph_period_steps
                                            and self.loop_settings.transit_commitment) else ""),
                "oracle_schema_repairs": 1, "clock": "global action ledger; room clock pauses off-floor"}

    def _room_visit_description(self):
        """The frozen configuration's one-line account of what a room visit is under the loop's ``visit`` mode."""
        loop = self.loop_settings
        if not loop.finishing:
            return "bounded frontier sweep of %d actions" % loop.local_steps
        finishing = ("finished rooms are shown to the oracle with their status and offered again%s once its pass "
                     "verdict says the home is covered (a revisit scans from a spot at least %.1f m from the earlier "
                     "scan points); every unfinished room is valued by the oracle whatever its type (no type "
                     "exclusion); a room's label is strong on two or more kinds of object and a confident model"
                     % ("" if loop.second_pass else " (DISABLED: second_pass off)", loop.revisit_standoff_m))
        if loop.exploring:
            return ("frontier walk: the room's frontier that reveals the most unknown floor per metre walked, then the "
                    "next, until none is left (bounded by %d actions); a room the agent stood in with no live frontier "
                    "is finished (explored), and so is a room a full rotation stood in or saw more than half of, one "
                    "with no live frontier the camera looked into or walked through, or a doorless fragment under "
                    "%.1f m2; every rotation is as many turns as the camera's field of view needs to see the circle%s; "
                    % (loop.explore_visit_steps, loop.fragment_max_m2, self._rotation_note())
                    + ("every exit of the storey is an attribute of the room it opens from and is walked by that "
                       "room's visit, and a finished room with GAPs left to walk is offered again for them "
                       "(room-level GAPs); " if loop.room_level_gaps else
                       "openings off a room that is itself a node are folded into its walk; ")
                    + finishing)
        return ("vantage point + full rotation (as many turns as the camera's field of view needs to see the circle%s); "
                "a scanned room (or one a scan saw more than half of) is finished for the first pass, and so is a room "
                "with no live frontier that the camera looked into or walked through, or that is a doorless fragment "
                "under %.1f m2; " % (self._rotation_note(), loop.fragment_max_m2) + finishing)

    def _rotation_note(self):
        """`` (N)`` once an episode's camera says how many turns that is; empty before the first reset."""
        return " (%d)" % self.rotation_turns() if getattr(self, "episode", None) is not None else ""


class LegacyRoomSearchPolicy(_RoomDescription, _ObservedSearchPolicy):
    """Original room segmentation, full-vocabulary landmarks and text-reasoning experiment."""

    name = "sparx-rpt-llm-host-sweep"
    variant = "rpt_room_llm"
    uses_room_loop = True
    suspended_phases = SUSPENDED_PHASES
    fallback_type = ExplorationFallback
    perception_type = PerceptionCycle

    def __init__(self, detector, llm_client, settings=None, segmenter=None, frame_log=None):
        super().__init__(detector, settings, segmenter, frame_log)
        self.llm_client = llm_client
        self.loop_settings = LoopSettings(**self.settings.room_search)
        self.supervisor_params = self.loop_settings.supervisor_params(self.settings.seed)
        self.door_settings, self.room_label_settings = DoorSettings(), RoomLabelSettings(**self.settings.room_labels)
        self.target_settings, self.extent_params = TargetEvidenceSettings(), ExtentParams()

    def _begin_search(self):
        s = self.loop_settings
        self._semantic_signature = self._resume_room = None
        self._semantic_signature_step = -1
        self.scans = RoomScanLedger(self, s.scan_seen_fraction, fragment_max_m2=s.fragment_max_m2,
                                    tiny_room_m2=s.tiny_room_m2, walkthrough_clearance_m=s.walkthrough_clearance_m,
                                    seen_range_m=s.scan_seen_range_m)
        self.openings = OpeningRegistry(match_m=s.openings.match_m, done_m=s.openings.done_m)
        self.floors = FloorContextBank(self)
        self._reset_floor()
        self.peek = DoorwayPeek(self)
        self.pursuit = TargetPursuit(self)

    def _change_floor(self, observation):
        self.peek.cancel(observation, "floor_changed", restore=False)
        if self.building:
            floor = self.mapping.floor_id
            fresh = floor not in self.floors.contexts and floor != self.floors.active
            self.floors.activate(floor)
            if fresh:
                self.warmup_actions, self._warmup_pending = 0, True
        else:
            self._reset_floor()

    @property
    def model_client(self):
        return self.llm_client

    def _observe_search(self, observation, world):
        pass

    def _explore(self, observation, world):
        return self.loop.plan(observation, world)

    def _reset_floor(self):
        self.floors.new()
        self._route = self._goal = self._target_xy = self._target_id = None
        self.route_memory.clear("floor_reset")

    def _search(self, observation, world, confirmed):
        s, pose = self.settings, observation.pose
        # Progress is a forward step's worth of displacement, not the slide a collision
        # leaves (Leonardo 2026-10-05: wedged for 350 actions, every blocked MOVE_FORWARD
        # shifted the agent 5-10 cm and that reset the clock the BLOCKED verdict waits on).
        if self._last_pose is not None and math.dist((pose.x, pose.y), self._last_pose) > 0.6 * self.episode.action_spec.forward_step_m:
            self._blocked_since = None
        self._last_pose = (pose.x, pose.y)
        self.graph.credit_time(world, pose, 0.0 if observation.step == 0 else s.action_time_s, step=observation.step)
        if self.hierarchy is None and confirmed and self._target_xy is not None and math.dist((pose.x, pose.y), self._target_xy) <= s.stop_distance_m:
            self.peek.cancel(observation, "target_priority")
            return NavigationCommand.stop_here(info={"reason": "fresh multi-view-confirmed target", "target_confirmed": True})
        cost = self._refresh_graph(observation, world)
        if self._target_xy is not None:
            self.peek.cancel(observation, "target_priority")
            self.glances.abort(observation, "target_priority")
            if self.hierarchy is None and observation.step - self._target_step <= self.target_settings.max_unseen_steps:
                approach = self._approach(observation, world)
                if approach is not None:
                    return approach
            elif self.hierarchy is not None:
                return self.hierarchy.plan(observation, world, confirmed)
        if self.glances.active is not None:
            if self.building and self.building.traversing:
                self.glances.abort(observation, "stairs_priority")
            else:
                # The look in force: the loop is not ticked and not charged while it
                # turns; the action it completes on goes to the search below.
                command = self.glances.continue_look(observation, world)
                if command is not None:
                    return command
        if self._target_xy is None:
            pursuit = self.pursuit.command(observation, world)
            if pursuit is not None:
                return pursuit
            discovery = self._discover(observation, world, cost)
            if discovery is not None:
                return discovery
        if self.building and self._target_xy is None:
            command = self.building.plan(observation, world)
            if command is not None:
                return self.glances.apply(observation, world, command)
        if self.hierarchy is not None:
            command = self.hierarchy.plan(observation, world, confirmed)
            if self.building and command.stop and self._target_xy is None and not self.hierarchy.errors:
                return self.building.plan(observation, world, exhausted=True) or command
            return command
        return self.glances.apply(observation, world, self._explore(observation, world))

    def _refresh_graph(self, observation, world, force=False):
        """Refresh geometry even on a stair approach, so new rooms revoke departure.

        The watershed, the registry, the frontier inventory and every
        per-room fact are rebuilt every ``graph_period_steps`` actions --
        every action by default -- EXCEPT while a committed transit route is
        being walked, where they are rebuilt every
        ``transit_graph_period_steps`` (:meth:`graph_period`): the follower
        needs the map, not the partition, and the full rebuild was the
        largest share of every decision's wall time (Corozal/000003,
        ``c1af8954a378``, 2026-10-22: 1.4 s of a 1.7 s median decision,
        500 times). The room-search loop forces a rebuild on the action a
        transit ends (:meth:`~rooms.room_search_loop.RoomSearchLoop._fresh_geometry`),
        and a loop point reasons over geometry refreshed on its own action
        regardless. Between rebuilds the frontier inventory is not refreshed
        either; the committed route does not read it.
        """
        pose = observation.pose
        cost = self.navigation_cost(world)
        if (force or observation.step - self._last_graph_step >= self.graph_period()
                or self.doors.revision != self._last_door_revision):
            started = time.monotonic()
            self.graph.update(world, self.landmarks.confirmed(), self.target, doors=self.doors.confirmed(),
                              step=observation.step, reason=False, cost=cost, here_xy=(pose.x, pose.y),
                              yaw=pose.yaw, ranking=self.sweep.settings.ranking, exclude=self.room_exclusion(world),
                              preferred_cost=self.preferred_cost(world))
            self.telemetry.latencies["scene_graph"].append((time.monotonic() - started) * 1000)
            self._last_graph_step, self._last_door_revision = observation.step, self.doors.revision
        elif self.graph_period() == self.settings.graph_period_steps:
            self.graph.refresh_accessibility(world, cost, (pose.x, pose.y), pose.yaw, self.sweep.settings.ranking,
                                             preferred_cost=self.preferred_cost(world))
        return cost

    def refresh_graph(self, observation, world, force=False):
        """The scene graph's geometry for this action (:meth:`_refresh_graph`); ``force`` rebuilds it whatever the cadence."""
        return self._refresh_graph(observation, world, force=force)

    def graph_period(self):
        """Actions between two rebuilds of the scene graph's geometry, as things stand this action.

        ``transit_graph_period_steps`` while the room-search loop walks a
        committed transit route under its commitment rule (the supervisor in
        TRANSIT, the route memory holding a ``transit/`` route), else
        ``graph_period_steps``.
        """
        s = self.settings
        if s.transit_graph_period_steps <= s.graph_period_steps:
            return s.graph_period_steps
        supervisor, loop, route = getattr(self, "supervisor", None), getattr(self, "loop", None), self.route_memory
        if (supervisor is not None and supervisor.state == "transit" and loop is not None
                and getattr(loop.settings, "transit_commitment", False)
                and route.path is not None and str(route.kind or "").startswith("transit/")):
            return s.transit_graph_period_steps
        return s.graph_period_steps

    def room_exclusion(self, world):
        """Cells that are never room floor: the footprint of every seen staircase near this storey's level.

        A staircase is a connector between storeys, not a room. Left in the
        free mask, the watershed carved a "room" on the landing that the
        search valued, peeked and waited on. None when the building knows
        no stairs here.
        """
        if self.building is None:
            return None
        mask = stair_peek_mask(self, world)
        return mask if mask.any() else None

    def _discover(self, obs, world, cost):
        return discover(self, obs, world, cost)

    def _perceive(self, observation):
        return self.perception.fuse(observation)

    def _approach(self, observation, world):
        p = observation.pose
        dx, dy = self._target_xy[0] - p.x, self._target_xy[1] - p.y
        distance = math.hypot(dx, dy)
        at_path_end = self.route_memory.kind == "target" and self.route_memory.arrived(observation)
        if distance <= self.settings.stop_distance_m or at_path_end or self.target_evidence.verification_started(self._target_id):
            self.route_memory.clear("target_route_arrived")
            self._route = self._goal = None
            yaw = self.target_evidence.verification_yaw(self._target_id, p.yaw, self.episode.action_spec.turn_angle_rad)
            if yaw is not None:
                return NavigationCommand.hold(final_yaw=yaw, info={"reason": "bounded target verification"})
            self.target_evidence.reject(self._target_id, observation.step)
            self._target_xy = self._target_id = None
            return None
        margin = self.converter_params.goal_tolerance_m + world.resolution + self.planner_params.goal_snap_radius_m
        standoff = max(self.settings.body_radius_m, self.settings.stop_distance_m - margin)
        scale = (distance - standoff) / distance
        command = self._navigate(observation, world, (p.x + dx * scale, p.y + dy * scale), "target", final_yaw=math.atan2(dy, dx))
        if command is None and self.route_memory.reason == "no_progress":
            self.target_evidence.reject(self._target_id, observation.step)
            self._target_xy = self._target_id = None
        return command

    def episode_info(self):
        return {"method": self.name, "pipeline_variant": self.variant, "rooms": len(self.graph.registry.rooms), "landmarks": len(self.landmarks),
                "warmup": {"actions": self.warmup_actions, "budget": self.warmup_turns, "bound": self.settings.warmup_steps},
                "doorway_peek": self.peek.diagnostics(),
                "llm_queries": self.graph.queries, "oracle_reuses": self.graph.oracle_reuses,
                **self._llm_answer_counts(),
                "supervisor": dict(self.supervisor.stats), "solver_records": self._solver_records,
                "blocked": self._blocked, "goals_retired_by_bumps": self._goals_retired_by_bumps,
                "doors": self.doors.diagnostics(), "max_rooms": self.graph.max_rooms,
                "room_label_queries": self.graph.label_tracker.queries, "room_label_history": list(self.graph.label_tracker.history),
                "room_labels_held": self.graph.label_tracker.held,
                "room_labels_dining_downgraded": self.graph.label_tracker.dining_downgrades,
                "room_labels_shape_ruled": self.graph.label_tracker.shape_ruled,
                "last_reasoning": self.graph.last_reasoning, "route_commitment": dict(self.route_memory.stats),
                "plan_calls": self._plan_calls, "duplicates_removed": self._duplicates_removed,
                "target_evidence": self.target_evidence.diagnostics(), "floor_revisions": self.mapping.floor_revision,
                "target_closing": self.closing.diagnostics(),
                "target_pursuit": self.pursuit.diagnostics(),
                "perception": self.perception.diagnostics(), "camera": dict(self.camera_control.last), "floor_maps": self.mapping.integrity(),
                "building": self.building.diagnostics() if self.building else None,
                "spawn_floor_guard": self.floor_guard.diagnostics(),
                "map_changes": self.map_ledger.diagnostics(),
                "frontier_sweep": self.sweep.diagnostics(),
                "room_search_loop": self.loop.diagnostics() if self.loop is not None else None,
                "room_scans": self.scans.diagnostics(),
                "sight": self.sight.diagnostics(),
                "openings": self.openings.diagnostics(),
                "glances": self.glances.diagnostics(),
                "exploration_fallback": self.fallback.diagnostics(),
                "escape_probe": self.escape.diagnostics(),
                "kinematic": self.kinematic.diagnostics(),
                "doorway_scrape": self.scrape_guard.diagnostics(),
                "doorway_look_down": self.look_down.diagnostics(),
                "exploration_metrics": self.telemetry.report(), "hierarchy": self.hierarchy.diagnostics() if self.hierarchy else None,
                "oracle_repair_attempts": getattr(self.graph.oracle, "repair_attempts", 0),
                "oracle_repair_successes": getattr(self.graph.oracle, "repair_successes", 0)}
