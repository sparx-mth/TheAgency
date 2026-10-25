"""Observed-only ObjectNav with persistent floors and frontier or planar FALCON.

Stairs are a declared exception to "observed-only": with
``RPTSettings.multifloor.stair_source == "ground_truth"`` (the default) the
building coordinator takes the simulator's navmesh connectors from the
episode metadata, and ``configuration()["ground_truth_stairs"]`` says so.
An optional local NavMesh target-standoff projector is declared separately by
``configuration()["target_navmesh_projection"]``; evaluator goals remain private.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import math
import random
import time

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.exploration.rpt_room_solver import RptStarRoomSolver
from sparx_agency.core.planning.exploration.view_gain import camera_rotation_turns
from sparx_agency.core.planning.exploration.falcon.params import SOURCE as FALCON_SOURCE
from sparx_agency.core.planning.interfaces.planner import PlanRequest
from sparx_agency.core.planning.objnav.action_converter.params import ActionConverterParams
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.planners.astar.params import WeightedAStarParams
from sparx_agency.core.planning.planners.astar.cost_grid_2d import assemble_cost_grid
from sparx_agency.core.planning.planners.astar.weighted_planner_2d import WeightedAStarPlanner2D
from sparx_agency.core.planning.object_search.world_model.observed_map import ObservedMap
from sparx_agency.core.planning.object_search.rooms.scene_graph import DEFAULT_SEGMENTATION
from sparx_agency.core.planning.object_search.perception.doors import DoorSettings
from sparx_agency.core.mapping.topology.revisable_room_labels import RoomLabelSettings
from sparx_agency.core.planning.object_search.route_memory import CommittedRoute, RouteSettings
from sparx_agency.core.mapping.objects.target_evidence import TargetEvidenceSettings
from sparx_agency.core.planning.object_search.exploration.escape_probe import EscapeProbe, EscapeSettings
from sparx_agency.core.planning.object_search.exploration.exploration_fallback import ExplorationFallback, FallbackSettings
from sparx_agency.core.planning.object_search.rooms.floor_context import FloorContextBank
from sparx_agency.core.planning.object_search.exploration.frontier_sweep import FrontierSweep, SweepSettings
from sparx_agency.core.planning.object_search.rooms.opening_nodes import OpeningRegistry
from sparx_agency.core.planning.object_search.exploration.path_glances import GlanceScheduler, GlanceSettings
from sparx_agency.core.planning.object_search.rooms.pattern_scan import PatternScan
from sparx_agency.core.planning.object_search.falcon.prescan import FalconPrescan
from sparx_agency.core.planning.object_search.falcon.room_falcon import RoomFalcon
from sparx_agency.core.planning.object_search.world_model.peek_stairs import stair_peek_mask
from sparx_agency.core.planning.object_search.rooms.room_scans import RoomScanLedger
from sparx_agency.core.planning.object_search.world_model.sightlines import SightLedger, SightSettings
from sparx_agency.core.planning.object_search.multifloor.spawn_floor_guard import SpawnFloorGuard
from sparx_agency.core.planning.object_search.rooms.room_search_loop import LoopSettings
from sparx_agency.core.planning.object_search.exploration.doorway_peek import DoorwayPeek
from sparx_agency.core.planning.object_search.rooms.discovery import SUSPENDED_PHASES, discover
from sparx_agency.core.planning.object_search.settings import ROOM_VISIT_LOOP, RPTSettings
from sparx_agency.core.planning.object_search.exploration.exploration_metrics import ExplorationMetrics
from sparx_agency.core.control.camera.pitch_controller import CameraController, CameraControlSettings
from sparx_agency.core.planning.object_search.perception.cycle import PerceptionCycle
from sparx_agency.core.planning.object_search.closing.target_closing import TargetClosing


class RPTSearchPolicy:
    name = "sparx-rpt-llm-host-sweep"

    def __init__(self, detector, llm_client, settings=None, segmenter=None):
        self.detector, self.llm_client, self.segmenter = detector, llm_client, segmenter
        self.settings = settings or RPTSettings()
        s = self.settings
        if s.segmentation and segmenter is None:
            raise ValueError("RPTSettings.segmentation is on but no segmenter was given; no silent box fallback")
        if s.local_exploration == "falcon":
            self.name = "sparx-rpt-llm-falcon-planar"
        self.converter_params = ActionConverterParams(heading_hysteresis_deg=1.0)
        # A ground agent squeezes rather than detours: a route at the preferred standoff more than
        # ``detour_factor`` times the body-radius route is a detour, and the squeeze is walked.
        self.planner_params = WeightedAStarParams(
            inflate_radius_m=s.preferred_clearance_m, inflate_floor_m=s.body_radius_m,
            detour_factor=s.detour_factor,
            unknown_blocked=True, goal_snap_radius_m=0.3, waypoint_spacing_m=0.25,
            start_skip_m=0.0, search_margin_m=s.map_size_m)
        self.loop_settings = LoopSettings(**ROOM_VISIT_LOOP[s.room_visit])
        self.supervisor_params = self.loop_settings.supervisor_params(s.seed)
        self.door_settings, self.room_label_settings = DoorSettings(), RoomLabelSettings()
        self.route_settings, self.target_settings = RouteSettings(), TargetEvidenceSettings()
        self.sweep_settings = SweepSettings()
        self.fallback_settings = FallbackSettings()
        self.escape_settings = EscapeSettings()
        self.glance_settings = GlanceSettings()
        self.sight_settings = SightSettings()
        self.target_projector = None  # optional simulator-local geometry, never goal annotations

    def configuration(self):
        return {"method": self.name, "adaptation": asdict(self.settings),
                "planner": asdict(self.planner_params), "converter": asdict(self.converter_params),
                "supervisor": asdict(self.supervisor_params), "doors": asdict(self.door_settings),
                "room_labels": asdict(self.room_label_settings), "room_segmentation": asdict(DEFAULT_SEGMENTATION),
                "route_commitment": asdict(self.route_settings), "target_evidence": asdict(self.target_settings),
                "frontier_sweep": asdict(self.sweep_settings), "room_search_loop": asdict(self.loop_settings),
                "exploration_fallback": asdict(self.fallback_settings),
                "escape_probe": asdict(self.escape_settings),
                "glances": dict(asdict(self.glance_settings),
                                rule="at most one in-place glance per route, where along the route a left, right or full "
                                     "look reveals the most unknown floor the walk itself will not, if that clears the "
                                     "gain and gain-per-action floors; a suspended phase (no loop charge, clocks paused)"
                                     + ("; a confident detection cut off by the frame's edge is a cue to turn toward it"
                                        if self.glance_settings.cue_enabled else "")
                                if self.glance_settings.enabled else "none"),
                "sight": dict(asdict(self.sight_settings),
                              rule="unknown looked through from %d poses within %.1f m without a depth return, and "
                                   "enclosed unknown pockets under %.1f m2 that nothing could stand in (inscribed width "
                                   "under %.1f m, or stood on by the agent), are settled: not frontier, not openings, "
                                   "not glance gain; the planner and the display keep the real map"
                                   % (self.sight_settings.min_looks, self.sight_settings.far_m,
                                      self.sight_settings.pocket_max_m2, self.sight_settings.pocket_max_width_m)
                              if self.sight_settings.enabled else "none: every unknown cell is an exit"),
                "target_closing": asdict(self.settings.target_closing),
                "target_navmesh_projection": self.target_projector is not None,
                "room_visit": self._room_visit_description(),
                "room_partition": "watershed of observed free space with seen stair footprints excluded; stairs are never a room; "
                                  "a detected door's cut is snapped to the nearest choke within %.1f m; furniture is credited to "
                                  "the nearest room floor; room numbers are unique across the building"
                                  % DEFAULT_SEGMENTATION.door_snap_reach_m,
                "openings": ("doorways and gaps at the edge of the mapped floor are nodes beside the rooms and the stairs, "
                             "visited by a peek (threshold, face the unknown, one look to each side) priced at %d actions; "
                             "valued by the oracle (uncertainty floor %s, elsewhere value %s, home floor %s)"
                             % (self.loop_settings.openings.service_steps,
                                "off" if not self.loop_settings.unexplored_floor else "p=%.2f" % self.loop_settings.unexplored_floor,
                                "off" if not self.loop_settings.unexplored_elsewhere else "p=%.2f" % self.loop_settings.unexplored_elsewhere,
                                "off" if not self.loop_settings.home_floor else "p=%.2f" % self.loop_settings.home_floor)
                             + ("; a confirmed landmark of the target's class is a node at p=%.2f without an oracle call, "
                                "visited by the same peek from %.1f m" % (self.loop_settings.openings.landmark_prob,
                                                                            self.loop_settings.openings.landmark_standoff_m)
                                if self.loop_settings.openings.landmarks_enabled else "")
                             if self.loop_settings.openings.enabled else "none: openings are the exploration fallback's alone"),
                "floor_exit_gate": ("unknown non-stair rooms get at most one peek attempt per episode before a floor change; "
                                    "initially classified, scanned and previously peeked rooms are exempt"
                                    if self.settings.doorway_peek.gate_floor_departure else
                                    "none: a floor change is the RPT* order's or the fallback rule's to take at any time"),
                "node_oracle": {"nodes": "every room of the floor in force (finished and type-excluded ones shown with "
                                         "their status) + its staircases" + (
                                    " + its openings" if self.loop_settings.openings.enabled else ""),
                                "asks": "independent P(searching there finds target), without a fixed action horizon; "
                                        "the model is briefed with the HOUSE line (room types found, rooms unidentified, "
                                        "openings unlooked, actions used of the budget) and writes home/house/stage and a "
                                        "first|second pass verdict before the numbers (LLM-first, 2026-10-07)",
                                "probability_model": "independent_search_success",
                                "code_floors": {"unexplored_floor": self.loop_settings.unexplored_floor,
                                                "unexplored_elsewhere": self.loop_settings.unexplored_elsewhere,
                                                "home_floor": self.loop_settings.home_floor},
                                "second_pass": self.loop_settings.second_pass, "type_prior": self.loop_settings.type_prior,
                                "route": "LLM_REASONING_MODEL", "cadence": "loop points and semantic discovery; unchanged prompts reused"},
                "floor_change_decision": ("RPT* over rooms and stair nodes; a staircase is charged its flight plus "
                                          "%.1f m on every arc; no allowance, no clock; the explicit floor_decision "
                                          "rule is the exploration fallback's last resort"
                                          % self.settings.multifloor.floor_change_cost_m)
                if self.loop_settings.stairs_as_nodes and self.settings.multifloor.stair_source == "ground_truth"
                else "exploration fallback only",
                "reasoning_cadence": "room LLM at loop points and semantic discovery events; geometry every %d action(s); "
                                     "synchronous frame processing, one full rotation (warm-up scan) before room selection "
                                     "on every storey first entered"
                                     % self.settings.graph_period_steps,
                "camera_control": asdict(CameraControlSettings()),
                "perception_fusion": "coherent-depth/floor-qualified-v1" + ("/target-masked" if self.settings.segmentation else ""),
                "segmentation": ({"enabled": True, "scope": "target boxes at or above detection_confidence",
                                  "depth": "robust_mask_depth over the mask within the box; near-surface component, no coherence gate",
                                  "unmasked_target_box": "not projected (segmentation_unavailable)"}
                                 if self.settings.segmentation else {"enabled": False, "depth": "robust_bbox_depth over the shrunken box"}),
                "oracle_schema_repairs": 1, "local_exploration": self.settings.local_exploration,
                "falcon_running": self.settings.local_exploration == "falcon",
                "falcon_source": dict(FALCON_SOURCE) if self.settings.local_exploration == "falcon" else None,
                "ground_truth_semantics": False,
                "ground_truth_stairs": self.settings.multifloor.enabled and self.settings.multifloor.stair_source == "ground_truth",
                "stair_source": self.settings.multifloor.stair_source, "training_free": True,
                "allow_stair_traversal": self.settings.allow_stair_traversal,
                "floor_plane_bound_m": self.settings.floor_plane_bound_m,
                "non_metric": False, "clock": "global action ledger; room clock pauses off-floor"}

    def _room_visit_description(self):
        """The frozen configuration's one-line account of what a room visit is under the loop's ``visit`` mode."""
        loop = self.loop_settings
        if not loop.finishing:
            return "bounded frontier sweep of %d actions" % loop.local_steps
        finishing = ("finished rooms are shown to the oracle with their status and offered again%s once its pass "
                     "verdict says the home is covered (a revisit scans from a spot at least %.1f m from the earlier "
                     "scan points); rooms whose STRONG type label cannot hold the target are %s; one signature object "
                     "(a bed, a toilet, an oven) makes a label strong"
                     % ("" if loop.second_pass else " (DISABLED: second_pass off)", loop.revisit_standoff_m,
                        "not nodes (type_prior on)" if loop.type_prior
                        else "valued by the oracle like every other room (type_prior off)"))
        if loop.exploring:
            return ("frontier walk: the room's frontier that reveals the most unknown floor per metre walked, then the "
                    "next, until none is left (bounded by %d actions); a room the agent stood in with no live frontier "
                    "is finished (explored), and so is a room a full rotation stood in or saw more than half of, one "
                    "with no live frontier the camera looked into or walked through, or a doorless fragment under "
                    "%.1f m2; every rotation is as many turns as the camera's field of view needs to see the circle%s; "
                    "openings off a room that is itself a node are folded into its walk; "
                    % (loop.explore_visit_steps, loop.fragment_max_m2, self._rotation_note())
                    + finishing)
        if loop.nearest_frontier_scan:
            return ("entered at the room's nearest frontier, then the nearest frontier until none is left, no walk "
                    "to the middle (bounded by %d actions); a revisit scans from a new spot instead; "
                    % loop.explore_visit_steps + finishing)
        return ("vantage point + full rotation (as many turns as the camera's field of view needs to see the circle%s); "
                "a scanned room (or one a scan saw more than half of) is finished for the first pass, and so is a room "
                "with no live frontier that the camera looked into or walked through, or that is a doorless fragment "
                "under %.1f m2; " % (self._rotation_note(), loop.fragment_max_m2) + finishing)

    def _rotation_note(self):
        """`` (N)`` once an episode's camera says how many turns that is; empty before the first reset."""
        return " (%d)" % self.rotation_turns() if getattr(self, "episode", None) is not None else ""

    def reset(self, episode, target):
        self.episode, self.target = episode, target
        s = self.settings
        self.mapping = ObservedMap(s.map_size_m, s.map_resolution_m, s.depth_stride, s.body_height_m, s.body_radius_m, s.multifloor)
        self.solver = RptStarRoomSolver(rng=random.Random("%s/%s" % (s.seed, episode.episode_id)), max_rooms=s.max_rooms)
        self.planner = WeightedAStarPlanner2D(self.planner_params)
        # The LOCKED target's last resort before a boxed-in release: the same planner with unknown
        # cells passable at a price (``target_closing.through_unknown``), for the standoff ring alone.
        self.optimistic_planner = WeightedAStarPlanner2D(replace(
            self.planner_params, unknown_blocked=False, unknown_cost=s.target_closing.through_unknown_cost))
        self.route_memory = CommittedRoute(episode.action_spec, self.converter_params, self.route_settings)
        self.fallback = ExplorationFallback(self, self.fallback_settings)
        # The fallback's rung before its idle turn: boxed in on the inflated map, one
        # probing step through the thinnest barrier -- the simulator says what the body passes.
        self.escape = EscapeProbe(self, self.escape_settings)
        # Where the camera has looked and what it looked through without a
        # return, per floor: settled unknown is not frontier, and the poses say
        # which rooms were walked through or looked into.
        self.sight = SightLedger(self, self.sight_settings)
        # Where every completed look-around stood, on every floor: the one memory
        # of "this room is finished" that survives the watershed renumbering rooms.
        # Under the explore visit a room is finished too when the agent has stood in
        # it and no live frontier is left (its frontiers were walked until none was).
        self.scans = RoomScanLedger(self, self.loop_settings.scan_seen_fraction,
                                    fragment_max_m2=self.loop_settings.fragment_max_m2,
                                    walkthrough_clearance_m=self.loop_settings.walkthrough_clearance_m)
        # Sticky ids for the floor's openings and the ones already peeked into,
        # building-wide like the room numbers: "O3" names one doorway in the recording.
        self.openings = OpeningRegistry(match_m=self.loop_settings.openings.match_m,
                                        done_m=self.loop_settings.openings.done_m)
        self.floors = FloorContextBank(self)
        self._reset_floor()
        self._last_plan_s = self._blocked_since = self._last_pose = None
        self._last_graph_step = -s.graph_period_steps
        self._plan_step = -s.replan_steps
        self._blocked = self._plan_calls = self._duplicates_removed = 0
        self._solver_records = []
        self.last_world = None
        self.telemetry = ExplorationMetrics()
        self.camera_control = CameraController(episode.action_spec)
        self.perception = PerceptionCycle(self)
        self.sweep = FrontierSweep(self, self.sweep_settings)
        self.hierarchy = None
        if s.local_exploration == "falcon":
            from sparx_agency.core.planning.object_search.falcon.policy import FalconObjectNav
            self.hierarchy = FalconObjectNav(self)
        self.building = None
        if s.multifloor.enabled:
            from sparx_agency.core.planning.object_search.multifloor.multifloor_policy import MultiFloorSearch
            self.building = MultiFloorSearch(self)
        # Spawn-floor confinement: inert when stair traversal is allowed.
        self.floor_guard = SpawnFloorGuard(self)
        self.peek = DoorwayPeek(self)
        self.closing = TargetClosing(self)
        # Where along the route in force a look to the side (or all round) is worth
        # its actions; performed as a suspended phase when the agent gets there.
        self.glances = GlanceScheduler(self, self.glance_settings)
        self.pattern_scan = PatternScan(self, s.pattern_scan)
        self.prescan = FalconPrescan(self, s.prescan) if s.prescan.steps else None
        self._room_falcon = None
        # One look-around is as many turns as the camera's field of view needs to see the
        # whole circle (ten at 30 degrees with a 79-degree field), bounded by ``warmup_steps``.
        self.warmup_turns = min(s.warmup_steps, self.rotation_turns())
        self.warmup_actions = self.warmup_turns if s.scan_kind == "falcon" else 0
        self._warmup_pending = s.scan_kind != "falcon"   # the warm-up rotation's scan has not been recorded yet
        self._action_owner = "search"
        self._notified_step = self._decision_step = -1
        self._decision_command = None
        self._semantic_signature = None
        self._resume_room = None

    def _reset_floor(self):
        self.floors.new()
        self._route = self._goal = self._target_xy = self._target_id = None
        self.route_memory.clear("floor_reset")

    def rotation_turns(self):
        """In-place turns one full look-around takes: what the camera's field of view needs to cover the circle.

        Every rotation of the method -- the warm-up, a room scan, a full
        glance, a footing sweep -- is complete when every bearing has been
        inside the field of view once, not when the body is back at its
        start heading (:meth:`~sparx_agency.core.planning.exploration.view_gain.ViewCone.rotation_turns`):
        ten turns at the benchmark's 30 degrees with its 79-degree camera,
        not twelve.
        """
        k = self.episode.camera.intrinsics
        return camera_rotation_turns(k.width, k.fx, self.episode.action_spec.turn_angle_rad)

    def notify_blocked(self, observation):
        if self.closing.active:
            self.closing.note_blocked()
            self.mapping.notify_blocked(observation, self.episode.action_spec.forward_step_m)
            self._blocked += 1
            self.route_memory.clear("target_forward_blocked")
            self._route = self._goal = None
            return
        if self.building and self.building.traversing:
            self._blocked += 1
            self.building.blocked(observation)
            return
        self.mapping.notify_blocked(observation, self.episode.action_spec.forward_step_m)
        self._blocked += 1
        self.escape.note_blocked(observation)
        if self._blocked_since is None:
            self._blocked_since = self._floor_time
        if self._goal is not None and not self._action_owner.startswith("doorway_peek"):
            self._visited_frontiers.append(self._goal)
        self.route_memory.clear("forward_blocked")
        self._route, self._goal = None, None
        if self.hierarchy is not None and self._action_owner not in SUSPENDED_PHASES:
            self.hierarchy.blocked()
        if self.prescan is not None and self._action_owner == "prescan":
            self.prescan.blocked(observation)
        if self._room_falcon is not None:
            self._room_falcon.blocked(observation)

    def plan(self, observation):
        if observation.step == self._decision_step:
            return self._decision_command
        self._action_owner = "search"
        started = time.monotonic()
        try:
            command = self._plan(observation)
            phase = self.hierarchy.machine.phase if self.hierarchy else str(command.info.get("kind", self.supervisor.state))
            if self._action_owner in SUSPENDED_PHASES:
                phase = self._action_owner
            if self.closing.active:
                phase = "target_" + self.closing.phase.lower()
            elif self.building and self.building.phase != "SEARCH":
                phase = self.building.phase
            transition = self.building.transition if self.building else None
            command = self.camera_control.apply(
                observation, command, "TARGET_CLOSING" if self.closing.active else self.building.phase if self.building else "SEARCH",
                transition.direction if transition else 0, transition.close_support if transition else False)
            self.telemetry.phase = phase
            self._decision_step = observation.step
            self._decision_command = replace(command, info=dict(command.info, explorer=self.settings.local_exploration,
                                                                phase=phase, floor_id=self.mapping.floor_id))
            return self._decision_command
        finally:
            self.telemetry.latencies["policy_decision"].append((time.monotonic() - started) * 1000)

    def filter_action(self, observation, action):
        if self.closing.active:
            return action  # closing A* and native collision checking, never a global room mask
        if self.building and self.building.committed:
            return self.building.filter_action(observation, action)
        if self.building:
            guarded = self.building.departure.filter_action(observation, action)
            if guarded != action:
                return guarded
        if self._action_owner == "prescan":
            return self.prescan.filter_action(observation, action)
        if self._action_owner in SUSPENDED_PHASES:
            return action  # A* already qualified this route; do not apply the suspended FALCON room mask.
        filtered = self.hierarchy.filter_action(observation, action) if self.hierarchy else action
        return self._room_falcon.filter_action(observation, filtered) if self._room_falcon is not None else filtered

    @property
    def room_falcon(self):
        """The in-room FALCON explorer, built the first time a room visit asks for it."""
        if self._room_falcon is None:
            self._room_falcon = RoomFalcon(self)
        return self._room_falcon

    def notify_action(self, observation, action):
        if observation.step == self._notified_step:
            return
        if observation.step < self._notified_step:
            raise ValueError("Action ledger regressed")
        self._notified_step = observation.step
        if self.closing.active:
            self.closing.note_action(action)
            self.telemetry.emitted(observation, action, self.telemetry.phase)
            return
        external = self._action_owner in SUSPENDED_PHASES
        # The closing's own release turn: decided inside ``closing.plan`` with ``active``
        # already False by now; the search did not take it, so the loop is not charged.
        released = self._action_owner == "target_closing"
        if not self.building or not self.building.committed:
            self._floor_time += self.settings.action_time_s
            if external or released:
                self.supervisor.pause(self.settings.action_time_s)
                # The blocked clock pauses with the supervisor's deadlines: a glance or a peek's
                # look turned in place after a blocked step is not the follower still blocked.
                # Corozal/000142 (``51eff3fa1637``, 2026-10-13): one blocked step, a ten-turn
                # glance, and the next four transits were released ``blocked`` one action in.
                if self._blocked_since is not None:
                    self._blocked_since += self.settings.action_time_s
            elif self.hierarchy is None:
                self.loop.charge()
        if external and self.warmup_actions < self.warmup_turns:
            self.warmup_actions += 1
        if self._action_owner == "doorway_peek":
            self.peek.charge()
        elif self._action_owner == "room_return" and self._resume_room is not None:
            floor, room, actions = self._resume_room
            self._resume_room = (floor, room, actions + 1)
        if self.hierarchy:
            self.hierarchy.machine.charge(observation.step, external_phase=self._action_owner if external else None)
            if external:
                self.hierarchy._transit_start += 1
        self.telemetry.emitted(observation, action, self.telemetry.phase)

    def _plan(self, observation):
        self.perception.observe(observation)
        self.closing.observe(observation)
        if self.closing.active:
            self._action_owner = "target_closing"
            if self.prescan is not None and self.prescan.active:
                self.prescan.suspend("target_closing")
            if self._room_falcon is not None:
                self._room_falcon.suspend("target_closing")
            # No global decision/fallback try block may catch closing failures.
            # Keep the current-floor collision map fresh without advancing any
            # building/room task or consuming privileged goal data.
            try:
                world = self.mapping.update(observation, arrival_allowed=False)
                world = self._confine(observation, world)
                self.last_world = world
                self.sight.observe(observation, world)
                # The landmark map keeps voting while the takeover walks in (the legacy
                # target evidence stands down, see ``PerceptionCycle.fuse``): the map
                # release -- the sofa from four metres that is the bed from two -- needs
                # the closer frames on the map, and until 2026-10-05 nothing fed them.
                self._perceive(observation)
                return self.closing.plan(observation, world)
            except Exception as exc:
                self.closing.phase = "FAILED"
                if self.closing.failure is None:
                    self.closing.failure = "%s: %s" % (type(exc).__name__, exc)
                raise
        if self.building:
            self.building.prepare_observation(observation)
        started = time.monotonic()
        floor_revision = self.mapping.floor_revision
        transition = self.building.transition if self.building else None
        try:
            world = self.mapping.update(observation, integrate=not (self.building and self.building.traversing),
                                        arrival_allowed=transition.arrival_allowed if transition else True)
        except ValueError as exc:
            # The agent walked within the camera's range of the 80 m map's edge: the frame
            # cannot be integrated, but the episode goes on with the map as it stands
            # (review of 2026-10-07: this raise stood outside the fallback's guard).
            if self.last_world is None:
                raise
            self.fallback.record_failure(observation, "mapping", exc)
            world = self.last_world
        world = self._confine(observation, world)
        self.telemetry.latencies["mapping"].append((time.monotonic() - started) * 1000)
        if self.mapping.floor_revision != floor_revision:
            self.peek.cancel(observation, "floor_changed", restore=False)
            if self.building:
                fresh = self.mapping.floor_id not in self.floors.contexts and self.mapping.floor_id != self.floors.active
                self.floors.activate(self.mapping.floor_id)
                if fresh:
                    # A storey never stood on: one full rotation at the stair head before any
                    # room is chosen -- the map gets its first rooms, the ledger its first scan.
                    self.warmup_actions = 0
                    self._warmup_pending = True
            else:
                self._reset_floor()
        self.last_world = world
        if self.building:
            self.building.observe(observation)
        if not self.building or not self.building.traversing:
            self.telemetry.observe(observation, world, self.mapping.floor_id)
            self.sight.observe(observation, world)
        confirmed = self._perceive(observation)
        try:
            return self._decide(observation, world, confirmed)
        except Exception as exc:  # A*, RPT*, the room LLM, or a bug in the decision: keep exploring, loudly
            self.peek.cancel(observation, "decision_failure")
            self._action_owner = "search"
            self.fallback.record_failure(observation, "decision", exc)
            try:
                return self.fallback.plan(observation, world, reason="decision failure: %s" % type(exc).__name__)
            except Exception as inner:  # the fallback itself: one hold, recorded, never a dead episode
                self.fallback.record_failure(observation, "fallback", inner)
                return NavigationCommand.hold(info={"kind": "fallback_hold",
                                                    "reason": "fallback failed: %s" % type(inner).__name__})

    def _decide(self, observation, world, confirmed):
        """The search decision proper; every exception out of here is a recorded fallback, not an idle spin."""
        if self.building and self.building.committed:
            if not self.building.traversing:
                self._refresh_graph(observation, world)
            self.peek.cancel(observation, "stairs_priority", restore=False)
            command = self.building.plan(observation, world)
            if command is not None:
                return command
        inspection = self.camera_control.inspection_command(observation)
        if inspection is not None:
            return inspection
        return self._search(observation, world, confirmed)

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
        return self.glances.apply(observation, world, self.loop.plan(observation, world))

    def _refresh_graph(self, observation, world):
        """Refresh geometry even on a stair approach, so new rooms revoke departure."""
        pose = observation.pose
        cost = self.navigation_cost(world)
        if (observation.step - self._last_graph_step >= self.settings.graph_period_steps
                or self.doors.revision != self._last_door_revision):
            started = time.monotonic()
            self.graph.update(world, self.landmarks.confirmed(), self.target, doors=self.doors.confirmed(),
                              step=observation.step, reason=False, cost=cost, here_xy=(pose.x, pose.y),
                              yaw=pose.yaw, ranking=self.sweep.settings.ranking, exclude=self.room_exclusion(world),
                              preferred_cost=self.preferred_cost(world))
            self.telemetry.latencies["scene_graph"].append((time.monotonic() - started) * 1000)
            self._last_graph_step, self._last_door_revision = observation.step, self.doors.revision
        else:
            self.graph.refresh_accessibility(world, cost, (pose.x, pose.y), pose.yaw, self.sweep.settings.ranking,
                                             preferred_cost=self.preferred_cost(world))
        return cost

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

    def navigation_cost(self, world):
        """The common clearance-qualified map used by counts, entries and routes (passable at the body radius)."""
        return assemble_cost_grid(self.planner.fields_for(world), self.planner_params, self.settings.body_radius_m)[0]

    def preferred_cost(self, world):
        """The cost map at the planner's PREFERRED standoff -- what A* flies wherever it can.

        Distances for the frontier inventory and the RPT* instance are
        measured on this where it reaches and on :meth:`navigation_cost`
        otherwise, so a node's charged distance is the route the planner
        will take, not the squeeze it would refuse (cached by the planner).
        """
        return self.planner.cost_for(world)[0]

    def _discover(self, obs, world, cost):
        return discover(self, obs, world, cost)

    def _confine(self, observation, world):
        """The world every decision plans on: seen stairs and observed drops impassable while traversal is forbidden.

        Stored back as the floor's world so the display panels, the peek
        copy and the terrain all read the same map; the observed log-odds
        grid underneath is untouched and rebuilds the world next action.
        """
        confined = self.floor_guard.observe(observation, world)
        if confined is not world:
            self.mapping.worlds[self.mapping.floor_id] = confined
        return confined

    def _navigate(self, observation, world, goal, kind, final_yaw=None, planner=None):
        planner = self.planner if planner is None else planner
        if not self.floor_guard.goal_allowed(observation, world, goal, kind):
            self._visited_frontiers.append(tuple(goal))
            self.route_memory.clear("goal off the spawn floor")
            self._route = self._goal = None
            return None
        if not self.route_memory.reusable(observation, world, planner, goal, kind):
            if self.route_memory.reason == "no_progress":
                self._visited_frontiers.append(tuple(goal))
                self._route = self._goal = None
                self._blocked_since = self._floor_time
                return None
            path = (self._plan_to(observation, world, goal) if planner is self.planner
                    else self._plan_to(observation, world, goal, planner=planner))
            if path is None:
                self._visited_frontiers.append(tuple(goal))
                self._route = self._goal = None
                return None
            self.route_memory.adopt(path, goal, kind, observation)
        self._route, self._goal = self.route_memory.path, self.route_memory.goal
        info = {"route": self.route_memory.reason, "kind": kind}
        if planner is not self.planner:
            info["through_unknown"] = True
        if self.route_memory.reason == "new_goal_or_invalid_route":
            info["route_replaced"] = self.route_memory.replaced
        return NavigationCommand.follow(self._route, final_yaw=final_yaw, info=info)

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

    def _plan_to(self, observation, world, goal, planner=None):
        p = observation.pose
        planner = self.planner if planner is None else planner
        self._plan_calls += 1
        started = time.monotonic()
        result = planner.plan(PlanRequest(Pose2D(p.x, p.y, p.yaw), Pose2D(*goal), frame_id="world"), world)
        self.telemetry.latencies["astar"].append((time.monotonic() - started) * 1000)
        self._plan_step = observation.step
        if not result.ok:
            return None
        self._last_plan_s = self._floor_time
        return result.path

    def episode_info(self):
        return {"method": self.name, "rooms": len(self.graph.registry.rooms), "landmarks": len(self.landmarks),
                "warmup": {"actions": self.warmup_actions, "budget": self.warmup_turns, "bound": self.settings.warmup_steps},
                "prescan": None if self.prescan is None else self.prescan.diagnostics(),
                "room_falcon": None if self._room_falcon is None else self._room_falcon.diagnostics(),
                "doorway_peek": self.peek.diagnostics(),
                "llm_queries": self.graph.queries, "oracle_reuses": self.graph.oracle_reuses,
                "supervisor": dict(self.supervisor.stats), "solver_records": self._solver_records,
                "blocked": self._blocked, "doors": self.doors.diagnostics(), "max_rooms": self.graph.max_rooms,
                "room_label_queries": self.graph.label_tracker.queries, "room_label_history": list(self.graph.label_tracker.history),
                "last_reasoning": self.graph.last_reasoning, "route_commitment": dict(self.route_memory.stats),
                "plan_calls": self._plan_calls, "duplicates_removed": self._duplicates_removed,
                "target_evidence": self.target_evidence.diagnostics(), "floor_revisions": self.mapping.floor_revision,
                "target_closing": self.closing.diagnostics(),
                "perception": self.perception.diagnostics(), "camera": dict(self.camera_control.last), "floor_maps": self.mapping.integrity(),
                "building": self.building.diagnostics() if self.building else None,
                "spawn_floor_guard": self.floor_guard.diagnostics(),
                "frontier_sweep": self.sweep.diagnostics(),
                "room_search_loop": self.loop.diagnostics(),
                "room_scans": self.scans.diagnostics(),
                "sight": self.sight.diagnostics(),
                "openings": self.openings.diagnostics(),
                "glances": self.glances.diagnostics(),
                "exploration_fallback": self.fallback.diagnostics(),
                "escape_probe": self.escape.diagnostics(),
                "exploration_metrics": self.telemetry.report(), "hierarchy": self.hierarchy.diagnostics() if self.hierarchy else None,
                "oracle_repair_attempts": self.graph.oracle.repair_attempts,
                "oracle_repair_successes": self.graph.oracle.repair_successes}
