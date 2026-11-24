"""Observed-only ObjectNav with persistent floors and VLM-scored frontier navigation.

**The agent is given no scene geometry.** What it knows of the building is
what its own RGB-D and pose have measured: the occupancy map, the storeys it
has stood on, the stairs it has seen. Since 2026-11-15 that holds without
exception -- two things used to reach it from the simulator and are gone:

* *Standing points.* Where the closing may stand beside a target was, from
  2026-10-14, projected onto the simulator's navigation mesh through a
  ``target_projector`` the evaluator bound to the policy. It is now read off
  the observed map (``world_model/standing_floor.py``): a cell observed free,
  with the body's clearance of observed free floor around it, reachable from
  the agent through such cells. ``configuration()["standing_points"]``
  records the rule.
* *Stairs and storeys.* ``RPTSettings.multifloor.stair_source`` is
  ``"observed"`` -- a staircase is a measured support surface rising within
  reach of the robot's own RGB-D (``world_model/stair_terrain.py``), masked
  on the planning grid from that measurement (ApexNav, SG-Nav and OSG-Nav
  have no staircase oracle either). The ``"ground_truth"`` ablation, which
  took the navmesh's stair connectors and storey levels from the episode
  metadata, is removed and raises; ``configuration()["ground_truth_stairs"]``
  is False always, kept so a run of this code can be told from an older one.

Nothing under ``sparx_agency/core`` may reach the navmesh, the pathfinder or
ground-truth storey data; ``tests/test_no_privileged_geometry.py`` fails if
anything does. Evaluator goals remain private. The simulator's ground-truth
semantics, when a run selects them, replace the DETECTOR at the perception
interface and are declared by the adapter, not here.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import math
import random
import time

import numpy as np

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.exploration.rpt_room_solver import RptStarRoomSolver
from sparx_agency.core.planning.exploration.view_gain import camera_rotation_turns
from sparx_agency.core.planning.exploration.falcon.params import SOURCE as FALCON_SOURCE
from sparx_agency.core.planning.interfaces.planner import PlanRequest
from sparx_agency.core.planning.environment import OccupancyGrid2D
from sparx_agency.core.planning.objnav.action_converter.params import ActionConverterParams
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.objnav.errors import ObjNavInternalError
from sparx_agency.core.planning.object_search.frame_log.episode_frame_log import EpisodeFrameLog
from sparx_agency.core.planning.object_search.frame_log.memory_store import MemoryFrameStore
from sparx_agency.core.planning.object_search.frontier_views.nodes import NodeSettings
from sparx_agency.core.planning.object_search.frontier_views.scoring import PROMPT_VERSION
from sparx_agency.core.planning.object_search.frontier_views.search import FrontierSearch
from sparx_agency.core.planning.object_search.frontier_views.settings import ViewSettings
from sparx_agency.core.planning.planners.astar.cost_grid_2d import assemble_cost_grid
from sparx_agency.core.planning.planners.astar.weighted_planner_2d import WeightedAStarPlanner2D
from sparx_agency.core.planning.planners.common.grid_geometry_2d import line_of_sight_clear
from sparx_agency.core.planning.object_search.world_model.observed_map import ObservedMap
from sparx_agency.core.planning.object_search.route_memory import CommittedRoute, RouteSettings
from sparx_agency.core.planning.object_search.exploration.escape_probe import EscapeProbe, EscapeSettings
from sparx_agency.core.planning.object_search.exploration.exploration_fallback import FallbackSettings
from sparx_agency.core.planning.object_search.exploration.geometric_fallback import GeometricFallback
from sparx_agency.core.planning.object_search.exploration.warmup import rotation_command
from sparx_agency.core.planning.object_search.interfaces import ground_truth_semantics
from sparx_agency.core.planning.object_search.exploration.frontier_sweep import FrontierSweep, SweepSettings
from sparx_agency.core.planning.object_search.exploration.path_glances import GlanceScheduler, GlanceSettings
from sparx_agency.core.planning.object_search.world_model.map_ledger import MapChangeLedger
from sparx_agency.core.planning.object_search.world_model.sightlines import SightLedger, SightSettings
from sparx_agency.core.planning.object_search.multifloor.spawn_floor_guard import SpawnFloorGuard
from sparx_agency.core.planning.object_search.navigation import (BottleneckLookDown, DoorwayScrapeGuard,
                                                                  KinematicFollower)
from sparx_agency.core.planning.object_search.settings import RPTSettings
from sparx_agency.core.planning.object_search.exploration.exploration_metrics import ExplorationMetrics
from sparx_agency.core.control.camera.pitch_controller import CameraController, CameraControlSettings
from sparx_agency.core.planning.object_search.perception.cycle import TargetStairCycle
from sparx_agency.core.planning.object_search.perception.target_prompts import target_stair_prompts
from sparx_agency.core.planning.object_search.closing.target_closing import TargetClosing


def __getattr__(name):
    """Compatibility for callers importing the room policy here; visual runs don't import rooms."""
    if name == "LegacyRoomSearchPolicy":
        from sparx_agency.core.planning.object_search.rooms.legacy_policy import LegacyRoomSearchPolicy
        return LegacyRoomSearchPolicy
    raise AttributeError(name)


class _ObservedSearchPolicy:
    """Shared observed mapping, collision handling, closing and route execution for both comparison arms."""

    def __init__(self, detector, settings=None, segmenter=None, frame_log=None):
        self.detector, self.segmenter = detector, segmenter
        self.frame_log = frame_log
        self.settings = settings or RPTSettings()
        s = self.settings
        if s.segmentation and segmenter is None:
            raise ValueError("RPTSettings.segmentation is on but no segmenter was given; no silent box fallback")
        if s.local_exploration == "falcon":
            self.name = "sparx-rpt-llm-falcon-planar"
        # Strict tracking (2026-10-28): the lattice follower plans every route on the benchmark's own step and
        # turn and re-plans from the exact pose, so the converter's ladder is only ever the fallback -- aimed one
        # step ahead, without hysteresis, so it cannot shortcut a planned turn. Off, the 1-degree hysteresis ladder.
        self.converter_params = ActionConverterParams(
            heading_hysteresis_deg=0.0 if s.kinematic.strict_tracking else 1.0,
            strict_plan_tracking=s.kinematic.strict_tracking)
        # The planner is the settings' own (``RPTSettings.planner_params``: squeeze rather than detour, the
        # medial-axis layer through doorways), so an offline replay of a recorded action builds the same one.
        self.planner_params = s.planner_params()
        self.route_settings = RouteSettings()
        self.sweep_settings = SweepSettings()
        self.fallback_settings = FallbackSettings(exits_first=self.uses_room_loop)
        self.escape_settings = EscapeSettings()
        self.glance_settings = GlanceSettings(geometric_enabled=s.geometric_glances, cue_enabled=self.uses_room_loop)
        self.sight_settings = SightSettings()

    def standing_retry_clearance_m(self):
        """The clearance the closing's second standing-point planner keeps: the setting, or the body radius by default."""
        retry = self.settings.target_closing.standing_retry_clearance_m
        return float(self.settings.body_radius_m if retry is None else retry)

    def configuration(self):
        return {"method": self.name, "adaptation": asdict(self.settings),
                "planner": asdict(self.planner_params), "converter": asdict(self.converter_params),
                "pipeline_variant": self.variant, "route_commitment": asdict(self.route_settings),
                "frontier_sweep": asdict(self.sweep_settings),
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
                                   "not glance gain; the planner and the display keep the real map; every free cell "
                                   "seen within %.1f m with a clear line of sight, at a pitch that puts an object body "
                                   "%.1f-%.1f m tall in the frame, is OBSERVED (per cell, per storey)"
                                   % (self.sight_settings.min_looks, self.sight_settings.far_m,
                                      self.sight_settings.pocket_max_m2, self.sight_settings.pocket_max_width_m,
                                      self.sight_settings.detect_range_m, self.sight_settings.object_band_m[0],
                                      self.sight_settings.object_band_m[1])
                              if self.sight_settings.enabled else "none: every unknown cell is an exit"),
                "target_closing": asdict(self.settings.target_closing),
                "kinematic": dict(asdict(self.settings.kinematic),
                                  rule="every committed route is walked as the action sequence a Hybrid A* over "
                                       "(x, y, heading) finds with the benchmark's own step and turn as its only "
                                       "primitives, collision-checked on the clearance field at the route's standoff "
                                       "and priced in actions; the converter executes it verbatim while the agent "
                                       "stands where the plan expects it; a refused step is a pose the lattice may "
                                       "not step forward from, not a cell on the map"
                                       if self.settings.kinematic.enabled else "none: the converter's waypoint ladder"),
                "doorway_scrape": asdict(self.settings.doorway_scrape),
                "doorway_look_down": asdict(self.settings.doorway_look_down),
                "standing_points": "observed map only: a standoff, a back-off spot or a nearest-floor goal is a cell "
                                   "observed free, with every cell within the body radius (%.2f m) of it observed "
                                   "free too, joined to the agent through such cells; it keeps %.2f m from every "
                                   "observed obstacle, and %.2f m when no route is found at that; unknown cells are "
                                   "never stood on, never counted as clearance and never planned through; no "
                                   "simulator geometry (world_model/standing_floor.py)"
                                   % (self.settings.body_radius_m, self.settings.min_clearance_m,
                                      self.standing_retry_clearance_m()),
                "camera_control": asdict(CameraControlSettings()),
                "perception_fusion": "coherent-depth/floor-qualified-v1" + ("/target-masked" if self.settings.segmentation else ""),
                "segmentation": ({"enabled": True, "scope": "target boxes at or above detection_confidence",
                                  "depth": "robust_mask_depth over the mask within the box; near-surface component, no coherence gate",
                                  "unmasked_target_box": "not projected (segmentation_unavailable)"}
                                 if self.settings.segmentation else {"enabled": False, "depth": "robust_bbox_depth over the shrunken box"}),
                "local_exploration": self.settings.local_exploration,
                "falcon_running": self.settings.local_exploration == "falcon",
                "falcon_source": dict(FALCON_SOURCE) if self.settings.local_exploration == "falcon" else None,
                "ground_truth_semantics": ground_truth_semantics(self.detector, self.segmenter),
                "ground_truth_stairs": False,
                "stair_source": self.settings.multifloor.stair_source, "training_free": True,
                "allow_stair_traversal": self.settings.allow_stair_traversal,
                "floor_plane_bound_m": self.settings.floor_plane_bound_m,
                "floor_change_decision": "building coordinator only", "non_metric": False}

    def reset(self, episode, target):
        self.episode, self.target = episode, target
        scope = getattr(self.model_client, "set_answer_scope", None)
        if scope is not None:
            scope(episode.episode_id)          # an answer store keys by episode (llm_answer_cache)
        self._llm_answers_at_reset = self._llm_answer_totals()
        if self.frame_log is not None:
            self.frame_log.begin(episode)      # the episode's own frames from here on (frame_log/)
        s = self.settings
        self.mapping = ObservedMap(s.map_size_m, s.map_resolution_m, s.depth_stride, s.body_height_m, s.body_radius_m, s.multifloor,
                                   sticky_bump_actions=s.sticky_bump_actions)
        self.solver = RptStarRoomSolver(rng=random.Random("%s/%s" % (s.seed, episode.episode_id)), max_rooms=s.max_rooms)
        self.planner = WeightedAStarPlanner2D(self.planner_params)
        # Every planner of the method blocks UNKNOWN cells (``unknown_blocked``): a route is walked on observed
        # floor or not at all. Until 2026-11-15 the closing kept one more, with unknown passable at a price, to
        # reach a standoff the simulator's navmesh had vouched for; with the navmesh gone nothing vouches for
        # floor the agent has not seen (``world_model/standing_floor.py``), and that planner went with it.
        # The exploration fallback's squeeze rung (since 2026-10-25): the same planner relaxing to the BODY radius,
        # for the frontier goals nothing reaches at ``min_clearance_m`` -- a doorway the map draws 0.4 m wide.
        self.squeeze_planner = WeightedAStarPlanner2D(replace(self.planner_params, inflate_floor_m=s.body_radius_m))
        # The closing's second planner for a standing point (its own since 2026-11-16): the same planner relaxing
        # to ``TargetClosingSettings.standing_retry_clearance_m``, asked when the default one (0.25 m) has no
        # route. By default that is the body radius (0.18 m), as the squeeze planner above: 0.20 m was measured
        # against it that day and gave the lower SR and SPL (see the setting).
        self.standing_retry_planner = WeightedAStarPlanner2D(
            replace(self.planner_params, inflate_floor_m=self.standing_retry_clearance_m()))
        self.route_memory = CommittedRoute(episode.action_spec, self.converter_params, self.route_settings)
        self.fallback = self.fallback_type(self, self.fallback_settings)
        # The fallback's rung before its idle turn: boxed in on the inflated map, one
        # probing step through the thinnest barrier -- the simulator says what the body passes.
        self.escape = EscapeProbe(self, self.escape_settings)
        # Where the camera has looked and what it looked through without a
        # return, per floor: settled unknown is not frontier, and the poses say
        # which rooms were walked through or looked into.
        self.sight = SightLedger(self, self.sight_settings)
        self._begin_search()
        self._last_plan_s = self._blocked_since = self._last_pose = None
        self._plan_step = -s.replan_steps
        self._blocked = self._plan_calls = self._duplicates_removed = 0
        self._goal_bumps = {}             # goal cell -> actions of the refused steps toward it (``_bumped_toward``)
        self._goals_retired_by_bumps = 0
        self._solver_records = []
        self.last_world = None
        self.telemetry = ExplorationMetrics()
        self.camera_control = CameraController(episode.action_spec)
        # The narrow-passage layer (2026-10-28): the lattice follower that walks every route on the benchmark's
        # own step and turn, the guard that keeps a doorway scrape off the map, the look-down at a bottleneck.
        self.kinematic = KinematicFollower(self, s.kinematic)
        self.scrape_guard = DoorwayScrapeGuard(self, s.doorway_scrape)
        self.look_down = BottleneckLookDown(self, s.doorway_look_down)
        self.perception = self.perception_type(self)
        self.sweep = FrontierSweep(self, self.sweep_settings)
        self.hierarchy = None
        if s.local_exploration == "falcon":
            from sparx_agency.core.planning.object_search.falcon.policy import FalconObjectNav
            self.hierarchy = FalconObjectNav(self)
        self.building = None
        if s.multifloor.enabled:
            from sparx_agency.core.planning.object_search.multifloor.multifloor_policy import MultiFloorSearch
            self.building = MultiFloorSearch(self, confinement_only=not self.uses_room_loop)
        # Spawn-floor confinement: inert when stair traversal is allowed.
        self.floor_guard = SpawnFloorGuard(self)
        # What changed on the planning grid each action, by layer and in or out of the camera's view (map_ledger.py).
        self.map_ledger = MapChangeLedger()
        self.closing = TargetClosing(self, target_only=not self.uses_room_loop)
        # Where along the route in force a look to the side (or all round) is worth
        # its actions; performed as a suspended phase when the agent gets there.
        self.glances = GlanceScheduler(self, self.glance_settings)
        # One look-around is as many turns as the camera's field of view needs to see the
        # whole circle (ten at 30 degrees with a 79-degree field), bounded by ``warmup_steps``.
        self.warmup_turns = self.full_rotation_turns
        self.warmup_actions = 0
        self._warmup_pending = True       # the warm-up rotation's scan has not been recorded yet
        self._action_owner = "search"
        self._notified_step = self._decision_step = -1
        self._decision_command = None

    @property
    def full_rotation_turns(self):
        return min(self.settings.warmup_steps, self.rotation_turns())

    def start_rotation(self, turns):
        self.warmup_turns = turns
        self.warmup_actions = 0

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
        # The refused step is a pose the lattice may not step forward from again, whatever else it is; and
        # inside a doorway, nearly aligned with it, it is a scrape the map is not told about (``scrape_guard``).
        self.kinematic.note_refused(observation)
        scrape = self.scrape_guard.judge(observation, self.last_world, self._route)
        if self.closing.active:
            self.closing.note_blocked()
            if scrape is None:
                self.mapping.notify_blocked(observation, self.episode.action_spec.forward_step_m)
            self._blocked += 1
            self.route_memory.clear("target_forward_blocked")
            self._route = self._goal = None
            return
        if self.building and self.building.traversing:
            self.building.blocked(observation)
            if self.building.transition is not None:
                self._blocked += 1
                return
            # No traversal of ours (the atlas alone reports the departure): the probe's memory and the
            # bump marks apply as on the storey, or the escape rung pushes into one wall for the rest of
            # the episode (00873/000002, 2026-10-25: 473 refused steps at one cell).
        self._blocked += 1
        if scrape is None:
            self.mapping.notify_blocked(observation, self.episode.action_spec.forward_step_m)
            self.escape.note_blocked(observation)
            if self._goal is not None and not self._action_owner.startswith("doorway_peek"):
                self._bumped_toward(observation, self._goal, self.route_memory.kind)
        if self._blocked_since is None:
            self._blocked_since = self._floor_time
        self.route_memory.clear("forward_blocked")
        self._route, self._goal = None, None
        if self.hierarchy is not None and self._action_owner not in self.suspended_phases:
            self.hierarchy.blocked()

    def _bumped_toward(self, observation, goal, kind):
        """A refused forward step on the way to ``goal``: the ROUTE is given up at once, the GOAL only after repeated bumps.

        Until 2026-10-28 one refused step retired the goal itself
        (``_visited_frontiers``), and with it every frontier within the
        sweep's ``visited_radius_m`` of it for the next hundred retirements.
        A doorway's jamb clipped by a 30-degree heading on the way to the
        exit behind it retired the exit; the next exit through the same
        doorway clipped the other jamb and was retired too, and the wing
        behind the door was gone from rung 1 of the fallback for good. The
        GT-oracle sweep of 2026-10-26 (2000 episodes) made the cost
        measurable: same-storey success ran 97 % with no refused step, 82 %
        with three to five and 30 % with eleven to twenty. The bump is still
        written on the map (``ObservedMap.notify_blocked``) and the route
        re-planned round it; the goal is retired when
        ``RPTSettings.goal_bump_retire`` refused steps toward the same place
        fall within ``goal_bump_window`` actions -- the place, not the way.
        Only a frontier goal is retired (the sweep's, the fallback's, a
        room's frontier entry), as in :meth:`_navigate`: a standoff or a
        vantage point has its owner's own stall rules.
        """
        s = self.settings
        if not self._frontier_route(kind):
            return
        key = (round(float(goal[0]) / s.goal_bump_match_m), round(float(goal[1]) / s.goal_bump_match_m))
        step = int(observation.step)
        steps = [t for t in self._goal_bumps.get(key, ()) if step - t <= s.goal_bump_window]
        steps.append(step)
        if len(steps) >= s.goal_bump_retire:
            self._visited_frontiers.append(tuple(goal))
            self._goal_bumps.pop(key, None)
            self._goals_retired_by_bumps += 1
            return
        self._goal_bumps[key] = steps

    def plan(self, observation):
        if observation.step == self._decision_step:
            return self._decision_command
        self._action_owner = "search"
        started = time.monotonic()
        try:
            if self.look_down.refreshed(observation):
                # The first frame with the pitched view: the doorway's entry is re-planned on it.
                self.kinematic.invalidate("look_down_refresh")
            command = self._plan(observation)
            # The look-down applies to a walk of the search's own: not the takeover's approach, not a stair
            # transaction, not a sweep or a glance turned in place.
            walking = (bool(command.waypoints) and not self.closing.active and self._action_owner == "search"
                       and not (self.building and self.building.phase != "SEARCH")
                       and self.camera_control.inspection is None)
            self.camera_control.bottleneck_pitch_deg = self.look_down.pitch_request(
                observation, self.last_world, command.waypoints if walking else None)
            idle_phase = self.supervisor.state if self.uses_room_loop else "search"
            phase = self.hierarchy.machine.phase if self.hierarchy else str(command.info.get("kind", idle_phase))
            if self._action_owner in self.suspended_phases:
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
        if self._action_owner in self.suspended_phases:
            return action  # A* already qualified this route; do not apply the suspended FALCON room mask.
        return self.hierarchy.filter_action(observation, action) if self.hierarchy else action

    def notify_action(self, observation, action):
        if observation.step == self._notified_step:
            return
        if observation.step < self._notified_step:
            raise ValueError("Action ledger regressed")
        self._notified_step = observation.step
        self.kinematic.note_action(observation, action)
        if self.closing.active:
            self.closing.note_action(action)
            self.telemetry.emitted(observation, action, self.telemetry.phase)
            return
        external = self._action_owner in self.suspended_phases
        # The closing's own release turn: decided inside ``closing.plan`` with ``active``
        # already False by now; the search did not take it, so the loop is not charged.
        released = self._action_owner == "target_closing"
        if not self.building or not self.building.committed:
            self._floor_time += self.settings.action_time_s
            if external or released:
                if self.uses_room_loop:
                    self.supervisor.pause(self.settings.action_time_s)
                # The blocked clock pauses with the supervisor's deadlines: a glance or a peek's
                # look turned in place after a blocked step is not the follower still blocked.
                # Corozal/000142 (``51eff3fa1637``, 2026-10-13): one blocked step, a ten-turn
                # glance, and the next four transits were released ``blocked`` one action in.
                if self._blocked_since is not None:
                    self._blocked_since += self.settings.action_time_s
            elif self.uses_room_loop and self.hierarchy is None:
                self.loop.charge()
        warmup_action = external if self.uses_room_loop else (
            self._action_owner == "warmup" and action in (DiscreteAction.TURN_LEFT, DiscreteAction.TURN_RIGHT))
        if warmup_action and self.warmup_actions < self.warmup_turns:
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
        if self.frame_log is not None:
            # Before perception, mapping or the decision: the log holds every frame a decision was taken on, and a
            # frame that cannot be kept ends the run (``FrameLogError``) instead of leaving a hole in the memory.
            self.frame_log.add(observation)
        self.perception.observe(observation)
        self.closing.observe(observation)
        if not self.uses_room_loop and self.building:
            self.building.prepare_observation(observation)  # also measure treads while closing or following an RGB bearing
        if self.closing.active:
            if self.uses_room_loop:
                self.pursuit.notify_takeover(observation)
            self._action_owner = "target_closing"
            # No global decision/fallback try block may catch closing failures.
            # Keep the current-floor collision map fresh without advancing any
            # building/room task or consuming privileged goal data.
            try:
                revision = self.mapping.floor_revision
                world = self.mapping.update(observation, arrival_allowed=False)
                if not self.uses_room_loop and self.building:
                    self.building.observe(observation)
                world = self._confine(observation, world)
                if not self.uses_room_loop and revision != self.mapping.floor_revision:
                    self._change_floor(observation)
                self.last_world = world
                self.sight.observe(observation, world)
                self._observe_search(observation, world)
                # The landmark map keeps voting while the takeover walks in (the legacy
                # target evidence stands down, see ``PerceptionCycle.fuse``): the map
                # release -- the sofa from four metres that is the bed from two -- needs
                # the closer frames on the map, and until 2026-10-05 nothing fed them.
                if self.uses_room_loop:
                    self._perceive(observation)
                return self.closing.plan(observation, world)
            except Exception as exc:
                self.closing.phase = "FAILED"
                if self.closing.failure is None:
                    self.closing.failure = "%s: %s" % (type(exc).__name__, exc)
                raise
        if self.building and self.uses_room_loop:
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
            self._change_floor(observation)
        self.last_world = world
        if self.building:
            self.building.observe(observation)
        if not self.building or not self.building.traversing:
            self.telemetry.observe(observation, world, self.mapping.floor_id)
            self.sight.observe(observation, world)
        confirmed = self._perceive(observation) if self.uses_room_loop else False
        self._observe_search(observation, world)
        try:
            return self._decide(observation, world, confirmed)
        except Exception as exc:  # A*, RPT* or a bug in the decision: keep exploring, loudly
            if self.uses_room_loop:
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
            if self.uses_room_loop and not self.building.traversing:
                self._refresh_graph(observation, world)
            if self.uses_room_loop:
                self.peek.cancel(observation, "stairs_priority", restore=False)
            command = self.building.plan(observation, world)
            if command is not None:
                return command
        inspection = self.camera_control.inspection_command(observation)
        if inspection is not None:
            return inspection
        return self._search(observation, world, confirmed)

    def navigation_cost(self, world):
        """The common clearance-qualified map used by counts, entries and routes (passable at the planner's floor standoff).

        Cached on the planner's distance fields for the map in force: the
        policy, the loop and the fallback all ask for it on one action, the
        map is rebuilt once per action, and ``planner.invalidate_cache()``
        drops this with the fields.
        """
        cached = getattr(self, "_navigation_cost_cache", None)
        fields = self.planner.fields_for(world)
        if cached is not None and cached[0] is fields:
            return cached[1]
        cost = assemble_cost_grid(fields, self.planner_params, self.planner_params.inflate_floor_m)[0]
        self._navigation_cost_cache = (fields, cost)
        return cost

    def preferred_cost(self, world):
        """The cost map at the planner's PREFERRED standoff -- what A* flies wherever it can.

        Distances for the frontier inventory and the RPT* instance are
        measured on this where it reaches and on :meth:`navigation_cost`
        otherwise, so a node's charged distance is the route the planner
        will take, not the squeeze it would refuse (cached by the planner).
        """
        return self.planner.cost_for(world)[0]

    def _confine(self, observation, world):
        """The world every decision plans on: seen stairs and observed drops impassable while traversal is forbidden, the trail free.

        Stored back as the floor's world so the display panels, the peek
        copy and the terrain all read the same map; the observed log-odds
        grid underneath is untouched and rebuilds the world next action.
        """
        original = world
        plane_before = getattr(self.mapping, "_anchor", None)
        world = self._walked_passable(observation, world)
        confined = self.floor_guard.observe(observation, world)
        layers = dict(self.floor_guard.layers)
        layers["bump"] = self._bump_bars(original)
        self.map_ledger.record(observation, self.mapping.floor_id, original, confined,
                               mask=self.floor_guard.mask, freed=self.floor_guard.freed, layers=layers,
                               plane_m=getattr(self.mapping, "_anchor", None), plane_before_m=plane_before)
        if confined is not original:
            self.mapping.worlds[self.mapping.floor_id] = confined
        return confined

    def _bump_bars(self, world):
        """``(H, W)`` bool of the observed map's sticky bump bars on this floor, for the ledger's attribution."""
        touched = getattr(self.mapping, "_touched", {}).get(self.mapping.floor_id) or {}
        if not touched:
            return None
        bars = np.zeros(world.grid.shape, dtype=bool)
        for tx, ty in touched:
            if 0 <= ty < bars.shape[0] and 0 <= tx < bars.shape[1]:
                bars[ty, tx] = True
        return bars

    def _walked_passable(self, observation, world):
        """Cells the body has stood on this storey are free for planning, whatever a depth frame painted there.

        The observed map clears the footprint under the agent on every
        action except where the cell already reads occupied, and a forward
        step the simulator refused writes an occupied cell a quarter metre
        ahead -- on the very cells the agent just walked when, as in HM3D
        without sliding, the refusal was a lateral touch and not a wall
        ahead (00824/000066, 2026-10-25: the agent's own bump marks fenced
        it into the pocket it had entered on foot, 460 actions). The trail
        (the sight ledger's poses, one cell about each) is the one fact no
        ray or bump outvotes. The sticky bars of repeated bumps lie two to
        three cells ahead of where the body stood and are untouched unless
        the body later stands on them -- in which case they were wrong.
        """
        sight = getattr(self, "sight", None)
        if sight is None:
            return world
        occupied = world.grid == world.values.occupied
        if not occupied.any():
            return world
        walked = sight.walked_mask(world, self.mapping.floor_id, world.resolution,
                                   include=[(float(observation.pose.x), float(observation.pose.y))])
        clear = walked & occupied
        if not clear.any():
            return world
        grid = world.grid.copy()
        grid[clear] = world.values.free
        return OccupancyGrid2D(grid, world.params, values=world.values)

    def _navigate(self, observation, world, goal, kind, final_yaw=None, planner=None, retire=True, exact_goal=False):
        """A follow command along a committed route to ``goal``, or None when the planner has none.

        ``exact_goal`` asks the planner not to move the goal (``PlanRequest.options``): the route ends on the
        goal's own cell or there is no route. The closing passes it for every standing point it chose on the
        observed map (``world_model/standing_floor.py``), which the planner's goal snap would otherwise trade
        for a cell nobody checked.
        """
        planner = self.planner if planner is None else planner
        # A goal nothing reaches is retired from the frontier memory when it IS a frontier (the
        # sweep's and the fallback's goals, a room's frontier entry). Until 2026-10-28 every kind
        # was: the closing's standoff ring alone asked for up to 72 routes per action, and their
        # refusals flushed the hundred-goal memory and hid every frontier within 0.4 m of the ring.
        # ``retire=False`` when this refusal is not the last word on the goal (the transit asks the
        # squeeze planner next, ``RoomSearchLoop._navigate_transit``).
        retiring = retire and self._frontier_route(kind)
        if not self.floor_guard.goal_allowed(observation, world, goal, kind):
            if retiring:
                self._visited_frontiers.append(tuple(goal))
            self.route_memory.clear("goal off the spawn floor")
            self._route = self._goal = None
            return None
        if not self.route_memory.reusable(observation, world, planner, goal, kind):
            if self.route_memory.reason == "no_progress":
                if retiring:
                    self._visited_frontiers.append(tuple(goal))
                self._route = self._goal = None
                self._blocked_since = self._floor_time
                return None
            # Keyword arguments only when they say something: a bare ``_plan_to(obs, world, goal)`` is the
            # call the regression rig's planner stubs were written against.
            options = {} if planner is self.planner else {"planner": planner}
            if exact_goal:
                options["exact_goal"] = True
            path = self._plan_to(observation, world, goal, **options)
            if path is None:
                if retiring:
                    self._visited_frontiers.append(tuple(goal))
                self._route = self._goal = None
                return None
            self.route_memory.adopt(path, goal, kind, observation)
        self._route, self._goal = self.route_memory.path, self.route_memory.goal
        info = {"route": self.route_memory.reason, "kind": kind}
        if planner is not self.planner:
            info["squeeze"] = True
        if self.route_memory.reason == "new_goal_or_invalid_route":
            info["route_replaced"] = self.route_memory.replaced
        command = NavigationCommand.follow(self._route, final_yaw=final_yaw, info=info)
        return self.kinematic.command(observation, world, planner, command)

    @staticmethod
    def _frontier_route(kind):
        return kind == "frontier" or (kind or "").startswith(("frontier/", "transit/"))

    def _plan_to(self, observation, world, goal, planner=None, exact_goal=False):
        p = observation.pose
        planner = self.planner if planner is None else planner
        self._plan_calls += 1
        started = time.monotonic()
        request = PlanRequest(Pose2D(p.x, p.y, p.yaw), Pose2D(*goal), frame_id="world",
                              options={"exact_goal": True} if exact_goal else {})
        result = planner.plan(request, world)
        self.telemetry.latencies["astar"].append((time.monotonic() - started) * 1000)
        self._plan_step = observation.step
        if not result.ok:
            return None
        self._last_plan_s = self._floor_time
        return result.path

    def _llm_answer_totals(self):
        stats = getattr(self.model_client, "answer_stats", None)
        return stats() if stats is not None else None

    def _llm_answer_counts(self):
        """This episode's LLM answers by origin (``llm_answer_cache``) when the client keeps a store; else nothing."""
        now = self._llm_answer_totals()
        if now is None:
            return {}
        then = getattr(self, "_llm_answers_at_reset", None) or {}
        return {"llm_answers": {k: v - then.get(k, 0) for k, v in now.items()}}


class RPTSearchPolicy(_ObservedSearchPolicy):
    """ObjectNav with independent visual frontier probabilities, RPT* ordering and observed-floor A*."""

    name = "sparx-rpt-vlm-frontiers"
    variant = "rpt"
    uses_room_loop = False
    suspended_phases = ("warmup", "glance")
    fallback_type = GeometricFallback
    perception_type = TargetStairCycle

    def __init__(self, detector, vlm_client, settings=None, segmenter=None, frame_log=None):
        settings = settings or RPTSettings()
        settings.validate_frontier_method()
        settings = replace(settings, target_closing=replace(settings.target_closing, support_surface_resilience=False),
                           doorway_scrape=replace(settings.doorway_scrape, doorways_only=False))
        if frame_log is None:
            frame_log = EpisodeFrameLog(MemoryFrameStore())
        super().__init__(detector, settings, segmenter=segmenter, frame_log=frame_log)
        self.vlm_client = vlm_client

    @property
    def model_client(self):
        return self.vlm_client

    def _begin_search(self):
        configure = getattr(self.detector, "set_episode_prompts", None)
        if configure is not None:
            configure(target_stair_prompts(self.target))
        self.frontiers = FrontierSearch(self)
        self._retired_by_floor = {}
        self._change_floor(None)

    def _plan(self, observation):
        # The Habitat adapter uses +inf for a clipped far pixel. Don't let another
        # adapter turn saturation into an occupied wall or a measured target.
        far = np.isfinite(observation.depth_m) & (observation.depth_m >= observation.camera.max_depth_m)
        if np.any(far):
            depth = observation.depth_m.copy()
            depth[far] = np.inf
            observation = replace(observation, depth_m=depth)
        return super()._plan(observation)

    def _change_floor(self, observation):
        floor = self.mapping.floor_id
        if floor not in self._retired_by_floor:
            self.start_rotation(self.full_rotation_turns)
        self._visited_frontiers = self._retired_by_floor.setdefault(floor, [])
        self._goal_bumps = {}
        self._floor_time = 0.0
        self._route = self._goal = self._last_pose = self._blocked_since = None
        self.route_memory.clear("floor_reset")

    def _observe_search(self, observation, world):
        # Observation never owns motion: in particular an unavailable score must not preempt closing or warm-up.
        try:
            self.frontiers.refresh(observation, world, on_floor=not self.mapping.atlas.in_transition)
        except ObjNavInternalError:
            raise
        except (ValueError, ArithmeticError, RuntimeError) as exc:
            self.frontiers.nodes = {}
            self.frontiers.blocked_cells.clear()
            self.frontiers.inventory = None
            self.fallback.record_failure(observation, "frontier_observation", exc)

    def _search(self, observation, world, confirmed):
        command = rotation_command(self, observation)
        if command is not None:
            return command
        if self.glances.active is not None:
            command = self.glances.continue_look(observation, world)
            if command is not None:
                return command
        if self.building is not None and not self.building.committed:
            inspection = self.building.plan(observation, world)
            if inspection is not None:
                return inspection
        return self.glances.apply(observation, world, self._explore(observation, world))

    def _explore(self, observation, world):
        command = self.frontiers.plan(observation, world)
        if command is not None:
            return command
        return self.fallback.plan(observation, world, reason="no reachable unvisited VLM-scored frontier")

    def filter_action(self, observation, action):
        # Direct closing/inspection steps do not always have an A* path. The measured
        # stair/drop exclusion applies to them too. A trail retreat returns to the storey.
        retreat = self.building and self.building.retreating and not self.closing.active
        if action == DiscreteAction.MOVE_FORWARD and not retreat:
            world, guard = self.last_world, self.floor_guard
            pose, step = observation.pose, self.episode.action_spec.forward_step_m
            start = world.world_to_grid(pose.x, pose.y)
            end = world.world_to_grid(pose.x + step * math.cos(pose.yaw), pose.y + step * math.sin(pose.yaw))
            if (guard.off_plane or self.mapping.atlas.in_transition or not world.in_bounds(*end) or
                    (guard.mask is not None and not line_of_sight_clear(guard.mask, *start, *end))):
                return DiscreteAction.TURN_LEFT
        return action

    def configuration(self):
        return {**super().configuration(), "frame_log": self.frame_log.describe(),
                "detector_scope": "episode target aliases + stairs", "floor_change_decision": "none: avoidance only",
                "perception_fusion": "target-only depth in TargetClosing; RGB-only bearing until depth is valid",
                "frontier_search": {"nodes": asdict(NodeSettings()), "views": asdict(ViewSettings()),
                                    "prompt": PROMPT_VERSION, "valuation": "one independent probability per node"}}

    def episode_info(self):
        return {"method": self.name, "pipeline_variant": self.variant, "frontier_search": self.frontiers.diagnostics(),
                "detector_prompts": list(target_stair_prompts(self.target)),
                "solver_records": self._solver_records, "plan_calls": self._plan_calls, **self._llm_answer_counts(),
                "warmup": {"actions": self.warmup_actions, "budget": self.warmup_turns, "bound": self.settings.warmup_steps},
                "target_closing": self.closing.diagnostics(), "perception": self.perception.diagnostics(),
                "exploration_fallback": self.fallback.diagnostics(), "exploration_metrics": self.telemetry.report(),
                "building": self.building.diagnostics() if self.building else None,
                "spawn_floor_guard": self.floor_guard.diagnostics(), "floor_maps": self.mapping.integrity(),
                "map_changes": self.map_ledger.diagnostics(), "route_commitment": dict(self.route_memory.stats),
                "kinematic": self.kinematic.diagnostics(), "doorway_scrape": self.scrape_guard.diagnostics(),
                "doorway_look_down": self.look_down.diagnostics(), "glances": self.glances.diagnostics(),
                "blocked": self._blocked, "goals_retired_by_bumps": self._goals_retired_by_bumps}
