"""Observed-only scene-graph/LLM/RPT* search for synchronous ObjectNav steps.

Uses existing core algorithms and host frontier sweeps, not privileged goals
or a second simulator policy. Door boundaries and room labels are revisable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random

import numpy as np

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.mapping.objects.landmarks import ObjectLandmarkMap
from sparx_agency.core.planning.exploration.object_search_supervisor import (
    ObjectSearchParams, ObjectSearchSupervisor, SEARCH, TRANSIT)
from sparx_agency.core.planning.exploration.room_costs import build_instance, in_room_frontier_goals
from sparx_agency.core.planning.exploration.rpt_room_solver import RptStarRoomSolver
from sparx_agency.core.planning.interfaces.planner import PlanRequest
from sparx_agency.core.planning.objnav.action_converter.params import ActionConverterParams
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.planners.astar.params import WeightedAStarParams
from sparx_agency.core.planning.planners.astar.weighted_planner_2d import WeightedAStarPlanner2D
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.observed_map import ObservedMap
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.perception import observed_objects
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.scene_graph import ObservedSceneGraph, DEFAULT_SEGMENTATION
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.doors import DOOR_LABELS, DoorSettings, ObservedDoors
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.room_labels import RoomLabelSettings


@dataclass(frozen=True)
class RPTSettings:
    """ObjectNav adaptation knobs; serialised before a run."""

    map_size_m: float = 80.0
    map_resolution_m: float = 0.1
    depth_stride: int = 8
    graph_period_steps: int = 10
    replan_steps: int = 5
    action_time_s: float = 1.0
    detection_confidence: float = 0.35
    stop_distance_m: float = 0.75
    target_memory_steps: int = 30
    seed: int = 0
    max_rooms: int = 12

    def __post_init__(self):
        for key in ("map_size_m", "map_resolution_m", "action_time_s", "stop_distance_m"):
            value = getattr(self, key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be positive and finite" % key)
        for key in ("depth_stride", "graph_period_steps", "replan_steps", "target_memory_steps", "max_rooms"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError("%s must be a positive integer" % key)
        if not 0 <= self.detection_confidence <= 1 or type(self.seed) is not int:
            raise ValueError("Invalid confidence or seed")


class RPTSearchPolicy:
    """Observed-only search using the existing RPT supervisor and room reasoning."""

    name = "sparx-rpt-llm-host-sweep"

    def __init__(self, detector, llm_client, settings=None):
        self.detector, self.llm_client = detector, llm_client
        self.settings = settings or RPTSettings()
        self.converter_params = ActionConverterParams()
        self.planner_params = WeightedAStarParams(
            inflate_radius_m=0.3, inflate_floor_m=0.18,
            unknown_blocked=True, goal_snap_radius_m=0.3,
            waypoint_spacing_m=0.25, start_skip_m=0.0,
            search_margin_m=self.settings.map_size_m)
        self.supervisor_params = ObjectSearchParams(seed=self.settings.seed)
        self.door_settings = DoorSettings()
        self.room_label_settings = RoomLabelSettings()

    def configuration(self):
        return {"method": self.name, "adaptation": asdict(self.settings),
                "planner": asdict(self.planner_params), "converter": asdict(self.converter_params),
                "supervisor": asdict(self.supervisor_params), "doors": asdict(self.door_settings),
                "room_labels": asdict(self.room_label_settings), "room_segmentation": asdict(DEFAULT_SEGMENTATION),
                "local_exploration": "observed-map host frontier sweep",
                "ground_truth_semantics": False, "training_free": True, "non_metric": False,
                "clock": "action_index * action_time_s"}

    def reset(self, episode, target):
        self.episode, self.target = episode, target
        s = self.settings
        self.mapping = ObservedMap(s.map_size_m, s.map_resolution_m, s.depth_stride)
        self.graph = ObservedSceneGraph(self.llm_client, label_settings=self.room_label_settings)
        self.doors = ObservedDoors(self.door_settings)
        self.landmarks = ObjectLandmarkMap()
        self.solver = RptStarRoomSolver(rng=random.Random("%s/%s" % (s.seed, episode.episode_id)), max_rooms=s.max_rooms)
        self.supervisor = ObjectSearchSupervisor(self.supervisor_params, solver=self.solver)
        self.planner = WeightedAStarPlanner2D(self.planner_params)
        self._route = self._goal = self._target_xy = None
        self._target_step = -s.target_memory_steps - 1
        self._last_plan_s = self._blocked_since = None
        self._last_graph_step = -s.graph_period_steps
        self._last_door_revision = 0
        self._plan_step = -s.replan_steps
        self._visited_frontiers = []
        self._blocked = 0
        self._solver_records = []
        self.last_world = None

    def notify_blocked(self, observation):
        self.mapping.notify_blocked(observation, self.episode.action_spec.forward_step_m)
        self._blocked += 1
        if self._blocked_since is None:
            self._blocked_since = observation.step * self.settings.action_time_s
        if self._goal is not None:
            self._visited_frontiers.append(self._goal)
        self._route, self._goal = None, None

    def plan(self, observation):
        s, pose = self.settings, observation.pose
        now = observation.step * s.action_time_s
        world = self.mapping.update(observation)
        self.last_world = world
        self.graph.credit_time(world, pose, 0.0 if observation.step == 0 else s.action_time_s)
        visible = self._perceive(observation)
        if visible and self._target_xy is not None and math.dist((pose.x, pose.y), self._target_xy) <= s.stop_distance_m:
            return NavigationCommand.stop_here(info={"reason": "confirmed predicted target in current view"})
        if (observation.step - self._last_graph_step >= s.graph_period_steps
                or self.doors.revision != self._last_door_revision):
            partition = self.graph.partition_revision
            self.graph.update(world, self.landmarks.confirmed(), self.target,
                              doors=self.doors.confirmed(), step=observation.step)
            if self.graph.partition_revision != partition:
                self.supervisor = ObjectSearchSupervisor(self.supervisor_params, solver=self.solver)
                self._route, self._goal = None, None
                self._visited_frontiers = []
            self._last_graph_step = observation.step
            self._last_door_revision = self.doors.revision
        cost = self.planner.cost_for(world)[0]
        if self._target_xy is not None and observation.step - self._target_step <= s.target_memory_steps:
            command = self._approach(observation, world)
            if command is not None:
                return command
        instance = None
        if self.graph.options and self.supervisor.state == "select":
            instance, _ = build_instance(
                world, cost, {r.room_id: r.xy for r in self.graph.options}, self.graph.probs,
                depot_xy=(pose.x, pose.y), cruise_speed_mps=self.episode.action_spec.forward_step_m / s.action_time_s)
        calls = self.solver.calls
        state = self.supervisor.update(
            self.graph.options, self.graph.facts, (pose.x, pose.y), now,
            last_plan_s=self._last_plan_s, instance=instance, blocked_since=self._blocked_since)
        if self.solver.calls != calls:
            report = asdict(self.solver.last)
            for key, value in report.items():
                if isinstance(value, float) and not math.isfinite(value):
                    report[key] = None  # undefined bounds must not erase the episode diagnostics
            self._solver_records.append(report)
        if state.changed or state.completed:
            self._route, self._goal = None, None
        if state.state == TRANSIT and state.goal_xy is not None:
            goal = state.goal_xy
        else:
            room = self.graph.registry.rooms.get(state.room_id)
            mask = room.mask if state.state == SEARCH and room is not None else np.ones(world.grid.shape, bool)
            goal = self._frontier(observation, world, cost, mask)
        if goal is None:
            return NavigationCommand.hold(info={"reason": "scan for more observed free space"})
        if self._route is None or goal != self._goal or observation.step - self._plan_step >= s.replan_steps:
            self._route = self._plan_to(observation, world, goal)
            self._goal = goal
        if self._route is None:
            self._visited_frontiers.append(goal)
            return NavigationCommand.hold(info={"reason": "no observed collision-free route"})
        return NavigationCommand.follow(self._route, info={"state": state.state})

    def _perceive(self, observation):
        visible = False
        seen = []
        detections = self.detector.detect(observation.rgb)
        self.doors.update(observation, detections)
        for label, xyz in observed_objects(observation, detections, self.settings.detection_confidence):
            if label in DOOR_LABELS:
                continue
            xy = (float(xyz[0]), float(xyz[1]))
            if any(label == old and math.dist(xy, pos) < 0.7 for old, pos in seen):
                continue
            seen.append((label, xy))
            landmark = self.landmarks.observe(label, xy)
            if self.target.accepts(label) and landmark.count >= 2:
                self._target_xy, self._target_step = landmark.xy, observation.step
                visible = True
        return visible

    def _approach(self, observation, world):
        p = observation.pose
        dx, dy = self._target_xy[0] - p.x, self._target_xy[1] - p.y
        distance = math.hypot(dx, dy)
        if distance <= self.settings.stop_distance_m:
            return NavigationCommand.hold(final_yaw=math.atan2(dy, dx))
        margin = self.converter_params.goal_tolerance_m + world.resolution
        standoff = max(0.0, self.settings.stop_distance_m - margin)
        scale = (distance - standoff) / distance
        goal = (p.x + dx * scale, p.y + dy * scale)
        route = self._plan_to(observation, world, goal)
        if route is not None:
            return NavigationCommand.follow(route, final_yaw=math.atan2(dy, dx))
        return None

    def _frontier(self, observation, world, cost, mask):
        xy = (observation.pose.x, observation.pose.y)
        arrival = self.converter_params.goal_tolerance_m + world.resolution
        if self._goal is not None and math.dist(xy, self._goal) <= arrival:
            self._visited_frontiers.append(self._goal)
            self._goal, self._route = None, None
        goals = in_room_frontier_goals(world, cost, mask)
        for goal in goals:
            if math.dist(xy, goal) > arrival and not any(math.dist(goal, used) < 0.4 for used in self._visited_frontiers[-100:]):
                return goal
        return None

    def _plan_to(self, observation, world, goal):
        p = observation.pose
        result = self.planner.plan(PlanRequest(Pose2D(p.x, p.y, p.yaw), Pose2D(*goal), frame_id="world"), world)
        self._plan_step = observation.step
        if not result.ok:
            return None
        self._last_plan_s = observation.step * self.settings.action_time_s
        self._blocked_since = None
        return result.path

    def episode_info(self):
        return {"method": self.name, "rooms": len(self.graph.registry.rooms),
                "landmarks": len(self.landmarks), "llm_queries": self.graph.queries,
                "supervisor": dict(self.supervisor.stats), "solver_records": self._solver_records,
                "blocked": self._blocked, "doors": self.doors.diagnostics(), "max_rooms": self.graph.max_rooms,
                "room_label_queries": self.graph.label_tracker.queries,
                "room_label_history": list(self.graph.label_tracker.history),
                "last_reasoning": self.graph.last_reasoning}

