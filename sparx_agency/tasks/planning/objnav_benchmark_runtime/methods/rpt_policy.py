"""Observed-only ObjectNav with persistent floors and frontier or planar FALCON."""
from __future__ import annotations

from dataclasses import asdict, replace
import math
import random
import time
import numpy as np

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.exploration.object_search_supervisor import ObjectSearchParams, SEARCH, TRANSIT
from sparx_agency.core.planning.exploration.room_costs import build_instance, in_room_frontier_goals
from sparx_agency.core.planning.exploration.rpt_room_solver import RptStarRoomSolver
from sparx_agency.core.planning.exploration.falcon.params import SOURCE as FALCON_SOURCE
from sparx_agency.core.planning.interfaces.planner import PlanRequest
from sparx_agency.core.planning.objnav.action_converter.params import ActionConverterParams
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.planners.astar.params import WeightedAStarParams
from sparx_agency.core.planning.planners.astar.cost_grid_2d import assemble_cost_grid
from sparx_agency.core.planning.planners.astar.weighted_planner_2d import WeightedAStarPlanner2D
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.observed_map import ObservedMap
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.scene_graph import DEFAULT_SEGMENTATION
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.doors import DoorSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.room_labels import RoomLabelSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.route_memory import CommittedRoute, RouteSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.object_evidence import TargetEvidenceSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.floor_context import FloorContextBank
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.rpt_settings import RPTSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.exploration_metrics import ExplorationMetrics
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.camera_control import CameraController, CameraControlSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.perception_cycle import PerceptionCycle


class RPTSearchPolicy:
    name = "sparx-rpt-llm-host-sweep"

    def __init__(self, detector, llm_client, settings=None):
        self.detector, self.llm_client = detector, llm_client
        self.settings = settings or RPTSettings()
        s = self.settings
        if s.local_exploration == "falcon":
            self.name = "sparx-rpt-llm-falcon-planar"
        self.converter_params = ActionConverterParams()
        self.planner_params = WeightedAStarParams(
            inflate_radius_m=s.preferred_clearance_m, inflate_floor_m=s.body_radius_m,
            unknown_blocked=True, goal_snap_radius_m=0.3, waypoint_spacing_m=0.25,
            start_skip_m=0.0, search_margin_m=s.map_size_m)
        self.supervisor_params = ObjectSearchParams(seed=s.seed)
        self.door_settings, self.room_label_settings = DoorSettings(), RoomLabelSettings()
        self.route_settings, self.target_settings = RouteSettings(), TargetEvidenceSettings()

    def configuration(self):
        return {"method": self.name, "adaptation": asdict(self.settings),
                "planner": asdict(self.planner_params), "converter": asdict(self.converter_params),
                "supervisor": asdict(self.supervisor_params), "doors": asdict(self.door_settings),
                "room_labels": asdict(self.room_label_settings), "room_segmentation": asdict(DEFAULT_SEGMENTATION),
                "route_commitment": asdict(self.route_settings), "target_evidence": asdict(self.target_settings),
                "camera_control": asdict(CameraControlSettings()), "perception_fusion": "coherent-depth/floor-qualified-v1",
                "oracle_schema_repairs": 1, "local_exploration": self.settings.local_exploration,
                "falcon_running": self.settings.local_exploration == "falcon",
                "falcon_source": dict(FALCON_SOURCE) if self.settings.local_exploration == "falcon" else None,
                "ground_truth_semantics": False, "training_free": True,
                "non_metric": False, "clock": "global action ledger; room clock pauses off-floor"}

    def reset(self, episode, target):
        self.episode, self.target = episode, target
        s = self.settings
        self.mapping = ObservedMap(s.map_size_m, s.map_resolution_m, s.depth_stride, s.body_height_m, s.body_radius_m, s.multifloor)
        self.solver = RptStarRoomSolver(rng=random.Random("%s/%s" % (s.seed, episode.episode_id)), max_rooms=s.max_rooms)
        self.planner = WeightedAStarPlanner2D(self.planner_params)
        self.route_memory = CommittedRoute(episode.action_spec, self.converter_params, self.route_settings)
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
        self.hierarchy = None
        if s.local_exploration == "falcon":
            from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.falcon_policy import FalconObjectNav
            self.hierarchy = FalconObjectNav(self)
        self.building = None
        if s.multifloor.enabled:
            from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.multifloor_policy import MultiFloorSearch
            self.building = MultiFloorSearch(self)

    def _reset_floor(self):
        self.floors.new()
        self._route = self._goal = self._target_xy = self._target_id = None
        self.route_memory.clear("floor_reset")

    def notify_blocked(self, observation):
        if self.building and self.building.traversing:
            self._blocked += 1
            self.building.blocked(observation)
            return
        self.mapping.notify_blocked(observation, self.episode.action_spec.forward_step_m)
        self._blocked += 1
        if self._blocked_since is None:
            self._blocked_since = self._floor_time
        if self._goal is not None:
            self._visited_frontiers.append(self._goal)
        self.route_memory.clear("forward_blocked")
        self._route, self._goal = None, None
        if self.hierarchy is not None:
            self.hierarchy.blocked()

    def plan(self, observation):
        started = time.monotonic()
        try:
            command = self._plan(observation)
            phase = self.hierarchy.machine.phase if self.hierarchy else str(command.info.get("kind", self.supervisor.state))
            if self.building and self.building.phase != "SEARCH":
                phase = self.building.phase
            transition = self.building.transition if self.building else None
            command = self.camera_control.apply(
                observation, command, self.building.phase if self.building else "SEARCH",
                transition.direction if transition else 0, transition.close_support if transition else False)
            self.telemetry.phase = phase
            return replace(command, info=dict(command.info, explorer=self.settings.local_exploration, phase=phase, floor_id=self.mapping.floor_id))
        finally:
            self.telemetry.latencies["policy_decision"].append((time.monotonic() - started) * 1000)

    def filter_action(self, observation, action):
        if self.building and self.building.committed:
            return self.building.filter_action(observation, action)
        return self.hierarchy.filter_action(observation, action) if self.hierarchy else action

    def notify_action(self, observation, action):
        if not self.building or not self.building.committed:
            self._floor_time += self.settings.action_time_s
        if self.hierarchy:
            self.hierarchy.machine.charge(observation.step)
        self.telemetry.emitted(observation, action, self.telemetry.phase)

    def _plan(self, observation):
        self.perception.observe(observation)
        if self.building:
            self.building.prepare_observation(observation)
        started = time.monotonic()
        floor_revision = self.mapping.floor_revision
        transition = self.building.transition if self.building else None
        world = self.mapping.update(observation, integrate=not (self.building and self.building.traversing),
                                    arrival_allowed=transition.arrival_allowed if transition else True)
        self.telemetry.latencies["mapping"].append((time.monotonic() - started) * 1000)
        if self.mapping.floor_revision != floor_revision:
            if self.building:
                self.floors.activate(self.mapping.floor_id)
            else:
                self._reset_floor()
        self.last_world = world
        if self.building:
            self.building.observe(observation)
        if not self.building or not self.building.traversing:
            self.telemetry.observe(observation, world, self.mapping.floor_id)
        confirmed = self._perceive(observation)
        if self.building and self.building.committed:
            command = self.building.plan(observation, world)
            if command is not None:
                return command
        inspection = self.camera_control.inspection_command(observation)
        if inspection is not None:
            return inspection
        return self._search(observation, world, confirmed)

    def _search(self, observation, world, confirmed):
        s, pose = self.settings, observation.pose
        if self._last_pose is not None and math.dist((pose.x, pose.y), self._last_pose) > 0.05:
            self._blocked_since = None
        self._last_pose = (pose.x, pose.y)
        self.graph.credit_time(world, pose, 0.0 if observation.step == 0 else s.action_time_s)
        if self.hierarchy is None and confirmed and self._target_xy is not None and math.dist((pose.x, pose.y), self._target_xy) <= s.stop_distance_m:
            return NavigationCommand.stop_here(info={"reason": "fresh multi-view-confirmed target", "target_confirmed": True})
        if observation.step - self._last_graph_step >= s.graph_period_steps or self.doors.revision != self._last_door_revision:
            self.graph.update(world, self.landmarks.confirmed(), self.target, doors=self.doors.confirmed(), step=observation.step,
                              reason=self.hierarchy is None)
            self._last_graph_step, self._last_door_revision = observation.step, self.doors.revision
        if self.building and self._target_xy is None:
            command = self.building.plan(observation, world)
            if command is not None:
                return command
        if self.hierarchy is not None:
            command = self.hierarchy.plan(observation, world, confirmed)
            if self.building and command.stop and self._target_xy is None and not self.hierarchy.errors:
                return self.building.plan(observation, world, exhausted=True) or command
            return command
        if self._target_xy is not None and observation.step - self._target_step <= self.target_settings.max_unseen_steps:
            approach = self._approach(observation, world)
            if approach is not None:
                return approach
        return self._local_frontier(observation, world)

    def _local_frontier(self, observation, world):
        s = self.settings
        cost = assemble_cost_grid(self.planner.fields_for(world), self.planner_params, s.body_radius_m)[0]
        state = self._search_state(observation, world, cost)
        if state.completed:
            self._route = self._goal = None
            self.route_memory.clear("room_completed")
        if state.state == TRANSIT and state.goal_xy is not None:
            goal, kind = state.goal_xy, "transit/%s" % state.room_id
        else:
            room = self.graph.registry.rooms.get(state.room_id)
            mask = room.mask if state.state == SEARCH and room is not None else np.ones(world.grid.shape, bool)
            goal, kind = self._frontier(observation, world, cost, mask), "frontier"
        if goal is None:
            if self.building:
                command = self.building.plan(observation, world, exhausted=True)
                if command is not None:
                    return command
            return NavigationCommand.hold(info={"reason": "no safe observed frontier; acquire another view"})
        return self._navigate(observation, world, goal, kind) or NavigationCommand.hold(info={"reason": "route unavailable"})

    def _search_state(self, observation, world, cost):
        p, s = observation.pose, self.settings
        instance = None
        if self.graph.options and self.supervisor.state == "select":
            instance, _ = build_instance(world, cost, {r.room_id: r.xy for r in self.graph.options}, self.graph.probs,
                                        depot_xy=(p.x, p.y), cruise_speed_mps=self.episode.action_spec.forward_step_m / s.action_time_s)
        calls = self.solver.calls
        state = self.supervisor.update(self.graph.options, self.graph.facts, (p.x, p.y), self._floor_time,
                                       last_plan_s=self._last_plan_s, instance=instance, blocked_since=self._blocked_since)
        if self.solver.calls != calls:
            data = asdict(self.solver.last)
            self._solver_records.append({k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in data.items()})
        return state

    def _navigate(self, observation, world, goal, kind, final_yaw=None):
        if not self.route_memory.reusable(observation, world, self.planner, goal, kind):
            if self.route_memory.reason == "no_progress":
                self._visited_frontiers.append(tuple(goal))
                self._route = self._goal = None
                self._blocked_since = self._floor_time
                return None
            path = self._plan_to(observation, world, goal)
            if path is None:
                self._visited_frontiers.append(tuple(goal))
                self._route = self._goal = None
                return None
            self.route_memory.adopt(path, goal, kind, observation)
        self._route, self._goal = self.route_memory.path, self.route_memory.goal
        return NavigationCommand.follow(self._route, final_yaw=final_yaw, info={"route": self.route_memory.reason, "kind": kind})

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

    def _frontier(self, observation, world, cost, mask):
        xy = (observation.pose.x, observation.pose.y)
        arrival = self.converter_params.goal_tolerance_m + world.resolution
        if self._goal is not None and self.route_memory.kind == "frontier":
            if math.dist(xy, self._goal) > arrival and not self.route_memory.arrived(observation):
                gx, gy = world.world_to_grid(*self._goal)
                if world.in_bounds(gx, gy) and mask[gy, gx] and np.isfinite(cost[gy, gx]):
                    return self._goal
            self._visited_frontiers.append(self._goal)
            self._goal = self._route = None
            self.route_memory.clear("frontier_completed_or_invalid")
        for goal in in_room_frontier_goals(world, cost, mask):
            if math.dist(xy, goal) > arrival and not any(math.dist(goal, used) < 0.4 for used in self._visited_frontiers[-100:]):
                return goal
        return None

    def _plan_to(self, observation, world, goal):
        p = observation.pose
        self._plan_calls += 1
        started = time.monotonic()
        result = self.planner.plan(PlanRequest(Pose2D(p.x, p.y, p.yaw), Pose2D(*goal), frame_id="world"), world)
        self.telemetry.latencies["astar"].append((time.monotonic() - started) * 1000)
        self._plan_step = observation.step
        if not result.ok:
            return None
        self._last_plan_s = self._floor_time
        return result.path

    def episode_info(self):
        return {"method": self.name, "rooms": len(self.graph.registry.rooms), "landmarks": len(self.landmarks),
                "llm_queries": self.graph.queries, "supervisor": dict(self.supervisor.stats), "solver_records": self._solver_records,
                "blocked": self._blocked, "doors": self.doors.diagnostics(), "max_rooms": self.graph.max_rooms,
                "room_label_queries": self.graph.label_tracker.queries, "room_label_history": list(self.graph.label_tracker.history),
                "last_reasoning": self.graph.last_reasoning, "route_commitment": dict(self.route_memory.stats),
                "plan_calls": self._plan_calls, "duplicates_removed": self._duplicates_removed,
                "target_evidence": self.target_evidence.diagnostics(), "floor_revisions": self.mapping.floor_revision,
                "perception": self.perception.diagnostics(), "camera": dict(self.camera_control.last), "floor_maps": self.mapping.integrity(),
                "building": self.building.diagnostics() if self.building else None,
                "exploration_metrics": self.telemetry.report(), "hierarchy": self.hierarchy.diagnostics() if self.hierarchy else None,
                "oracle_repair_attempts": self.graph.oracle.repair_attempts,
                "oracle_repair_successes": self.graph.oracle.repair_successes}
