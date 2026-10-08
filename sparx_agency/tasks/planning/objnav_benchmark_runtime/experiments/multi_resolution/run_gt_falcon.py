"""Run planar FALCON in Habitat with evaluator-only HM3D goal viewpoints."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import cv2

from sparx_agency.core.planning.exploration.falcon.params import FalconParams
from sparx_agency.core.planning.exploration.falcon.planner import FalconPlanner
from sparx_agency.core.planning.objnav.action_converter.params import ActionConverterParams
from sparx_agency.core.planning.objnav.agent.headless_agent import HeadlessObjNavAgent
from sparx_agency.core.planning.objnav.labels.datasets.hm3d import hm3d_label_mapper
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.objnav.types.episode import ObjNavEpisode
from sparx_agency.core.planning.objnav.types.observation import ObjNavObservation
from sparx_agency.core.planning.exploration.floor_atlas import MultiFloorParams
from sparx_agency.core.planning.planners.astar.params import WeightedAStarParams
from sparx_agency.core.planning.planners.astar.weighted_planner_2d import WeightedAStarPlanner2D
from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator import HabitatRGBDSimulator, habitat_pose
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.dataset import HM3DDataset, xyzw_to_wxyz
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import protocol_for
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.falcon_motion import GroundMotion
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.observed_map import ObservedMap
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.rpt_settings import RPTSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.stair_ground_truth import GroundTruthStairs


@dataclass(frozen=True)
class GoalView:
    x: float
    y: float
    z: float
    yaw: float
    habitat_position: tuple
    rotation_wxyz: tuple


class SampleVideo:
    """Stream native Habitat RGB frames to an H.264 MP4."""

    def __init__(self, path, width, height, fps):
        self.path = Path(path)
        encoder = shutil.which("ffmpeg")
        if encoder is None:
            raise RuntimeError("ffmpeg is required to write sample.mp4")
        self.temporary = self.path.with_name(self.path.stem + ".partial.mp4")
        self.log = self.path.with_suffix(".encoder.log").open("wb")
        self.process = subprocess.Popen(
            [encoder, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
             "-pixel_format", "rgb24", "-video_size", "%dx%d" % (width, height),
             "-framerate", str(fps), "-i", "-", "-an", "-c:v", "libx264",
             "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             str(self.temporary)], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=self.log)

    def write(self, frame, labels=()):
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Video frames must be HxWx3 uint8 RGB")
        annotated = frame.copy()
        for index, label in enumerate(labels):
            y = 28 + index * 28
            cv2.putText(annotated, str(label), (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.58, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(annotated, str(label), (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.58, (255, 255, 255), 1, cv2.LINE_AA)
        self.process.stdin.write(np.ascontiguousarray(annotated).tobytes())

    def close(self):
        try:
            self.process.stdin.close()
            if self.process.wait(timeout=60) != 0:
                raise RuntimeError("ffmpeg failed; see %s" % self.log.name)
            self.temporary.replace(self.path)
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            self.log.close()


def write_trajectory_csv(path, rows):
    fields = ("step", "action", "x", "y", "z", "yaw", "target_distance_m",
              "target_yaw_error_deg", "planner_status")
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_occupancy_map_png(path, world, trace, route_history, plan_records=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    grid = world.grid
    known_y, known_x = np.nonzero(grid >= 0)
    if not len(known_x):
        x0, x1, y0, y1 = 0, world.width, 0, world.height
    else:
        padding = max(4, int(round(2.0 / world.resolution)))
        x0 = max(0, int(known_x.min()) - padding)
        x1 = min(world.width, int(known_x.max()) + padding + 1)
        y0 = max(0, int(known_y.min()) - padding)
        y1 = min(world.height, int(known_y.max()) + padding + 1)
    cropped = grid[y0:y1, x0:x1]
    left = world.origin_x + x0 * world.resolution
    bottom = world.origin_y + y0 * world.resolution
    extent = (left, left + cropped.shape[1] * world.resolution,
              bottom, bottom + cropped.shape[0] * world.resolution)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
    image = np.where(cropped < 0, 0, np.where(cropped == 0, 1, 2))
    ax.imshow(image, origin="lower", extent=extent, interpolation="nearest",
              cmap=ListedColormap(("#aeb7bd", "#f5f3ed", "#252b30")), vmin=0, vmax=2)
    if trace:
        xs = [row["x"] if "x" in row else row["pose"][0] for row in trace]
        ys = [row["y"] if "y" in row else row["pose"][1] for row in trace]
        ax.plot(xs, ys, color="#007c91", linewidth=2, label="Executed trajectory")
        ax.scatter(xs[0], ys[0], color="#27864a", edgecolor="white", label="Start", zorder=5)
    for index, route in enumerate(route_history):
        points = route["route_xy_world"]
        if len(points) > 1:
            status = route.get("status", "planned")
            styles = {
                "reached": ("#318a52", ":", 1.1, 0.35, "Reached viewpoint routes"),
                "invalidated": ("#bd5147", "--", 1.2, 0.55, "Invalidated routes"),
                "planned": ("#c3572b", "-", 2.0, 0.9, "Unfinished route"),
            }
            color, linestyle, linewidth, alpha, label = styles.get(
                status, ("#c3572b", "--", 1.2, 0.55, "Other route"))
            ax.plot([p[0] for p in points], [p[1] for p in points], linestyle,
                    color=color, alpha=alpha, linewidth=linewidth,
                    label=label if not any(item.get("status") == status
                                           for item in route_history[:index]) else None)
    if plan_records:
        record = plan_records[-1]
        groups = record.get("connectivity_groups_world", [])
        groups_by_id = {group["id"]: group for group in groups}
        for edge in record.get("connectivity_edges_world", []):
            start_group = groups_by_id.get(edge["from"])
            end_group = groups_by_id.get(edge["to"])
            if start_group and end_group:
                ax.plot([start_group["center_xy"][0], end_group["center_xy"][0]],
                        [start_group["center_xy"][1], end_group["center_xy"][1]],
                        color="#3656a8", linewidth=0.65, alpha=0.22, zorder=2)
        free_groups = [group for group in groups if not group["unknown"]]
        unknown_groups = [group for group in groups if group["unknown"]]
        if free_groups:
            ax.scatter([group["center_xy"][0] for group in free_groups],
                       [group["center_xy"][1] for group in free_groups],
                       marker="s", s=16, color="#3656a8", alpha=0.75,
                       label="Free connectivity groups", zorder=4)
        if unknown_groups:
            ax.scatter([group["center_xy"][0] for group in unknown_groups],
                       [group["center_xy"][1] for group in unknown_groups],
                       marker="^", s=17, color="#7a8791", alpha=0.7,
                       label="Unknown connectivity groups", zorder=4)
        frontiers = record.get("frontier_clusters_world", [])
        frontier_points = [point for cluster in frontiers
                           for point in cluster.get("boundary_samples_xy", [])]
        if frontier_points:
            ax.scatter([point[0] for point in frontier_points],
                       [point[1] for point in frontier_points],
                       marker="x", s=16, color="#db8b16", alpha=0.75,
                       label="Frontier samples", zorder=5)
        candidates = record.get("viewpoint_candidates_world", [])
        if candidates:
            ax.scatter([view["xy"][0] for view in candidates],
                       [view["xy"][1] for view in candidates],
                       marker="*", s=32, color="#c3572b", alpha=0.8,
                       label="Candidate viewpoints", zorder=5)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")
    ax.set_title("FALCON occupancy and routes (%.2f m cells)" % world.resolution)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def target_view_error(pose, views):
    """Best Euclidean distance and heading error to an evaluator POV."""
    errors = [(math.dist(pose.position(), (view.x, view.y, view.z)),
               abs(math.atan2(math.sin(pose.yaw - view.yaw),
                              math.cos(pose.yaw - view.yaw))))
              for view in views]
    return min(errors, key=lambda error: (error[0], error[1]))


def target_view_match(pose, views, distance_threshold_m, yaw_threshold_rad):
    """Return the best threshold-satisfying POV, or the closest candidate."""
    errors = [(math.dist(pose.position(), (view.x, view.y, view.z)),
               abs(math.atan2(math.sin(pose.yaw - view.yaw),
                              math.cos(pose.yaw - view.yaw))))
              for view in views]
    matches = [error for error in errors if error[0] <= distance_threshold_m
               and error[1] <= yaw_threshold_rad]
    return bool(matches), min(matches or errors, key=lambda error: (error[0], error[1]))


def target_view_outcome(pose, views, distance_threshold_m, yaw_threshold_rad):
    """Return the success state and best evaluator POV error for one pose."""
    matched, error = target_view_match(pose, views, distance_threshold_m, yaw_threshold_rad)
    return matched, "gt_viewpoint" if matched else "action_limit", error


def final_video_labels(success, target_category, distance_m, yaw_error_deg):
    """Make outcome and evaluator-only POV status unmistakable on the last frame."""
    if success:
        return ("SUCCESS: TARGET POV REACHED",
                "Target: %s | GT POV: %.2f m, %.1f deg" %
                (target_category, distance_m, yaw_error_deg))
    return ("GT POV REFERENCE - NOT TRAVERSED",
            "Target: %s | evaluator reference frame only" % target_category)


def target_viewpoints(goals, start_height_m, height_tolerance_m=0.35):
    """Convert one floor's HM3D goal viewpoints to world-frame target poses."""
    import quaternion

    views = []
    for goal in goals:
        for viewpoint in goal.get("view_points") or ():
            state = viewpoint.get("agent_state") or {}
            position = np.asarray(state.get("position"), dtype=float)
            rotation = state.get("rotation")
            if position.shape != (3,) or not np.isfinite(position).all():
                continue
            if abs(float(position[1]) - start_height_m) > height_tolerance_m:
                continue
            rotation_matrix = quaternion.as_rotation_matrix(
                quaternion.from_float_array(xyzw_to_wxyz(rotation)))
            pose = habitat_pose(position, rotation_matrix, rotation_matrix)
            views.append(GoalView(pose.x, pose.y, pose.z, pose.yaw,
                                  tuple(float(v) for v in position),
                                  xyzw_to_wxyz(rotation)))
    if not views:
        raise ValueError("No goal viewpoints on the episode's starting floor")
    return tuple(views)


def resolve_scene_assets(episode, scene_release_root, scene_directory):
    """Resolve episode paths against Habitat's versioned HM3D asset layout."""
    relative = Path(episode.scene_relative_path)
    if len(relative.parts) != 4:
        raise ValueError("Unexpected HM3D scene path %r" % episode.scene_relative_path)
    scene = Path(scene_release_root) / scene_directory / Path(*relative.parts[1:])
    navmesh = scene.with_name(scene.name[:-4] + ".navmesh")
    for asset in (scene, navmesh):
        if not asset.is_file() or asset.stat().st_size == 0:
            raise FileNotFoundError("Missing or empty Habitat asset: %s" % asset)
    return scene, navmesh


class StandaloneFalcon:
    """RGB-D mapping and FALCON planning without RPT, LLM or detector services."""

    name = "sparx-falcon-gt-viewpoint"

    def __init__(self, settings, stair_exclusion_margin_m, bootstrap_turns=0):
        self.settings = settings
        self.stair_exclusion_margin_m = stair_exclusion_margin_m
        self.bootstrap_turns = bootstrap_turns
        self.bootstrap_turns_remaining = bootstrap_turns
        self.converter_params = ActionConverterParams()

    def reset(self, episode, target):
        del target
        self.episode = episode
        settings = self.settings
        self.mapping = ObservedMap(settings.map_size_m, settings.map_resolution_m,
                                   settings.depth_stride, settings.body_height_m,
                                   settings.body_radius_m,
                                   MultiFloorParams(enabled=False))
        self.planner_params = WeightedAStarParams(
            inflate_radius_m=settings.preferred_clearance_m,
            inflate_floor_m=settings.body_radius_m, unknown_blocked=True,
            goal_snap_radius_m=0.3, waypoint_spacing_m=0.25,
            start_skip_m=0.0, search_margin_m=settings.map_size_m)
        self.planner = WeightedAStarPlanner2D(self.planner_params)
        self.explorer = FalconPlanner(settings.falcon)
        self.motion = GroundMotion(self)
        self.stairs = GroundTruthStairs.from_metadata(episode.metadata)
        if settings.stay_on_start_floor and not self.stairs.available:
            raise ValueError("Same-floor FALCON requires navmesh stair metadata")
        self.floor_height_m = None
        self.scope = None
        self.current_view = None
        self.current_command = None
        self.current_route_world = []
        self.current_goal_world = None
        self.route_history = []
        self.last_plan_status = "not_started"
        self.stop_reason = None
        self.plan_count = 0
        self.bootstrap_turns_remaining = self.bootstrap_turns

    def plan(self, observation):
        world = self.mapping.update(observation)
        if self.floor_height_m is None:
            self.floor_height_m = observation.pose.z
        self.explorer.observe(world)
        cost = self.motion.update(observation, world)
        if self.scope is None:
            self.scope = self._scope_mask(world, observation.pose.x, observation.pose.y)
        self.motion.scope = self.scope
        if self.bootstrap_turns_remaining:
            self.bootstrap_turns_remaining -= 1
            next_yaw = observation.pose.yaw + self.episode.action_spec.turn_angle_rad
            next_yaw = math.atan2(math.sin(next_yaw), math.cos(next_yaw))
            return NavigationCommand.hold(
                final_yaw=next_yaw,
                info={"kind": "bootstrap_scan", "remaining": self.bootstrap_turns_remaining})
        if self._arrived(observation, world):
            self.explorer.retire(self.current_view)
            if self.route_history and self.route_history[-1]["status"] == "planned":
                self.route_history[-1].update(status="reached", reached_step=observation.step)
            self.current_view = self.current_command = None
            self.current_route_world = []
            self.current_goal_world = None
        if self.current_command is not None:
            return self.current_command
        request = type("PlanRequest", (), {})()
        request.pose = observation.pose
        request.camera = observation.camera
        request.step = observation.step
        request.action_spec = self.episode.action_spec
        request.topology_cost = self.motion.topology_cost
        request.arrival_m = self.episode.action_spec.forward_step_m
        result = self.explorer.plan(world, cost, self.scope, request,
                                    self.settings.body_height_m)
        self.last_plan_status = result.status
        self.plan_count += 1
        if result.status != "planned":
            self.stop_reason = "no_executable_falcon_route:%s" % result.status
            return NavigationCommand.stop_here(info={"kind": "falcon", "status": result.status})
        self.current_view = result.viewpoints[0]
        points = [(observation.pose.x, observation.pose.y)]
        points.extend(world.grid_to_world(*cell) for cell in result.routes[0][1:])
        goal = world.grid_to_world(*self.current_view.cell)
        if len(points) == 1:
            points.append(goal)
        self.current_route_world = [list(point) for point in points]
        self.current_goal_world = list(goal)
        self.route_history.append({
            "plan_index": self.plan_count,
            "status": result.status,
            "planned_step": observation.step,
            "frontier_id": self.current_view.frontier,
            "goal_xy_world": self.current_goal_world,
            "goal_yaw_rad": self.current_view.yaw,
            "route_xy_world": self.current_route_world,
        })
        self.current_command = NavigationCommand.follow(
            points, final_yaw=self.current_view.yaw,
            info={"kind": "falcon", "frontier": self.current_view.frontier})
        return self.current_command

    def filter_action(self, observation, action):
        if action != DiscreteAction.MOVE_FORWARD:
            return action
        if self._approaches_stairs(observation):
            self._invalidate_view(rejected=True)
            return DiscreteAction.TURN_RIGHT
        if not self.motion.permits(observation, action, local=True):
            self._invalidate_view(rejected=True)
            return DiscreteAction.TURN_LEFT
        return action

    def notify_blocked(self, observation):
        self.mapping.notify_blocked(observation, self.episode.action_spec.forward_step_m)
        self._invalidate_view(rejected=True)

    def configuration(self):
        return {"method": self.name, "settings": asdict(self.settings),
                "target_pose_source": "HM3D goal viewpoints, evaluator only",
                "floor_lock": "navmesh stair connectors, ground truth"}

    def episode_info(self):
        return {"falcon_plans": self.explorer.records,
                "visited_viewpoints_world": [
                    {"xy": list(point), "yaw_rad": yaw} for point, yaw in self.explorer.visited],
                "rejected_viewpoints_world": [
                    {"xy": list(point), "yaw_rad": yaw} for point, yaw in self.explorer.rejected],
                "route_history": self.route_history,
                "current_route_xy_world": self.current_route_world,
                "last_plan_status": self.last_plan_status,
                "stop_reason": self.stop_reason,
                "bootstrap_turns_requested": self.bootstrap_turns,
                "bootstrap_turns_remaining": self.bootstrap_turns_remaining}

    def _scope_mask(self, world, anchor_x, anchor_y):
        anchor_x, anchor_y = world.world_to_grid(anchor_x, anchor_y)
        rows, columns = np.ogrid[:world.grid.shape[0], :world.grid.shape[1]]
        distance2 = ((columns - anchor_x) ** 2 + (rows - anchor_y) ** 2) * world.resolution ** 2
        return distance2 <= self.settings.falcon.scope_radius_m ** 2

    def _arrived(self, observation, world):
        if self.current_view is None:
            return False
        goal_x, goal_y = world.grid_to_world(*self.current_view.cell)
        distance = math.hypot(observation.pose.x - goal_x, observation.pose.y - goal_y)
        heading = abs(math.atan2(math.sin(observation.pose.yaw - self.current_view.yaw),
                                 math.cos(observation.pose.yaw - self.current_view.yaw)))
        return (distance <= self.episode.action_spec.forward_step_m
                and heading <= self.episode.action_spec.turn_angle_rad / 2 + 1e-6)

    def _approaches_stairs(self, observation):
        if self.floor_height_m is None or self.stairs is None:
            return False
        pose = observation.pose
        step = self.episode.action_spec.forward_step_m
        next_x = pose.x + step * math.cos(pose.yaw)
        next_y = pose.y + step * math.sin(pose.yaw)
        connectors = self.stairs.touching(
            self.floor_height_m, self.settings.multifloor.floor_match_m)
        return any(connector.distance_xy(next_x, next_y) <= self.stair_exclusion_margin_m
                   and connector.distance_xy(next_x, next_y) < connector.distance_xy(pose.x, pose.y)
                   for connector in connectors)

    def _invalidate_view(self, rejected):
        if self.current_view is not None:
            self.explorer.retire(self.current_view, rejected=rejected)
            if self.route_history and self.route_history[-1]["status"] == "planned":
                self.route_history[-1].update(status="invalidated")
        self.current_view = self.current_command = None
        self.current_route_world = []
        self.current_goal_world = None


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--episodes-dir", type=Path, required=True)
    command.add_argument("--scene-release-root", type=Path, required=True)
    command.add_argument("--scene-directory", default="hm3d")
    command.add_argument("--scene", action="append", required=True,
                         help="Scene folder; repeat to include several houses")
    command.add_argument("--episode-id", action="append",
                         help="Episode identity; repeat for a multi-episode batch")
    command.add_argument("--version", choices=("v1", "v2"), default="v2")
    command.add_argument("--split", default="val")
    command.add_argument("--profile", type=Path, default=Path(__file__).with_name("coarse_falcon.json"))
    command.add_argument("--map-resolution-m", type=float,
                         help="Override occupancy resolution for a paired coarse/fine trial")
    command.add_argument("--output-dir", "--output", dest="output_root", type=Path,
                         required=True,
                         help="Root directory; one artifact directory is created per episode")
    command.add_argument("--max-actions", type=int, default=100)
    command.add_argument("--bootstrap-turns", type=int, default=12,
                         help="RGB-D yaw-only scan turns before first FALCON plan")
    command.add_argument("--distance-threshold-m", type=float, default=0.5)
    command.add_argument("--yaw-threshold-deg", type=float, default=39.5)
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--gpu-device", type=int, default=0)
    command.add_argument("--fps", type=int, default=6)
    return command


def _load_settings(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    return RPTSettings(**config)


def _run_episode(args, dataset, episode_id, protocol, settings, run_dir):
    row = dataset.episodes[episode_id]
    scene, navmesh = resolve_scene_assets(row, args.scene_release_root, args.scene_directory)
    simulator = HabitatRGBDSimulator(
        protocol.camera(), protocol.actions(), protocol.agent_radius_m,
        protocol.allow_sliding, args.gpu_device, height_m=protocol.agent_height_m,
        navmesh=protocol.navmesh)
    video = SampleVideo(run_dir / "sample.mp4", protocol.width, protocol.height, args.fps)
    started = time.monotonic()
    try:
        rgb, depth, pose = simulator.reset(scene, navmesh, row.start_position,
                                           row.start_rotation_wxyz, args.seed)
        metadata = simulator.scene_structure()
        episode = ObjNavEpisode(
            row.episode_id, row.scene_key, "hm3d_" + args.version, args.split,
            row.category, protocol.camera(), protocol.actions(), args.max_actions,
            metadata=metadata)
        target_views = target_viewpoints(dataset.goals_for(row), row.start_position[1])
        policy = StandaloneFalcon(settings, settings.stair_exclusion_margin_m,
                      bootstrap_turns=args.bootstrap_turns)
        agent = HeadlessObjNavAgent(policy, hm3d_label_mapper(), name=policy.name)
        agent.reset(episode)
        observation = ObjNavObservation(rgb, depth, pose, protocol.camera(), row.category, 0)
        actions, trace = 0, []
        matched, termination, (best_distance, best_yaw_error) = target_view_outcome(
            pose, target_views, args.distance_threshold_m, math.radians(args.yaw_threshold_deg))
        success = matched
        video.write(rgb, ("Target: %s" % row.category,
                          "START | EVAL GT POV: %.2f m, %.1f deg" %
                          (best_distance, math.degrees(best_yaw_error))))
        trace = [{"step": 0, "action": "START", "x": pose.x, "y": pose.y,
                  "z": pose.z, "yaw": pose.yaw,
                  "target_distance_m": best_distance,
                  "target_yaw_error_deg": math.degrees(best_yaw_error),
                  "planner_status": policy.last_plan_status}]
        while actions < args.max_actions and not matched:
            if matched:
                success = True
                termination = "gt_viewpoint"
                break
            decision = agent.act(observation)
            if decision.action == DiscreteAction.STOP:
                rgb, depth, pose = simulator.step(decision.action)
                actions += 1
                video.write(rgb, ("Target: %s | Action: STOP" % row.category,
                                  "EVAL GT POV: %.2f m, %.1f deg" %
                                  (best_distance, math.degrees(best_yaw_error))))
                trace.append({"step": actions, "action": decision.action.name,
                              "x": pose.x, "y": pose.y, "z": pose.z, "yaw": pose.yaw,
                              "target_distance_m": best_distance,
                              "target_yaw_error_deg": math.degrees(best_yaw_error),
                              "planner_status": policy.last_plan_status})
                termination = policy.stop_reason or "policy_stop"
                break
            frame = simulator.step(decision.action)
            actions += 1
            rgb, depth, pose = frame
            observation = ObjNavObservation(rgb, depth, pose, protocol.camera(),
                                            row.category, actions)
            matched, termination, (best_distance, best_yaw_error) = target_view_outcome(
                pose, target_views, args.distance_threshold_m,
                math.radians(args.yaw_threshold_deg))
            success = matched
            video.write(rgb, ("Target: %s | Action %d/%d: %s" %
                              (row.category, actions, args.max_actions, decision.action.name),
                              "FALCON: %s | EVAL GT POV: %.2f m, %.1f deg" %
                              (policy.last_plan_status, best_distance,
                               math.degrees(best_yaw_error))))
            trace.append({"step": actions, "action": decision.action.name,
                          "x": pose.x, "y": pose.y, "z": pose.z, "yaw": pose.yaw,
                          "target_distance_m": best_distance,
                          "target_yaw_error_deg": math.degrees(best_yaw_error),
                          "planner_status": policy.last_plan_status})
        world = policy.mapping.worlds[policy.mapping.floor_id]
        occupancy_path = run_dir / "occupancy_map.npz"
        np.savez_compressed(
            occupancy_path, occupancy=world.grid,
            resolution_m=np.float32(world.resolution),
            origin_xy_m=np.asarray([world.origin_x, world.origin_y], dtype=np.float64),
            floor_height_m=np.float32(policy.floor_height_m),
            floor_id=np.int32(policy.mapping.floor_id))
        routes_path = run_dir / "routes.json"
        routes_path.write_text(json.dumps(policy.route_history, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
        trajectory_path = run_dir / "trajectory.csv"
        write_trajectory_csv(trajectory_path, trace)
        map_image_path = run_dir / "occupancy_map.png"
        write_occupancy_map_png(map_image_path, world, trace, policy.route_history,
                                policy.explorer.records)
        frontiers_path = run_dir / "frontiers.json"
        frontiers_path.write_text(
            json.dumps({"plans": policy.explorer.records}, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        final_pose = pose
        if success:
            video.write(rgb, final_video_labels(
                True, row.category, best_distance, math.degrees(best_yaw_error)))
            reference_frame = {"kind": "verified_live_frame", "traversed": True}
        else:
            reference_view = min(
                target_views,
                key=lambda view: (math.dist(final_pose.position(), (view.x, view.y, view.z)),
                                  abs(math.atan2(math.sin(final_pose.yaw - view.yaw),
                                                 math.cos(final_pose.yaw - view.yaw)))))
            ref_rgb, _, _ = simulator.reset(
                scene, navmesh, reference_view.habitat_position,
                reference_view.rotation_wxyz, args.seed)
            video.write(ref_rgb, final_video_labels(
                False, row.category, best_distance, math.degrees(best_yaw_error)))
            reference_frame = {"kind": "nearest_gt_viewpoint_reference", "traversed": False,
                               "video_final_frame_label": "GT POV REFERENCE - NOT TRAVERSED"}
        result = {"episode_id": row.episode_id, "scene": row.scene_key,
                "target_category": row.category, "seed": args.seed,
                "success": success, "termination": termination,
                "actions": actions, "distance_threshold_m": args.distance_threshold_m,
                "yaw_threshold_deg": args.yaw_threshold_deg,
                "start_floor_height_habitat_m": row.start_position[1],
                "target_viewpoints_same_floor": len(target_views),
                "bootstrap_turns": args.bootstrap_turns,
                "final_target_distance_m": best_distance,
                "final_target_yaw_error_deg": math.degrees(best_yaw_error),
                "wall_time_s": time.monotonic() - started,
                "transfer_state": {
                    "schema": "sparx-falcon-coarse-state/1",
                    "map_file": occupancy_path.name,
                    "map_resolution_m": float(world.resolution),
                    "map_origin_xy_m": [float(world.origin_x), float(world.origin_y)],
                    "map_shape_yx": list(world.grid.shape),
                    "occupancy_encoding": {"unknown": -1, "free": 0, "occupied": 100},
                    "floor_height_habitat_m": float(policy.floor_height_m),
                    "current_pose_world_xyz_yaw": [final_pose.x, final_pose.y, final_pose.z, final_pose.yaw],
                    "visited_viewpoints_world": policy.episode_info()["visited_viewpoints_world"],
                    "rejected_viewpoints_world": policy.episode_info()["rejected_viewpoints_world"],
                    "route_history": policy.route_history,
                    "falcon_parameters": asdict(settings.falcon),
                    "target_ground_truth_included": False,
                    "fine_session_action": "rebuild fine occupancy, frontiers, zones, graph, and routes",
                },
                "reference_frame": reference_frame,
                "policy": agent.episode_info(), "trace": trace,
                "artifacts": {"video": "sample.mp4", "occupancy_npz": "occupancy_map.npz",
                              "occupancy_png": "occupancy_map.png", "trajectory_csv": "trajectory.csv",
                              "routes_json": "routes.json", "frontiers_json": "frontiers.json"}}
        (run_dir / "run.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                          encoding="utf-8")
        return result
    finally:
        video.close()
        simulator.close()


def main(argv=None):
    args = parser().parse_args(argv)
    if (args.max_actions <= 0 or args.bootstrap_turns < 0
            or args.distance_threshold_m <= 0 or args.yaw_threshold_deg <= 0 or args.fps <= 0):
        parser().error("Action limits, video FPS and POV thresholds must be positive; bootstrap turns cannot be negative")
    protocol = protocol_for(args.version).with_split(args.split)
    dataset = HM3DDataset(args.episodes_dir, None, protocol, full=False,
                          scenes=args.scene, require_scenes=False,
                          episodes_per_scene=1 if args.episode_id is None else None)
    if args.episode_id:
        episode_ids = args.episode_id
    else:
        episode_ids = []
        for scene in args.scene:
            candidate = next((row.episode_id for row in dataset.episodes.values()
                              if row.scene_key == scene), None)
            if candidate is None:
                raise ValueError("No selected episode found for scene %r" % scene)
            episode_ids.append(candidate)
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("Episode IDs must be unique within a batch")
    missing = [episode_id for episode_id in episode_ids if episode_id not in dataset.episodes]
    if missing:
        raise ValueError("Episode IDs were not loaded: %s" % ", ".join(missing))
    settings = _load_settings(args.profile)
    if args.map_resolution_m is not None:
        settings = replace(settings, map_resolution_m=args.map_resolution_m)
    if settings.local_exploration != "falcon":
        raise ValueError("This runner requires the FALCON local exploration profile")
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for episode_id in episode_ids:
        row = dataset.episodes[episode_id]
        run_dir = args.output_root / row.scene_key / episode_id.rsplit("/", 1)[-1]
        run_dir.mkdir(parents=True, exist_ok=False)
        result = _run_episode(args, dataset, episode_id, protocol, settings, run_dir)
        summaries.append({key: result[key] for key in (
            "episode_id", "target_category", "success", "termination", "actions",
            "final_target_distance_m", "final_target_yaw_error_deg")})
        print(json.dumps(summaries[-1], indent=2), flush=True)
    print(json.dumps({"run_directories": [
        str(args.output_root / dataset.episodes[episode_id].scene_key / episode_id.rsplit("/", 1)[-1])
        for episode_id in episode_ids]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())