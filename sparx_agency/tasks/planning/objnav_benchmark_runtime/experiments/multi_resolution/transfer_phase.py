"""One explore-only FALCON phase of the coarse-to-fine transfer, optionally seeded by a prior map."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import numpy as np

from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from sparx_agency.core.planning.exploration.floor_atlas import MultiFloorParams
from sparx_agency.core.planning.objnav.agent.headless_agent import HeadlessObjNavAgent
from sparx_agency.core.planning.objnav.labels.datasets.hm3d import hm3d_label_mapper
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.episode import ObjNavEpisode
from sparx_agency.core.planning.objnav.types.observation import ObjNavObservation
from sparx_agency.tasks.planning.objnav_benchmark_runtime.experiments.multi_resolution.run_gt_falcon import (
    StandaloneFalcon, write_occupancy_map_png, write_trajectory_csv)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.experiments.multi_resolution.transfer_eval import (
    coverage_metrics)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.observed_map import ObservedMap
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.rpt_settings import RPTSettings

MAP_FILE = "occupancy_map.npz"
ENCODING = OccupancyValues(free=0, occupied=100, unknown=-1)


@dataclass(frozen=True)
class PhaseConfig:
    label: str
    settings: RPTSettings
    max_actions: int
    bootstrap_turns: int
    checkpoints: tuple = (100, 200, 400, 600, 800)


class TransferFalcon(StandaloneFalcon):
    """Explore-only FALCON whose map frame and starting evidence can come from an earlier phase."""

    name = "sparx-falcon-explore-transfer"

    def __init__(self, settings, stair_exclusion_margin_m, bootstrap_turns=0, prior=None):
        super().__init__(settings, stair_exclusion_margin_m, bootstrap_turns)
        self.prior = prior

    def reset(self, episode, target):
        super().reset(episode, target)
        s, prior = self.settings, self.prior
        origin = None if prior is None else (prior.origin_x, prior.origin_y)
        self.mapping = ObservedMap(s.map_size_m, s.map_resolution_m, s.depth_stride, s.body_height_m,
                                   s.body_radius_m, MultiFloorParams(enabled=False), origin_xy=origin,
                                   free_log_odds=s.free_log_odds, obstacle_min_height_m=s.obstacle_min_height_m)
        if prior is not None:
            self.mapping.seed_prior(prior)


def save_phase_map(path, world, observed_mask, floor_height_m, floor_id=0):
    np.savez_compressed(
        path, occupancy=world.grid, observed=observed_mask, resolution_m=np.float64(world.resolution),
        origin_xy_m=np.asarray([world.origin_x, world.origin_y], dtype=np.float64),
        floor_height_m=np.float32(floor_height_m), floor_id=np.int32(floor_id))


def load_prior(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("no phase map at %s" % path)
    with np.load(str(path)) as data:
        params = OccupancyGrid2DParams(float(data["resolution_m"]), *map(float, data["origin_xy_m"]), "world")
        return OccupancyGrid2D(data["occupancy"], params, values=ENCODING)


def plan_statistics(records):
    millis = np.array([r["planning_ms"] for r in records], dtype=float)
    clusters = [r["frontier_clusters"] for r in records if "frontier_clusters" in r]
    return {"plans": len(records), "statuses": dict(Counter(r["status"] for r in records)),
            "max_frontier_clusters": max(clusters, default=0),
            "planning_ms_median": float(np.median(millis)) if len(millis) else 0.0,
            "planning_ms_max": float(millis.max()) if len(millis) else 0.0}


def _path_length_m(trace):
    return sum(math.hypot(b["x"] - a["x"], b["y"] - a["y"]) for a, b in zip(trace, trace[1:]))


def _drive(simulator, agent, policy, row, protocol, config, first_frame, probe=None):
    rgb, depth, pose = first_frame
    trace = [{"step": 0, "action": "START", "x": pose.x, "y": pose.y, "z": pose.z, "yaw": pose.yaw,
              "planner_status": policy.last_plan_status}]
    actions, stop, curve = 0, "action_limit", {}
    while actions < config.max_actions:
        decision = agent.act(ObjNavObservation(rgb, depth, pose, protocol.camera(), row.category, actions))
        if decision.action == DiscreteAction.STOP:
            stop = policy.stop_reason or "policy_stop"
            break
        rgb, depth, pose = simulator.step(decision.action)
        actions += 1
        trace.append({"step": actions, "action": decision.action.name, "x": pose.x, "y": pose.y, "z": pose.z,
                      "yaw": pose.yaw, "planner_status": policy.last_plan_status})
        if probe is not None and actions in config.checkpoints:
            curve[actions] = probe()
    return trace, stop, curve


def _write_artifacts(run_dir, policy, trace, metrics):
    world = policy.mapping.worlds[policy.mapping.floor_id]
    save_phase_map(run_dir / MAP_FILE, world, policy.mapping.grid.observed_mask, policy.floor_height_m)
    write_occupancy_map_png(run_dir / "occupancy_map.png", world, trace, policy.route_history, policy.explorer.records)
    write_trajectory_csv(run_dir / "trajectory.csv", trace)
    (run_dir / "phase.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return world


def run_phase(simulator, protocol, row, scene, navmesh, config, prior, run_dir, seed, gt_points):
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    first_frame = simulator.reset(scene, navmesh, row.start_position, row.start_rotation_wxyz, seed)
    episode = ObjNavEpisode(row.episode_id, row.scene_key, protocol.benchmark, protocol.split, row.category,
                            protocol.camera(), protocol.actions(), config.max_actions,
                            metadata=simulator.scene_structure())
    policy = TransferFalcon(config.settings, config.settings.stair_exclusion_margin_m, config.bootstrap_turns, prior)
    agent = HeadlessObjNavAgent(policy, hm3d_label_mapper(), name=policy.name)
    agent.reset(episode)
    probe = lambda: coverage_metrics(  # noqa: E731
        policy.mapping.worlds[policy.mapping.floor_id], policy.mapping.grid.observed_mask, gt_points)["observed_coverage"]
    trace, stop, curve = _drive(simulator, agent, policy, row, protocol, config, first_frame, probe)
    metrics = {"label": config.label, "resolution_m": config.settings.map_resolution_m, "episode_id": row.episode_id,
               "actions": len(trace) - 1, "wall_s": time.monotonic() - started, "termination": stop,
               "last_plan_status": policy.last_plan_status, "path_length_m": _path_length_m(trace),
               "prior_seeded_cells": 0 if prior is None else int(np.count_nonzero(prior.grid != ENCODING.unknown)),
               "plan_statistics": plan_statistics(policy.explorer.records),
               "settings": asdict(config.settings)}
    world = policy.mapping.worlds[policy.mapping.floor_id]
    metrics["coverage"] = coverage_metrics(world, policy.mapping.grid.observed_mask, gt_points)
    metrics["coverage_curve"] = {str(k): v for k, v in curve.items()}
    _write_artifacts(run_dir, policy, trace, metrics)
    return metrics
