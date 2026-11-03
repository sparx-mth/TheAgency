"""Where each HM3D episode starts and ends in its building: on a storey, on the stairs, or on a raised or sunken part.

Evaluator side only: reads the navmesh habitat-lab's ObjectNav runs on (``navmesh="agent"``: the shipped mesh
recomputed at the agent's 0.18 m radius and 0.88 m height) and the published goal view points, never anything a
policy sees. For every episode it records:

* the scene's storeys -- :func:`habitat.stair_connectors.floor_levels`, the area-supported modes of the navmesh
  height, the same rule the multi-floor protocol uses;
* where the start stands: ``floor`` (within :data:`ON_LEVEL_M` of a storey), ``stairs`` (off every storey, on a
  patch of level navmesh smaller than :data:`MIN_PLATFORM_M2` -- a tread or a landing), or ``platform`` (off every
  storey on a larger level patch -- a raised or sunken part of a storey, a half-level);
* the nearest goal view point by geodesic distance, its storey, and how far the shortest path to it climbs or
  descends.

Usage::

    python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.episode_levels \\
        --episodes-dir ~/datasets/objectnav/hm3d/v1/val --scenes-dir ~/datasets/scene_datasets/hm3d/val \\
        --out runs/stair_spawn/episode_levels.csv
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.stair_connectors import floor_levels, triangles
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.geodesic import view_point_positions

#: A start this close (m) to a storey's height stands on that storey.
ON_LEVEL_M = 0.15
#: Level navmesh (within :data:`PATCH_DZ_M` of the start's height, within :data:`PATCH_RADIUS_M`) of at least this
#: area makes an off-storey start a platform rather than a tread or a landing.
MIN_PLATFORM_M2 = 3.0
PATCH_DZ_M = 0.06
PATCH_RADIUS_M = 2.5
AGENT_RADIUS_M = 0.18
AGENT_HEIGHT_M = 0.88


def open_pathfinder(scene_glb: Path):
    """A renderer-free simulator on the scene, its navmesh recomputed at the ObjectNav agent's size."""
    import habitat_sim

    config = habitat_sim.SimulatorConfiguration()
    config.scene_id = str(scene_glb)
    config.enable_physics = False
    config.create_renderer = False
    agent = habitat_sim.agent.AgentConfiguration()
    agent.height, agent.radius = AGENT_HEIGHT_M, AGENT_RADIUS_M
    sim = habitat_sim.Simulator(habitat_sim.Configuration(config, [agent]))
    settings = habitat_sim.NavMeshSettings()
    settings.set_defaults()
    settings.agent_height, settings.agent_radius = AGENT_HEIGHT_M, AGENT_RADIUS_M
    if not sim.recompute_navmesh(sim.pathfinder, settings):
        sim.close()
        raise RuntimeError("cannot recompute the navmesh of %s" % scene_glb)
    return sim


def classify_start(start, tris, levels) -> Dict[str, object]:
    """The start's storey offset and whether it stands on a floor, the stairs, or a platform."""
    heights = [lv["height_m"] for lv in levels]
    nearest = int(np.argmin([abs(start[1] - h) for h in heights]))
    dz = float(start[1] - heights[nearest])
    centroids = tris.mean(axis=1)
    area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    near = ((np.abs(centroids[:, 1] - start[1]) <= PATCH_DZ_M)
            & (np.hypot(centroids[:, 0] - start[0], centroids[:, 2] - start[2]) <= PATCH_RADIUS_M))
    patch = float(area[near].sum())
    if abs(dz) <= ON_LEVEL_M:
        kind = "floor"
    else:
        kind = "platform" if patch >= MIN_PLATFORM_M2 else "stairs"
    below = [i for i, h in enumerate(heights) if h < start[1] - ON_LEVEL_M]
    above = [i for i, h in enumerate(heights) if h > start[1] + ON_LEVEL_M]
    return {"start_kind": kind, "start_level": nearest, "start_dz_m": round(dz, 3), "start_patch_m2": round(patch, 2),
            "level_below": below[-1] if below else "", "level_above": above[0] if above else ""}


def goal_row(sim, start, view_points, heights) -> Dict[str, object]:
    """The nearest goal view point by geodesic, its storey, and the vertical extent of the path to it."""
    import habitat_sim

    query = habitat_sim.MultiGoalShortestPath()
    query.requested_start = np.asarray(start, dtype=np.float32)
    query.requested_ends = [np.asarray(v, dtype=np.float32) for v in view_points]
    if not sim.pathfinder.find_path(query) or not math.isfinite(query.geodesic_distance):
        return {"geodesic_m": "", "goal_z_m": "", "goal_level": "", "goal_dz_m": "", "path_max_rise_m": "",
                "path_max_drop_m": ""}
    points = np.asarray(query.points, dtype=float)
    goal = points[-1]
    level = int(np.argmin([abs(goal[1] - h) for h in heights]))
    return {"geodesic_m": round(float(query.geodesic_distance), 3), "goal_z_m": round(float(goal[1]), 3),
            "goal_level": level, "goal_dz_m": round(float(goal[1] - start[1]), 3),
            "path_max_rise_m": round(float(points[:, 1].max() - start[1]), 3),
            "path_max_drop_m": round(float(start[1] - points[:, 1].min()), 3)}


def survey_scene(shard: Path, scene_dir: Path) -> List[Dict[str, object]]:
    data = json.load(gzip.open(shard))
    glb = next(scene_dir.glob("*.basis.glb"))
    sim = open_pathfinder(glb)
    try:
        tris = triangles(sim.pathfinder.build_navmesh_vertices())
        levels = floor_levels(tris)
        heights = [lv["height_m"] for lv in levels]
        rows = []
        for index, episode in enumerate(data["episodes"]):
            start = [float(v) for v in episode["start_position"]]
            key = "%s_%s" % (Path(episode["scene_id"]).name, episode["object_category"])
            views = view_point_positions(data["goals_by_category"][key])
            row = {"episode": "%s/%06d" % (scene_dir.name, index), "category": episode["object_category"],
                   "levels": len(levels), "level_heights_m": " ".join("%.2f" % h for h in heights),
                   "start_z_m": round(start[1], 3)}
            row.update(classify_start(start, tris, levels))
            row.update(goal_row(sim, start, views, heights))
            row["goal_same_level"] = int(row["goal_level"] == row["start_level"]) if row["goal_level"] != "" else ""
            rows.append(row)
        return rows
    finally:
        sim.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--episodes-dir", type=Path, required=True)
    p.add_argument("--scenes-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    scenes = {d.name.split("-", 1)[1]: d for d in args.scenes_dir.expanduser().iterdir() if "-" in d.name}
    rows = []
    for shard in sorted((args.episodes_dir.expanduser() / "content").glob("*.json.gz")):
        code = shard.name.split(".")[0]
        if code not in scenes:
            raise SystemExit("no scene directory for %s under %s" % (code, args.scenes_dir))
        rows.extend(survey_scene(shard, scenes[code]))
        print("%s: %d episodes" % (scenes[code].name, len(rows)), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("%d episodes -> %s" % (len(rows), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
