"""Deterministic, evaluator-only selection of genuinely connected multi-story starts."""
from __future__ import annotations

import hashlib
import json
import math
import numpy as np

from sparx_agency.core.planning.objnav.labels.datasets.gibson import CATEGORIES
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.development_dataset import file_identity
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.multifloor_dataset import MULTIFLOOR_SCHEMA, MULTIFLOOR_SPLIT
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.multifloor_distance import MultiFloorDistance, goal_region_points
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run import source_fingerprint


def sampled_levels(pathfinder, seed, count=6000, min_area_m2=8.0):
    """Area-supported height modes, not isolated stair treads or tiny mesh islands."""
    pathfinder.seed(seed)
    points = np.asarray([pathfinder.get_random_navigable_point() for _ in range(count)], dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        raise ValueError("Navmesh has no finite support samples")
    bins, counts = np.unique(np.round(points[:, 1] / 0.10), return_counts=True)
    levels = []
    for index in np.argsort(-counts, kind="stable"):
        height = float(bins[index] * 0.10)
        if any(abs(height - level["height_m"]) < 1.5 for level in levels):
            continue
        near = np.abs(points[:, 1] - height) <= 0.30
        area = float(near.sum()) / len(points) * float(pathfinder.navigable_area)
        if area >= min_area_m2:
            levels.append({"height_m": float(np.median(points[near, 1])), "sample_count": int(near.sum()),
                           "estimated_area_m2": area})
    return sorted(levels, key=lambda level: level["height_m"]), points


def cross_floor_starts(scene, floors, pathfinder, seed, count=3):
    rng = np.random.RandomState(seed)
    levels, samples = sampled_levels(pathfinder, seed)
    if len(levels) < 2:
        raise ValueError("Fewer than two area-supported storeys")
    floor_id = min(floors, key=lambda key: abs(float(floors[key]["floor_height"])))
    floor = floors[floor_id]
    categories = [k for k in range(6) if np.asarray(floor["sem_map"])[k + 1].any()]
    rng.shuffle(categories)
    regions, fields = {}, {}
    for category in categories:
        try:
            goals = goal_region_points(pathfinder, floor, category)
        except ValueError:
            continue
        height = float(np.median(np.asarray(goals)[:, 1]))
        regions[str(category)] = {"floor_id": int(floor_id), "height_m": height, "points": goals}
        fields[category] = MultiFloorDistance(pathfinder, goals, floor["sem_map"], floor["origin"], category, height)
    if not fields:
        raise ValueError("No labelled, navigable goal regions")
    episodes, audits = [], []
    for episode_index in range(count):
        category_order = list(fields)
        category_order = category_order[episode_index % len(category_order):] + category_order[:episode_index % len(category_order)]
        selected = None
        for category in category_order:
            field = fields[category]
            other_levels = [level["height_m"] for level in levels if abs(level["height_m"] - field.goal_height) >= 1.5]
            candidates = [point for point in samples[rng.permutation(len(samples))]
                          if any(abs(point[1] - height) <= 0.15 for height in other_levels)]
            for point in candidates:
                if any(np.linalg.norm(point[[0, 2]] - np.asarray(e["start_position"])[[0, 2]]) < 2.0 for e in episodes):
                    continue
                try:
                    distance = field.distance(point, start=True)
                except ValueError:
                    continue  # disconnected storeys are explicitly ineligible
                if not 4.0 <= distance <= 60.0:
                    continue
                yaw = float(rng.uniform(0, 2 * math.pi))
                selected = {"scene": scene, "object_category": CATEGORIES[category], "object_id": int(category),
                            "floor_id": int(floor_id), "start_position": point.tolist(),
                            "start_rotation": [math.cos(yaw / 2), 0.0, math.sin(yaw / 2), 0.0]}
                audits.append({"episode_index": episode_index, "start_height_m": float(point[1]),
                               "goal_height_m": field.goal_height, "initial_geodesic_m": distance,
                               "connected_cross_floor": True})
                break
            if selected is not None:
                episodes.append(selected)
                break
        if selected is None:
            raise ValueError("Insufficient distinct connected cross-floor starts")
    used = {str(row["object_id"]) for row in episodes}
    return episodes, {k: v for k, v in regions.items() if k in used}, {"levels": levels, "episodes": audits, "seed": seed}


def generate_multifloor(args, maps):
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.generate_development import select_buildings, stage_building
    import habitat_sim
    if args.episodes_per_building < 1:
        raise ValueError("--episodes-per-building must be positive")
    order = select_buildings(maps, len(maps), args.seed)
    scenes, episodes, assets, regions, audits, rejected = [], [], [], {}, {}, []
    for scene in order:
        pair = stage_building(args, scene, maps)
        pathfinder = habitat_sim.PathFinder()
        if not pathfinder.load_nav_mesh(str(args.scenes_dir / (scene + ".navmesh"))):
            raise ValueError("Cannot load original navmesh: " + scene)
        seed = int.from_bytes(hashlib.sha256((str(args.seed) + "/multifloor/" + scene).encode()).digest()[:4], "big") & 0x7fffffff
        try:
            rows, goals, audit = cross_floor_starts(scene, maps[scene], pathfinder, seed, args.episodes_per_building)
        except ValueError as exc:
            rejected.append({"scene": scene, "reason": str(exc)})
            print("Ineligible:", scene, str(exc), flush=True)
            continue
        scenes.append(scene)
        episodes.extend(rows)
        assets.extend(pair)
        regions[scene], audits[scene] = goals, audit
        print("Selected:", scene, "levels", [round(x["height_m"], 2) for x in audit["levels"]], "episodes", len(rows), flush=True)
        if len(scenes) == args.buildings:
            break
    if len(scenes) != args.buildings:
        raise ValueError("Not enough genuinely connected multi-story buildings; no campaign written")
    data = {"schema": MULTIFLOOR_SCHEMA, "split": MULTIFLOOR_SPLIT, "scenes": scenes,
            "seed": args.seed, "train_info": file_identity(args.train_info),
            "scenes_dir": str(args.scenes_dir.resolve()), "assets": assets,
            "episodes": episodes, "goal_regions": regions, "generation": {
                "source_sha256": source_fingerprint(), "policy_feedback_used": False,
                "selection": "seeded SHA256 scene order, area-supported distinct levels, connected geodesics; no policy feedback",
                "episodes_per_building": args.episodes_per_building, "scene_audits": audits,
                "ineligible_scenes": rejected, "min_level_separation_m": 1.5, "min_level_area_m2": 8.0,
                "min_distance_m": 4.0, "max_distance_m": 60.0,
                "semantic_limit": "Only reference-floor targets are annotated; not a complete full-building ObjectNav benchmark"}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("Frozen multi-story development manifest:", args.output, flush=True)

