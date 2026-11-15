"""Deterministic, evaluator-only selection of genuinely connected multi-story starts."""
from __future__ import annotations

import hashlib
import json
import math
import numpy as np

from sparx_agency.core.planning.objnav.labels.datasets.gibson import CATEGORIES
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.development_dataset import file_identity
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.generate_development import (
    MIN_START_CLEARANCE_M, start_clearance)
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


def cross_floor_starts(scene, floors, pathfinder, seed, count=3, min_clearance_m=MIN_START_CLEARANCE_M,
                       categories=None, same_storey=False, distinct_categories=True):
    """Starts connected to the goals and not boxed in (``min_clearance_m``), on another storey or the goals' own.

    ``categories`` restricts the goal categories to the given dataset indices
    (a couch-only campaign: the one category that is never upstairs in a
    house, so the storey change is the test); None offers every category
    annotated on the reference floor, in a seeded order. Each episode takes
    the next category in that order, so a building's episodes look for
    different objects, and every start lies 2 m or more from the others.

    ``same_storey`` (since 2026-10-05) samples the start on the GOALS' storey
    instead of another one -- the SemExp-style same-floor protocol on the
    multistory harness, whose 3D geodesic region metrics then score a STOP
    at an annotated instance: the cross-floor starts score 0 by construction
    whenever the agent stops at a real but unannotated upstairs instance.
    The 4-60 m geodesic band is the same, and a single-storey building is
    eligible.

    ``distinct_categories`` (the default) makes a building with fewer
    annotated categories than episodes ineligible, so no two of a
    building's episodes look for the same object; False lets the
    rotation repeat a category (the manifests generated before 2026-10-05).
    """
    rng = np.random.RandomState(seed)
    levels, samples = sampled_levels(pathfinder, seed)
    if len(levels) < 2 and not same_storey:
        raise ValueError("Fewer than two area-supported storeys")
    floor_id = min(floors, key=lambda key: abs(float(floors[key]["floor_height"])))
    floor = floors[floor_id]
    annotated = [k for k in range(6) if np.asarray(floor["sem_map"])[k + 1].any()]
    if categories is not None:
        wanted = [int(k) for k in categories]
        annotated = [k for k in annotated if k in wanted]
        if not annotated:
            raise ValueError("None of %s annotated on the reference floor"
                             % ", ".join(CATEGORIES[k] for k in wanted))
    categories = annotated
    rng.shuffle(categories)
    if distinct_categories and len(categories) < int(count):
        raise ValueError("Only %d categor%s annotated on the reference floor for %d episodes that must differ"
                         % (len(categories), "y" if len(categories) == 1 else "ies", int(count)))
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
    if distinct_categories and len(fields) < int(count):
        raise ValueError("Only %d navigable goal region%s for %d episodes that must differ"
                         % (len(fields), "" if len(fields) == 1 else "s", int(count)))
    episodes, audits = [], []
    for episode_index in range(count):
        category_order = list(fields)
        category_order = category_order[episode_index % len(category_order):] + category_order[:episode_index % len(category_order)]
        if distinct_categories:
            # A category whose starts all failed must not be replaced by one already used:
            # Onaga (2026-10-05) fell back to the couch three times over.
            used = {int(e["object_id"]) for e in episodes}
            category_order = [c for c in category_order if int(c) not in used]
        selected = None
        for category in category_order:
            field = fields[category]
            if same_storey:
                candidates = [point for point in samples[rng.permutation(len(samples))]
                              if abs(point[1] - field.goal_height) <= 0.15]
            else:
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
                clearance = start_clearance(pathfinder, point) if min_clearance_m > 0 else None
                if clearance is not None and clearance < min_clearance_m:
                    continue
                yaw = float(rng.uniform(0, 2 * math.pi))
                selected = {"scene": scene, "object_category": CATEGORIES[category], "object_id": int(category),
                            "floor_id": int(floor_id), "start_position": point.tolist(),
                            "start_rotation": [math.cos(yaw / 2), 0.0, math.sin(yaw / 2), 0.0]}
                audits.append({"episode_index": episode_index, "start_height_m": float(point[1]),
                               "goal_height_m": field.goal_height, "initial_geodesic_m": distance,
                               "start_clearance_m": clearance, "min_start_clearance_m": float(min_clearance_m),
                               "connected_cross_floor": not same_storey,
                               "start_storey": "same" if same_storey else "other"})
                break
            if selected is not None:
                episodes.append(selected)
                break
        if selected is None:
            raise ValueError("Insufficient distinct connected %s starts" % ("same-storey" if same_storey else "cross-floor"))
    used = {str(row["object_id"]) for row in episodes}
    return episodes, {k: v for k, v in regions.items() if k in used}, {"levels": levels, "episodes": audits, "seed": seed}


def generate_multifloor(args, maps):
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.generate_development import select_buildings, stage_building
    import habitat_sim
    if args.episodes_per_building < 1:
        raise ValueError("--episodes-per-building must be positive")
    wanted = None
    if getattr(args, "categories", None):
        names = [name.strip() for name in str(args.categories).split(",") if name.strip()]
        unknown = [name for name in names if name not in CATEGORIES]
        if unknown:
            raise ValueError("Unknown goal categories %s; the Gibson goals are %s" % (unknown, list(CATEGORIES)))
        wanted = [CATEGORIES.index(name) for name in names]
    same_storey = getattr(args, "start_storey", "other") == "same"
    order = select_buildings(maps, len(maps), args.seed)
    scenes, episodes, assets, regions, audits, rejected = [], [], [], {}, {}, []
    for scene in order:
        pair = stage_building(args, scene, maps)
        pathfinder = habitat_sim.PathFinder()
        if not pathfinder.load_nav_mesh(str(args.scenes_dir / (scene + ".navmesh"))):
            raise ValueError("Cannot load original navmesh: " + scene)
        seed = int.from_bytes(hashlib.sha256((str(args.seed) + "/multifloor/" + scene).encode()).digest()[:4], "big") & 0x7fffffff
        try:
            rows, goals, audit = cross_floor_starts(scene, maps[scene], pathfinder, seed, args.episodes_per_building,
                                                    categories=wanted, same_storey=same_storey)
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
        raise ValueError("Not enough %s buildings; no campaign written"
                         % ("eligible" if same_storey else "genuinely connected multi-story"))
    data = {"schema": MULTIFLOOR_SCHEMA, "split": MULTIFLOOR_SPLIT, "scenes": scenes,
            "seed": args.seed, "train_info": file_identity(args.train_info),
            "scenes_dir": str(args.scenes_dir.resolve()), "assets": assets,
            "episodes": episodes, "goal_regions": regions, "generation": {
                "source_sha256": source_fingerprint(), "policy_feedback_used": False,
                "selection": ("seeded SHA256 scene order, starts on the annotated reference storey, connected geodesics; no policy feedback"
                              if same_storey else
                              "seeded SHA256 scene order, area-supported distinct levels, connected geodesics; no policy feedback"),
                "start_storey": "same" if same_storey else "other", "distinct_categories": True,
                "episodes_per_building": args.episodes_per_building, "scene_audits": audits,
                "goal_categories": None if wanted is None else [CATEGORIES[k] for k in wanted],
                "min_start_clearance_m": MIN_START_CLEARANCE_M,
                "ineligible_scenes": rejected, "min_level_separation_m": 1.5, "min_level_area_m2": 8.0,
                "min_distance_m": 4.0, "max_distance_m": 60.0,
                "semantic_limit": ("Only reference-floor targets are annotated; the starts lie on that storey, so a STOP at an "
                                   "annotated instance scores -- a same-storey development protocol, not published Gibson val"
                                   if same_storey else
                                   "Only reference-floor targets are annotated; not a complete full-building ObjectNav benchmark")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("Frozen multi-story development manifest:", args.output, flush=True)

