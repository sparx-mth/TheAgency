"""Freeze ObjectNav starts in distinct Gibson TRAIN buildings before evaluation.

Correspondence: SemExp 5d76902, objectgoal_env.py:156-264 and arguments.py:89-94.
Retains floor/category selection, navmesh sampling, free semantic cells and the
1.5-100 m reference distance band. Adds deterministic seeds, bounded attempts,
explicit unreachable rejection and immutable manifests. No policy is consulted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from sparx_agency.core.planning.objnav.labels.datasets.gibson import CATEGORIES
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.assets import import_scene_zip
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.development_dataset import SCHEMA, SPLIT, file_identity, load_training_maps
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.distance import GibsonDistanceField
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import PROTOCOL
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run import source_fingerprint

REFERENCE = {"repository": "https://github.com/devendrachaplot/Object-Goal-Navigation",
             "revision": "5d76902fe9be821926a1de32557ca9a8dc21d0f5",
             "sampler": "envs/habitat/objectgoal_env.py:156-264",
             "defaults": "arguments.py:89-94"}


def select_buildings(names, count, seed):
    names = set(names)
    if count < 1 or count > len(names) or seed < 0:
        raise ValueError("Need a nonnegative seed and enough distinct training buildings")
    return sorted(names, key=lambda name: hashlib.sha256((str(seed) + "/" + name).encode()).hexdigest())[:count]


def generate_start(scene, floors, pathfinder, seed, min_distance=1.5, max_distance=100.0,
                   floor_tolerance=0.5, attempts=20000):
    """Bounded reference-style evaluator-only sampling; never a nav-policy oracle."""
    rng = np.random.RandomState(seed)
    pathfinder.seed(seed)
    floor_ids = sorted(floors)
    rng.shuffle(floor_ids)
    tried = 0
    for floor_id in floor_ids:
        floor = floors[floor_id]
        semantic = np.asarray(floor["sem_map"])
        categories = [k for k in range(6) if semantic[k + 1].any()]
        rng.shuffle(categories)
        for category in categories:
            field = GibsonDistanceField(semantic, floor["origin"], category)
            distances = field.cells * PROTOCOL.map_resolution_m + PROTOCOL.success_radius_m
            valid = (semantic[0] > 0) & ~field.unreachable & (distances > min_distance) & (distances < max_distance)
            if not valid.any():
                continue
            for _ in range(attempts):
                tried += 1
                position = np.asarray(pathfinder.get_random_navigable_point(), dtype=float)
                if position.shape != (3,) or not np.isfinite(position).all():
                    continue
                if abs(position[1] - float(floor["floor_height"])) >= floor_tolerance:
                    continue
                try:
                    cell = field.map_cell(position)
                except ValueError:
                    continue
                if not valid[cell] or not pathfinder.is_navigable(position):
                    continue
                yaw = float(rng.uniform(0.0, 2 * math.pi))
                row = {"scene": scene, "object_category": CATEGORIES[category], "object_id": int(category),
                       "floor_id": int(floor_id), "start_position": position.tolist(),
                       "start_rotation": [math.cos(yaw / 2), 0.0, math.sin(yaw / 2), 0.0]}
                audit = {"scene": scene, "seed": seed, "samples_tried": tried,
                         "reference_start_distance_m": float(distances[cell]),
                         "floor_height_m": float(floor["floor_height"]), "unreachable": False}
                return row, audit
    raise RuntimeError("No valid ObjectNav start found in selected building %s; no replacement building selected" % scene)


def stage_building(args, scene, maps):
    """Reuse complete local asset pairs, or extract without overwriting files."""
    args.scenes_dir.mkdir(parents=True, exist_ok=True)
    paths = [args.scenes_dir / (scene + suffix) for suffix in (".glb", ".navmesh")]
    if any(path.exists() for path in paths):
        if not all(path.is_file() for path in paths):
            raise FileExistsError("Incomplete existing asset pair for " + scene)
    else:
        import_scene_zip(args.archive, scene, args.scenes_dir, allowed_scenes=tuple(maps))
    return [file_identity(path) for path in paths]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-info", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--scenes-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--buildings", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--multistory", action="store_true")
    parser.add_argument("--episodes-per-building", type=int, default=1)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Not overwriting an existing episode manifest")
    maps = load_training_maps(args.train_info)
    if args.multistory:
        if not 1 <= args.buildings <= len(maps) or args.seed < 0:
            raise ValueError("Invalid building count or seed")
        from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.multifloor_generation import generate_multifloor
        return generate_multifloor(args, maps)
    if args.episodes_per_building != 1:
        raise ValueError("Multiple generated starts require --multistory")
    scenes = select_buildings(maps, args.buildings, args.seed)
    print("Selected buildings before any policy run:", ", ".join(scenes), flush=True)
    args.scenes_dir.mkdir(parents=True, exist_ok=True)
    assets = []
    for scene in scenes:
        assets.extend(stage_building(args, scene, maps))
    import habitat_sim  # PathFinder needs no renderer/GPU context
    episodes, audits = [], []
    for scene in scenes:
        pathfinder = habitat_sim.PathFinder()
        if not pathfinder.load_nav_mesh(str(args.scenes_dir / (scene + ".navmesh"))):
            raise RuntimeError("Cannot load original navmesh for " + scene)
        seed = int.from_bytes(hashlib.sha256((str(args.seed) + "/start/" + scene).encode()).digest()[:4], "big") & 0x7fffffff
        episode, audit = generate_start(scene, maps[scene], pathfinder, seed)
        episodes.append(episode)
        audits.append(audit)
        print(scene, episode["object_category"], "start distance", round(audit["reference_start_distance_m"], 2), flush=True)
    manifest = {"schema": SCHEMA, "split": SPLIT, "scenes": scenes, "seed": args.seed,
                "train_info": file_identity(args.train_info), "scenes_dir": str(args.scenes_dir.resolve()),
                "assets": assets, "episodes": episodes, "generation": {
                    "reference": REFERENCE, "source_sha256": source_fingerprint(),
                    "min_distance_m": 1.5, "max_distance_m": 100.0, "floor_tolerance_m": 0.5,
                    "max_samples_per_category": 20000, "reject_unreachable": True,
                    "selection": "seeded SHA256 order, first N distinct training scenes",
                    "policy_feedback_used": False, "audit": audits}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("Frozen generated development episodes:", args.output)


if __name__ == "__main__":
    main()


