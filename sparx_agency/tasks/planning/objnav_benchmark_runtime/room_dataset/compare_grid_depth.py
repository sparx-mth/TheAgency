"""For labelled Matterport3D regions, score the same candidate poses by the rendered depth image and by rays on the occupancy grid, and keep both for comparison."""
import argparse
import json
import traceback
from pathlib import Path

import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import protocol_for
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.geometry import compass_yaws, spread_points
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.grid_views import grid_candidates
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.select_grid_views import HEADINGS, MIN_ROOM_CELLS, SPOTS, Scene
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.view_selection import describe_candidates

SCORES = "grid_vs_depth.npz"


def stacked(candidates, prefix):
    return {prefix + "cells": np.array([c["cells"] for c in candidates]), prefix + "near": np.array([c["near"] for c in candidates]),
            prefix + "inside": np.array([c["inside"] for c in candidates])}


def score_room(scene, room, protocol):
    room_cells = scene.room_cells(room)
    if len(room_cells) < MIN_ROOM_CELLS:
        return None
    poses = [(room_cells[i], yaw) for i in spread_points(room_cells[:, [0, 2]], SPOTS) for yaw in compass_yaws(HEADINGS)]
    by_depth = describe_candidates(poses, scene.render, room_cells, protocol, None)
    by_grid = grid_candidates(scene.grid(room["floor_y"]), room_cells, [(c["cell"], c["yaw"]) for c in by_depth], protocol.width, protocol.height, protocol.hfov_deg, protocol.max_depth_m)
    return {**stacked(by_depth, "depth_"), **stacked(by_grid, "grid_"), "cell": np.array([c["cell"] for c in by_depth]), "yaw": np.array([c["yaw"] for c in by_depth])}


def score_scene(name, rooms, args, protocol):
    scene = Scene(name, args.scenes_dir, protocol, args.seed)
    arrays = {}
    for room in rooms[::args.every]:
        scored = score_room(scene, room, protocol)
        arrays.update({"%s|%s" % (room["room"], key): value for key, value in (scored or {}).items()})
    scene.close()
    folder = args.out / name
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / SCORES, **arrays)
    return len({key.split("|")[0] for key in arrays})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rooms", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scenes-dir", type=Path, default=Path.home() / "datasets/mp3d/scene_datasets/mp3d")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--scenes", nargs="*")
    p.add_argument("--every", type=int, default=3)
    args = p.parse_args(argv)
    protocol = protocol_for("v1").with_split("train")
    for folder in sorted(d for d in args.rooms.iterdir() if (d / "regions.json").exists() and (not args.scenes or d.name in args.scenes)):
        if (args.out / folder.name / SCORES).exists():
            continue
        try:
            print("%s: %d regions" % (folder.name, score_scene(folder.name, json.loads((folder / "regions.json").read_text())["rooms"], args, protocol)), flush=True)
        except Exception:
            print("%s: FAILED\n%s" % (folder.name, traceback.format_exc()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
