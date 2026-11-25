"""Render frames of HM3D rooms (annotated regions) with depth and the exact visible instances and their regions.

Per scene writes ``rooms.json``, ``cells.npz`` (navmesh floor cells with their region), ``views.jsonl``, ``frames/`` and ``depth/``.
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from sparx_agency.core.planning.objnav.errors import EnvContractError
from sparx_agency.tasks.planning.objnav_benchmark_runtime.detector_sweep.sample_frames import yaw_quaternion_wxyz
from sparx_agency.tasks.planning.objnav_benchmark_runtime.ground_truth.semantic_frame import instance_extents
from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator import HabitatRGBDSimulator
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import protocol_for
from sparx_agency.tasks.planning.objnav_benchmark_runtime.object_dataset.geometry import is_held_out
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.annotations import (
    instance_centers, read_semantic_table, read_vertices)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.geometry import UNASSIGNED, assign_cells, compass_yaws, spread_points

STRUCTURAL = ("ceiling", "floor", "unknown")
FLOOR_SAMPLES = 40000
CELL_M = 0.25
MAX_GAP_M = 2.0
MIN_ROOM_AREA_M2 = 4.0
MIN_VISIBLE_PIXELS = 300
DEPTH_MM_MAX = 65000


def floor_cells(finder, count, seed):
    """Random navmesh points on islands wider than 1.5 m, one per ``CELL_M`` grid cell and floor."""
    finder.seed(seed)
    cells = {}
    for _ in range(count):
        p = np.asarray(finder.get_random_navigable_point(), dtype=float)
        if finder.island_radius(p.astype(np.float32)) > 1.5:
            cells.setdefault((round(p[0] / CELL_M), round(p[1] / 1.0), round(p[2] / CELL_M)), p)
    return np.array(list(cells.values()))


def rooms_of_scene(cells, regions, area_per_cell_m2, count, rng):
    """Up to ``count`` random regions with at least ``MIN_ROOM_AREA_M2`` of floor."""
    big = [int(r) for r in sorted(set(regions) - {UNASSIGNED}) if (regions == r).sum() * area_per_cell_m2 >= MIN_ROOM_AREA_M2]
    return [big[i] for i in sorted(rng.permutation(len(big))[:count])]


def build_scene(scene, args, protocol, out):
    import habitat_sim

    scenes = Path(args.scenes_dir)
    stem = scene.split("-", 1)[1]
    folder_in = scenes / "hm3d" / "train" / scene
    glb, navmesh = (folder_in / (stem + suffix) for suffix in (".basis.glb", ".basis.navmesh"))
    table = read_semantic_table(folder_in / (stem + ".semantic.txt"))
    positions, colours = read_vertices(folder_in / (stem + ".semantic.glb"))
    objects = [o for o in instance_centers(positions, colours, table) if o["name"] not in STRUCTURAL]
    finder = habitat_sim.nav.PathFinder()
    finder.load_nav_mesh(str(navmesh))
    cells = floor_cells(finder, FLOOR_SAMPLES, args.seed)
    regions = assign_cells(cells, objects, MAX_GAP_M)
    rng = np.random.default_rng(args.seed)
    chosen = rooms_of_scene(cells, regions, finder.navigable_area / len(cells), args.rooms_per_scene, rng)
    simulator = HabitatRGBDSimulator(
        protocol.camera(), protocol.actions(), protocol.agent_radius_m, protocol.allow_sliding, 0, height_m=protocol.agent_height_m,
        navmesh=protocol.navmesh, depth_holes=protocol.depth_holes(), semantic=True,
        scene_dataset_config=scenes / "hm3d" / "hm3d_annotated_basis.scene_dataset_config.json")
    folder = args.out / scene
    for sub in ("frames", "depth"):
        (folder / sub).mkdir(parents=True, exist_ok=True)
    names = {entry["instance"]: (entry["name"], entry["region"]) for entry in table.values()}
    rooms, views = [], []
    for region in chosen:
        room_cells = cells[regions == region]
        spots = spread_points(room_cells[:, [0, 2]], args.spots_per_room)
        room = "%s_r%d" % (scene.split("-")[0], region)
        rooms.append({"room": room, "region": region, "scene": scene, "cells": len(room_cells),
                      "area_m2": float(len(room_cells) * finder.navigable_area / len(cells)), "floor_y": float(np.median(room_cells[:, 1])),
                      "objects": [{"instance": o["instance"], "name": o["name"], "center": o["center"]} for o in objects if o["region"] == region]})
        for spot, index in enumerate(spots):
            for heading, yaw in enumerate(compass_yaws(args.headings)):
                try:
                    rgb, depth, pose = simulator.reset(glb, navmesh, room_cells[index], yaw_quaternion_wxyz(yaw), args.seed)
                except EnvContractError:
                    continue
                name = "%s_s%d_h%d" % (room, spot, heading)
                cv2.imwrite(str(folder / "frames" / (name + ".jpg")), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(folder / "depth" / (name + ".png")), np.clip(np.nan_to_num(depth.astype(np.float64)) * 1000, 0, DEPTH_MM_MAX).astype(np.uint16))
                visible = [{"instance": i, "name": names[i][0], "region": names[i][1], "pixels": e.pixels}
                           for i, e in visible_instances(simulator, names).items()]
                views.append({"room": room, "scene": scene, "region": region, "spot": spot, "heading": heading, "frame": name,
                              "x": float(pose.x), "y": float(pose.y), "z": float(pose.z), "yaw": yaw, "visible": visible})
    simulator.close()
    (folder / "rooms.json").write_text(json.dumps(rooms))
    np.savez_compressed(folder / "cells.npz", cells=cells, regions=regions)
    (folder / "views.jsonl").write_text("".join(json.dumps(v) + "\n" for v in views))
    return len(rooms), len(views)


def visible_instances(simulator, names):
    return {i: e for i, e in instance_extents(simulator.last_semantic, min_pixels=MIN_VISIBLE_PIXELS).items() if i in names}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes-dir", default=os.environ.get("HM3D_SCENES_DIR"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scenes", nargs="+", required=True)
    p.add_argument("--rooms-per-scene", type=int, default=10)
    p.add_argument("--spots-per-room", type=int, default=3)
    p.add_argument("--headings", type=int, default=8)
    p.add_argument("--held-out-fraction", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args(argv)
    protocol = protocol_for("v1").with_split("train")
    for scene in args.scenes:
        if is_held_out(scene, args.held_out_fraction):
            print("%s: held out, skipped" % scene)
        elif not (args.out / scene / "views.jsonl").exists():
            print("%s: %d rooms, %d views" % ((scene,) + build_scene(scene, args, protocol, args.out)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
