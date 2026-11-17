"""For the labelled Matterport3D regions already built, render a pool of candidate views and keep 6 by four methods: random, random after the close-up filter, room floor seen, room floor and objects seen."""
import argparse
import json
import traceback
import zlib
from pathlib import Path

import cv2
import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.detector_sweep.run_detectors import DEFAULT_VOCABULARY
from sparx_agency.tasks.planning.objnav_benchmark_runtime.detector_sweep.sample_frames import yaw_quaternion_wxyz
from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator import HabitatRGBDSimulator
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import protocol_for
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.build_mp3d_frames import FLOOR_SAMPLES, OBJECT_GAP_M, interior_cells
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.geometry import assign_cells, compass_yaws, spread_points
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_house import read_house, to_habitat
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.view_selection import describe_candidates, random_pick, select
from sparx_agency.tasks.planning.objnav_benchmark_runtime.services.http_detector import HttpDetector

METHODS = ("random", "random_filtered", "floor", "floor_objects")
FRAMES = 6
MIN_ROOM_CELLS = 10


def picks_of(candidates, rng):
    return {"random": random_pick(candidates, FRAMES, rng, filtered=False), "random_filtered": random_pick(candidates, FRAMES, rng, filtered=True),
            "floor": select(candidates, FRAMES, with_objects=False), "floor_objects": select(candidates, FRAMES, with_objects=True)}


def select_scene(scene, rooms, args, protocol, detector):
    import habitat_sim

    folder_in = args.scenes_dir / scene
    glb, navmesh = folder_in / (scene + ".glb"), folder_in / (scene + ".navmesh")
    house = read_house(folder_in / (scene + ".house"))
    finder = habitat_sim.nav.PathFinder()
    finder.load_nav_mesh(str(navmesh))
    finder.seed(args.seed)
    cells = np.array([np.asarray(finder.get_random_navigable_point(), dtype=float) for _ in range(FLOOR_SAMPLES)])
    cells = cells[np.isfinite(cells).all(axis=1)]
    objects = [{"region": o["region"], "center": to_habitat(o["center"]), "category": o["category"]} for o in house["objects"]]
    simulator = HabitatRGBDSimulator(protocol.camera(), protocol.actions(), protocol.agent_radius_m, protocol.allow_sliding, 0,
                                     height_m=protocol.agent_height_m, navmesh=protocol.navmesh, depth_holes=protocol.depth_holes())

    def render(cell, yaw):
        rgb, depth, pose = simulator.reset(glb, navmesh, cell, yaw_quaternion_wxyz(yaw), args.seed)
        return rgb, depth, (float(pose.x), float(pose.y), float(pose.z))

    folder = args.out / scene
    (folder / "frames").mkdir(parents=True, exist_ok=True)
    result = {}
    for room in rooms:
        owner = lambda points, r=room["region"]: assign_cells(points, objects, OBJECT_GAP_M) == r
        room_cells = interior_cells(cells, tuple(room["box_xz"]), room["floor_y"], owner, margin=0.0)
        if len(room_cells) < MIN_ROOM_CELLS:
            result[room["room"]] = {"skipped": "room cells %d" % len(room_cells)}
            continue
        poses = [(room_cells[i], yaw) for i in spread_points(room_cells[:, [0, 2]], args.candidates) for yaw in compass_yaws(args.headings)]
        candidates = describe_candidates(poses, render, room_cells, protocol, detector)
        picks = picks_of(candidates, np.random.default_rng([args.seed, zlib.crc32(room["room"].encode())]))
        frames = {}
        for index in sorted({i for chosen in picks.values() for i in chosen}):
            c = candidates[index]
            rgb, _, _ = render(c["cell"], c["yaw"])
            name = "%s_c%d" % (room["room"], index)
            cv2.imwrite(str(folder / "frames" / (name + ".jpg")), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            frames[name] = {"seen_cells": int(c["cells"].sum()), "seen_objects": len(c["objects"]), "near": c["near"]}
        result[room["room"]] = {"group": room["group"], "room_cells": len(room_cells), "candidates": len(candidates),
                                "usable": sum(c["near"] <= 0.4 for c in candidates), "frames": frames,
                                "methods": {m: ["%s_c%d" % (room["room"], i) for i in chosen] for m, chosen in picks.items()}}
    simulator.close()
    (folder / "selection.json").write_text(json.dumps(result))
    return len(result)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rooms", type=Path, required=True, help="the built Matterport3D regions (regions.json per scene)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scenes-dir", type=Path, default=Path.home() / "datasets/mp3d/scene_datasets/mp3d")
    p.add_argument("--detector-url", required=True)
    p.add_argument("--backend", default="yolo_world")
    p.add_argument("--candidates", type=int, default=24)
    p.add_argument("--headings", type=int, default=8)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--scenes", nargs="*", help="only these scenes")
    args = p.parse_args(argv)
    protocol = protocol_for("v1").with_split("train")
    detector = HttpDetector(args.detector_url, [w for w in DEFAULT_VOCABULARY.read_text().strip().split(",") if w], timeout_s=120.0, expected_backend=args.backend)
    for folder in sorted(d for d in args.rooms.iterdir() if (d / "regions.json").exists() and (not args.scenes or d.name in args.scenes)):
        if (args.out / folder.name / "selection.json").exists():
            continue
        try:
            print("%s: %d regions" % (folder.name, select_scene(folder.name, json.loads((folder / "regions.json").read_text())["rooms"], args, protocol, detector)), flush=True)
        except Exception:
            print("%s: FAILED\n%s" % (folder.name, traceback.format_exc()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
