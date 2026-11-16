"""Render frames of labelled Matterport3D regions that look mostly inside the region and not far away, with depth and the region's label and objects.

Per scene writes ``regions.json``, ``frames/`` (jpg) and ``depth/`` (half resolution, millimetres, uint16 png).
"""
import argparse
import gzip
import json
import traceback
from pathlib import Path

import cv2
import numpy as np

from sparx_agency.core.planning.objnav.errors import EnvContractError
from sparx_agency.tasks.planning.objnav_benchmark_runtime.detector_sweep.sample_frames import yaw_quaternion_wxyz
from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator import HabitatRGBDSimulator
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import protocol_for
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.geometry import assign_cells, compass_yaws, in_box_share, spread_points
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_house import read_house, region_box_xz, to_habitat
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_room_types import group_of_label
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.review_sheets import far_fraction

STRUCTURAL = ("wall", "floor", "ceiling")
FLOOR_SAMPLES = 30000
MIN_REGION_AREA_M2 = 4.0
MIN_INTERIOR_CELLS = 10
SAME_FLOOR_M = 0.6
EDGE_MARGIN_M = 0.4
OBJECT_GAP_M = 2.5
MIN_IN_REGION_SHARE = 0.7
MIN_KEPT_FRAMES = 4
FRAMES_PER_SPOT = 3
DEPTH_MM_MAX = 65000


def train_scenes(episodes_dir):
    return sorted(p.name.removesuffix(".json.gz") for p in (Path(episodes_dir) / "content").glob("*.json.gz"))


def interior_cells(cells, box_xz, floor_y, owner):
    """Navmesh points on the region's floor, away from its box edges, whose nearest objects belong to the region."""
    x0, z0, x1, z1 = box_xz
    inside = ((cells[:, 0] > x0 + EDGE_MARGIN_M) & (cells[:, 0] < x1 - EDGE_MARGIN_M) & (cells[:, 2] > z0 + EDGE_MARGIN_M)
              & (cells[:, 2] < z1 - EDGE_MARGIN_M) & (np.abs(cells[:, 1] - floor_y) < SAME_FLOOR_M))
    return cells[inside][owner(cells[inside])]


def build_scene(scene, args, protocol, out):
    import habitat_sim

    folder_in = Path(args.scenes_dir) / scene
    glb, navmesh = folder_in / (scene + ".glb"), folder_in / (scene + ".navmesh")
    house = read_house(folder_in / (scene + ".house"))
    finder = habitat_sim.nav.PathFinder()
    finder.load_nav_mesh(str(navmesh))
    finder.seed(args.seed)
    cells = np.array([np.asarray(finder.get_random_navigable_point(), dtype=float) for _ in range(FLOOR_SAMPLES)])
    cells = cells[np.isfinite(cells).all(axis=1)]
    objects = [{"region": o["region"], "center": to_habitat(o["center"]), "category": o["category"]} for o in house["objects"]]
    rng = np.random.default_rng(args.seed)
    labelled = [r for r in house["regions"] if group_of_label(r["label"])
                and (r["hi"][0] - r["lo"][0]) * (r["hi"][1] - r["lo"][1]) >= MIN_REGION_AREA_M2]
    chosen = [labelled[i] for i in sorted(rng.permutation(len(labelled))[:args.regions_per_scene])]
    simulator = HabitatRGBDSimulator(protocol.camera(), protocol.actions(), protocol.agent_radius_m, protocol.allow_sliding, 0,
                                     height_m=protocol.agent_height_m, navmesh=protocol.navmesh, depth_holes=protocol.depth_holes())
    folder = args.out / scene
    for sub in ("frames", "depth"):
        (folder / sub).mkdir(parents=True, exist_ok=True)
    rooms, skipped = [], []
    for region in chosen:
        name = "%s_r%d" % (scene, region["region"])
        box = region_box_xz(region)
        floor_y = region["lo"][2]
        owner = lambda points, r=region["region"]: assign_cells(points, objects, OBJECT_GAP_M) == r
        spots = interior_cells(cells, box, floor_y, owner)
        if len(spots) < MIN_INTERIOR_CELLS:
            skipped.append((name, "interior cells %d" % len(spots)))
            continue
        kept, per_spot = [], {}
        for spot, index in enumerate(spread_points(spots[:, [0, 2]], args.spots_per_region)):
            for heading, yaw in enumerate(compass_yaws(args.headings)):
                if per_spot.get(spot, 0) >= FRAMES_PER_SPOT or len(kept) >= args.frames_per_region:
                    break
                try:
                    rgb, depth, _ = simulator.reset(glb, navmesh, spots[index], yaw_quaternion_wxyz(yaw), args.seed)
                except EnvContractError:
                    continue
                far, share = far_fraction(depth), in_box_share(depth, (spots[index][0], spots[index][2], yaw), box, protocol.width, protocol.height, protocol.hfov_deg)
                if far <= args.max_far_fraction and share >= MIN_IN_REGION_SHARE:
                    per_spot[spot] = per_spot.get(spot, 0) + 1
                    kept.append({"frame": "%s_s%d_h%d" % (name, spot, heading), "spot": spot, "heading": heading, "x": float(spots[index][0]),
                                 "z": float(spots[index][2]), "yaw": yaw, "far_fraction": far, "in_region_share": share, "rgb": rgb, "depth": depth})
        if len(kept) < MIN_KEPT_FRAMES:
            skipped.append((name, "frames kept %d" % len(kept)))
            continue
        for k in kept:
            cv2.imwrite(str(folder / "frames" / (k["frame"] + ".jpg")), cv2.cvtColor(k.pop("rgb"), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            millimetres = np.clip(np.nan_to_num(k.pop("depth").astype(np.float64)[::2, ::2], nan=0.0, posinf=0.0, neginf=0.0) * 1000, 0, DEPTH_MM_MAX).astype(np.uint16)
            cv2.imwrite(str(folder / "depth" / (k["frame"] + ".png")), millimetres)
        rooms.append({"room": name, "scene": scene, "region": region["region"], "label": region["label"], "group": group_of_label(region["label"]),
                      "level": region["level"], "box_xz": box, "floor_y": floor_y, "area_m2": (box[2] - box[0]) * (box[3] - box[1]),
                      "objects": [o["category"] for o in objects if o["region"] == region["region"] and o["category"] not in STRUCTURAL],
                      "frames": kept})
    simulator.close()
    (folder / "regions.json").write_text(json.dumps({"rooms": rooms, "skipped": skipped}))
    return len(rooms), len(skipped)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes-dir", type=Path, default=Path.home() / "datasets/mp3d/scene_datasets/mp3d")
    p.add_argument("--episodes-dir", type=Path, default=Path.home() / "datasets/objectnav/mp3d/v1/train")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scenes", nargs="*")
    p.add_argument("--regions-per-scene", type=int, default=20)
    p.add_argument("--spots-per-region", type=int, default=4)
    p.add_argument("--headings", type=int, default=8)
    p.add_argument("--frames-per-region", type=int, default=8)
    p.add_argument("--max-far-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args(argv)
    protocol = protocol_for("v1").with_split("train")
    for scene in args.scenes or train_scenes(args.episodes_dir):
        if (args.out / scene / "regions.json").exists():
            continue
        try:
            print("%s: %d regions kept, %d skipped" % ((scene,) + build_scene(scene, args, protocol, args.out)), flush=True)
        except Exception:
            print("%s: FAILED\n%s" % (scene, traceback.format_exc()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
