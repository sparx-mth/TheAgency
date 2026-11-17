"""Pick the views of chosen HM3D rooms that together see the most of the room's floor, and draw them beside the old compass-heading views for a visual check."""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from sparx_agency.core.planning.objnav.errors import EnvContractError
from sparx_agency.tasks.planning.objnav_benchmark_runtime.detector_sweep.sample_frames import yaw_quaternion_wxyz
from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator import HabitatRGBDSimulator
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import protocol_for
from sparx_agency.tasks.planning.objnav_benchmark_runtime.object_dataset.geometry import habitat_position
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.build_room_frames import DEPTH_MM_MAX
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.geometry import compass_yaws, greedy_views, near_share, spread_points, visible_cells
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.review_sheets import FONT, far_fraction, floor_map, pick_frames

MIN_NEW_CELL_SHARE = 0.02
MIN_NEW_CELLS = 3
NEAR_M = 1.0
MAX_NEAR_SHARE = 0.4
SHOWN_PER_SIDE = 6
THUMBNAIL = (320, 240)


def room_in(views_dir, room_name):
    for folder in sorted(d for d in views_dir.iterdir() if (d / "rooms.json").exists()):
        for room in json.loads((folder / "rooms.json").read_text()):
            if room["room"] == room_name:
                return folder, room
    raise SystemExit("no room %s in %s" % (room_name, views_dir))


def render(simulator, glb, navmesh, cell, yaw):
    rgb, depth, pose = simulator.reset(glb, navmesh, cell, yaw_quaternion_wxyz(yaw), 0)
    return rgb, depth, (float(pose.x), float(pose.y), float(pose.z))


def select_views(simulator, glb, navmesh, room_cells, protocol, args):
    poses = [(room_cells[i], yaw) for i in spread_points(room_cells[:, [0, 2]], args.candidates) for yaw in compass_yaws(args.headings)]
    rendered, masks = [], []
    for cell, yaw in poses:
        try:
            _, depth, pose = render(simulator, glb, navmesh, cell, yaw)
        except EnvContractError:
            continue
        if near_share(depth, NEAR_M) > MAX_NEAR_SHARE:
            continue
        rendered.append((cell, yaw))
        masks.append(visible_cells(depth, (*habitat_position(dict(zip("xyz", pose))), yaw), room_cells, protocol.width, protocol.height, protocol.hfov_deg))
    chosen = greedy_views(masks, args.frames, max(MIN_NEW_CELLS, int(MIN_NEW_CELL_SHARE * len(room_cells))))
    return [(rendered[i], int(masks[i].sum())) for i in chosen]


def write_view(simulator, glb, navmesh, folder, room, order, pose, seen):
    (cell, yaw) = pose
    rgb, depth, (x, y, z) = render(simulator, glb, navmesh, cell, yaw)
    name = "%s_v%d" % (room["room"], order)
    cv2.imwrite(str(folder / "frames" / (name + ".jpg")), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(folder / "depth" / (name + ".png")), np.clip(np.nan_to_num(depth.astype(np.float64)) * 1000, 0, DEPTH_MM_MAX).astype(np.uint16))
    return {"room": room["room"], "scene": room["scene"], "region": room["region"], "spot": order, "heading": 0, "frame": name, "x": x, "y": y, "z": z,
            "yaw": yaw, "seen_cells": seen, "far_fraction": far_fraction(depth)}


def side(folder, cells, room, views, title):
    frames = [cv2.resize(cv2.imread(str(folder / "frames" / (v["frame"] + ".jpg"))), THUMBNAIL) for v in views]
    for letter, frame in zip("ABCDEF", frames):
        cv2.putText(frame, letter, (10, 34), FONT, 1.2, (30, 30, 220), 3)
    frames += [np.full(THUMBNAIL[::-1] + (3,), 255, np.uint8)] * (SHOWN_PER_SIDE - len(frames))
    grid = np.vstack([np.hstack(frames[:3]), np.hstack(frames[3:])])
    top = np.hstack([cv2.resize(floor_map(cells["cells"], cells["regions"], room, views), (grid.shape[0], grid.shape[0])), grid])
    banner = np.full((40, top.shape[1], 3), 255, np.uint8)
    cv2.putText(banner, title, (10, 28), FONT, 0.8, (0, 0, 0), 2)
    return np.vstack([banner, top])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes-dir", default=os.environ.get("HM3D_SCENES_DIR"))
    p.add_argument("--views", type=Path, required=True, help="the existing room dataset: its rooms.json, cells.npz and old views")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rooms", nargs="+", required=True)
    p.add_argument("--candidates", type=int, default=60)
    p.add_argument("--headings", type=int, default=12)
    p.add_argument("--frames", type=int, default=SHOWN_PER_SIDE)
    args = p.parse_args(argv)
    protocol = protocol_for("v1").with_split("train")
    scenes = Path(args.scenes_dir)
    (args.out / "sheets").mkdir(parents=True, exist_ok=True)
    for room_name in args.rooms:
        old_folder, room = room_in(args.views, room_name)
        scene = room["scene"]
        stem = scene.split("-", 1)[1]
        glb, navmesh = (scenes / "hm3d" / "train" / scene / (stem + suffix) for suffix in (".basis.glb", ".basis.navmesh"))
        cells = np.load(old_folder / "cells.npz")
        room_cells = cells["cells"][(cells["regions"] == room["region"]) & (np.abs(cells["cells"][:, 1] - room["floor_y"]) < 1.0)]
        folder = args.out / scene
        for sub in ("frames", "depth"):
            (folder / sub).mkdir(parents=True, exist_ok=True)
        simulator = HabitatRGBDSimulator(
            protocol.camera(), protocol.actions(), protocol.agent_radius_m, protocol.allow_sliding, 0, height_m=protocol.agent_height_m,
            navmesh=protocol.navmesh, depth_holes=protocol.depth_holes())
        picked = select_views(simulator, glb, navmesh, room_cells, protocol, args)
        new = [write_view(simulator, glb, navmesh, folder, room, i, pose, seen) for i, (pose, seen) in enumerate(picked)]
        simulator.close()
        (folder / ("views_%s.json" % room_name)).write_text(json.dumps(new))
        old = pick_frames([json.loads(line) for line in (old_folder / "views.jsonl").read_text().splitlines() if json.loads(line)["room"] == room_name], count=SHOWN_PER_SIDE)
        image = np.vstack([side(old_folder, cells, room, old, "%s before: spots in the room, compass headings" % room_name),
                           side(folder, cells, room, new, "%s after: views chosen to see the most room floor (%d of %d cells)" % (room_name, len(new), len(room_cells)))])
        cv2.imwrite(str(args.out / "sheets" / (room_name + ".jpg")), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(room_name, [seen for _, seen in picked], "of", len(room_cells), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
