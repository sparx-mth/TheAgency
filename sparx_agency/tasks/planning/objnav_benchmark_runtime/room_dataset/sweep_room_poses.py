"""Render every room of HM3D scenes from an even set of positions and headings, with what each view sees (room floor, nearness, targets in view) and its distance to the entrance, to find out how much the choice of place and direction matters."""
import argparse
import json
import os
import traceback
from pathlib import Path

import cv2
import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import protocol_for
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.report_sweep import load_sweep
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.select_door_views import render_at, visible_targets
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.select_grid_views import MIN_ROOM_CELLS, find_entrance
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.select_hm3d_views import Hm3dScene
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.sweep_poses import pose_grid
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.view_selection import describe_candidates

SPACING_M = 1.5
HEADINGS = 4
MAX_POSITIONS = 6
METHOD = "sweep_all"


def sweep_room(scene, room, protocol, seed, folder):
    room_cells = scene.room_cells(room)
    if len(room_cells) < MIN_ROOM_CELLS:
        return {"skipped": "room cells %d" % len(room_cells)}
    entrance = find_entrance(scene, room, room_cells, seed)
    poses = pose_grid(room_cells, SPACING_M, HEADINGS, MAX_POSITIONS, seed)
    described = describe_candidates(poses, scene.render, room_cells, protocol, None)
    frames, visible = {}, {}
    for view in described:
        name = "%s_s%d" % (room["room"], view["index"])
        rendered = render_at(scene, poses, view["index"])
        cv2.imwrite(str(folder / "frames" / (name + ".jpg")), cv2.cvtColor(rendered[0], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        cell = np.asarray(view["cell"])
        frames[name] = {"x": float(cell[0]), "z": float(cell[2]), "yaw": float(view["yaw"]), "near": float(view["near"]), "inside": float(view["inside"]),
                        "floor_share": float(np.mean(view["cells"])), "entrance_m": None if entrance is None else float(np.hypot(cell[0] - entrance[0][0], cell[2] - entrance[0][2]))}
        visible[name] = visible_targets(scene, room, rendered, protocol)
    return {"group": "", "frames": frames, "visible": visible, "methods": {METHOD: sorted(frames)}, "actions": {METHOD: None}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rooms", type=Path, required=True, help="the HM3D rooms folder (rooms.json and cells.npz per scene)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scenes-dir", default=os.environ.get("HM3D_SCENES_DIR"))
    p.add_argument("--every", type=int, default=2, help="sweep every this-many-th room, in name order")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--only-rooms", type=Path, help="sweep only the rooms found in this views folder")
    p.add_argument("--skip-rooms", type=Path, help="leave out the rooms already in this views folder")
    args = p.parse_args(argv)
    protocol = protocol_for("v1").with_split("train")
    only = set(load_sweep(args.only_rooms)) if args.only_rooms else None
    skip = set(load_sweep(args.skip_rooms)) if args.skip_rooms else set()
    for folder in sorted(d for d in args.rooms.iterdir() if (d / "rooms.json").exists()):
        scene = folder.name
        if (args.out / scene / "selection.json").exists():
            continue
        try:
            rooms = sorted(json.loads((folder / "rooms.json").read_text()), key=lambda r: r["room"])[::args.every]
            rooms = [r for r in rooms if (only is None or r["room"] in only) and r["room"] not in skip]
            (args.out / scene / "frames").mkdir(parents=True, exist_ok=True)
            hm3d = Hm3dScene(scene, args.scenes_dir, protocol, args.seed, folder)
            result = {room["room"]: sweep_room(hm3d, room, protocol, args.seed, args.out / scene) for room in rooms}
            hm3d.close()
            (args.out / scene / "selection.json").write_text(json.dumps(result))
            print("%s: %d rooms, %d frames" % (scene, len(result), sum(len(r.get("frames", ())) for r in result.values())), flush=True)
        except Exception:
            print("%s: FAILED\n%s" % (scene, traceback.format_exc()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
