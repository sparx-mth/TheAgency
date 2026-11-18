"""For each labelled Matterport3D region, walk from the door to the room's middle and on to its farthest spot, and keep three of the views passed on the way by two rules: evenly spread in time, and the online floor-coverage keeper."""
import argparse
import json
import math
import traceback
from pathlib import Path

import cv2
import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import protocol_for
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.geometry import spread_points
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.select_grid_views import MIN_ROOM_CELLS, SPOTS, Scene, find_entrance
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.view_selection import describe_candidates
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.walk_views import even_picks, keep_best, walk_poses

FRAMES = 3


def path_length(path):
    return sum(math.dist((a[0], a[2]), (b[0], b[2])) for a, b in zip(path, path[1:]))


def walk_through(scene, room_cells, entrance):
    """The walk from the entrance to the room's middle and on to its spot farthest from the middle by path; None if there is no such path."""
    spots = [room_cells[i] for i in spread_points(room_cells[:, [0, 2]], SPOTS)]
    to_middle = scene.path(entrance[0], spots[0])
    legs = [(scene.path(spots[0], s), s) for s in spots[1:]]
    legs = [(p, s) for p, s in legs if p]
    if not to_middle or not legs:
        return None
    far, _ = max(legs, key=lambda leg: path_length(leg[0]))
    return to_middle + far[1:]


def select_room(scene, room, protocol, seed, folder):
    room_cells = scene.room_cells(room)
    if len(room_cells) < MIN_ROOM_CELLS:
        return {"skipped": "room cells %d" % len(room_cells)}
    entrance = find_entrance(scene, room, room_cells, seed)
    path = None if entrance is None else walk_through(scene, room_cells, entrance)
    if path is None:
        return {"skipped": "no walk through the room"}
    poses = walk_poses(path, entrance[1])
    frames = describe_candidates(poses, scene.render, room_cells, protocol, None)
    picks = {"walk_even3": [frames[i]["index"] for i in even_picks(len(frames), FRAMES)], "walk_keep3": [frames[i]["index"] for i in keep_best(frames, FRAMES)]}
    by_index = {f["index"]: f for f in frames}
    named = {}
    for index in sorted({i for chosen in picks.values() for i in chosen}):
        rgb, _, _ = scene.render(poses[index][0], poses[index][1])
        name = "%s_w%d" % (room["room"], index)
        cv2.imwrite(str(folder / "frames" / (name + ".jpg")), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        named[name] = {"seen_cells": int(by_index[index]["cells"].sum()), "near": by_index[index]["near"], "step": index}
    return {"group": room["group"], "room_cells": len(room_cells), "frames": named, "walk_actions": len(poses) - 1, "usable_frames": sum(c["near"] <= 0.4 and c["inside"] >= 0.7 for c in frames),
            "methods": {m: ["%s_w%d" % (room["room"], i) for i in chosen] for m, chosen in picks.items()}, "actions": {m: 0 for m in picks}}


def select_scene(name, rooms, args, protocol):
    scene = Scene(name, args.scenes_dir, protocol, args.seed)
    folder = args.out / name
    (folder / "frames").mkdir(parents=True, exist_ok=True)
    result = {room["room"]: select_room(scene, room, protocol, args.seed, folder) for room in rooms[::args.every]}
    scene.close()
    (folder / "selection.json").write_text(json.dumps(result))
    return len(result)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rooms", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scenes-dir", type=Path, default=Path.home() / "datasets/mp3d/scene_datasets/mp3d")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--scenes", nargs="*")
    p.add_argument("--every", type=int, default=1)
    args = p.parse_args(argv)
    protocol = protocol_for("v1").with_split("train")
    for folder in sorted(d for d in args.rooms.iterdir() if (d / "regions.json").exists() and (not args.scenes or d.name in args.scenes)):
        if (args.out / folder.name / "selection.json").exists():
            continue
        try:
            print("%s: %d regions" % (folder.name, select_scene(folder.name, json.loads((folder / "regions.json").read_text())["rooms"], args, protocol)), flush=True)
        except Exception:
            print("%s: FAILED\n%s" % (folder.name, traceback.format_exc()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
