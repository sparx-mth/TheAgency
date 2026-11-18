"""For the rooms whose walk-through views were scored, render the door views (at the entrance, and four steps in) and mark which targets are in view in them and in the kept walk views."""
import argparse
import json
import re
import traceback
from pathlib import Path

import cv2

from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import protocol_for
from sparx_agency.tasks.planning.objnav_benchmark_runtime.object_dataset.geometry import habitat_position
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_target_score import kept_frames
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.select_grid_views import Scene, find_entrance
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.select_walk_views import walk_through
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.target_score import TARGETS
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.target_visibility import steps_in_index, target_visible
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.walk_views import walk_poses

MARGIN_PX = 60
TOLERANCE_M = 0.7
STEPS_IN = 4


def visible_targets(scene, room, rendered, protocol):
    """``{target: bool}``: some annotated instance of the target in this region is in view of the rendered ``(rgb, depth, pose)`` and not hidden."""
    _, depth, pose = rendered
    camera = (*habitat_position(dict(zip("xyz", pose[:3]))), pose[3])
    centres = {t: [o["center"] for o in scene.objects if o["region"] == room["region"] and scene.is_target(o, t)] for t in TARGETS}
    return {t: any(target_visible(camera, depth, c, protocol.width, protocol.height, protocol.hfov_deg, MARGIN_PX, TOLERANCE_M, protocol.max_depth_m) for c in cs) for t, cs in centres.items()}


def render_at(scene, poses, index):
    rgb, depth, pose = scene.render(poses[index][0], poses[index][1])
    return rgb, depth, (*pose, poses[index][1])


def select_room(scene, room, kept, protocol, seed, folder):
    room_cells = scene.room_cells(room)
    entrance = find_entrance(scene, room, room_cells, seed)
    path = None if entrance is None else walk_through(scene, room_cells, entrance)
    if path is None:
        return {"skipped": "no walk through the room"}
    poses = walk_poses(path, entrance[1])
    door = {"door0": 0, "door4": steps_in_index(poses, STEPS_IN)}
    frames, visible, methods = {}, {}, {}
    for method, index in door.items():
        rendered = render_at(scene, poses, index)
        name = "%s_d%d" % (room["room"], index)
        cv2.imwrite(str(folder / "frames" / (name + ".jpg")), cv2.cvtColor(rendered[0], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        frames[name], visible[name], methods[method] = {"step": index}, visible_targets(scene, room, rendered, protocol), [name]
    for name in kept:
        visible[name] = visible_targets(scene, room, render_at(scene, poses, int(re.search(r"_w(\d+)$", name).group(1))), protocol)
    return {"group": room["group"], "frames": frames, "methods": methods, "visible": visible, "actions": {"door0": 0, "door4": door["door4"]}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rooms", type=Path, required=True)
    p.add_argument("--walk", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--every", type=int, default=4)
    p.add_argument("--scenes-dir", type=Path, default=Path.home() / "datasets/mp3d/scene_datasets/mp3d")
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args(argv)
    protocol = protocol_for("v1").with_split("train")
    wanted = kept_frames(args.walk, args.every)
    for folder in sorted(d for d in args.rooms.iterdir() if (d / "regions.json").exists()):
        rooms = [r for r in json.loads((folder / "regions.json").read_text())["rooms"] if r["room"] in wanted]
        if not rooms or (args.out / folder.name / "selection.json").exists():
            continue
        try:
            scene = Scene(folder.name, args.scenes_dir, protocol, args.seed)
            out = args.out / folder.name
            (out / "frames").mkdir(parents=True, exist_ok=True)
            result = {r["room"]: select_room(scene, r, wanted[r["room"]][1], protocol, args.seed, out) for r in rooms}
            scene.close()
            (out / "selection.json").write_text(json.dumps(result))
            print("%s: %d rooms" % (folder.name, len(result)), flush=True)
        except Exception:
            print("%s: FAILED\n%s" % (folder.name, traceback.format_exc()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
