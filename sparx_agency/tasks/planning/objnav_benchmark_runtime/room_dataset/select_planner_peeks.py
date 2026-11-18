"""Peek frames from the planner's own recorded runs: for the first visit to each HM3D room, the frame at arrival, the frame four actions in, and the best three frames of the first thirty actions; written in the layout the target-score tools read."""
import argparse
import csv
import json
import os
import traceback
from pathlib import Path

import cv2
import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import protocol_for
from sparx_agency.tasks.planning.objnav_benchmark_runtime.object_dataset.geometry import habitat_position
from sparx_agency.tasks.planning.objnav_benchmark_runtime.review_recording import events_command, load_steps
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.planner_peeks import parse_events, peek_steps, region_at, visit_intervals
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.select_door_views import visible_targets
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.select_hm3d_views import Hm3dScene
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.view_selection import describe_candidates
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.walk_views import keep_best

KEPT = 3
MAX_PITCH_RAD = 0.05


def trajectory(directory):
    with open(Path(directory) / "trajectory.csv") as handle:
        return [{k: float(v) if v not in ("", None) and k in ("x", "y", "z", "yaw", "camera_pitch") else v for k, v in row.items()} for row in csv.DictReader(handle)]


def habitat_pose(row):
    return np.array(habitat_position({"x": row["x"], "y": row["y"], "z": row["z"]}), dtype=float), row["yaw"]


def render_pose(scene, position, yaw):
    rgb, depth, pose = scene.render(position, yaw)
    return rgb, depth, (*pose, yaw)


def first_visits(directory, scene, rooms):
    """``{GT room name: (arrival, end)}`` for the first planner visit to each sampled room, by the room its arrival pose lies in."""
    rows, lines = load_steps(directory), []
    events_command(rows, lines.append)
    steps = trajectory(directory)
    by_region = {r["region"]: r["room"] for r in rooms}
    found = {}
    for _, arrival, end in visit_intervals(parse_events("\n".join(lines)), last_step=len(steps) - 1):
        region = region_at(habitat_pose(steps[arrival])[0], scene.cells, scene.regions)
        if region in by_region and by_region[region] not in found:
            found[by_region[region]] = (arrival, end)
    return found


def peeks_of_room(scene, room, steps, arrival, end, protocol, folders):
    walk_folder, door_folder = folders
    inside_step, window = peek_steps(arrival, end)[1], peek_steps(arrival, end)[2]
    level = [s for s in window if abs(steps[s]["camera_pitch"]) <= MAX_PITCH_RAD]
    poses = [habitat_pose(steps[s]) for s in level]
    frames = describe_candidates(poses, scene.render, scene.room_cells(room), protocol, None)
    kept = [level[frames[i]["index"]] for i in keep_best(frames, KEPT)]
    names, visible = {}, {}
    for step in sorted({arrival, inside_step, *kept}):
        position, yaw = habitat_pose(steps[step])
        rendered = render_pose(scene, position, yaw)
        name = "%s_p%d" % (room["room"], step)
        for folder in (walk_folder, door_folder):
            cv2.imwrite(str(folder / "frames" / (name + ".jpg")), cv2.cvtColor(rendered[0], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        names[step], visible[name] = name, visible_targets(scene, room, rendered, protocol)
    frames_entry = {names[s]: {"step": s} for s in names}
    walk = {"group": "", "frames": frames_entry, "visible": visible, "methods": {"walk_keep3": [names[s] for s in kept]}, "actions": {"walk_keep3": 0}}
    door = {"group": "", "frames": frames_entry, "visible": visible, "methods": {"door0": [names[arrival]], "door4": [names[inside_step]]},
            "actions": {"door0": 0, "door4": inside_step - arrival}}
    return walk, door


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=Path, required=True, help="a planner run folder: <scene>/recordings/<episode>/")
    p.add_argument("--rooms", type=Path, required=True, help="the HM3D rooms of those scenes (rooms.json and cells.npz per scene)")
    p.add_argument("--walk-out", type=Path, required=True)
    p.add_argument("--door-out", type=Path, required=True)
    p.add_argument("--scenes-dir", default=os.environ.get("HM3D_SCENES_DIR"))
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args(argv)
    protocol = protocol_for("v1").with_split("train")
    for scene_dir in sorted(d for d in args.batch.iterdir() if (d / "recordings").is_dir()):
        name = scene_dir.name
        if not (args.rooms / name / "rooms.json").exists() or (args.door_out / name / "selection.json").exists():
            continue
        try:
            rooms = json.loads((args.rooms / name / "rooms.json").read_text())
            by_name = {r["room"]: r for r in rooms}
            scene = Hm3dScene(name, args.scenes_dir, protocol, args.seed, args.rooms / name)
            folders = []
            for out in (args.walk_out, args.door_out):
                (out / name / "frames").mkdir(parents=True, exist_ok=True)
                folders.append(out / name)
            walk_result, door_result = {}, {}
            for recording in sorted((scene_dir / "recordings").iterdir()):
                steps = trajectory(recording)
                for room_name, (arrival, end) in first_visits(recording, scene, rooms).items():
                    if room_name not in walk_result:
                        walk_result[room_name], door_result[room_name] = peeks_of_room(scene, by_name[room_name], steps, arrival, end, protocol, folders)
            scene.close()
            (args.walk_out / name / "selection.json").write_text(json.dumps(walk_result))
            (args.door_out / name / "selection.json").write_text(json.dumps(door_result))
            print("%s: %d rooms visited" % (name, len(walk_result)), flush=True)
        except Exception:
            print("%s: FAILED\n%s" % (name, traceback.format_exc()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
