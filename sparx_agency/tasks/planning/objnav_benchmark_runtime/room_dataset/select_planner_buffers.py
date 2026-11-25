"""A buffer of about twenty frames along the planner's own path from the moment it first walks into each HM3D room, with what each frame sees, in the layout the target-score tools read."""
import argparse
import json
import os
import traceback
from pathlib import Path

import cv2
import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import protocol_for
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.select_door_views import visible_targets
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.select_grid_views import MIN_ROOM_CELLS
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.select_hm3d_views import Hm3dScene
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.select_planner_peeks import MAX_PITCH_RAD, first_visits, habitat_pose, render_pose, trajectory
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.view_selection import describe_candidates

BUFFER_FRAMES = 20
BUFFER_SPAN_STEPS = 40
METHOD = "buffer20"


def buffer_steps(steps, arrival, end):
    """Up to ``BUFFER_FRAMES`` steps, evenly spread over the first ``BUFFER_SPAN_STEPS`` of the visit, with the camera level."""
    level = [s for s in range(arrival, min(end, arrival + BUFFER_SPAN_STEPS) + 1) if abs(steps[s]["camera_pitch"]) <= MAX_PITCH_RAD]
    spots = np.linspace(0, len(level) - 1, min(BUFFER_FRAMES, len(level))).round().astype(int) if level else []
    return [level[i] for i in dict.fromkeys(spots)]


def buffer_of_room(scene, room, steps, arrival, end, protocol, folder):
    room_cells = scene.room_cells(room)
    if len(room_cells) < MIN_ROOM_CELLS:
        return None
    chosen = buffer_steps(steps, arrival, end)
    described = describe_candidates([habitat_pose(steps[s]) for s in chosen], scene.render, room_cells, protocol, None)
    frames, visible = {}, {}
    for view in described:
        step = chosen[view["index"]]
        name = "%s_b%d" % (room["room"], step)
        rendered = render_pose(scene, *habitat_pose(steps[step]))
        cv2.imwrite(str(folder / "frames" / (name + ".jpg")), cv2.cvtColor(rendered[0], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        frames[name] = {"step": step - arrival, "yaw": float(view["yaw"]), "near": float(view["near"]), "inside": float(view["inside"]), "floor_share": float(np.mean(view["cells"]))}
        visible[name] = visible_targets(scene, room, rendered, protocol)
    return {"group": "", "frames": frames, "visible": visible, "methods": {METHOD: sorted(frames)}, "actions": {METHOD: 0}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=Path, action="append", required=True, help="a planner run folder: <scene>/recordings/<episode>/ (repeatable)")
    p.add_argument("--rooms", type=Path, action="append", required=True, help="the HM3D rooms of those scenes, one per --batch")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scenes-dir", default=os.environ.get("HM3D_SCENES_DIR"))
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args(argv)
    protocol = protocol_for("v1").with_split("train")
    for batch, rooms_dir in zip(args.batch, args.rooms):
        for scene_dir in sorted(d for d in batch.iterdir() if (d / "recordings").is_dir()):
            name = scene_dir.name
            if not (rooms_dir / name / "rooms.json").exists() or (args.out / name / "selection.json").exists():
                continue
            try:
                rooms = json.loads((rooms_dir / name / "rooms.json").read_text())
                by_name = {r["room"]: r for r in rooms}
                scene = Hm3dScene(name, args.scenes_dir, protocol, args.seed, rooms_dir / name)
                (args.out / name / "frames").mkdir(parents=True, exist_ok=True)
                result = {}
                for recording in sorted((scene_dir / "recordings").iterdir()):
                    steps = trajectory(recording)
                    for room_name, (arrival, end) in first_visits(recording, scene, rooms, "poses").items():
                        if room_name not in result:
                            buffered = buffer_of_room(scene, by_name[room_name], steps, arrival, end, protocol, args.out / name)
                            if buffered is not None:
                                result[room_name] = buffered
                scene.close()
                (args.out / name / "selection.json").write_text(json.dumps(result))
                print("%s: %d rooms, %d frames" % (name, len(result), sum(len(r["frames"]) for r in result.values())), flush=True)
            except Exception:
                print("%s: FAILED\n%s" % (name, traceback.format_exc()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
