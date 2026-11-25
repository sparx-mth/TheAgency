"""Frames from a spin in place at the planner's arrival pose in each HM3D room: the arrival view and a view after each further 30-degree turn to the left, up to a full warm-up, with the turns each shorter spin costs."""
import argparse
import json
import os
import traceback
from pathlib import Path

import cv2
import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import protocol_for
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.select_door_views import visible_targets
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.select_hm3d_views import Hm3dScene
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.select_planner_peeks import first_visits, habitat_pose, render_pose, trajectory

TURN_DEG = 30
TURNS = (3, 6, 10)


def spin_offsets(turns):
    return [TURN_DEG * i for i in range(turns + 1)]


def spin_methods(turns_list):
    """``({method: view offsets in degrees}, {method: turns it costs})``; ``spin0`` is the arrival view alone."""
    methods = {"spin%d" % t: spin_offsets(t) for t in (0, *turns_list)}
    return methods, {name: len(offsets) - 1 for name, offsets in methods.items()}


def spin_of_room(scene, room, steps, arrival, protocol, folder):
    position, yaw = habitat_pose(steps[arrival])
    methods, actions = spin_methods(TURNS)
    names, visible = {}, {}
    for offset in spin_offsets(max(TURNS)):
        name = "%s_s%d" % (room["room"], offset)
        rendered = render_pose(scene, position, yaw + np.deg2rad(offset))
        cv2.imwrite(str(folder / "frames" / (name + ".jpg")), cv2.cvtColor(rendered[0], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        names[offset], visible[name] = name, visible_targets(scene, room, rendered, protocol)
    return {"group": "", "frames": {n: {"offset_deg": o} for o, n in names.items()}, "visible": visible,
            "methods": {m: [names[o] for o in offsets] for m, offsets in methods.items()}, "actions": actions}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=Path, action="append", required=True)
    p.add_argument("--rooms", type=Path, action="append", required=True)
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
                    for room_name, (arrival, _) in first_visits(recording, scene, rooms, "poses").items():
                        if room_name not in result:
                            result[room_name] = spin_of_room(scene, by_name[room_name], steps, arrival, protocol, args.out / name)
                scene.close()
                (args.out / name / "selection.json").write_text(json.dumps(result))
                print("%s: %d rooms" % (name, len(result)), flush=True)
            except Exception:
                print("%s: FAILED\n%s" % (name, traceback.format_exc()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
