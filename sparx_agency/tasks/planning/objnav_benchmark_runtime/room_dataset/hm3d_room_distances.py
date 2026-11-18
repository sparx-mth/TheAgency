"""Walking distance between the middles of the HM3D rooms of each scene, along the navmesh: the travel costs of the room-ordering test."""
import argparse
import json
import os
import traceback
from pathlib import Path

from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import protocol_for
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.geometry import spread_points
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.select_hm3d_views import Hm3dScene
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.select_walk_views import path_length

FILE = "distances.json"


def scene_distances(scene):
    rooms = json.loads((scene.rooms_folder / "rooms.json").read_text())
    middles = []
    for room in rooms:
        cells = scene.room_cells(room)
        middles.append(cells[spread_points(cells[:, [0, 2]], 1)[0]] if len(cells) else None)
    matrix = [[0.0 if i == j else None for j in range(len(rooms))] for i in range(len(rooms))]
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            path = scene.path(middles[i], middles[j]) if middles[i] is not None and middles[j] is not None else None
            matrix[i][j] = matrix[j][i] = None if path is None else path_length(path)
    return {"rooms": [r["room"] for r in rooms], "matrix": matrix}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes-dir", default=os.environ.get("HM3D_SCENES_DIR"))
    p.add_argument("--rooms", type=Path, required=True, help="folder of scenes with rooms.json and cells.npz")
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args(argv)
    protocol = protocol_for("v1").with_split("train")
    result = {}
    for folder in sorted(d for d in args.rooms.iterdir() if (d / "rooms.json").exists()):
        try:
            scene = Hm3dScene(folder.name, args.scenes_dir, protocol, args.seed, folder)
            scene.rooms_folder = folder
            result[folder.name] = scene_distances(scene)
            scene.close()
            print("%s: %d rooms" % (folder.name, len(result[folder.name]["rooms"])), flush=True)
        except Exception:
            print("%s: FAILED\n%s" % (folder.name, traceback.format_exc()), flush=True)
    (args.rooms / FILE).write_text(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
