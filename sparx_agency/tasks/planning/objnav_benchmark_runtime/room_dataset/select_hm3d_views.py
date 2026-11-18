"""HM3D rooms (annotated regions) for the target-in-room test: build them from the semantic annotation, then walk each from its door, keep the best three frames, take the door views, and mark which targets are in view."""
import argparse
import json
import os
import traceback
from pathlib import Path

import cv2
import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator import HabitatRGBDSimulator
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import protocol_for
from sparx_agency.tasks.planning.objnav_benchmark_runtime.object_dataset.geometry import is_held_out
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.annotations import instance_centers, read_semantic_table, read_vertices
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.build_room_frames import FLOOR_SAMPLES, MAX_GAP_M, STRUCTURAL, floor_cells, rooms_of_scene
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.geometry import assign_cells
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.hm3d_targets import category_of, target_categories
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.select_door_views import STEPS_IN, render_at, visible_targets
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.select_grid_views import MIN_ROOM_CELLS, Scene, find_entrance
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.select_walk_views import FRAMES, path_length, walk_through
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.target_score import TARGETS
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.target_visibility import steps_in_index
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.view_selection import describe_candidates
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.walk_views import even_picks, keep_best, walk_poses


def build_rooms(scene, scenes_dir, count, seed, folder):
    """Write ``rooms.json`` and ``cells.npz`` of up to ``count`` random annotated regions of the scene; no rendering."""
    import habitat_sim

    inside = Path(scenes_dir) / "hm3d" / "train" / scene
    stem = scene.split("-", 1)[1]
    table = read_semantic_table(inside / (stem + ".semantic.txt"))
    positions, colours = read_vertices(inside / (stem + ".semantic.glb"))
    objects = [o for o in instance_centers(positions, colours, table) if o["name"] not in STRUCTURAL]
    finder = habitat_sim.nav.PathFinder()
    finder.load_nav_mesh(str(inside / (stem + ".basis.navmesh")))
    cells = floor_cells(finder, FLOOR_SAMPLES, seed)
    regions = assign_cells(cells, objects, MAX_GAP_M)
    chosen = rooms_of_scene(cells, regions, finder.navigable_area / len(cells), count, np.random.default_rng(seed))
    rooms = []
    for region in chosen:
        room_cells = cells[regions == region]
        held = [o for o in objects if o["region"] == region]
        rooms.append({"room": "%s_r%d" % (scene.split("-")[0], region), "region": region, "scene": scene, "cells": len(room_cells),
                      "area_m2": float(len(room_cells) * finder.navigable_area / len(cells)), "floor_y": float(np.median(room_cells[:, 1])),
                      "box_xz": [float(room_cells[:, 0].min()), float(room_cells[:, 2].min()), float(room_cells[:, 0].max()), float(room_cells[:, 2].max())],
                      "objects": [{"instance": o["instance"], "name": o["name"], "center": o["center"], "category": category_of(o["name"])} for o in held],
                      "target_categories": target_categories([o["name"] for o in held])})
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "rooms.json").write_text(json.dumps(rooms))
    np.savez_compressed(folder / "cells.npz", cells=cells, regions=regions)


class Hm3dScene(Scene):
    """An HM3D scene whose rooms were built by ``build_rooms``: the same interface as the Matterport3D ``Scene``."""

    def __init__(self, scene, scenes_dir, protocol, seed, folder):
        import habitat_sim

        inside = Path(scenes_dir) / "hm3d" / "train" / scene
        stem = scene.split("-", 1)[1]
        self.glb, self.navmesh, self.seed, self.protocol = inside / (stem + ".basis.glb"), inside / (stem + ".basis.navmesh"), seed, protocol
        loaded = np.load(folder / "cells.npz")
        self.cells, self.regions = loaded["cells"], loaded["regions"]
        self.objects = [dict(o, region=r["region"]) for r in json.loads((folder / "rooms.json").read_text()) for o in r["objects"]]
        self.finder = habitat_sim.nav.PathFinder()
        self.finder.load_nav_mesh(str(self.navmesh))
        self.finder.seed(seed)
        self.simulator = HabitatRGBDSimulator(protocol.camera(), protocol.actions(), protocol.agent_radius_m, protocol.allow_sliding, 0,
                                              height_m=protocol.agent_height_m, navmesh=protocol.navmesh, depth_holes=protocol.depth_holes())
        self.grids = {}

    def room_cells(self, room):
        return self.cells[(self.regions == room["region"]) & (np.abs(self.cells[:, 1] - room["floor_y"]) < 1.0)]

    def is_target(self, obj, target):
        return obj["category"] == target


def save(scene, folder, name, rendered):
    cv2.imwrite(str(folder / "frames" / (name + ".jpg")), cv2.cvtColor(rendered[0], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])


def views_of_room(scene, room, protocol, seed, walk_folder, door_folder):
    room_cells = scene.room_cells(room)
    if len(room_cells) < MIN_ROOM_CELLS:
        return {"skipped": "room cells %d" % len(room_cells)}, {"skipped": "room cells %d" % len(room_cells)}
    entrance = find_entrance(scene, room, room_cells, seed)
    path = None if entrance is None else walk_through(scene, room_cells, entrance)
    if path is None:
        return {"skipped": "no walk through the room"}, {"skipped": "no walk through the room"}
    poses = walk_poses(path, entrance[1])
    frames = describe_candidates(poses, scene.render, room_cells, protocol, None)
    picks = {"walk_even3": [frames[i]["index"] for i in even_picks(len(frames), FRAMES)], "walk_keep3": [frames[i]["index"] for i in keep_best(frames, FRAMES)]}
    walk_named, visible = {}, {}
    for index in sorted({i for chosen in picks.values() for i in chosen}):
        rendered = render_at(scene, poses, index)
        name = "%s_w%d" % (room["room"], index)
        save(scene, walk_folder, name, rendered)
        walk_named[name] = {"step": index}
        visible[name] = visible_targets(scene, room, rendered, protocol)
    walk = {"group": "", "frames": walk_named, "walk_actions": len(poses) - 1, "usable_frames": sum(c["near"] <= 0.4 and c["inside"] >= 0.7 for c in frames),
            "methods": {m: ["%s_w%d" % (room["room"], i) for i in chosen] for m, chosen in picks.items()}, "actions": {m: 0 for m in picks}}
    door_named, methods = {}, {}
    for method, index in {"door0": 0, "door4": steps_in_index(poses, STEPS_IN)}.items():
        rendered = render_at(scene, poses, index)
        name = "%s_d%d" % (room["room"], index)
        save(scene, door_folder, name, rendered)
        door_named[name], visible[name], methods[method] = {"step": index}, visible_targets(scene, room, rendered, protocol), [name]
    door = {"group": "", "frames": door_named, "methods": methods, "visible": visible, "actions": {"door0": 0, "door4": steps_in_index(poses, STEPS_IN)}}
    return walk, door


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes-dir", default=os.environ.get("HM3D_SCENES_DIR"))
    p.add_argument("--rooms-out", type=Path, required=True)
    p.add_argument("--walk-out", type=Path, required=True)
    p.add_argument("--door-out", type=Path, required=True)
    p.add_argument("--scenes", type=int, default=30, help="the first this many train scenes that are not held out")
    p.add_argument("--rooms-per-scene", type=int, default=10)
    p.add_argument("--held-out-fraction", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args(argv)
    protocol = protocol_for("v1").with_split("train")
    names = [d.name for d in sorted((Path(args.scenes_dir) / "hm3d" / "train").iterdir()) if not is_held_out(d.name, args.held_out_fraction)][:args.scenes]
    for scene in names:
        if (args.door_out / scene / "selection.json").exists():
            continue
        try:
            folder = args.rooms_out / scene
            if not (folder / "rooms.json").exists():
                build_rooms(scene, args.scenes_dir, args.rooms_per_scene, args.seed, folder)
            hm3d = Hm3dScene(scene, args.scenes_dir, protocol, args.seed, folder)
            walk_result, door_result = {}, {}
            for sub in (args.walk_out, args.door_out):
                (sub / scene / "frames").mkdir(parents=True, exist_ok=True)
            for room in json.loads((folder / "rooms.json").read_text()):
                walk_result[room["room"]], door_result[room["room"]] = views_of_room(hm3d, room, protocol, args.seed, args.walk_out / scene, args.door_out / scene)
            hm3d.close()
            (args.walk_out / scene / "selection.json").write_text(json.dumps(walk_result))
            (args.door_out / scene / "selection.json").write_text(json.dumps(door_result))
            print("%s: %d rooms" % (scene, len(walk_result)), flush=True)
        except Exception:
            print("%s: FAILED\n%s" % (scene, traceback.format_exc()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
