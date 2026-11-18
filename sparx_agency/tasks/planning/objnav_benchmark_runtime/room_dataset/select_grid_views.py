"""Plan the views of each labelled Matterport3D region from an occupancy grid, as a robot entering by the door would, under action budgets; render only the views planned."""
import argparse
import json
import traceback
import zlib
from pathlib import Path

import cv2
import numpy as np

from sparx_agency.core.planning.objnav.errors import EnvContractError
from sparx_agency.tasks.planning.objnav_benchmark_runtime.detector_sweep.sample_frames import yaw_quaternion_wxyz
from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator import HabitatRGBDSimulator
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import protocol_for
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.build_mp3d_frames import FLOOR_SAMPLES, OBJECT_GAP_M, interior_cells
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.geometry import assign_cells, compass_yaws, spread_points
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.grid_views import OccupancyGrid, entrance_pose, grid_candidates, leg_actions, plan_views, turns_between
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_house import read_house, to_habitat
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.target_score import TARGETS

SPOTS = 24
HEADINGS = 12
FRAMES = 3
BUDGETS = (12, 24, 48)
GRID_CELL_M = 0.1
MIN_ROOM_CELLS = 10
ENTRY_TRIES = 6
ENTRY_GAP_M = 0.5
MIN_GRID_AGREEMENT = 0.9


class Scene:
    """One Matterport3D scene: navmesh, sampled floor, house objects, a plain simulator and the occupancy grid of each floor."""

    def __init__(self, scene, scenes_dir, protocol, seed):
        import habitat_sim

        folder = scenes_dir / scene
        self.glb, self.navmesh, self.seed, self.protocol = folder / (scene + ".glb"), folder / (scene + ".navmesh"), seed, protocol
        house = read_house(folder / (scene + ".house"))
        self.finder = habitat_sim.nav.PathFinder()
        self.finder.load_nav_mesh(str(self.navmesh))
        self.finder.seed(seed)
        cells = np.array([np.asarray(self.finder.get_random_navigable_point(), dtype=float) for _ in range(FLOOR_SAMPLES)])
        self.cells = cells[np.isfinite(cells).all(axis=1)]
        self.objects = [{"region": o["region"], "center": to_habitat(o["center"]), "category": o["category"]} for o in house["objects"]]
        self.simulator = HabitatRGBDSimulator(protocol.camera(), protocol.actions(), protocol.agent_radius_m, protocol.allow_sliding, 0,
                                              height_m=protocol.agent_height_m, navmesh=protocol.navmesh, depth_holes=protocol.depth_holes())
        self.grids = {}

    def render(self, cell, yaw):
        rgb, depth, pose = self.simulator.reset(self.glb, self.navmesh, cell, yaw_quaternion_wxyz(yaw), self.seed)
        return rgb, depth, (float(pose.x), float(pose.y), float(pose.z))

    def grid(self, floor_y):
        key = round(float(floor_y), 1)
        if key not in self.grids:
            lower, _ = self.finder.get_bounds()
            self.grids[key] = OccupancyGrid(self.finder.get_topdown_view(GRID_CELL_M, float(floor_y)), (float(lower[0]), float(lower[2])), GRID_CELL_M)
        return self.grids[key]

    def room_cells(self, room):
        owner = lambda points, r=room["region"]: assign_cells(points, self.objects, OBJECT_GAP_M) == r
        return interior_cells(self.cells, tuple(room["box_xz"]), room["floor_y"], owner, margin=0.0)

    def is_target(self, obj, target):
        return obj["category"] in TARGETS[target]

    def path(self, start, end):
        import habitat_sim

        shortest = habitat_sim.ShortestPath()
        shortest.requested_start, shortest.requested_end = np.asarray(start, dtype=np.float32), np.asarray(end, dtype=np.float32)
        return [np.asarray(p, dtype=float) for p in shortest.points] if self.finder.find_path(shortest) else None

    def close(self):
        self.simulator.close()


def find_entrance(scene, room, room_cells, seed):
    """Pose where a walk from elsewhere on the floor first reaches the room: ``(cell, yaw)`` or None. The walk starts at a seeded random floor point outside the room's box."""
    x0, z0, x1, z1 = room["box_xz"]
    outside = scene.cells[(np.abs(scene.cells[:, 1] - room["floor_y"]) < 0.6) & ~((scene.cells[:, 0] >= x0) & (scene.cells[:, 0] <= x1) & (scene.cells[:, 2] >= z0) & (scene.cells[:, 2] <= z1))]
    if not len(outside):
        return None
    middle = room_cells[spread_points(room_cells[:, [0, 2]], 1)[0]]
    in_room = lambda p: float(np.hypot(room_cells[:, 0] - p[0], room_cells[:, 2] - p[1]).min()) <= ENTRY_GAP_M
    rng = np.random.default_rng([seed, zlib.crc32(room["room"].encode())])
    for source in rng.permutation(len(outside))[:ENTRY_TRIES]:
        path = scene.path(outside[source], middle)
        ground = [(p[0], p[2]) for p in path] if path else []
        found = entrance_pose(ground, in_room) if ground else None
        if found:
            point, yaw = found
            return path[ground.index(point)], yaw
    return None


class Planner:
    """Costs in actions between candidate poses of one room, from the entrance."""

    def __init__(self, scene, candidates, entrance):
        self.scene, self.candidates, self.entrance, self.paths = scene, candidates, entrance, {}

    def leg(self, candidate, index, previous):
        start_cell, start_yaw = (self.entrance if previous is None else (self.candidates[previous]["cell"], self.candidates[previous]["yaw"]))
        if np.allclose(start_cell, candidate["cell"]):
            return turns_between(start_yaw, candidate["yaw"])
        key = (tuple(np.round(start_cell, 3)), tuple(np.round(candidate["cell"], 3)))
        if key not in self.paths:
            self.paths[key] = self.scene.path(start_cell, candidate["cell"])
        path = self.paths[key]
        return 10 ** 6 if path is None else leg_actions([(p[0], p[2]) for p in path], start_yaw, candidate["yaw"])

    def total(self, picks):
        previous, total = None, 0
        for i in picks:
            total += self.leg(self.candidates[i], i, previous)
            previous = i
        return total


def only_spots(candidates, spots):
    return [dict(c, near=1.0) if c["spot"] not in spots else c for c in candidates]


def plan_renderable(scene, candidates, plan, rendered):
    """Run ``plan`` until every view it picks renders; a pose the simulator refuses is dropped from the pool."""
    while True:
        picks = plan(candidates)
        failed = []
        for i in picks:
            if i not in rendered:
                try:
                    rendered[i] = scene.render(candidates[i]["cell"], candidates[i]["yaw"])
                except EnvContractError:
                    failed.append(i)
        if not failed:
            return picks
        for i in failed:
            candidates[i] = dict(candidates[i], near=1.0)


def select_room(scene, room, protocol, seed, folder):
    room_cells = scene.room_cells(room)
    if len(room_cells) < MIN_ROOM_CELLS:
        return {"skipped": "room cells %d" % len(room_cells)}
    entrance = find_entrance(scene, room, room_cells, seed)
    if entrance is None:
        return {"skipped": "no path into the room"}
    grid = scene.grid(room["floor_y"])
    agreement = float(grid.is_free(room_cells[:, 0], room_cells[:, 2]).mean())
    if agreement < MIN_GRID_AGREEMENT:
        return {"skipped": "grid covers %.2f of the room cells" % agreement}
    spots = [room_cells[i] for i in spread_points(room_cells[:, [0, 2]], SPOTS)] + [entrance[0]]
    reachable = [scene.path(entrance[0], spot) is not None for spot in spots]
    poses = [(cell, yaw) for cell in spots for yaw in compass_yaws(HEADINGS)]
    candidates = grid_candidates(grid, room_cells, poses, protocol.width, protocol.height, protocol.hfov_deg, protocol.max_depth_m)
    for i, c in enumerate(candidates):
        c["spot"] = i // HEADINGS
        c["near"] = c["near"] if reachable[c["spot"]] else 1.0
    planner = Planner(scene, candidates, entrance)
    entrance_spot = len(spots) - 1
    plans = {"grid_free": lambda cs: plan_views(cs, FRAMES, None, planner.leg)}
    for b in BUDGETS:
        plans["grid_b%d" % b] = lambda cs, b=b: plan_views(cs, FRAMES, b, planner.leg)
    plans["door_spin"] = lambda cs: plan_views(only_spots(cs, {entrance_spot}), FRAMES, None, planner.leg)
    plans["middle_spin"] = lambda cs: plan_views(only_spots(cs, {0}), FRAMES, None, planner.leg)
    rendered, picks = {}, {}
    for method, plan in plans.items():
        picks[method] = plan_renderable(scene, candidates, plan, rendered)
    frames = {}
    for index in sorted({i for chosen in picks.values() for i in chosen}):
        name = "%s_c%d" % (room["room"], index)
        cv2.imwrite(str(folder / "frames" / (name + ".jpg")), cv2.cvtColor(rendered[index][0], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        frames[name] = {"seen_cells": int(candidates[index]["cells"].sum()), "near": candidates[index]["near"], "spot": candidates[index]["spot"]}
    return {"group": room["group"], "room_cells": len(room_cells), "frames": frames, "entrance": [float(entrance[0][0]), float(entrance[0][2])],
            "methods": {m: ["%s_c%d" % (room["room"], i) for i in chosen] for m, chosen in picks.items()},
            "actions": {m: planner.total(chosen) for m, chosen in picks.items()}}


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
