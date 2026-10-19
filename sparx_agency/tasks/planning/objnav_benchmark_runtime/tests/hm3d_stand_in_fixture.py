"""Build an HM3D-format development split on a Gibson mesh, to run the HM3D adapter where the HM3D scenes are absent.

The HM3D scenes are licensed and not on every machine; the Gibson meshes often
are. This tool writes a split the HM3D loader accepts -- ``<split>.json.gz``,
``content/<stem>.json.gz`` with ``goals_by_category`` and view points, and a
scene tree ``hm3d_v0.2/<split>/00001-<Scene>/<Scene>.basis.{glb,navmesh}`` of
symlinks to the Gibson files -- so that ``hm3d.run`` executes its real path end
to end: habitat-sim's navmesh recompute at the agent's dimensions, the
view-point geodesic, habitat-lab's success rule, the recording and the report.

What it is **not**: an HM3D number, or a Gibson one. The episodes are Gibson's
published starts; the goal objects are read off SemExp's evaluator semantic
map (``val_info.pbz2``) and the view points are the navigable points within
1 m of an object's footprint -- habitat-lab's generator also requires a view
IoU, which no semantic map can give. The split is named so that the protocol
id and the results say ``dev_gibson``, never ``val``.

Example::

    "$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.hm3d_stand_in_fixture \\
        --scene Darden --categories toilet chair bed --root ~/datasets/objectnav/hm3d_stand_in
    SPLIT=dev_gibson HM3D_EPISODES_DIR=~/datasets/objectnav/hm3d_stand_in/episodes \\
        HM3D_SCENES_DIR=~/datasets/objectnav/hm3d_stand_in/scenes \\
        hm3d/run_benchmark.sh --scene 00001-Darden --limit 3 --record --episodes-per-scene 3

Needs the Habitat interpreter (habitat-sim loads the mesh to build the navmesh
and measure the geodesics) and a rendering device, like a run.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np

from sparx_agency.core.planning.objnav.labels.datasets import hm3d as hm3d_labels
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.dataset import GibsonDataset
from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator import HabitatRGBDSimulator
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import HM3D_V2

#: Gibson's category spellings to HM3D's (the same six objects).
GIBSON_TO_HM3D = {"chair": "chair", "couch": "sofa", "potted plant": "plant",
                  "bed": "bed", "toilet": "toilet", "tv": "tv_monitor"}
HM3D_TO_GIBSON = {v: k for k, v in GIBSON_TO_HM3D.items()}
#: habitat-lab's view-point generator keeps points within this distance of the object.
VIEW_POINT_MAX_DISTANCE_M = 1.0
VIEW_POINT_GRID_M = 0.1
#: SemExp's map resolution: 5 cm cells, ``row = (x - origin_x) * 20``.
CELLS_PER_METRE = 20.0


def _world_xz(row, col, origin_m):
    """The centre of a SemExp map cell in Habitat (x, z); the inverse of ``GibsonDistanceField.map_cell``."""
    return (row + 0.5) / CELLS_PER_METRE + origin_m[1], (col + 0.5) / CELLS_PER_METRE + origin_m[0]


def _objects(mask):
    """Connected components of the category's footprint, as lists of (row, col)."""
    from scipy import ndimage
    labels, count = ndimage.label(mask)
    return [np.argwhere(labels == index) for index in range(1, count + 1)]


def _view_points(pathfinder, cells, origin_m, floor_y, start_island):
    """Navigable points within 1 m of the footprint, on the start's island, one per 10 cm grid cell."""
    centres = np.array([_world_xz(r, c, origin_m) for r, c in cells])
    lo, hi = centres.min(axis=0) - VIEW_POINT_MAX_DISTANCE_M, centres.max(axis=0) + VIEW_POINT_MAX_DISTANCE_M
    points, seen = [], set()
    for x in np.arange(lo[0], hi[0] + 1e-6, VIEW_POINT_GRID_M):
        for z in np.arange(lo[1], hi[1] + 1e-6, VIEW_POINT_GRID_M):
            snapped = np.asarray(pathfinder.snap_point(np.array([x, floor_y, z], dtype=np.float32)), dtype=float)
            if not np.isfinite(snapped).all() or abs(snapped[1] - floor_y) > 0.5:
                continue
            if pathfinder.get_island(snapped) != start_island:
                continue
            key = (round(float(snapped[0]), 2), round(float(snapped[2]), 2))
            if key in seen:
                continue
            if np.min(np.hypot(centres[:, 0] - snapped[0], centres[:, 1] - snapped[2])) > VIEW_POINT_MAX_DISTANCE_M:
                continue
            seen.add(key)
            points.append([float(snapped[0]), float(snapped[1]), float(snapped[2])])
    return points


def _geodesic(pathfinder, start, targets):
    import habitat_sim
    path = habitat_sim.MultiGoalShortestPath()
    path.requested_start = np.asarray(start, dtype=np.float32)
    path.requested_ends = np.asarray(targets, dtype=np.float32)
    return float(path.geodesic_distance) if pathfinder.find_path(path) else math.inf


def build(root, scene, categories, episodes_dir, scenes_dir, split, gpu_device):
    root = Path(root).expanduser()
    protocol = HM3D_V2.with_split(split)
    gibson = GibsonDataset(Path(episodes_dir).expanduser(), Path(scenes_dir).expanduser(), full=False, scene=scene)
    folder = "00001-%s" % scene
    scene_tree = root / "scenes" / protocol.scene_root_name / split / folder
    scene_tree.mkdir(parents=True, exist_ok=True)
    glb, navmesh = scene_tree / (scene + ".basis.glb"), scene_tree / (scene + ".basis.navmesh")
    for link, target in ((glb, gibson.scenes_dir / (scene + ".glb")), (navmesh, gibson.scenes_dir / (scene + ".navmesh"))):
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target.resolve())
    scene_id = "%s/%s/%s/%s.basis.glb" % (protocol.scene_root_name, split, folder, scene)

    bridge = HabitatRGBDSimulator(protocol.camera(), protocol.actions(), protocol.agent_radius_m, protocol.allow_sliding,
                                  gpu_device, height_m=protocol.agent_height_m, navmesh="agent")
    rows, goals_by_category, report = [], {}, []
    try:
        for category in categories:
            gibson_category = HM3D_TO_GIBSON[category]
            episode = next((e for e in gibson.episodes.values()
                            if e.scene == scene and e.category == gibson_category), None)
            if episode is None:
                raise ValueError("No published Gibson %s episode in %s" % (gibson_category, scene))
            bridge.reset(glb, navmesh, episode.start_position, episode.start_rotation, 0)
            pathfinder = bridge.pathfinder
            start = np.asarray(pathfinder.snap_point(np.asarray(episode.start_position, dtype=np.float32)), dtype=float)
            start_island = pathfinder.get_island(start)
            semantic, origin_cm = gibson.floor(episode)
            origin_m = np.asarray(origin_cm, dtype=float) / 100.0
            goals = []
            for index, cells in enumerate(_objects(semantic[episode.category_index + 1].astype(bool))):
                points = _view_points(pathfinder, cells, origin_m, float(start[1]), start_island)
                if not points:
                    continue
                centre = np.mean([_world_xz(r, c, origin_m) for r, c in cells], axis=0)
                goals.append({"object_category": category, "object_id": index,
                              "object_name": "%s_%d" % (category, index), "object_name_id": None,
                              "position": [float(centre[0]), float(start[1]), float(centre[1])],
                              "radius": None, "room_id": None, "room_name": None,
                              "view_points": [{"agent_state": {"position": p, "rotation": [0.0, 0.0, 0.0, 1.0]},
                                               "iou": None} for p in points]})
            if not goals:
                raise ValueError("No %s object of %s/%s has a navigable view point on the start's island"
                                 % (category, scene, episode.floor_id))
            targets = [vp["agent_state"]["position"] for goal in goals for vp in goal["view_points"]]
            geodesic = _geodesic(pathfinder, episode.start_position, targets)
            if not math.isfinite(geodesic):
                raise ValueError("No %s view point is reachable from the start of %s" % (category, episode.episode_id))
            w, x, y, z = episode.start_rotation
            rows.append({"episode_id": str(len(rows)), "scene_id": scene_id,
                         "scene_dataset_config": "./data/scene_datasets/%s/hm3d_annotated_basis.scene_dataset_config.json" % protocol.scene_root_name,
                         "additional_obj_config_paths": [], "start_position": [float(v) for v in episode.start_position],
                         "start_rotation": [float(x), float(y), float(z), float(w)], "start_room": None,
                         "shortest_paths": None, "object_category": category, "goals": [],
                         "info": {"geodesic_distance": geodesic, "closest_goal_object_id": None,
                                  "source": "gibson %s %s on SemExp's semantic map; not an HM3D episode" % (episode.episode_id, gibson_category)}})
            goals_by_category["%s.basis.glb_%s" % (scene, category)] = goals
            report.append({"episode": "%s/%06d" % (folder, len(rows) - 1), "from_gibson": episode.episode_id,
                           "category": category, "objects": len(goals),
                           "view_points": len(targets), "geodesic_m": round(geodesic, 3)})
    finally:
        bridge.close()

    episodes_root = root / "episodes"
    (episodes_root / "content").mkdir(parents=True, exist_ok=True)
    category_ids = {c: i for i, c in enumerate(hm3d_labels.CATEGORIES)}
    with gzip.open(episodes_root / ("%s.json.gz" % split), "wt") as stream:
        json.dump({"episodes": [], "category_to_task_category_id": category_ids}, stream)
    with gzip.open(episodes_root / "content" / ("%s.json.gz" % scene), "wt") as stream:
        json.dump({"episodes": rows, "goals_by_category": goals_by_category,
                   "category_to_task_category_id": category_ids,
                   "category_to_scene_annotation_category_id": {}}, stream)
    (root / "README.txt").write_text(
        "HM3D-format development split on the Gibson mesh %s (hm3d_stand_in_fixture.py).\n"
        "Not an HM3D result: the starts are Gibson's published ones, the goals are SemExp's semantic map, "
        "the view points are the navigable points within 1 m of a footprint.\n"
        "Run with SPLIT=%s HM3D_EPISODES_DIR=%s HM3D_SCENES_DIR=%s.\n" % (scene, split, episodes_root, root / "scenes"))
    return {"episodes_dir": str(episodes_root), "scenes_dir": str(root / "scenes"), "split": split,
            "scene": folder, "episodes": report}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="Where episodes/ and scenes/ are written")
    parser.add_argument("--scene", default="Darden", help="A Gibson val scene with a mesh on this machine")
    parser.add_argument("--categories", nargs="+", default=["toilet", "chair", "bed"], choices=hm3d_labels.CATEGORIES,
                        help="One episode per category, in this order: the first published Gibson episode of it")
    parser.add_argument("--gibson-episodes-dir", type=Path,
                        default=Path.home() / "datasets/objectnav/gibson/objectnav/gibson/v1.1/val")
    parser.add_argument("--gibson-scenes-dir", type=Path, default=Path.home() / "datasets/gibson/scenes")
    parser.add_argument("--split", default="dev_gibson", help="Anything but val: the results must say what they are")
    parser.add_argument("--gpu-device", type=int, default=0)
    args = parser.parse_args(argv)
    if args.split == "val":
        parser.error("--split val would label a Gibson mesh an HM3D validation result")
    summary = build(args.root, args.scene, args.categories, args.gibson_episodes_dir, args.gibson_scenes_dir,
                    args.split, args.gpu_device)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
