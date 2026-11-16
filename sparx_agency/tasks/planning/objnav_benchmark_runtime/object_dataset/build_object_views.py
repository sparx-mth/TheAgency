"""Render object-centred views of HM3D scenes with exact boxes: approach lines toward each goal object at several headings.

Per scene writes ``<out>/<scene>/frames/NNNNN.jpg`` and ``views.jsonl`` (one row per view). Needs the annotated scene files.
"""
import argparse
import gzip
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.detector_sweep.kinds import kind_of_annotation
from sparx_agency.tasks.planning.objnav_benchmark_runtime.detector_sweep.sample_frames import navigable_points, yaw_quaternion_wxyz
from sparx_agency.tasks.planning.objnav_benchmark_runtime.ground_truth.semantic_frame import instance_extents
from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator import HabitatRGBDSimulator
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import protocol_for
from sparx_agency.tasks.planning.objnav_benchmark_runtime.object_dataset.geometry import (
    approach_positions, is_held_out, jittered_yaws, touches_border, yaw_toward)

MIN_BOX_PIXELS = 150
DISTANCE_TOLERANCE_M = 0.35
SAME_FLOOR_M = 1.0
MAX_JITTER_RAD = math.radians(30)


def goal_objects(episodes_dir, scene):
    """The goal-set objects of the scene's ObjectNav episodes: instance id, benchmark category, habitat position."""
    content = json.load(gzip.open(Path(episodes_dir) / "content" / (scene.split("-", 1)[1] + ".json.gz")))
    return [{"instance": int(goal["object_id"]), "category": goal["object_category"], "position": goal["position"]}
            for goals in content["goals_by_category"].values() for goal in goals]


def boxes_in_view(simulator, goal_instances, width, height):
    names = simulator.semantic_objects()
    rows = []
    for instance, extent in instance_extents(simulator.last_semantic, min_pixels=MIN_BOX_PIXELS).items():
        kind = kind_of_annotation(names.get(instance, ""))
        if kind:
            xyxy = [float(v) for v in extent.xyxy]
            rows.append({"instance": instance, "kind": kind, "name": names[instance], "xyxy": xyxy, "pixels": extent.pixels,
                         "is_goal": instance in goal_instances, "truncated": touches_border(xyxy, width, height)})
    return rows


def camera_ground_point(finder, xz, floor_y, distance_m, center_xz):
    """The navigable point at the wanted range from the object on the object's floor, or None."""
    point = np.asarray(finder.snap_point(np.array([xz[0], floor_y, xz[1]], dtype=np.float32)), dtype=np.float64)
    reached = math.dist((point[0], point[2]), center_xz)
    if not np.isfinite(point).all() or abs(point[1] - floor_y) > SAME_FLOOR_M or abs(reached - distance_m) > DISTANCE_TOLERANCE_M:
        return None
    return point


def build_scene(scene, args, protocol, out):
    import habitat_sim

    scenes = Path(args.scenes_dir)
    stem = scene.split("-", 1)[1]
    glb, navmesh = (scenes / "hm3d" / "train" / scene / (stem + suffix) for suffix in (".basis.glb", ".basis.navmesh"))
    finder = habitat_sim.nav.PathFinder()
    finder.load_nav_mesh(str(navmesh))
    simulator = HabitatRGBDSimulator(
        protocol.camera(), protocol.actions(), protocol.agent_radius_m, protocol.allow_sliding, 0, height_m=protocol.agent_height_m,
        navmesh=protocol.navmesh, depth_holes=protocol.depth_holes(), semantic=True,
        scene_dataset_config=scenes / "hm3d" / "hm3d_annotated_basis.scene_dataset_config.json")
    width, height = protocol.width, protocol.height
    rng = np.random.default_rng(args.seed)
    goals = goal_objects(args.episodes_dir, scene)
    goal_instances = {g["instance"] for g in goals}
    chosen = [goals[i] for i in rng.permutation(len(goals))[:args.objects_per_scene]]
    folder = out / scene
    (folder / "frames").mkdir(parents=True, exist_ok=True)
    rows = []

    def render(position, yaw, **fields):
        rgb, _, pose = simulator.reset(glb, navmesh, position, yaw_quaternion_wxyz(yaw), args.seed)
        name = "%05d.jpg" % len(rows)
        cv2.imwrite(str(folder / "frames" / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        boxes = boxes_in_view(simulator, goal_instances, width, height)
        target_visible = any(b["instance"] == fields.get("target_instance") for b in boxes)
        rows.append(dict(fields, frame=name, scene=scene, x=float(pose.x), y=float(pose.y), z=float(pose.z), yaw=yaw,
                         target_visible=target_visible, boxes=boxes))

    for goal in chosen:
        center = (goal["position"][0], goal["position"][2])
        for sequence in range(args.sequences_per_object):
            azimuth = float(rng.uniform(0, 2 * math.pi))
            for step, xz in enumerate(approach_positions(center, azimuth)):
                distance = math.dist(xz, center)
                point = camera_ground_point(finder, xz, goal["position"][1], distance, center)
                if point is None:
                    continue
                for yaw, jitter in jittered_yaws(yaw_toward((point[0], point[2]), center), args.headings, MAX_JITTER_RAD, rng):
                    render(point, yaw, role="object", target_instance=goal["instance"], category=goal["category"],
                           sequence=sequence, step=step, distance_m=distance, jitter_rad=float(jitter))
    for point in navigable_points(navmesh, args.random_views, 1.5, args.seed):
        render(point, float(rng.uniform(0, 2 * math.pi)), role="random")
    simulator.close()
    with open(folder / "views.jsonl", "w") as handle:
        handle.writelines(json.dumps(r) + "\n" for r in rows)
    return len(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes-dir", default=os.environ.get("HM3D_SCENES_DIR"))
    p.add_argument("--episodes-dir", default=os.environ.get("HM3D_EPISODES_DIR"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scenes", nargs="+", required=True)
    p.add_argument("--objects-per-scene", type=int, default=12)
    p.add_argument("--sequences-per-object", type=int, default=4)
    p.add_argument("--headings", type=int, default=3)
    p.add_argument("--random-views", type=int, default=40)
    p.add_argument("--held-out-fraction", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args(argv)
    protocol = protocol_for("v1").with_split("train")
    for scene in args.scenes:
        if (args.out / scene / "views.jsonl").exists():
            print("%s: already built" % scene)
            continue
        split = "held_out" if is_held_out(scene, args.held_out_fraction) else "dev"
        print("%s (%s): %d views" % (scene, split, build_scene(scene, args, protocol, args.out)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
