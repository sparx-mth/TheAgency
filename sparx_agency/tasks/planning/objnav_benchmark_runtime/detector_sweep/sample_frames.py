"""Render frames at random navigable poses of HM3D scenes, with the exact boxes of every annotated object in view.

For each scene: ``--positions`` navmesh points at least ``--spacing`` m apart, ``--headings`` evenly spaced headings each.
Writes ``<out>/<scene>/frames/NNNNN.jpg`` and ``<out>/<scene>/gt.json`` (pose, floor height, boxes of the kinds).
"""
import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.detector_sweep.kinds import kind_of_annotation
from sparx_agency.tasks.planning.objnav_benchmark_runtime.ground_truth.semantic_frame import instance_extents
from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator import HabitatRGBDSimulator
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import protocol_for

MIN_BOX_PIXELS = 400


def navigable_points(navmesh, count, spacing, seed):
    """Up to ``count`` random navmesh points on a large island, at least ``spacing`` m apart."""
    import habitat_sim

    finder = habitat_sim.nav.PathFinder()
    if not finder.load_nav_mesh(str(navmesh)):
        raise RuntimeError("cannot load navmesh %s" % navmesh)
    finder.seed(seed)
    points = []
    for _ in range(count * 40):
        p = np.asarray(finder.get_random_navigable_point(), dtype=np.float64)
        far_enough = all(math.dist((p[0], p[2]), (q[0], q[2])) >= spacing or abs(p[1] - q[1]) > 1.0 for q in points)
        if far_enough and finder.island_radius(p.astype(np.float32)) > 3.0:
            points.append(p)
        if len(points) == count:
            break
    return points


def yaw_quaternion_wxyz(yaw):
    return (math.cos(yaw / 2), 0.0, math.sin(yaw / 2), 0.0)


def boxes_in_view(simulator):
    """Boxes of the annotated objects of the kinds in the current frame."""
    names = simulator.semantic_objects()
    boxes = []
    for instance, extent in instance_extents(simulator.last_semantic, min_pixels=MIN_BOX_PIXELS).items():
        kind = kind_of_annotation(names.get(instance, ""))
        if kind:
            boxes.append({"kind": kind, "name": names[instance], "xyxy": [float(v) for v in extent.xyxy], "pixels": extent.pixels})
    return boxes


def sweep_scene(scene, args, protocol, out):
    scenes_dir = Path(args.scenes_dir)
    stem = scene.split("-", 1)[1]
    glb, navmesh = (scenes_dir / "hm3d" / "val" / scene / (stem + suffix) for suffix in (".basis.glb", ".basis.navmesh"))
    simulator = HabitatRGBDSimulator(
        protocol.camera(), protocol.actions(), protocol.agent_radius_m, protocol.allow_sliding, 0, height_m=protocol.agent_height_m,
        navmesh=protocol.navmesh, depth_holes=protocol.depth_holes(), semantic=True,
        scene_dataset_config=scenes_dir / "hm3d" / "hm3d_annotated_basis.scene_dataset_config.json")
    folder = out / scene
    (folder / "frames").mkdir(parents=True, exist_ok=True)
    rows = []
    for position in navigable_points(navmesh, args.positions, args.spacing, args.seed):
        for h in range(args.headings):
            yaw = 2 * math.pi * h / args.headings
            try:
                rgb, _, pose = simulator.reset(glb, navmesh, position, yaw_quaternion_wxyz(yaw), args.seed)
            except Exception as exc:
                print("skip", scene, position, repr(exc)[:80])
                break
            name = "%05d.jpg" % len(rows)
            cv2.imwrite(str(folder / "frames" / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append({"frame": name, "x": float(pose.x), "y": float(pose.y), "z": float(pose.z), "yaw": yaw, "boxes": boxes_in_view(simulator)})
    simulator.close()
    (folder / "gt.json").write_text(json.dumps(rows))
    return len(rows), sum(len(r["boxes"]) for r in rows)


def annotated_scenes(scenes_dir):
    """The val scenes that ship HM3D-Semantics annotations (most of the release does not)."""
    root = scenes_dir / "hm3d" / "val"
    return sorted(d.name for d in root.iterdir() if d.is_dir() and "-" in d.name and (d / (d.name.split("-", 1)[1] + ".semantic.glb")).is_file())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes-dir", default=os.environ.get("HM3D_SCENES_DIR"), help="holds hm3d/val/<scene>/ and the annotated scene config")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scenes", nargs="+")
    p.add_argument("--positions", type=int, default=75)
    p.add_argument("--headings", type=int, default=4)
    p.add_argument("--spacing", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args(argv)
    protocol = protocol_for("v1").with_split("val")
    scenes = args.scenes or annotated_scenes(Path(args.scenes_dir))
    for scene in scenes:
        if (args.out / scene / "gt.json").exists():
            print("%s: already sampled" % scene)
            continue
        frames, boxes = sweep_scene(scene, args, protocol, args.out)
        print("%s: %d frames, %d boxes of the kinds" % (scene, frames, boxes), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
