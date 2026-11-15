"""Measure habitat-sim's step-filter placements on HM3D scenes with the harness's own simulator and kinematic check.

Seeded random walks (MOVE_FORWARD 60 %, a turn either way 20 % each) from every published start of each scene;
every action is scored with :func:`check_motion` under the protocol's tolerance, once as the runner did before
2026-10-27 (no placement) and once as it does now (``placed`` = the simulator's own projection of the pose,
``HabitatRGBDSimulator.last_placed_pose``), and the placements themselves -- the projection differing from the
pose -- are counted and sized. This is how the two refusals that ended scenes of the 2026-10-26 v1 sweep were
reproduced (LESSONS.md 2026-10-27): 148,500 actions per scene, 00878 turns displaced up to 0.313 m and strides
carried up to 0.299 m across their heading, 00832 strides climbed up to 0.740 m; all pass with the placement
reported. Needs the Habitat interpreter, the episodes and the licensed scenes (hm3d/README.md section 1).

    "$HAB_PY" -m sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.placement_probe \\
        00878-XB4GS9ShBRE 00832-qyAac8rV8Zk --steps 1500 --seed 17 --out probe.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

from sparx_agency.core.planning.objnav.errors import EnvContractError
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.tasks.planning.objnav_benchmark.kinematics import check_motion
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.dataset import HM3DDataset
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import protocol_for


def _quantile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def probe_scene(simulator, dataset, scene, spec, tolerance, steps, rng, seed):
    """Walk every published start of ``scene`` for ``steps`` actions; return the placements and the refusals."""
    rows = [row for row in dataset.episodes.values() if row.scene_key == scene]
    if not rows:
        raise SystemExit("no published episode of %s in %s" % (scene, dataset.episodes_dir))
    scene_path, navmesh_path = dataset.scene_assets(rows[0])
    actions = 0
    placement_xy, placement_dz = [], []
    turn_moves, across, climbs = [], [], []
    refused_plain, refused_placed = [], []
    for row in rows:
        _, _, before = simulator.reset(scene_path, navmesh_path, row.start_position, row.start_rotation_wxyz, seed)
        for index in range(steps):
            draw = rng.random()
            action = (DiscreteAction.MOVE_FORWARD if draw < 0.6
                      else DiscreteAction.TURN_LEFT if draw < 0.8 else DiscreteAction.TURN_RIGHT)
            placed = getattr(simulator, "last_placed_pose", None)
            _, _, after = simulator.step(action)
            actions += 1
            dx, dy, dz = after.x - before.x, after.y - before.y, after.z - before.z
            if action == DiscreteAction.MOVE_FORWARD:
                cos_yaw, sin_yaw = math.cos(before.yaw), math.sin(before.yaw)
                across.append(abs(dy * cos_yaw - dx * sin_yaw))
                climbs.append(abs(dz))
            else:
                turn_moves.append(math.sqrt(dx * dx + dy * dy + dz * dz))
            if placed is not None and (placed.x, placed.y, placed.z) != (before.x, before.y, before.z):
                placement_xy.append(math.hypot(placed.x - before.x, placed.y - before.y))
                placement_dz.append(abs(placed.z - before.z))
            if index >= 1:    # the first body action may still settle a published start
                try:
                    check_motion(action, before, after, spec, tolerance)
                except EnvContractError as exc:
                    refused_plain.append(str(exc)[:240])
                try:
                    check_motion(action, before, after, spec, tolerance, placed=placed)
                except EnvContractError as exc:
                    refused_placed.append(str(exc)[:240])
            before = after
    return {"actions": actions, "starts": len(rows),
            "placements": {"count": len(placement_xy), "xy_max_m": max(placement_xy, default=None),
                           "xy_p99_m": _quantile(placement_xy, 0.99), "dz_max_m": max(placement_dz, default=None)},
            "turn_move_m": {"max": max(turn_moves, default=None), "p999": _quantile(turn_moves, 0.999)},
            "forward_across_m": {"max": max(across, default=None), "p999": _quantile(across, 0.999)},
            "forward_abs_dz_m": {"max": max(climbs, default=None), "p999": _quantile(climbs, 0.999)},
            "refused_without_placement": {"count": len(refused_plain), "examples": refused_plain[:5]},
            "refused_with_placement": {"count": len(refused_placed), "examples": refused_placed[:5]}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("scenes", nargs="+", help="scene keys, e.g. 00878-XB4GS9ShBRE")
    parser.add_argument("--version", default="v1", choices=("v1", "v2"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--episodes-dir", default=None, help="default $HM3D_EPISODES_DIR or ~/datasets/objectnav/hm3d/<version>/<split>")
    parser.add_argument("--scenes-dir", default=None, help="default $HM3D_SCENES_DIR or ~/datasets/scene_datasets")
    parser.add_argument("--steps", type=int, default=300, help="actions per published start (default 300)")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--out", required=True, help="the JSON report")
    args = parser.parse_args(argv)
    data = Path(os.environ.get("DATA", os.path.expanduser("~/datasets")))
    episodes_dir = Path(args.episodes_dir or os.environ.get("HM3D_EPISODES_DIR")
                        or data / "objectnav" / "hm3d" / args.version / args.split)
    scenes_dir = Path(args.scenes_dir or os.environ.get("HM3D_SCENES_DIR") or data / "scene_datasets")
    protocol = protocol_for(args.version).with_split(args.split)
    dataset = HM3DDataset(episodes_dir, scenes_dir, protocol)
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator import HabitatRGBDSimulator
    spec, tolerance = protocol.actions(), protocol.kinematics()
    simulator = HabitatRGBDSimulator(protocol.camera(), spec, protocol.agent_radius_m, protocol.allow_sliding,
                                     args.gpu_device, height_m=protocol.agent_height_m, navmesh=protocol.navmesh,
                                     depth_holes=protocol.depth_holes())
    rng = random.Random(args.seed)
    report = {"version": args.version, "split": args.split, "steps_per_start": args.steps, "seed": args.seed,
              "tolerance": {"climb_m": tolerance.climb_m, "reproject_m": tolerance.reproject_m}, "scenes": {}}
    started = time.time()
    try:
        for scene in args.scenes:
            result = probe_scene(simulator, dataset, scene, spec, tolerance, args.steps, rng, args.seed)
            report["scenes"][scene] = result
            print("%s: %d actions | placements %d (xy max %s m) | turn move max %s | forward across max %s | "
                  "|dz| max %s | refused without placement %d, with placement %d  (%.0f s)"
                  % (scene, result["actions"], result["placements"]["count"], result["placements"]["xy_max_m"],
                     result["turn_move_m"]["max"], result["forward_across_m"]["max"], result["forward_abs_dz_m"]["max"],
                     result["refused_without_placement"]["count"], result["refused_with_placement"]["count"],
                     time.time() - started), flush=True)
            Path(args.out).write_text(json.dumps(report, indent=1))
    finally:
        simulator.close()
    return 1 if any(s["refused_with_placement"]["count"] for s in report["scenes"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
