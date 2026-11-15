"""Visualization-only smooth replay of a finished HM3D recording.

The evaluation's own video is the benchmark's: one frame per discrete action,
0.25 m and 30 degrees at a time, which looks like teleporting because it *is*.
Changing those observations or actions would change the benchmark, so the
smoother motion lives here instead -- rendered afterwards, from the recorded
poses, into a separate ``smooth_rgb.mp4``. It never touches the original
trajectory, metrics or video, and no interpolated frame was ever shown to the
policy.

Run it only once the evaluation has released the rendering GPU.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess

from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator import (
    HabitatRGBDSimulator)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.replay import render_replay
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import protocol_for


def _scene_assets(config, scene_key):
    """The recorded mesh and navmesh of one scene, from the run's own manifest."""
    meshes = [Path(row["path"]) for row in (config.get("dataset") or {}).get("files", [])
              if row["path"].endswith(".basis.glb")
              and Path(row["path"]).parent.name == scene_key]
    if len(meshes) != 1:
        raise ValueError("Cannot identify the exact recorded mesh of scene %r; the "
                         "run manifest lists %d candidates" % (scene_key, len(meshes)))
    glb = meshes[0]
    return glb, glb.with_name(glb.name[:-len(".glb")] + ".navmesh")


def _gpu_is_free():
    used = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True)
    return int(used.strip().splitlines()[0]) <= 512


def replay_run(run_dir, *, subframes=4, allow_shared_gpu=False):
    """Render a smooth replay for every completed recording under ``run_dir``.

    Args:
        run_dir: A results directory that was produced with ``--record``.
        subframes: Interpolated frames per recorded action, 1..12.
        allow_shared_gpu: Skip the "is the evaluation still rendering" check.

    Returns:
        The paths written. A recording that already has one is an error, not a
        silent overwrite.
    """
    root = Path(run_dir).expanduser()
    config = json.loads((root / "run.json").read_text())["config"]
    protocol = protocol_for(config["protocol"]["dataset_version"])
    if not allow_shared_gpu and not _gpu_is_free():
        raise RuntimeError("Wait for the evaluation to release its rendering GPU, "
                           "or pass --allow-shared-gpu")
    bridge = HabitatRGBDSimulator(
        protocol.camera(), protocol.actions(), protocol.agent_radius_m,
        protocol.allow_sliding, height_m=protocol.agent_height_m,
        navmesh=protocol.navmesh)
    outputs, already = [], []
    try:
        for metrics_file in sorted((root / "recordings").glob("*/metrics.json")):
            if (metrics_file.parent / "smooth_rgb.mp4").exists():
                # render_replay refuses to overwrite one, which is right for a
                # single directory and wrong for a sweep: a second pass over a
                # run that gained recordings would otherwise die on the first
                # one it had already done and never reach the new ones.
                already.append(str(metrics_file.parent))
                continue
            row = json.loads(metrics_file.read_text())
            glb, navmesh = _scene_assets(config, row["scene_id"])

            def render_pose(pose, glb=glb, navmesh=navmesh):
                # Our ENU pose back into Habitat's +Y-up frame, and a yaw-only
                # WXYZ quaternion about Habitat's +Y. Camera pitch is not
                # replayed: a teleport reset restores the sensor to level.
                position = (-pose.y, pose.z, -pose.x)
                rotation = (math.cos(pose.yaw / 2), 0.0, math.sin(pose.yaw / 2), 0.0)
                rgb, _, _ = bridge.reset(glb, navmesh, position, rotation, seed=0)
                return rgb

            outputs.append(str(render_replay(metrics_file.parent, render_pose,
                                             subframes=subframes)))
    finally:
        bridge.close()
        # Written even when a recording failed, so a partial pass leaves an
        # honest manifest rather than none at all.
        (root / "smooth_replay.json").write_text(json.dumps(
            {"visual_only": True, "subframes": subframes,
             "policy_observations_changed": False,
             "original_results_changed": False,
             "already_rendered": already, "outputs": outputs}, indent=2))
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--subframes", type=int, default=4)
    parser.add_argument("--allow-shared-gpu", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(replay_run(args.run_dir, subframes=args.subframes,
                                allow_shared_gpu=args.allow_shared_gpu), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
