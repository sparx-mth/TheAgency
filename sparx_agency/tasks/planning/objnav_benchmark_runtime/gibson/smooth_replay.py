"""Visualization-only smooth RGB replay from a finished Gibson trajectory.

Intermediate camera views are rendered AFTER evaluation and never reach the
policy or metrics. Mesh texture seams remain part of the original Gibson scan.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess

import cv2
import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import PROTOCOL
from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator import HabitatRGBDSimulator
from sparx_agency.tasks.planning.objnav_benchmark_runtime.recording import VideoSink


def render_replay(run_dir, subframes=4):
    """Render smoother motion without modifying any evaluation artifact."""
    if not 1 <= subframes <= 12:
        raise ValueError("subframes must be in 1..12")
    root = Path(run_dir).expanduser()
    info = json.loads((root / "run.json").read_text())
    config = info["config"]
    scene = config.get("scene")
    if scene is None:
        raise ValueError("Replay needs a completed single-scene run")
    meshes = [Path(row["path"]) for row in config["dataset"]["files"] if row["path"].endswith(scene + ".glb")]
    if len(meshes) != 1:
        raise ValueError("Cannot identify the exact recorded scene mesh")
    used = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    if int(used.strip()) > 512:
        raise RuntimeError("Wait for the evaluation to release its rendering GPU")
    bridge = HabitatRGBDSimulator(PROTOCOL.camera(), PROTOCOL.actions(), PROTOCOL.agent_radius_m, PROTOCOL.allow_sliding)
    outputs = []
    try:
        for trajectory in sorted((root / "recordings").glob("*/trajectory.csv")):
            with trajectory.open() as stream:
                rows = list(csv.DictReader(stream))
            if not (trajectory.parent / "metrics.json").exists() or len(rows) < 2:
                continue
            output = trajectory.parent / "smooth_rgb.mp4"
            if output.exists():
                raise FileExistsError("Do not overwrite an existing replay")
            sink = VideoSink(output, 6 * subframes)
            try:
                for first, last in zip(rows, rows[1:]):
                    yaw0, yaw1 = float(first["yaw"]), float(last["yaw"])
                    dyaw = math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0))
                    for index in range(subframes):
                        fraction = index / subframes
                        xyz = [(1 - fraction) * float(first[key]) + fraction * float(last[key]) for key in ("x", "y", "z")]
                        yaw = yaw0 + fraction * dyaw
                        quaternion = (math.cos(yaw / 2), 0.0, math.sin(yaw / 2), 0.0)
                        rgb, _, _ = bridge.reset(meshes[0], meshes[0].with_suffix(".navmesh"),
                                                 (-xyz[1], xyz[2], -xyz[0]), quaternion, seed=0)
                        canvas = np.zeros((900, 1600, 3), np.uint8)
                        canvas[0:900, 200:1400] = cv2.resize(rgb[..., ::-1], (1200, 900))
                        cv2.putText(canvas, "VISUAL-ONLY INTERPOLATION - not policy observations", (20, 32),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                        sink.write(canvas)
            finally:
                sink.close()
            outputs.append(str(output))
    finally:
        bridge.close()
    (root / "smooth_replay.json").write_text(json.dumps({"visual_only": True, "subframes": subframes,
        "policy_observations_changed": False, "outputs": outputs}, indent=2))
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--subframes", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(render_replay(args.run_dir, args.subframes), indent=2))


if __name__ == "__main__":
    main()

