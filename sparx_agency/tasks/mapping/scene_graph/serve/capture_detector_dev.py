"""Capture annotated TRAIN-only detector fixtures, not navigation episodes.

This small optional AI2-THOR data collector is deliberately separate from the
navigation adapter: it observes masks for detector evaluation, never gives them
to a policy, and does not change the shared policy's camera control.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import requests

from sparx_agency.tasks.common.model_registry.download.verify import sha256_of

TARGET_LABELS = {
    "AlarmClock": "alarm clock", "Apple": "apple", "BaseballBat": "baseball bat",
    "BasketBall": "basketball", "Bowl": "bowl", "GarbageCan": "garbage can",
    "HousePlant": "house plant", "Laptop": "laptop", "Mug": "mug",
    "SprayBottle": "spray bottle", "Television": "television", "Vase": "vase"}
BUILD = "bad5bc2b250615cb766ffb45d455c211329af17e"


def mask_annotations(event):
    """Tight visible-mask boxes, including distant objects outside STOP visibility."""
    objects = {obj["objectId"]: obj for obj in event.metadata["objects"]}
    boxes = []
    for object_id, mask in event.instance_masks.items():
        obj = objects.get(object_id, {})
        object_type = obj.get("objectType", object_id.split("|")[0])
        if object_type not in TARGET_LABELS:
            continue
        ys, xs = np.where(mask)
        if len(xs):
            boxes.append({"label": TARGET_LABELS[object_type], "object_id": object_id,
                          "xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                          "mask_pixels": len(xs), "ignore": len(xs) < 4,
                          "stop_visible": bool(obj.get("visible", False))})
    return boxes


def capture(args):
    if args.output.exists():
        raise FileExistsError("Use a new output directory; captures are immutable")
    if (len(args.scenes) < 3 or len(set(args.scenes)) != len(args.scenes)
            or any(not s.startswith("FloorPlan_Train") for s in args.scenes)):
        raise ValueError("Use distinct training scenes, with two reserved for calibration")
    families = [s.rsplit("_", 1)[0] for s in args.scenes]
    if set(families[:2]) & set(families[2:]):
        raise ValueError("Calibration and measurement must use different training layout families")
    used = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    if max(int(v.strip()) for v in used.splitlines()) > 512:
        raise RuntimeError("Rendering GPU is occupied; do not displace its owner")
    binary = Path.home() / ".ai2thor" / "releases" / ("thor-Linux64-" + BUILD)
    executable = binary / binary.name
    if not executable.is_file():
        raise FileNotFoundError("Provision the reference AI2-THOR build explicitly first")
    response = requests.get(args.detector_url.rstrip("/") + "/health", timeout=10)
    response.raise_for_status()
    vocabulary = response.json()["classes"]
    from ai2thor.controller import Controller

    args.output.mkdir(parents=True)
    init = {"width": 640, "height": 480, "agentMode": "locobot", "gridSize": 0.25,
            "rotateStepDegrees": 30, "visibilityDistance": 1.0, "snapToGrid": False,
            "fieldOfView": 63.453048374758716, "renderDepthImage": True,
            "renderInstanceSegmentation": True}
    controller = Controller(local_executable_path=str(executable), **init)
    frames, sources = [], {}
    try:
        for scene_index, scene in enumerate(args.scenes):
            source = args.episodes / "train" / "episodes" / (scene + ".json.gz")
            with gzip.open(source, "rt") as stream:
                episodes = json.load(stream)
            if isinstance(episodes, dict):
                episodes = episodes["episodes"]
            sources[scene] = sha256_of(source)
            controller.reset(scene)
            starts = []
            for episode in episodes:
                position = episode["initial_position"]
                if position not in starts:
                    starts.append(position)
                if len(starts) == args.starts:
                    break
            for start_index, position in enumerate(starts):
                for horizon in (0, 30):
                    for yaw in range(0, 360, 60):
                        event = controller.step(action="TeleportFull", **position,
                                                rotation={"x": 0, "y": yaw, "z": 0},
                                                horizon=horizon, standing=True)
                        if not event.metadata["lastActionSuccess"]:
                            raise RuntimeError(event.metadata.get("errorMessage"))
                        name = "%s-%d-h%d-y%d.npz" % (scene, start_index, horizon, yaw)
                        path = args.output / name
                        np.savez_compressed(path, rgb=event.frame, depth=event.depth_frame)
                        agent = event.metadata["agent"]
                        p = agent["position"]
                        # Capture schema is ENU; this is not another simulator adapter.
                        pose = [p["x"], p["z"], p["y"] - 0.901,
                                math.radians(90 - agent["rotation"]["y"]),
                                math.radians(agent["cameraHorizon"])]
                        frames.append({"file": name, "sha256": sha256_of(path), "scene": scene,
                                       "partition": "calibration" if scene_index < 2 else "measurement",
                                       "pose": pose, "annotations": mask_annotations(event)})
    finally:
        controller.stop()
    manifest = {"schema": 1, "dataset": "RoboTHOR", "split": "train",
                "purpose": "development detector calibration/check; NOT held-out benchmark",
                "annotation": "visible instance-mask bounds; <4 mask pixels ignored; all distances included",
                "selection": "first distinct training starts; fixed 60-degree bearings; horizons 0/30",
                "build": BUILD, "binary_sha256": sha256_of(executable),
                "ai2thor_version": importlib.metadata.version("ai2thor"),
                "initialization": init, "episode_source_sha256": sources,
                "vocabulary": vocabulary, "targets": list(TARGET_LABELS.values()), "frames": frames}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Captured", len(frames), "annotated training frames at", args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector-url", required=True, help="Read vocabulary only; never reconfigure")
    parser.add_argument("--scenes", nargs="+", default=["FloorPlan_Train1_1", "FloorPlan_Train2_1",
                                                      "FloorPlan_Train3_1", "FloorPlan_Train4_1"])
    parser.add_argument("--starts", type=int, default=1)
    args = parser.parse_args()
    if len(args.scenes) < 3 or args.starts <= 0:
        parser.error("Need two calibration scenes, a separate measurement scene, and positive starts")
    capture(args)


if __name__ == "__main__":
    main()
