"""The unchanged method on chosen episodes, with a trace of what it sees, asks and credits to rooms.

Run with the repo on PYTHONPATH; TRACE_OUT names the directory. Nothing about the method's decisions is touched.
"""
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

from sparx_agency.core.planning.objnav.camera_geometry import world_T_camera_optical
from sparx_agency.tasks.planning.objnav_benchmark_runtime.evaluation import run_evaluation
from sparx_agency.tasks.planning.objnav_benchmark_runtime.services.http_detector import HttpDetector
from sparx_agency.tasks.planning.objnav_benchmark_runtime.trace_tools.agreement import agreeing
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import protocol_for
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.run import parser, prepare

FRAME_EVERY = 5
DEPTH_STRIDE = 16
FLOOR_BAND_M = 0.25


def pose_dict(pose):
    return {"x": round(float(pose.x), 3), "y": round(float(pose.y), 3), "z": round(float(pose.z), 3),
            "yaw": round(float(pose.yaw), 3), "pitch": round(float(pose.camera_pitch), 3)}


def room_shares(pids, total):
    shares = {str(int(p) - 1) if p > 0 else "outside": int(n) / total for p, n in zip(*np.unique(pids, return_counts=True))}
    return {key: round(value, 3) for key, value in shares.items()}


def frame_rooms(observation, world, labels):
    """Share of the frame's valid depth pixels that land in each room's cells: all pixels, and floor-level ones."""
    if labels is None or world is None:
        return None
    camera, k = observation.camera, observation.camera.intrinsics
    depth = np.asarray(observation.depth_m)[::DEPTH_STRIDE, ::DEPTH_STRIDE]
    v, u = (grid * DEPTH_STRIDE for grid in np.mgrid[0:depth.shape[0], 0:depth.shape[1]])
    ok = np.isfinite(depth) & (depth > camera.min_depth_m) & (depth < camera.max_depth_m)
    if not ok.any():
        return None
    d = depth[ok]
    optical = np.stack(((u[ok] - k.cx) * d / k.fx, (v[ok] - k.cy) * d / k.fy, d, np.ones_like(d)))
    points = (world_T_camera_optical(observation.pose, camera) @ optical)[:3].T
    pids = np.zeros(len(points), dtype=int)
    for i, (x, y, _) in enumerate(points):
        gx, gy = world.world_to_grid(float(x), float(y))
        pids[i] = labels[gy, gx] if world.in_bounds(gx, gy) else 0
    low = points[:, 2] - observation.pose.z < FLOOR_BAND_M
    return {"all": room_shares(pids, len(pids)), "floor": room_shares(pids[low], max(1, int(low.sum()))) if low.any() else {}}


def install_agreement(policy, url, backend):
    """Ask a second detector about every frame and keep only the target-kind boxes it confirms."""
    second = HttpDetector(url, policy.detector.vocabulary, timeout_s=120.0, expected_backend=backend)
    first_detect = policy.detector.detect
    policy.detector.detect = lambda rgb: agreeing(first_detect(rgb), second.detect(rgb))


def snapshot(policy, world, pose):
    g = policy.graph
    if g.labels is None or world is None:
        return None
    rooms = {}
    for value in np.unique(g.labels):
        if value <= 0:
            continue
        ys, xs = np.nonzero(g.labels == value)
        stride = max(1, len(xs) // 150)
        cells = [[round(float(c), 2) for c in world.grid_to_world(int(x), int(y))] for x, y in zip(xs[::stride], ys[::stride])]
        rooms[int(value) - 1] = {"area_cells": int(len(xs)), "cells": cells}
    marks = [{"cls": m.class_name, "xy": [round(float(m.xy[0]), 2), round(float(m.xy[1]), 2)], "n": m.count, "r": m.radius_m,
              "pid": g.object_room(world, m)} for m in policy.landmarks.confirmed()]
    return {"rooms": rooms, "landmarks": marks, "pose": pose, "floor_id": policy.mapping.floor_id}


class Tracer:
    """Wraps the policy's plan, detector and LLM client to write trace.jsonl; the method's decisions are untouched."""

    def __init__(self, policy, out):
        self.policy, self.out = policy, out
        self.trace = open(out / "trace.jsonl", "a", buffering=1)
        self.state = {"episode": None, "calls": 0, "obs": None, "pose": None, "world": None}

    def write(self, kind, **fields):
        state = self.state
        self.trace.write(json.dumps(dict(kind=kind, episode=state["episode"], call=state["calls"], pose=state["pose"], **fields),
                                    default=str) + "\n")

    def install(self):
        original_plan, original_detect = self.policy.plan, self.policy.detector.detect

        def plan(observation):
            self.state["obs"], self.state["pose"] = observation, pose_dict(observation.pose)
            return original_plan(observation)

        def detect(rgb):
            self._hook_graph()
            detections = original_detect(rgb)
            self._record_detection(rgb, detections)
            return detections

        self.policy.plan, self.policy.detector.detect = plan, detect
        for method in [m for m in ("chat_json", "chat_text") if hasattr(self.policy.llm_client, m)]:
            self._trace_llm(method)

    def _hook_graph(self):
        graph = getattr(self.policy, "graph", None)
        if graph is None or getattr(graph, "_traced", False):
            return
        original_update = graph.update

        def update(world, *a, **kw):
            self.state["world"] = world
            return original_update(world, *a, **kw)

        graph.update, graph._traced = update, True

    def _record_detection(self, rgb, detections):
        state, policy = self.state, self.policy
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(getattr(policy.episode, "episode_id", "episode")))
        if name != state["episode"]:
            state["episode"], state["calls"] = name, 0
        state["calls"] += 1
        boxes = [{"cls": d.cls, "conf": round(float(d.conf), 3), "xyxy": [round(float(v), 1) for v in d.xyxy]} for d in detections]
        obs = state["obs"]
        rooms = frame_rooms(obs, state["world"], getattr(policy.graph, "labels", None)) if obs else None
        file = "%s_%04d.jpg" % (name, state["calls"]) if state["calls"] % FRAME_EVERY == 0 else None
        if file:
            small = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (320, 240))
            cv2.imwrite(str(self.out / "frames" / file), small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        self.write("detect", boxes=boxes, frame=file, frame_rooms=rooms, step=int(obs.step) if obs else None,
                   floor_id=policy.mapping.floor_id, target=str(getattr(policy, "target", None)))

    def _trace_llm(self, method):
        client, original = self.policy.llm_client, getattr(self.policy.llm_client, method)

        def wrapped(system, user, *a, **kw):
            reply = original(system, user, *a, **kw)
            asked = user.startswith("Room observed objects")
            extra = {"snapshot": snapshot(self.policy, self.state["world"], self.state["pose"])} if asked else {}
            self.write("llm", method=method, system=system, user=user, reply=reply, **extra)
            return reply

        setattr(client, method, wrapped)


def main():
    out = Path(os.environ["TRACE_OUT"])
    args = parser().parse_args(sys.argv[1:])
    protocol = protocol_for(args.version).with_split(args.split)
    env, policy, label_mapper, config, issues = prepare(args)
    if issues:
        print("not ready:", *issues, sep="\n  ")
        return 1
    (out / "frames").mkdir(parents=True, exist_ok=True)
    if os.environ.get("TRACE_AGREE_URL"):
        install_agreement(policy, os.environ["TRACE_AGREE_URL"], os.environ.get("TRACE_AGREE_BACKEND"))
    Tracer(policy, out).install()
    run_evaluation(env, policy, label_mapper, output=args.output, config=config, settings=protocol.evaluation_settings(),
                   episode_ids=tuple(config["selected_episode_ids"]), record=False, video_fps=args.video_fps,
                   save_frames=False, progress=lambda i, n, row: print("%d/%d %s steps=%d success=%s dtg=%.1f" % (
                       i, n, row.episode_id, row.steps, row.success, row.distance_to_goal_m), flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
