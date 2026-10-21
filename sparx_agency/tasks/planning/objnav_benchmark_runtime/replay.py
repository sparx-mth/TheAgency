"""Visualization-only replay from completed recordings, independent of simulator.

The adapter supplies render_pose(AgentPose) -> RGB, including camera pitch.
It owns scene/assets, rendering-context safety and cleanup. Interpolated frames
never enter policy observations, actions, scoring or original artifacts.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import cv2
import numpy as np

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.objnav_benchmark_runtime.recording import VideoSink
from sparx_agency.tasks.planning.objnav_benchmark_runtime.visualization import FRAME_SIZE


def replay_poses(rows, subframes):
    """Interpolate translation/pitch and the shortest yaw arc; include the final pose."""
    if type(subframes) is not int or not 1 <= subframes <= 12:
        raise ValueError("subframes must be an integer in 1..12")
    poses = [AgentPose(**{key: float(row[key]) for key in ("x", "y", "z", "yaw", "camera_pitch")})
             for row in rows]
    if len(poses) < 2:
        raise ValueError("Replay requires at least two recorded poses")
    for first, last in zip(poses, poses[1:]):
        dyaw = normalize_angle(last.yaw - first.yaw)
        for index in range(subframes):
            fraction = index / subframes
            xyz = [(1 - fraction) * a + fraction * b for a, b in zip(first.position(), last.position())]
            yield AgentPose(*xyz, yaw=first.yaw + fraction * dyaw,
                            camera_pitch=(1 - fraction) * first.camera_pitch + fraction * last.camera_pitch)
    yield poses[-1]


def render_replay(recording_dir, render_pose, *, subframes=4, decision_fps=6, writer_factory=VideoSink):
    """Write a separate letterboxed RGB replay; do not overwrite any existing output."""
    root = Path(recording_dir).expanduser()
    if not (root / "metrics.json").is_file():
        raise ValueError("Replay requires a completed recording with episode metrics")
    if not math.isfinite(decision_fps) or decision_fps <= 0:
        raise ValueError("decision_fps must be positive and finite")
    with (root / "trajectory.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    poses = list(replay_poses(rows, subframes))
    output = root / "smooth_rgb.mp4"
    if output.exists():
        raise FileExistsError("Do not overwrite an existing smooth replay")
    sink = writer_factory(output, decision_fps * subframes)
    try:
        for pose in poses:
            rgb = np.asarray(render_pose(pose))
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
                raise ValueError("render_pose must return nonempty uint8 RGB")
            width, height = FRAME_SIZE
            scale = min(width / rgb.shape[1], height / rgb.shape[0])
            w, h = max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale))
            frame = np.zeros((height, width, 3), np.uint8)
            x, y = (width - w) // 2, (height - h) // 2
            frame[y:y + h, x:x + w] = cv2.resize(rgb[..., ::-1], (w, h))
            cv2.putText(frame, "VISUAL-ONLY INTERPOLATION - not policy observations", (20, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            sink.write(frame)
    finally:
        sink.close()
    return output

