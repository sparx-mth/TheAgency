#!/usr/bin/env python3
"""Render a synthetic run to MP4 -- proves the recorder works without the stack.

No ROS, no Gazebo, no model. It fakes a drone exploring a floor plan: a moving
camera panel on the left, N1's committed route and a growing trail on the right,
and the System-1 / System-2 FPS drawn on -- the exact layout a live run records,
so the output format can be seen before wiring the whole stack up.

    python -m sparx_agency.tasks.planning.sjtu_internvla_n1.scripts.demo_recording \
        --output /tmp/sjtu_n1/demo.mp4 --seconds 12

The FPS defaults are this machine's measured InternVLA-N1 numbers (see
`~/trt/internnav/REPORT.md`): System 1 22.99 Hz (TensorRT), System 2 ~1.4 Hz.
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

from sparx_agency.tasks.planning.sjtu_internvla_n1.recording import (
    OverlayInfo,
    TopDownRenderer,
    compose,
    draw_camera_panel,
)

INSTRUCTION = "Explore the entire hospital, enter all the rooms, reach every area at least once"


def _fake_camera(w, h, t, action):
    """A cheap moving scene so the left panel is not a flat colour."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    frame[:, :, 0] = ((xx + t * 40) % 255).astype(np.uint8)     # drifting blue
    frame[:, :, 1] = ((yy + t * 12) % 160).astype(np.uint8)     # green
    # a "corridor" that slides, to read as forward motion
    cx = int(w / 2 + 120 * np.sin(t * 0.6))
    cv2.rectangle(frame, (cx - 60, 0), (cx + 60, h), (60, 60, 70), -1)
    cv2.putText(frame, "SYNTHETIC", (w // 2 - 90, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 0, 0), 3, cv2.LINE_AA)
    return frame


def _route(x, y, yaw, n=16, step=0.25):
    """A short curving body-frame route, anchored at the pose, in world xy."""
    pts = []
    curve = 0.15
    fx, fy, fyaw = x, y, yaw
    for i in range(n):
        fyaw += curve * np.sin(i * 0.5)
        fx += step * np.cos(fyaw)
        fy += step * np.sin(fyaw)
        pts.append((fx, fy))
    return np.array(pts, dtype=float)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="/tmp/sjtu_n1/demo.mp4")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--panel-width", type=int, default=640)
    ap.add_argument("--panel-height", type=int, default=480)
    ap.add_argument("--s1-fps", type=float, default=22.99)
    ap.add_argument("--s2-fps", type=float, default=1.41)
    args = ap.parse_args(argv)

    w, h = args.panel_width, args.panel_height
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (w * 2, h), True)
    if not writer.isOpened():
        raise SystemExit("could not open VideoWriter at %s" % (args.output,))

    topdown = TopDownRenderer(size=(w, h))
    n_frames = int(args.seconds * args.fps)
    x, y, yaw = 1.0, 1.0, 0.0
    actions = ["MOVE_FORWARD", "MOVE_FORWARD", "TURN_LEFT", "MOVE_FORWARD", "TURN_RIGHT"]

    for i in range(n_frames):
        t = i / args.fps
        # wander like an exploration sweep
        yaw += 0.05 * np.sin(t * 0.4)
        x += 0.08 * np.cos(yaw)
        y += 0.08 * np.sin(yaw)
        topdown.add_pose(x, y)
        action = actions[(i // 8) % len(actions)]

        committed = _route(x, y, yaw, n=8)
        full = _route(x, y, yaw, n=16)
        info = OverlayInfo(
            instruction=INSTRUCTION, action=action, status="navigating",
            s1_fps=args.s1_fps, s2_fps=args.s2_fps,
            s1_ms=1000.0 / args.s1_fps, s2_ms=1000.0 / args.s2_fps,
            pixel_goal=(int(w / 2 + 120 * np.sin(t * 0.6)), int(h * 0.42)),
            pixel_goal_frame=(w, h))

        left = draw_camera_panel(_fake_camera(w, h, t, action), info, (w, h))
        right = topdown.render((x, y, yaw), committed, full)
        writer.write(compose(left, right))

    writer.release()
    print("wrote %d frames (%.1fs) to %s" % (n_frames, args.seconds, args.output))


if __name__ == "__main__":
    main()

