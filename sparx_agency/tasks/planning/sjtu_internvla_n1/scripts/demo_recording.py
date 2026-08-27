#!/usr/bin/env python3
"""Render a synthetic run to MP4 -- proves the recorder works without the stack.

No ROS, no Gazebo, no model. It fakes a drone exploring a floor plan: a moving
camera panel on the left, N1's committed route and a growing trail on the right,
the floor the camera has looked at washed in blue with its percentage, and the
System-1 / System-2 FPS drawn on -- the exact layout a live run records, so the
output format can be seen before wiring the whole stack up.

The coverage here is computed the same way a real run computes it, off the same
map, from the fake poses -- so the demo exercises the measurement, not only the
drawing.

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

from sparx_agency.core.planning.environment.occupancy_io import occupancy_from_mask
from sparx_agency.core.planning.exploration.visibility_coverage import (
    VisibilityCoverage,
    cone_from_intrinsics,
)
from sparx_agency.tasks.planning.sjtu_internvla_n1.map_backdrop import (
    load_map_backdrop,
)
from sparx_agency.tasks.planning.sjtu_internvla_n1.recording import (
    CoverageOverlay,
    OverlayInfo,
    TopDownRenderer,
    compose,
    draw_camera_panel,
)

INSTRUCTION = ("Explore the entire hospital. Enter every room you pass, look around "
               "inside to see what is in it, then come back out into the corridor and "
               "go on to the next room.")


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
    ap.add_argument("--map", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), *([os.pardir] * 5),
        "sparx_agency", "robots", "SJTU", "maps", "hospital.yaml"),
        help="occupancy map to draw the route on; empty for graph paper")
    args = ap.parse_args(argv)

    w, h = args.panel_width, args.panel_height
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (w * 2, h), True)
    if not writer.isOpened():
        raise SystemExit("could not open VideoWriter at %s" % (args.output,))

    # Draw on the real hospital map when it is there, so the demo shows the
    # format a recording actually has rather than a simplified one.
    backdrop = load_map_backdrop(args.map)
    topdown = TopDownRenderer(size=(w, h), backdrop=backdrop)
    coverage = None
    if backdrop is not None:
        coverage = VisibilityCoverage(
            occupancy_from_mask(backdrop.occupied_mask, backdrop.resolution,
                                backdrop.origin_x, backdrop.origin_y,
                                known=backdrop.known_mask),
            cone_from_intrinsics(600, 390.642735, max_range_m=10.0,
                                 forward_offset_m=0.2))
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
        if coverage is not None:
            coverage.observe(x, y, yaw)
        action = actions[(i // 8) % len(actions)]

        committed = _route(x, y, yaw, n=8)
        full = _route(x, y, yaw, n=16)
        info = OverlayInfo(
            instruction=INSTRUCTION, action=action, status="navigating",
            s1_fps=args.s1_fps, s2_fps=args.s2_fps,
            s1_ms=1000.0 / args.s1_fps, s2_ms=1000.0 / args.s2_fps,
            pixel_goal=(int(w / 2 + 120 * np.sin(t * 0.6)), int(h * 0.42)),
            pixel_goal_frame=(w, h))

        overlay = None if coverage is None else CoverageOverlay(
            seen=coverage.seen_mask, fraction=coverage.fraction_seen,
            area_seen_m2=coverage.area_seen_m2,
            area_total_m2=coverage.area_total_m2)
        left = draw_camera_panel(_fake_camera(w, h, t, action), info, (w, h))
        right = topdown.render((x, y, yaw), committed, full, None, overlay)
        writer.write(compose(left, right))

    writer.release()
    print("wrote %d frames (%.1fs) to %s" % (n_frames, args.seconds, args.output))
    if coverage is not None:
        print("coverage: %s" % (coverage.summary(),))


if __name__ == "__main__":
    main()

