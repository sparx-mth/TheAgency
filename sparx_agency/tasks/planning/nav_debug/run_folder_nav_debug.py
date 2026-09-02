#!/usr/bin/env python3
"""run_folder_nav_debug.py -- offline replay of a recorded FALCON navigation run.

Loads a run folder written by ``nav_debug_recorder_node`` (BEV maps + the three
route layers + replan events) together with the per-tick certainty CSV, and lets
you step through it frame by frame or play it back, rendering for every instant
the nav-debug screen (:mod:`.render`): the BEV map with the raw/corrected/final
routes, the target waypoint, the pose + localization trail and the drift vector,
beside two ROLL/PITCH/YAW gauge stacks (the command we send vs. the command the
converter sends the drone), confidence, the "why", and history strips.

No ROS, no drone -- just a finished recording. Run on the dev PC:

    python -m sparx_agency.tasks.planning.nav_debug.run_folder_nav_debug \\
        --run /path/to/nav_debug_YYYYmmdd_HHMMSS [--csv /path/to/certainty.csv]

Interactive keys: n / -> next, p / <- prev, SPACE play/pause, +/- speed,
s save PNG, q quit. Headless? Use ``--export DIR`` to write annotated frames
(+ an mp4) instead of opening a window.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np

from typing import Optional

from sparx_agency.tasks.planning.nav_debug.frame import GaugeScales
from sparx_agency.tasks.planning.nav_debug.render import ROOSTER_SCALES, render
from sparx_agency.tasks.planning.nav_debug.session import NavSession

_WINDOW = "FALCON nav debug (offline replay)"


def build_scales(which: str = "auto") -> Optional[GaugeScales]:
    """Gauge full-scales for the run being replayed.

    ``auto`` (the default) returns ``None`` and lets :func:`render.render` pick
    per frame: a Sphera run carries per-axis actuator traces and gets the Rooster
    envelope, an XTEND run gets the XTEND one. The envelopes differ by 3-4x, so
    forcing the wrong one silently mis-scales every command gauge.

    Args:
        which: ``auto``, ``xtend`` or ``rooster``.

    Returns:
        The chosen scales, or ``None`` to let the renderer decide.
    """
    if which == "auto":
        return None      # resolved once per run by :func:`resolve_scales`
    if which == "rooster":
        return ROOSTER_SCALES
    try:
        from sparx_agency.robots.XTEND.adapters.axis_calibration import XTEND_CALIBRATION as C
        return GaugeScales(
            our_vx=C.forward.max_velocity, our_vy=C.lateral.max_velocity,
            our_vz=C.vertical.max_velocity, our_wz=C.yaw.max_velocity,
            drone_forward=float(C.forward.max_counts),
            drone_lateral=float(C.lateral.max_counts),
            drone_vertical=float(C.vertical.max_counts),
            drone_yaw=float(C.yaw.max_counts))
    except Exception:      # pragma: no cover - robots pkg optional off-target
        return GaugeScales()


def _footer(img, i: int, n: int, t: float, playing: bool, map_px: int):
    """Append a thin status/help strip BELOW the image (never over the caption)."""
    strip = np.full((26, img.shape[1], 3), (12, 12, 12), np.uint8)
    tag = "PLAY" if playing else "PAUSE"
    cv2.putText(strip, "frame %d/%d  t=%.2fs  [%s]  zoom %dpx    |    "
                "n/p step   space play   z/x zoom   +/- speed   s save   q quit"
                % (i + 1, n, t, tag, map_px), (12, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)
    return np.vstack([img, strip])


def export(session: NavSession, scales: Optional[GaugeScales], out_dir: str, stride: int,
           map_px: int, video: bool, fps: float) -> None:
    """Write an annotated PNG per (strided) frame, and optionally an mp4."""
    frames_dir = Path(out_dir) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    n = len(session)
    indices = list(range(0, n, stride))
    size = _canvas_size(session, scales, map_px, indices)
    for i in indices:
        img = _fit(render(session.build(i), scales, map_px), size)
        cv2.imwrite(str(frames_dir / ("%06d.png" % i)), img)
        if video:
            if writer is None:
                writer = cv2.VideoWriter(str(Path(out_dir) / "replay.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            writer.write(img)
    if writer is not None:
        writer.release()
    print("[export] %d frames -> %s%s" % (len(range(0, n, stride)), frames_dir,
                                          " (+ replay.mp4)" if video else ""))


def _canvas_size(session: NavSession, scales, map_px: int, indices) -> tuple:
    """The widest/tallest frame layout in the run, as ``(w, h)``.

    The screen grows a lane column only on frames that have Sphera lane data, so
    sizing the export from frame 0 alone would squash every later frame. Sample
    across the run instead: a handful of renders is cheap next to the export.
    """
    probes = indices[:: max(1, len(indices) // 12)][:12] or [0]
    shapes = [render(session.build(i), scales, map_px).shape for i in probes]
    return (max(sh[1] for sh in shapes), max(sh[0] for sh in shapes))


def _fit(img: np.ndarray, size: tuple) -> np.ndarray:
    """Place ``img`` on a ``size`` canvas -- padded, not rescaled, when smaller.

    Rescaling a narrower frame up to the canvas would silently change the pixel
    scale of the map and every gauge from frame to frame.
    """
    width, height = size
    if (img.shape[1], img.shape[0]) == size:
        return img
    if img.shape[1] > width or img.shape[0] > height:
        img = cv2.resize(img, (min(img.shape[1], width), min(img.shape[0], height)))
    canvas = np.zeros((height, width, 3), np.uint8)
    canvas[: img.shape[0], : img.shape[1]] = img
    return canvas


def resolve_scales(session: NavSession) -> GaugeScales:
    """Pick one envelope for the whole run, from whichever lanes it recorded.

    Which airframe flew is a property of the run; deciding per frame would flip
    between envelopes that differ by 3.5x whenever a lane momentarily blanks.
    """
    step = max(1, len(session) // 40)
    for i in range(0, len(session), step):
        frame = session.build(i)
        if frame.axes or frame.actuator is not None or frame.altitude is not None:
            return ROOSTER_SCALES
    return GaugeScales()


def play(session: NavSession, scales: Optional[GaugeScales], map_px: int,
         start: int) -> None:
    """Interactive window: step or play through the run."""
    n = len(session)
    i = max(0, min(start, n - 1))
    playing = False
    speed = 1.0
    # WINDOW_NORMAL = a resizable window: drag any corner to enlarge (OpenCV scales
    # to fit) for a quick zoom; z/x re-render the BEV larger/smaller for a crisp one.
    cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_WINDOW, 1500, 950)
    last = time.time()
    while True:
        fr = session.build(i)
        cv2.imshow(_WINDOW, _footer(render(fr, scales, map_px), i, n, fr.stamp,
                                    playing, map_px))
        key = cv2.waitKey(15) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key in (ord("n"), 83, 84):        # n / right / down
            i = min(n - 1, i + 1)
        elif key in (ord("p"), 81, 82):         # p / left / up
            i = max(0, i - 1)
        elif key == ord(" "):
            playing = not playing
            last = time.time()
        elif key == ord("z"):                   # crisp zoom in (re-render larger)
            map_px = min(2400, map_px + 150)
        elif key == ord("x"):                   # crisp zoom out
            map_px = max(360, map_px - 150)
        elif key in (ord("+"), ord("=")):
            speed = min(8.0, speed * 2.0)
        elif key == ord("-"):
            speed = max(0.125, speed / 2.0)
        elif key == ord("s"):
            path = "nav_debug_frame_%06d.png" % i
            cv2.imwrite(path, render(fr, scales, map_px))
            print("[saved]", path)
        if playing and i < n - 1:
            dt_frames = session.rows[i + 1]["t"] - fr.stamp
            if time.time() - last >= max(0.0, dt_frames) / speed:
                i += 1
                last = time.time()
        elif playing:
            playing = False        # reached the end
    cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run folder from nav_debug_recorder_node")
    ap.add_argument("--csv", default=None,
                    help="certainty CSV (default: from manifest.json, else auto-found)")
    ap.add_argument("--export", default=None, metavar="DIR",
                    help="headless: write annotated frames (+ mp4) here instead of a window")
    ap.add_argument("--stride", type=int, default=1, help="export every Nth frame")
    ap.add_argument("--no-video", action="store_true", help="export frames only, no mp4")
    ap.add_argument("--fps", type=float, default=15.0, help="export mp4 frame rate")
    ap.add_argument("--map-px", type=int, default=900,
                    help="BEV display size, longest edge (bigger = larger window; z/x adjust live)")
    ap.add_argument("--start", type=int, default=0, help="first frame index (interactive)")
    ap.add_argument("--ros2", default=None, metavar="DIR",
                    help="the ROS2 recorder's lane directory, when it was not "
                         "collected into the run folder (default: <run>/ros2)")
    ap.add_argument("--scales", choices=("auto", "xtend", "rooster"), default="auto",
                    help="gauge full-scales (default: auto -- Rooster for a run with "
                         "actuator traces, XTEND otherwise)")
    args = ap.parse_args()

    session = NavSession(args.run, args.csv, args.ros2)
    dur = session.rows[-1]["t"] - session.rows[0]["t"]
    print("[nav_debug] %d frames over %.1fs | csv=%s" % (
        len(session), dur, os.path.basename(session.csv_path or "-- (telemetry only)")))

    scales = build_scales(args.scales) or resolve_scales(session)
    if args.export:
        export(session, scales, args.export, max(1, args.stride), args.map_px,
               not args.no_video, args.fps)
    else:
        play(session, scales, args.map_px, args.start)


if __name__ == "__main__":
    main()
