"""Render a flight's map panel: what the policy proposed, over the real map.

    python -m ...world_goal.track_video --track flights/trained/mission_06_track.json \\
        --scene warehouse_shelves --out mission_06_track.mp4

The camera stream says what the policy saw. This says what it decided. Each
frame draws the surveyed occupancy map, the goal, the path flown so far, and the
24-waypoint trajectory NavDP proposed at that instant — the thing that is gone a
quarter of a second later and that no camera can show.

The trajectory is drawn in world coordinates, so a plan that heads into a shelf
is visibly a plan that heads into a shelf, whatever the aircraft did afterwards.
That is the distinction the whole comparison rests on: a policy that flies into
geometry because it planned to is a different failure from one that planned well
and was carried in by momentum.

Frames are rendered at the inference rate and encoded with ffmpeg. matplotlib
and numpy only — no torch, no simulator, so this runs on a laptop from the JSON.
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.track_log import TrackLog

FLOWN = "#2a78d6"
PROPOSED = "#eb6834"
GOAL = "#d81b60"
MARGIN_M = 4.0
"""Padding around the flight's extent, metres. Enough that the proposed
trajectory stays on screen when it heads away from the goal."""


def load_map(scene: str, altitude_m: float = 1.5, map_dir: Optional[Path] = None):
    """The surveyed occupancy grid for a scene: ``(grid, resolution, origin)``."""
    if map_dir is None:
        map_dir = (Path(__file__).resolve().parents[6]
                   / "robots" / "PEGASUS" / "maps")
    path = Path(map_dir) / f"{scene}_alt{int(round(altitude_m * 100)):04d}cm.npz"
    if not path.is_file():
        raise FileNotFoundError(f"no surveyed map at {path}")
    data = np.load(path)
    return data["grid"], float(data["resolution"]), data["origin"]


def _bounds(log: Dict, pad: float = MARGIN_M):
    """Axis limits covering the whole flight, the goal and every proposal."""
    points = [np.asarray(log["flown"], dtype=float).reshape(-1, 2)] if log.get("flown") else []
    points.append(np.asarray([log["goal_xy"], log["start_xy"]], dtype=float))
    for entry in log["inferences"]:
        if "traj" in entry:
            points.append(np.asarray(entry["traj"], dtype=float))
    stacked = np.concatenate([p for p in points if p.size], axis=0)
    low, high = stacked.min(axis=0) - pad, stacked.max(axis=0) + pad
    span = max(high[0] - low[0], high[1] - low[1])          # keep it square
    centre = (low + high) / 2.0
    return (centre[0] - span / 2, centre[0] + span / 2,
            centre[1] - span / 2, centre[1] + span / 2)


def render_frames(log: Dict, scene: str, out_dir: Path, altitude_m: float = 1.5,
                  size_px: int = 540, dpi: int = 100) -> int:
    """One PNG per inference. Returns how many were written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid, resolution, origin = load_map(scene, altitude_m)
    height, width = grid.shape
    extent = (origin[0], origin[0] + width * resolution,
              origin[1], origin[1] + height * resolution)
    x0, x1, y0, y1 = _bounds(log)

    flown = np.asarray(log.get("flown") or [], dtype=float).reshape(-1, 2)
    # The flown path is sampled at the physics rate and the inferences at ~4 Hz,
    # so this maps one to the other by fraction of the flight rather than by
    # time -- the log does not promise the two clocks share a zero.
    entries = log["inferences"]
    inches = size_px / dpi

    for index, entry in enumerate(entries):
        figure, axis = plt.subplots(figsize=(inches, inches), dpi=dpi)
        axis.imshow(grid > 0, cmap="Greys", origin="lower", extent=extent,
                    interpolation="nearest", vmin=0, vmax=1.6)

        upto = int(round((index + 1) / len(entries) * flown.shape[0])) if flown.size else 0
        if upto > 1:
            axis.plot(flown[:upto, 0], flown[:upto, 1], color=FLOWN,
                      linewidth=2.0, label="flown", zorder=3)
        if "traj" in entry:
            proposed = np.asarray(entry["traj"], dtype=float)
            axis.plot(proposed[:, 0], proposed[:, 1], color=PROPOSED,
                      linewidth=2.2, label="NavDP plan", zorder=4)
            axis.plot(proposed[-1, 0], proposed[-1, 1], "o", color=PROPOSED,
                      markersize=4, zorder=4)
        else:
            axis.text(0.5, 0.94, "inference dropped", transform=axis.transAxes,
                      ha="center", color=PROPOSED, fontsize=9)

        pose = entry["pose"]
        axis.plot([pose[0]], [pose[1]], "o", color="black", markersize=7, zorder=5)
        axis.arrow(pose[0], pose[1], 1.2 * np.cos(pose[2]), 1.2 * np.sin(pose[2]),
                   head_width=0.45, color="black", zorder=5, length_includes_head=True)
        axis.plot([log["goal_xy"][0]], [log["goal_xy"][1]], "*", color=GOAL,
                  markersize=15, zorder=6, label="goal")

        axis.set_xlim(x0, x1)
        axis.set_ylim(y0, y1)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.legend(loc="lower right", fontsize=7, framealpha=0.85)
        axis.set_title(f"t = {entry['t']:.1f} s", fontsize=9)
        figure.tight_layout(pad=0.2)
        figure.savefig(out_dir / f"frame_{index:05d}.png")
        plt.close(figure)
    return len(entries)


def encode(frame_dir: Path, out_path: Path, fps: float) -> bool:
    """Encode the rendered frames into an MP4."""
    command = ["ffmpeg", "-y", "-framerate", f"{fps:.3f}",
               "-i", str(frame_dir / "frame_%05d.png"),
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
               "-pix_fmt", "yuv420p", str(out_path)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[track] ffmpeg failed:\n{result.stderr[-800:]}")
        return False
    return out_path.is_file()


def render(track_path: Path, out_path: Path, scene: Optional[str] = None,
           altitude_m: float = 1.5, size_px: int = 540) -> Optional[Path]:
    """Render one flight's map panel to an MP4."""
    log = TrackLog.read(Path(track_path))
    if not log.get("inferences"):
        print(f"[track] {track_path.name} has no inferences")
        return None
    scene = scene or log.get("scene")
    if not scene:
        raise ValueError(f"{track_path} does not name a scene; pass --scene")

    duration = max(1e-3, log["inferences"][-1]["t"] - log["inferences"][0]["t"])
    fps = max(1.0, len(log["inferences"]) / duration)
    with tempfile.TemporaryDirectory() as work:
        count = render_frames(log, scene, Path(work), altitude_m, size_px)
        print(f"[track] {track_path.name}: {count} frames at {fps:.1f} fps")
        if not encode(Path(work), Path(out_path), fps):
            return None
    return Path(out_path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--track", required=True, help="a mission's *_track.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene", default=None,
                        help="defaults to the scene named in the log")
    parser.add_argument("--altitude", type=float, default=1.5)
    parser.add_argument("--size", type=int, default=540, help="panel side, pixels")
    args = parser.parse_args(argv)

    result = render(Path(args.track).expanduser(), Path(args.out).expanduser(),
                    args.scene, args.altitude, args.size)
    return 0 if result else 1


if __name__ == "__main__":
    raise SystemExit(main())
