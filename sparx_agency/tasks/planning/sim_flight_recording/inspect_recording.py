"""Look at a collected recording without opening a simulator.

A campaign produces directories of numpy arrays and JPEGs, which is the right
format for training and a useless one for answering "did that flight actually
work?". This prints the numbers that matter and draws two pictures per
recording:

* a **contact sheet** -- evenly spaced RGB frames over their depth maps, which
  is where a black camera, a frozen image or a drone facing a wall the whole
  way is immediately obvious;
* a **plan view** -- the flown path drawn over the scene's surveyed map next to
  the route that was planned, which is where clipping a wall or never leaving
  the pad is immediately obvious.

Runs in the repo venv; needs no GPU and no Isaac Sim::

    .venv/bin/python sparx_agency/tasks/planning/sim_flight_recording/inspect_recording.py \\
        ~/flight_dataset/office --out-dir /tmp/review
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np

from sparx_agency.tasks.planning.vlas.common.finetune.datasets.recording import (
    load_recording,
)

CONTACT_COLUMNS = 6
TILE_WIDTH = 240
PLAN_SCALE = 6          # pixels per map cell
FLOWN_COLOUR = (0, 200, 255)     # BGR, amber
PLANNED_COLOUR = (0, 220, 0)     # BGR, green


def is_recording(path: Path) -> bool:
    """Whether ``path`` looks like a recording directory."""
    return (path / "poses.npy").exists() and (path / "meta.json").exists()


def find_recordings(root: Path) -> list:
    """Every recording at or below ``root``, sorted by name."""
    root = Path(root)
    if is_recording(root):
        return [root]
    return sorted(p for p in root.iterdir() if p.is_dir() and is_recording(p))


def summarize(recording, meta: dict) -> dict:
    """The numbers worth printing for one recording."""
    poses = recording.poses
    stamps = poses[:, 0]
    intervals = np.diff(stamps)
    return {
        "frames": recording.num_frames,
        "duration_s": float(stamps[-1] - stamps[0]),
        "rate_hz": float(1.0 / intervals.mean()) if intervals.size else 0.0,
        "rate_jitter_s": float(intervals.std()) if intervals.size else 0.0,
        "flown_m": float(meta.get("path_length_m", 0.0)),
        "planned_m": float(meta.get("planned_path_length_m", 0.0)),
        "goal_error_m": float(meta.get("goal_error_m", float("nan"))),
        "yaw_span_deg": float(np.degrees(np.ptp(np.unwrap(poses[:, 3])))),
        "altitude_m": (float(poses[:, 4].mean()) if poses.shape[1] > 4 else float("nan")),
        "outcome": meta.get("outcome", "?"),
        "has_rgb": bool(meta.get("has_rgb", False)),
    }


def contact_sheet(recording, columns: int = CONTACT_COLUMNS) -> np.ndarray:
    """A grid of evenly spaced frames: RGB on top, colourised depth beneath."""
    indices = np.linspace(0, recording.num_frames - 1, columns).astype(int)
    rgb_row, depth_row = [], []
    for index in indices:
        depth = recording.depth(index)
        scale = TILE_WIDTH / depth.shape[1]
        size = (TILE_WIDTH, max(int(depth.shape[0] * scale), 1))

        rgb = recording.rgb(index)
        colour = (cv2.resize(rgb[:, :, ::-1], size) if rgb is not None
                  else np.zeros((size[1], size[0], 3), np.uint8))
        rgb_row.append(colour)

        # Near = bright. A fixed 0-10 m range, not per-frame normalisation, so
        # frames are comparable and an all-far-plane frame looks empty rather
        # than being stretched into something that looks like content.
        normalised = np.clip(1.0 - depth / 10.0, 0.0, 1.0)
        heat = cv2.applyColorMap((normalised * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        depth_row.append(cv2.resize(heat, size))

    return np.vstack([np.hstack(rgb_row), np.hstack(depth_row)])


def plan_view(recording, meta: dict, grid=None) -> np.ndarray:
    """The flown path (amber) and the planned route (green), over the map."""
    poses = recording.poses
    if grid is not None:
        values = grid.values
        base = np.full(grid.grid.shape, 128, np.uint8)
        base[grid.grid == values.free] = 255
        base[grid.grid == values.occupied] = 0
        canvas = cv2.cvtColor(cv2.resize(base, None, fx=PLAN_SCALE, fy=PLAN_SCALE,
                                         interpolation=cv2.INTER_NEAREST),
                              cv2.COLOR_GRAY2BGR)

        def to_pixel(x, y):
            gx = (x - grid.origin_x) / grid.resolution
            gy = (y - grid.origin_y) / grid.resolution
            return int(gx * PLAN_SCALE), int(gy * PLAN_SCALE)
    else:
        # No map available: draw into a square box scaled to the flight itself.
        size = 600
        margin = 2.0
        xs, ys = poses[:, 1], poses[:, 2]
        span = max(np.ptp(xs), np.ptp(ys), 1.0) + 2 * margin
        canvas = np.full((size, size, 3), 40, np.uint8)

        def to_pixel(x, y):
            return (int((x - xs.min() + margin) / span * size),
                    int((y - ys.min() + margin) / span * size))

    planned = meta.get("planned_waypoints") or []
    if planned:
        start = meta.get("start_xy", [poses[0, 1], poses[0, 2]])
        points = [to_pixel(start[0], start[1])] + [to_pixel(w[0], w[1]) for w in planned]
        cv2.polylines(canvas, [np.array(points, np.int32)], False, PLANNED_COLOUR, 2)
        for point in points[1:]:
            cv2.circle(canvas, point, 4, PLANNED_COLOUR, -1)

    flown = np.array([to_pixel(x, y) for x, y in poses[:, 1:3]], np.int32)
    cv2.polylines(canvas, [flown], False, FLOWN_COLOUR, 2)
    cv2.circle(canvas, tuple(flown[0]), 7, (255, 255, 255), -1)
    cv2.circle(canvas, tuple(flown[-1]), 7, (0, 0, 255), -1)
    return np.flipud(canvas)  # +y up, so the picture is a plan view


def _load_map(meta: dict):
    """The scene map this recording was flown against, or None if unavailable."""
    from sparx_agency.robots.PEGASUS.adapters.scene_map import load_scene_map

    try:
        grid, _meta, _layers = load_scene_map(meta["scene"], float(meta["altitude_m"]))
        return grid
    except (KeyError, FileNotFoundError, TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path,
                    help="a recording directory, or a campaign directory of them")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where to write the pictures (default: inside each recording)")
    args = ap.parse_args()

    recordings = find_recordings(args.root)
    if not recordings:
        print(f"no recordings under {args.root}", file=sys.stderr)
        return 1

    header = (f"{'recording':<26} {'outcome':<14} {'frames':>7} {'s':>7} {'Hz':>6} "
              f"{'flown m':>8} {'plan m':>7} {'goal m':>7} {'yaw deg':>8}")
    print(header)
    print("-" * len(header))

    for path in recordings:
        recording = load_recording(path)
        meta = json.loads((path / "meta.json").read_text())
        stats = summarize(recording, meta)
        print(f"{path.name:<26} {stats['outcome']:<14} {stats['frames']:>7} "
              f"{stats['duration_s']:>7.1f} {stats['rate_hz']:>6.2f} "
              f"{stats['flown_m']:>8.1f} {stats['planned_m']:>7.1f} "
              f"{stats['goal_error_m']:>7.2f} {stats['yaw_span_deg']:>8.0f}")

        out_dir = Path(args.out_dir) / path.name if args.out_dir else path
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "contact_sheet.jpg"), contact_sheet(recording))
        cv2.imwrite(str(out_dir / "plan_view.png"),
                    plan_view(recording, meta, _load_map(meta)))

    where = args.out_dir if args.out_dir else "each recording directory"
    print(f"\nwrote contact_sheet.jpg and plan_view.png to {where}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
