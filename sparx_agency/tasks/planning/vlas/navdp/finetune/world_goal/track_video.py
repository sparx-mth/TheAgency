"""Render a flight's map panel: what the policy proposed, over the real map.

    python -m ...world_goal.track_video --track flights/trained/mission_06_track.json \\
        --scene warehouse_shelves --out mission_06_track.mp4

The camera stream says what the policy saw. This says what it decided. Each
frame draws the surveyed occupancy map, the goal, the path flown so far, and the
24-waypoint trajectory NavDP proposed at that instant — the thing no camera can
show.

Two claims, drawn differently. The aircraft **commits** to roughly the first
half of each prediction and flies it as a route before asking again (see
``core/planning/vlas/common/plan_commit``); the rest is the policy's guess about
what comes after. Drawing both as one line says the aircraft promised the whole
thing, which it did not — so the committed prefix is solid green and the
speculative tail is a dashed orange. A plan that stands still on screen while
the aircraft moves along it is the commitment working; a plan that is replaced
every frame is the failure this drawing exists to make visible.

The trajectory is drawn in world coordinates, so a plan that heads into a shelf
is visibly a plan that heads into a shelf, whatever the aircraft did afterwards.
That is the distinction the whole comparison rests on: a policy that flies into
geometry because it planned to is a different failure from one that planned well
and was carried in by momentum.

## The panel runs on the flight's clock, and that is the whole point

Every frame is placed at a real simulation time, taken from the log, and shows
what was true then: the flown path up to that moment and the most recent
proposal. It used to be one frame per *inference*, with the flown path indexed by
**fraction of the flight** — and those are not the same timeline. The aircraft is
recorded from the moment it leaves the ground while inference is held off until it
reaches cruise altitude, so the first inference happens some ten seconds in. The
panel therefore drew the trail metres behind the aircraft marker for most of the
video, and the video itself started with the marker already displaced from the
takeoff point. Both were artefacts. The reading they invited — that the aircraft
was mislocalised, and had drifted before it left the ground — was wrong, and it
cost a campaign's worth of data being thrown away before it was checked.

A log written before that was understood has no clock (``flown_dt`` absent, or
``schema`` below 2). Those still render, the old way, with a warning: the shape is
right and the timing is not.

Rendering at the camera's frame rate rather than the inference rate also makes the
panel the same length as the chase-cam clip, so ``compare_videos`` can stack the
two without one running ahead of the other.

matplotlib and numpy only — no torch, no simulator, so this runs on a laptop from
the JSON.
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Sequence

import numpy as np

from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.track_log import TrackLog

FLOWN = "#2a78d6"
PROPOSED = "#eb6834"
COMMITTED = "#1b7f3b"
GOAL = "#d81b60"
MARGIN_M = 4.0
"""Padding around the flight's extent, metres. Enough that the proposed
trajectory stays on screen when it heads away from the goal."""

PANEL_FPS = 10.0
"""Default panel frame rate.

Matches ``FlightRecorder``'s default capture rate, which is what the chase-cam
clip is encoded at -- so the panel and the camera come out the same length and
stack without drifting apart.
"""


def timeline(log: Dict, fps: float) -> List[float]:
    """The simulation times to draw a frame at, one per output frame.

    Spans the whole flight, from the first flown sample to the last, so the panel
    covers exactly what the camera recorded.

    Args:
        log: A track log with timing (``flown_dt`` present and non-zero).
        fps: Output frame rate.

    Returns:
        Simulation times, ascending. Empty if the log has no flown path.
    """
    flown = log.get("flown") or []
    if not flown:
        return []
    start = float(log.get("started_s", 0.0))
    end = start + (len(flown) - 1) * float(log["flown_dt"])
    count = max(1, int(round((end - start) * fps)) + 1)
    return [start + index / fps for index in range(count)]


def has_timing(log: Dict) -> bool:
    """Whether this log can place its flown positions in time."""
    return bool(log.get("flown_dt")) and bool(log.get("flown"))


def latest_inference(entries: Sequence[Dict], when: float) -> Optional[Dict]:
    """The most recent inference at or before ``when``, or None before the first.

    Args:
        entries: The log's inferences, in time order.
        when: Simulation time.

    Returns:
        The entry, or None -- which is the correct answer for the climb, when the
        policy has not been asked anything yet.
    """
    found = None
    for entry in entries:
        if entry["t"] > when:
            break
        found = entry
    return found


STOP_ARC_M = 0.05
"""A proposal shorter than this is the policy declining to move.

Not a dropped request and not a drawing fault -- a real answer, with all its
waypoints stacked on one point. It is most of what the pretrained checkpoint
emits in these scenes, so a panel that renders it as a bare dot and says nothing
is indistinguishable from a broken video. It gets said out loud instead."""


def is_stop(entry: Dict) -> bool:
    """Whether this inference returned a route with no length in it."""
    path = entry.get("traj")
    if not path:
        return False
    return float(np.linalg.norm(np.diff(np.asarray(path, dtype=float),
                                        axis=0), axis=1).sum()) < STOP_ARC_M


def committed_prefix(entry: Dict) -> Optional[np.ndarray]:
    """The part of a proposal the aircraft promised to fly: ``(k, 2)`` or None.

    ``entry["traj"]`` holds the predicted waypoints only — the anchor pose is not
    one of them — so the first ``entry["commit"]`` of them are the commitment.

    A log written before commitments existed (schema < 3) carries no ``commit``
    and gets ``None``. That is deliberate: every waypoint really was equally
    provisional on those flights, and drawing half of one as a promise would
    describe a flight that never made one.

    Args:
        entry: One inference from a track log.

    Returns:
        The committed waypoints, or ``None`` when the entry claims none.
    """
    held = int(entry.get("commit", 0))
    if held <= 0 or "traj" not in entry:
        return None
    return np.asarray(entry["traj"], dtype=float)[:held]


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


def _frame_plan(log: Dict, fps: float) -> List[Tuple[float, int, Optional[Dict]]]:
    """What each output frame shows: ``(time, flown samples so far, inference)``.

    On a log with timing this is the flight's own clock. On one without it falls
    back to the old one-frame-per-inference behaviour, where the flown prefix can
    only be guessed at from the fraction of the way through -- see this module's
    docstring for what that misrepresents.

    Args:
        log: A track log.
        fps: Output frame rate, used only when the log has timing.

    Returns:
        One tuple per output frame, in order.
    """
    entries = log["inferences"]
    total = len(log.get("flown") or [])

    if has_timing(log):
        plan = []
        for when in timeline(log, fps):
            elapsed = when - float(log.get("started_s", 0.0))
            upto = min(total, int(elapsed / float(log["flown_dt"])) + 1)
            plan.append((when, upto, latest_inference(entries, when)))
        return plan

    print("[track] this log predates flown-path timing (schema < 2), so the "
          "flown trail can only be spaced evenly across the inferences -- it will "
          "not line up with the aircraft. Re-fly to get an aligned panel.")
    return [(entry["t"], int(round((index + 1) / len(entries) * total)), entry)
            for index, entry in enumerate(entries)]


def render_frames(log: Dict, scene: str, out_dir: Path, altitude_m: float = 1.5,
                  size_px: int = 540, dpi: int = 100,
                  fps: float = PANEL_FPS) -> int:
    """One PNG per output frame. Returns how many were written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid, resolution, origin = load_map(scene, altitude_m)
    height, width = grid.shape
    extent = (origin[0], origin[0] + width * resolution,
              origin[1], origin[1] + height * resolution)
    x0, x1, y0, y1 = _bounds(log)

    flown = np.asarray(log.get("flown") or [], dtype=float).reshape(-1, 2)
    plan = _frame_plan(log, fps)
    inches = size_px / dpi

    for index, (when, upto, entry) in enumerate(plan):
        figure, axis = plt.subplots(figsize=(inches, inches), dpi=dpi)
        axis.imshow(grid > 0, cmap="Greys", origin="lower", extent=extent,
                    interpolation="nearest", vmin=0, vmax=1.6)

        if upto > 1:
            axis.plot(flown[:upto, 0], flown[:upto, 1], color=FLOWN,
                      linewidth=2.0, label="flown", zorder=3)
        if entry is not None and is_stop(entry):
            here = np.asarray(entry["traj"], dtype=float)[0]
            axis.plot(here[0], here[1], "x", color=PROPOSED, markersize=9,
                      markeredgewidth=2, zorder=5)
            axis.text(0.5, 0.94, "policy returned no route (stop)",
                      transform=axis.transAxes, ha="center", color=PROPOSED,
                      fontsize=9)
        elif entry is not None and "traj" in entry:
            proposed = np.asarray(entry["traj"], dtype=float)
            held = committed_prefix(entry)
            axis.plot(proposed[:, 0], proposed[:, 1], color=PROPOSED,
                      linewidth=1.6, linestyle="--", label="NavDP plan", zorder=4)
            axis.plot(proposed[-1, 0], proposed[-1, 1], "o", color=PROPOSED,
                      markersize=4, zorder=4)
            if held is not None:
                axis.plot(held[:, 0], held[:, 1], color=COMMITTED,
                          linewidth=2.8, label="committed", zorder=5)
                axis.plot(held[-1, 0], held[-1, 1], "o", color=COMMITTED,
                          markersize=6, zorder=5)
            if entry.get("why"):
                axis.text(0.02, 0.02, entry["why"], transform=axis.transAxes,
                          fontsize=7, color="0.25")
        elif entry is not None:
            axis.text(0.5, 0.94, "inference dropped", transform=axis.transAxes,
                      ha="center", color=PROPOSED, fontsize=9)
        else:
            # Before the first inference the aircraft is still climbing and the
            # policy has not been asked anything. Saying so beats an empty panel
            # that reads as a dropped request.
            axis.text(0.5, 0.94, "climbing to cruise altitude",
                      transform=axis.transAxes, ha="center", color="0.35", fontsize=9)

        # The marker is the aircraft's own position at this instant, from the
        # flown path -- not the pose the last inference was made at, which by now
        # is up to an inference period stale and was the other half of why the
        # marker and the trail disagreed. The heading arrow still comes from the
        # inference, since that is the only place a yaw is recorded.
        here = flown[upto - 1] if upto > 0 else None
        if here is not None:
            axis.plot([here[0]], [here[1]], "o", color="black", markersize=7, zorder=5)
            if entry is not None:
                yaw = entry["pose"][2]
                axis.arrow(here[0], here[1], 1.2 * np.cos(yaw), 1.2 * np.sin(yaw),
                           head_width=0.45, color="black", zorder=5,
                           length_includes_head=True)
        axis.plot([log["goal_xy"][0]], [log["goal_xy"][1]], "*", color=GOAL,
                  markersize=15, zorder=6, label="goal")

        axis.set_xlim(x0, x1)
        axis.set_ylim(y0, y1)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.legend(loc="lower right", fontsize=7, framealpha=0.85)
        axis.set_title(f"t = {when:.1f} s", fontsize=9)
        figure.tight_layout(pad=0.2)
        figure.savefig(out_dir / f"frame_{index:05d}.png")
        plt.close(figure)
    return len(plan)


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
           altitude_m: float = 1.5, size_px: int = 540,
           fps: float = PANEL_FPS) -> Optional[Path]:
    """Render one flight's map panel to an MP4.

    Args:
        track_path: A mission's ``*_track.json``.
        out_path: Where to write the MP4.
        scene: Scene key; defaults to the one named in the log.
        altitude_m: Which surveyed map to draw.
        size_px: Panel side, pixels.
        fps: Output frame rate. Leave at :data:`PANEL_FPS` to match the chase-cam
            clip, which is what makes the two stackable.

    Returns:
        The written path, or None if there was nothing to draw.
    """
    log = TrackLog.read(Path(track_path))
    if not log.get("inferences"):
        print(f"[track] {track_path.name} has no inferences")
        return None
    scene = scene or log.get("scene")
    if not scene:
        raise ValueError(f"{track_path} does not name a scene; pass --scene")

    if not has_timing(log):
        # An old log has no clock, so the only rate that means anything is the one
        # it was drawn at: one frame per inference.
        duration = max(1e-3, log["inferences"][-1]["t"] - log["inferences"][0]["t"])
        fps = max(1.0, len(log["inferences"]) / duration)
    with tempfile.TemporaryDirectory() as work:
        count = render_frames(log, scene, Path(work), altitude_m, size_px, fps=fps)
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
    parser.add_argument("--fps", type=float, default=PANEL_FPS,
                        help="panel frame rate. The default matches the chase-cam "
                             "clip's, which is what lets compare_videos stack the "
                             "two without one running ahead")
    args = parser.parse_args(argv)

    result = render(Path(args.track).expanduser(), Path(args.out).expanduser(),
                    args.scene, args.altitude, args.size, args.fps)
    return 0 if result else 1


if __name__ == "__main__":
    raise SystemExit(main())
