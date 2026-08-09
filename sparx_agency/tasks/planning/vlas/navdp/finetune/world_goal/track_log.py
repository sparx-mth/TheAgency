"""What the policy proposed, every time it was asked, in world coordinates.

The chase-camera video shows where the aircraft *went*. It cannot show what the
policy *wanted* — and the difference between the two is the whole question when
one set of weights flies into a shelf and another does not. NavDP re-plans at
around 4 Hz and each plan is 24 waypoints in the body frame at that instant, so
by the time the aircraft has moved the plan is gone unless something records it.

This is that recorder: one entry per inference, holding the pose it was made
from and the proposed trajectory already rotated into the world, so a map panel
can be drawn later without needing the aircraft, the simulator, or the policy.

Small on purpose — a 60-second flight is about 240 entries of 24 points, well
under a megabyte of JSON — because it is written next to multi-gigabyte imagery
and must never be the reason a recording is dropped.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


SCHEMA_VERSION = 3
"""Version 2 added the flown path's own clock to :meth:`TrackLog.set_flown`;
version 1 logs carry positions with no way to say *when* -- see ``track_video``
for what that cost. Version 3 adds ``commit`` and ``why`` to each inference: the
aircraft now flies half of a plan before asking for another, so *which* half it
promised to fly, and what ended the promise, are part of the story. Both are
optional and a reader must cope without them."""


class TrackLog:
    """Accumulates one entry per inference, then writes it as JSON."""

    def __init__(self, goal_xy: Sequence[float], start_xy: Sequence[float]) -> None:
        """
        Args:
            goal_xy: The mission goal, world frame.
            start_xy: Where the mission was planned to start from.
        """
        self.goal_xy = [float(goal_xy[0]), float(goal_xy[1])]
        self.start_xy = [float(start_xy[0]), float(start_xy[1])]
        self.started_s = 0.0
        self.flown_dt = 0.0
        self.entries: List[Dict] = []

    def add(self, sim_time: float, pose: Sequence[float],
            trajectory: Optional[np.ndarray], target_world: Sequence[float],
            commit_index: Optional[int] = None,
            reason: Optional[str] = None) -> None:
        """Record one inference.

        Args:
            sim_time: Simulation clock, so the panel can be matched to the video.
            pose: ``(x, y, yaw)`` in the world, FLU, at inference time.
            trajectory: The chosen body-frame trajectory ``(T, >=2)``, or None
                when the request failed — a dropped inference is part of the
                story and is recorded as an entry with no trajectory.
            target_world: The carrot the follower is actually chasing.
            commit_index: How many of the trajectory's waypoints the aircraft
                committed to flying before it would ask again. Recorded because
                the committed prefix and the speculative tail are two different
                claims, and a panel that draws them identically says the policy
                promised more than it did.
            reason: What ended the previous commitment and caused this
                inference — flown, expired, off route. One phrase; the panel
                shows it so a flight that is thrashing is visible as thrashing.
        """
        entry = {
            "t": round(float(sim_time), 3),
            "pose": [round(float(pose[0]), 3), round(float(pose[1]), 3),
                     round(float(pose[2]), 4)],
            "target": [round(float(target_world[0]), 3),
                       round(float(target_world[1]), 3)],
        }
        if trajectory is not None:
            entry["traj"] = [[round(float(x), 3), round(float(y), 3)]
                             for x, y in self.to_world(trajectory, pose)]
        if commit_index is not None:
            entry["commit"] = int(commit_index)
        if reason is not None:
            entry["why"] = str(reason)
        self.entries.append(entry)

    @staticmethod
    def to_world(trajectory: np.ndarray, pose: Sequence[float]) -> np.ndarray:
        """Body-frame ``(T, >=2)`` at ``pose`` -> world ``(T, 2)``.

        The same rotation the loss uses, kept here as plain numpy so a panel can
        be redrawn from a log without importing torch.
        """
        path = np.asarray(trajectory, dtype=np.float64)[:, :2]
        cos, sin = math.cos(float(pose[2])), math.sin(float(pose[2]))
        return np.stack([
            float(pose[0]) + path[:, 0] * cos - path[:, 1] * sin,
            float(pose[1]) + path[:, 0] * sin + path[:, 1] * cos,
        ], axis=-1)

    def set_flown(self, track: Sequence[Sequence[float]], started_s: float,
                  dt: float, stride: int = 10) -> None:
        """The path the aircraft actually took, and when each sample was taken.

        Sampled every ``stride``-th point: the control loop appends at the
        physics rate (250 Hz), so a minute of flight is 15,000 positions about
        2 mm apart — far finer than anything a map panel can draw, and enough
        JSON to matter next to the imagery.

        **The timing is not optional.** Without it the only way to place a stored
        position in time is to assume the flight's inferences and its positions
        cover the same span. They do not: the aircraft is recorded from the
        moment it leaves the ground, while inference is held off until it reaches
        cruise altitude, so the inferences start about ten seconds in. Drawing
        the two as if they were coextensive put the flown trail metres away from
        the aircraft marker for most of a comparison video and made a correctly
        located aircraft look badly mislocalised.

        Args:
            track: One position per control step, in order.
            started_s: The simulation clock at ``track[0]`` — the same instant the
                onboard recording and the chase camera start, to within one
                render period.
            dt: The control step, seconds.
            stride: Keep every ``stride``-th sample.
        """
        self.flown = [[round(float(x), 3), round(float(y), 3)]
                      for x, y in list(track)[::max(1, stride)]]
        self.started_s = float(started_s)
        self.flown_dt = float(dt) * max(1, stride)

    def flown_time(self, index: int) -> float:
        """The simulation clock at flown sample ``index``."""
        return self.started_s + index * self.flown_dt

    def write(self, path: Path, extra: Optional[Dict] = None) -> None:
        """Write the log, with whatever the caller knows about the flight.

        ``extra`` is merged **first** so this log's own keys always win. A flight
        result carries its own ``inferences`` — the *count* — and merging it last
        replaced the list of trajectories with an integer, destroying the only
        copy of what the policy proposed. Structure beats annotation.
        """
        payload = dict(extra or {})
        payload.update({"schema": SCHEMA_VERSION,
                        "goal_xy": self.goal_xy, "start_xy": self.start_xy,
                        "started_s": self.started_s, "flown_dt": self.flown_dt,
                        "flown": getattr(self, "flown", []),
                        "inference_count": len(self.entries),
                        "inferences": self.entries})
        Path(path).write_text(json.dumps(payload))

    @staticmethod
    def read(path: Path) -> Dict:
        """Read a log back."""
        return json.loads(Path(path).read_text())
