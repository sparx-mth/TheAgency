"""Scoring a trajectory against the ground truth, not against its own teacher.

The previous evaluation measured the fine-tuned policy's distance to the label
it had been trained on, using the single depth frame the label was built from.
Both halves of that are circular: a model that memorised its teacher scores
perfectly, and a wall the camera could not see is a wall the ruler cannot see
either.

Here every number comes from the **surveyed map**, which no part of the training
loop can influence, and which knows about geometry outside the field of view.
Nine numbers per trajectory, in three groups:

*safety*      ``min_clear_m``, ``p5_clear_m``, ``frac_below_safe``, ``collides``.
              The minimum is what actually kills an aircraft, so it is reported
              rather than a mean that a single excursion cannot move.
*destination* ``goal_progress_m`` -- how much closer to the true world goal the
              trajectory ends than it started -- and ``goal_gap_m``.
*quality*     ``centre_offset_m``, the distance from the corridor's medial axis,
              which is the direct measurement of "does it fly down the middle";
              ``bending``, total heading change, since a safe path that
              oscillates is not a good one; and ``path_len_m``.

**Everything is densified to 2 cm first.** NavDP's waypoints are ~20 cm apart and
a 10 cm wall fits comfortably between two of them, so a trajectory scored at its
own resolution can pass clean through a partition.

Pure numpy against a :class:`~.scene.Scene`; no torch, so the same scoring runs
on offline trajectories and on a live flight log.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.polyline import (
    arclength, resample, to_world,
)
from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.scene import Scene

# Which direction is an improvement, per metric. Passed to the paired-statistics
# module so a positive delta always means "the fine-tune helped", whichever way
# the underlying number happens to point.
HIGHER_IS_BETTER = {
    "min_clear_m": True,
    "p5_clear_m": True,
    "mean_clear_m": True,
    "frac_below_safe": False,
    "collides": False,
    "goal_progress_m": True,
    "goal_gap_m": False,
    "centre_offset_m": False,
    "bending": False,
    "path_len_m": False,
}


@dataclass(frozen=True)
class TrajectoryMetrics:
    """One trajectory, scored against the surveyed map."""

    min_clear_m: float
    p5_clear_m: float
    mean_clear_m: float
    frac_below_safe: float
    collides: float
    goal_progress_m: float
    goal_gap_m: float
    centre_offset_m: float
    bending: float
    path_len_m: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def densify(points: np.ndarray, step_m: float = 0.02) -> np.ndarray:
    """Resample a polyline to ``step_m`` so no obstacle can hide between samples."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] < 2 or float(arclength(points)[-1]) <= step_m:
        return points
    return resample(points, step_m)


def bending(points: np.ndarray) -> float:
    """Total absolute heading change along a polyline, radians.

    Zero for a straight line. Measured on the *sparse* waypoints, not the
    densified copy, since densification would report interpolation noise.
    """
    steps = np.diff(np.asarray(points, dtype=np.float64).reshape(-1, 2), axis=0)
    keep = np.linalg.norm(steps, axis=1) > 1e-9
    if keep.sum() < 2:
        return 0.0
    headings = np.arctan2(steps[keep, 1], steps[keep, 0])
    delta = np.diff(headings)
    return float(np.abs(np.arctan2(np.sin(delta), np.cos(delta))).sum())


def centre_offset(world_points: np.ndarray, scene: Scene, search_m: float = 1.2,
                  step_m: float = 0.05) -> float:
    """Mean distance from the corridor's medial axis, metres.

    For each waypoint, clearance is sampled along the local path normal over
    ``+-search_m``; the offset to the maximum is how far that waypoint sits from
    the locus of greatest clearance, which is the medial axis. Zero means dead
    centre. This is the direct measurement of the behaviour being trained for,
    and it is deliberately independent of the expert: a trajectory can be
    perfectly centred while looking nothing like the label.

    Waypoints whose corridor is wider than ``search_m`` either side are skipped
    -- in an open hall there is no meaningful centre to be off.
    """
    points = np.asarray(world_points, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] < 2:
        return 0.0
    steps = np.diff(points, axis=0, prepend=points[:1])
    steps[0] = steps[1] if points.shape[0] > 1 else steps[0]
    norms = np.linalg.norm(steps, axis=1, keepdims=True)
    tangent = steps / np.maximum(norms, 1e-9)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)

    offsets = np.arange(-search_m, search_m + step_m * 0.5, step_m)
    probes = points[:, None, :] + normal[:, None, :] * offsets[None, :, None]
    clearance = scene.clearance(probes[..., 0].ravel(), probes[..., 1].ravel())
    clearance = clearance.reshape(points.shape[0], offsets.size)

    best = np.argmax(clearance, axis=1)
    distance = np.abs(offsets[best])
    interior = (best > 0) & (best < offsets.size - 1)     # a real maximum, not the edge
    return float(distance[interior].mean()) if interior.any() else 0.0


def score(waypoints_body: np.ndarray, pose: Sequence[float], goal_world: Sequence[float],
          scene: Scene, d_safe_m: float = 0.5, step_m: float = 0.02) -> TrajectoryMetrics:
    """Score one body-frame trajectory against the surveyed map.

    Args:
        waypoints_body: ``(T, 2)`` ``[forward, left]`` metres, the policy's output.
        pose: ``(x, y, yaw)`` world pose the trajectory was produced at.
        goal_world: ``(x, y)`` the true world goal it was aimed at.
        scene: The surveyed building.
        d_safe_m: Clearance below which a waypoint counts as unsafe.
        step_m: Densification spacing before clearance is measured.
    """
    sparse = to_world(np.asarray(waypoints_body, dtype=np.float64).reshape(-1, 2), pose)
    dense = densify(sparse, step_m)
    clearance = scene.clearance(dense[:, 0], dense[:, 1])

    goal = np.asarray(goal_world, dtype=np.float64)
    start_gap = float(np.hypot(pose[0] - goal[0], pose[1] - goal[1]))
    end_gap = float(np.hypot(sparse[-1, 0] - goal[0], sparse[-1, 1] - goal[1]))

    return TrajectoryMetrics(
        min_clear_m=float(clearance.min()),
        p5_clear_m=float(np.percentile(clearance, 5)),
        mean_clear_m=float(clearance.mean()),
        frac_below_safe=float((clearance < d_safe_m).mean()),
        collides=float(clearance.min() < 0.0),
        goal_progress_m=start_gap - end_gap,
        goal_gap_m=end_gap,
        centre_offset_m=centre_offset(sparse, scene),
        bending=bending(sparse),
        path_len_m=float(arclength(sparse)[-1]),
    )


def summarise(rows: List[TrajectoryMetrics]) -> Dict[str, float]:
    """Mean of every field, plus the tail statistics a mean would hide."""
    if not rows:
        return {}
    table = {name: np.array([getattr(row, name) for row in rows], dtype=np.float64)
             for name in TrajectoryMetrics.__annotations__}
    out = {name: float(values.mean()) for name, values in table.items()}
    out["min_clear_m_p05"] = float(np.percentile(table["min_clear_m"], 5))
    out["min_clear_m_worst"] = float(table["min_clear_m"].min())
    out["collision_rate"] = float(table["collides"].mean())
    out["n"] = float(len(rows))
    return out
