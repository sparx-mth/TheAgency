"""Where to send the drone: goals that are always somewhere it could actually go.

This module exists because of one concrete failure in the previous fine-tune.
Goals were made by picking a pixel and back-projecting it with depth, so a goal
*was* the surface the ray hit -- a wall, a desk, the floor. Roughly a third of
the labels asked the policy to fly into geometry, and the network dutifully
learned to.

Here a goal is drawn from the surveyed map instead of from the image, and it can
only come from :attr:`Scene.goal_region`: cells that are free, at least
``goal_clearance_m`` from anything, marked landable, and all in one connected
component. A goal on an obstacle is not filtered out -- it cannot be produced.
Reachability is then settled by the expert planner (:mod:`.expert`): if A* finds
no route, the candidate is discarded and another is drawn.

Because the goal comes from a map rather than a camera, it does **not** have to
be visible. That is the point: a goal 20 m away behind a corner is exactly the
supervision that teaches a policy to head for the doorway instead of the wall in
front of it. The long-range intent travels in the goal token; the camera only
has to solve the next few metres.

Five kinds are drawn in a fixed mixture so the policy sees the whole job:

``route``   a point on the flight's own remaining path -- the most natural goal.
``near``    inside the prediction horizon: teaches arriving and slowing down.
``mid``     just beyond it: ordinary corridor cruising.
``far``     across the building: teaches committing to a direction.
``corner``  deliberately off to one side: teaches turning toward something the
            camera cannot see.

Pure numpy. No planner call happens here -- proposing is cheap, planning is not,
so the caller plans only the candidates it actually wants.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, cos, degrees, radians, sin
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from sparx_agency.core.common.types.geometry import normalize_angle
from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.scene import Scene

GOAL_KINDS: Tuple[str, ...] = ("route", "near", "mid", "far", "corner")


def _default_weights() -> Dict[str, float]:
    return {"route": 0.25, "near": 0.15, "mid": 0.25, "far": 0.15, "corner": 0.20}


def _default_bearings() -> Dict[str, Tuple[float, float]]:
    """Where off the nose each kind may draw from, degrees.

    Per-kind rather than one global cap, because the cap alone produces a goal
    bearing that is close to uniform over its whole range -- and since the label
    ends up pointing at the goal, that makes the *median* training sample a
    45-degree turn and leaves under 10 % of samples going roughly straight. A
    policy needs "keep going" examples as much as it needs corners.

    ``far`` is the tightest: a distant goal is a direction to commit to, and a
    distant goal 70 degrees off the nose mostly produces a near-reversal that
    the label auditor then throws away -- expensive to plan and useless to keep.
    ``corner`` is the one kind that deliberately draws off-axis; raising its
    weight is how you ask for more turning, rather than widening everything.
    """
    return {"near": (0.0, 45.0), "mid": (0.0, 45.0),
            "far": (0.0, 30.0), "corner": (30.0, 75.0)}


@dataclass(frozen=True)
class GoalSamplerConfig:
    """How goals are drawn for one frame.

    Attributes:
        goals_per_frame: How many accepted labels to produce per frame. One frame
            supports many goals because the *image* is shared and only the goal
            token and the target route change -- which is what multiplies a few
            thousand frames into a real dataset.
        kind_weights: Mixture over :data:`GOAL_KINDS`; renormalised.
        bearing_deg: Per-kind ``(min, max)`` bearing off the nose, degrees. This
            is what controls how much of the dataset is turning; see
            :func:`_default_bearings`.
        near_range_m / mid_range_m / far_range_m: Distance bands, metres. The
            ``far`` ceiling is also a cost knob: a route across the whole
            building is the most expensive A* call in the pipeline.
        route_range_m: Along-path distance band for the ``route`` kind.
        min_forward_m: Reject a goal less than this far in front of the aircraft.
            NavDP's goal encoder collapses anything at or behind the camera plane
            to a fixed tiny straight-ahead goal, so such a sample carries no
            information about where it was actually meant to go.
        max_bearing_deg: Reject goals further off the nose than this. Beyond it
            the goal token stops being informative for the same reason.
        corner_bearing_deg: Bearing band the ``corner`` kind draws from -- far
            enough off-axis that the aircraft must turn to reach it.
        goal_stride_cells: Sub-sample the goal-eligible cells by this stride
            before searching. Goals do not need 10 cm granularity, and a 5x
            stride makes the per-frame distance pass 25x cheaper.
        snap_radius_m: How far a ``route`` point may be pulled to reach a
            goal-eligible cell. The flown path can pass closer to a wall than a
            goal is allowed to sit.
        max_attempts_per_goal: Proposals to try before giving up on one goal.
    """

    goals_per_frame: int = 10
    kind_weights: Dict[str, float] = field(default_factory=_default_weights)
    bearing_deg: Dict[str, Tuple[float, float]] = field(default_factory=_default_bearings)
    near_range_m: Tuple[float, float] = (1.5, 5.0)
    mid_range_m: Tuple[float, float] = (5.0, 12.0)
    far_range_m: Tuple[float, float] = (12.0, 25.0)
    route_range_m: Tuple[float, float] = (2.0, 20.0)
    min_forward_m: float = 0.5
    max_bearing_deg: float = 75.0
    goal_stride_cells: int = 5
    snap_radius_m: float = 1.0
    max_attempts_per_goal: int = 12


@dataclass(frozen=True)
class GoalCandidate:
    """One proposed goal, before the planner has had a say."""

    x: float
    y: float
    kind: str
    distance_m: float
    bearing_rad: float


class GoalSampler:
    """Draws goal candidates for frames of one scene.

    The scene's goal-eligible cells are sub-sampled and cached on construction,
    so per-frame sampling is a single vectorised distance pass.
    """

    def __init__(self, scene: Scene, config: Optional[GoalSamplerConfig] = None) -> None:
        self.scene = scene
        self.config = config or GoalSamplerConfig()
        stride = max(1, int(self.config.goal_stride_cells))
        cells = np.argwhere(scene.goal_region)
        keep = (cells[:, 0] % stride == 0) & (cells[:, 1] % stride == 0)
        cells = cells[keep] if keep.any() else cells
        xs = (cells[:, 1] + 0.5) * scene.resolution + scene.grid.origin_x
        ys = (cells[:, 0] + 0.5) * scene.resolution + scene.grid.origin_y
        self.points = np.stack([xs, ys], axis=1).astype(np.float64)
        if self.points.shape[0] == 0:
            raise ValueError(f"scene {scene.name!r} has no goal-eligible cell")

        weights = np.array([max(0.0, float(self.config.kind_weights.get(k, 0.0)))
                            for k in GOAL_KINDS], dtype=np.float64)
        if weights.sum() <= 0.0:
            raise ValueError("kind_weights must have at least one positive entry")
        self._kind_p = weights / weights.sum()

    # ------------------------------------------------------------------ util
    def _relative(self, pose: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Distance and body-frame bearing from ``pose`` to every candidate cell."""
        dx = self.points[:, 0] - float(pose[0])
        dy = self.points[:, 1] - float(pose[1])
        distance = np.hypot(dx, dy)
        bearing = np.arctan2(dy, dx) - float(pose[2])
        bearing = np.arctan2(np.sin(bearing), np.cos(bearing))
        return distance, bearing

    def snap(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        """Nearest goal-eligible point, or None if none is within the snap radius."""
        offsets = self.points - np.array([x, y], dtype=np.float64)
        distance = np.hypot(offsets[:, 0], offsets[:, 1])
        best = int(np.argmin(distance))
        if distance[best] > self.config.snap_radius_m:
            return None
        return float(self.points[best, 0]), float(self.points[best, 1])

    # -------------------------------------------------------------- proposals
    def _band(self, distance: np.ndarray, bearing: np.ndarray, band: Tuple[float, float],
              bearing_band: Optional[Tuple[float, float]] = None) -> np.ndarray:
        cfg = self.config
        forward = distance * np.cos(bearing)
        ok = ((distance >= band[0]) & (distance <= band[1])
              & (forward >= cfg.min_forward_m)
              & (np.abs(bearing) <= radians(cfg.max_bearing_deg)))
        if bearing_band is not None:
            magnitude = np.abs(bearing)
            ok &= ((magnitude >= radians(bearing_band[0]))
                   & (magnitude <= radians(bearing_band[1])))
        return np.flatnonzero(ok)

    def _from_route(self, pose: Sequence[float], rng: np.random.Generator,
                    route_ahead: Optional[np.ndarray]) -> Optional[GoalCandidate]:
        """A point on the flight's own remaining path, snapped to a legal cell."""
        if route_ahead is None or route_ahead.shape[0] < 2:
            return None
        steps = np.linalg.norm(np.diff(route_ahead, axis=0), axis=1)
        along = np.concatenate([[0.0], np.cumsum(steps)])
        lo, hi = self.config.route_range_m
        hi = min(hi, float(along[-1]))
        if hi < lo:
            return None
        target = float(rng.uniform(lo, hi))
        x = float(np.interp(target, along, route_ahead[:, 0]))
        y = float(np.interp(target, along, route_ahead[:, 1]))
        snapped = self.snap(x, y)
        if snapped is None:
            return None
        return self._make(pose, snapped[0], snapped[1], "route")

    def _make(self, pose: Sequence[float], x: float, y: float,
              kind: str) -> Optional[GoalCandidate]:
        cfg = self.config
        dx, dy = x - float(pose[0]), y - float(pose[1])
        distance = float(np.hypot(dx, dy))
        bearing = normalize_angle(atan2(dy, dx) - float(pose[2]))
        if distance * cos(bearing) < cfg.min_forward_m:
            return None
        if abs(degrees(bearing)) > cfg.max_bearing_deg:
            return None
        return GoalCandidate(x=x, y=y, kind=kind, distance_m=distance,
                             bearing_rad=bearing)

    def candidates(self, pose: Sequence[float], rng: np.random.Generator,
                   route_ahead: Optional[np.ndarray] = None,
                   limit: int = 200) -> Iterator[GoalCandidate]:
        """Yield goal proposals for one frame, in the configured mixture.

        Args:
            pose: ``(x, y, yaw)`` world pose of the aircraft.
            rng: Seeded generator; the only source of randomness.
            route_ahead: ``(M, 2)`` remaining world path of the recording, for the
                ``route`` kind. ``None`` re-rolls that kind as ``mid``.
            limit: Stop after this many proposals, so a frame in a dead end
                cannot loop forever.

        Yields:
            :class:`GoalCandidate` values, unvalidated by any planner.
        """
        cfg = self.config
        distance, bearing = self._relative(pose)
        ranges = {"near": cfg.near_range_m, "mid": cfg.mid_range_m,
                  "far": cfg.far_range_m, "corner": cfg.mid_range_m}
        bands = {kind: self._band(distance, bearing, span,
                                  cfg.bearing_deg.get(kind))
                 for kind, span in ranges.items()}
        for _ in range(int(limit)):
            kind = GOAL_KINDS[int(rng.choice(len(GOAL_KINDS), p=self._kind_p))]
            if kind == "route":
                candidate = self._from_route(pose, rng, route_ahead)
                if candidate is not None:
                    yield candidate
                    continue
                kind = "mid"
            pool = bands.get(kind)
            if pool is None or pool.size == 0:
                continue
            index = int(rng.choice(pool))
            candidate = self._make(pose, float(self.points[index, 0]),
                                   float(self.points[index, 1]), kind)
            if candidate is not None:
                yield candidate


def route_ahead_world(poses: np.ndarray, frame: int,
                      max_points: int = 400) -> Optional[np.ndarray]:
    """The recording's own remaining world path from ``frame`` onward.

    Args:
        poses: ``(N, >=3)`` array whose columns 1 and 2 are world ``x, y``.
        frame: Index to start from.
        max_points: Cap, so a long flight does not make the arc-length pass slow.

    Returns:
        ``(M, 2)`` world path, or ``None`` if fewer than two points remain.
    """
    tail = poses[frame:, 1:3]
    if tail.shape[0] < 2:
        return None
    if tail.shape[0] > max_points:
        step = int(np.ceil(tail.shape[0] / max_points))
        tail = tail[::step]
    return np.asarray(tail, dtype=np.float64)
