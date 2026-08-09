"""One VLA prediction, frozen in the world and flown as a route.

A policy answers in its own body frame at the instant it was asked. Anchoring
that answer at the pose it was asked from turns it from a steering suggestion
that expires with the next frame into a **route with a start, an end and an arc
length** -- something an aircraft can be measured against while it flies it.

The commitment is the first ``fraction`` of that route. The rest is kept, and
drawn, and used for the lookahead when the aircraft nears the end of the
committed part, but it is not what the aircraft promises to fly: the far end of
a learned trajectory is the part the policy is least sure of and the part the
world has had the most time to change under. Half is the default for the same
reason a pilot flies the leg they can see.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from sparx_agency.core.planning.vlas.common.plan_commit.progress import (
    DEGENERATE_M,
    cumulative_arc,
    project,
)


class CommittedPlan(object):
    """A body-frame trajectory anchored in the world, with a commit point.

    The polyline carries the anchor pose as its **first vertex**, so index ``k``
    is trajectory waypoint ``k`` and arc length is measured from the aircraft
    rather than from the policy's first step. That is what makes
    ``commit_index`` mean what an operator expects: point 8 of a 16-point
    prediction is ``world_xy[8]``.

    Attributes:
        world_xy: ``(T + 1, 2)`` world polyline -- the anchor, then every
            predicted waypoint.
        anchor: ``(x, y, yaw)`` the prediction was made from.
        issued_s: The clock the prediction was made at, in whatever timebase the
            caller uses consistently (simulation seconds, ROS time, ...).
        commit_index: Index into ``world_xy`` the commitment ends at.
        arc: ``(T + 1,)`` cumulative arc length along ``world_xy``.
    """

    def __init__(self, world_xy: np.ndarray, anchor: Sequence[float],
                 issued_s: float, commit_index: int) -> None:
        self.world_xy = np.asarray(world_xy, dtype=np.float64).reshape(-1, 2)
        self.anchor = (float(anchor[0]), float(anchor[1]), float(anchor[2]))
        self.issued_s = float(issued_s)
        self.commit_index = int(commit_index)
        self.arc = cumulative_arc(self.world_xy)

    # ── shape ────────────────────────────────────────────────────────
    @property
    def waypoints(self) -> int:
        """How many predicted waypoints the plan holds (the anchor is not one)."""
        return max(0, self.world_xy.shape[0] - 1)

    @property
    def committed_xy(self) -> np.ndarray:
        """The part being flown: anchor through the commit point, ``(k + 1, 2)``."""
        return self.world_xy[:self.commit_index + 1]

    @property
    def commit_point(self) -> Tuple[float, float]:
        """Where the commitment ends, world frame."""
        point = self.world_xy[self.commit_index]
        return float(point[0]), float(point[1])

    @property
    def commit_arc_m(self) -> float:
        """Arc length of the committed part, metres."""
        return float(self.arc[self.commit_index])

    @property
    def total_arc_m(self) -> float:
        """Arc length of the whole prediction, metres."""
        return float(self.arc[-1])

    # ── flying it ────────────────────────────────────────────────────
    def progress(self, x: float, y: float,
                 from_segment: int = 0) -> Tuple[float, float, int]:
        """``(arc_m, lateral_m, segment)`` of the aircraft against this plan.

        ``arc_m`` is how far along the *whole* polyline the aircraft has got --
        compare it with :attr:`commit_arc_m` to know whether the commitment has
        been flown. ``lateral_m`` is how far it is from the polyline, which is
        how a plan that is not being flown announces itself. ``segment`` is the
        cursor to pass back as ``from_segment`` next time; see
        :func:`~sparx_agency.core.planning.vlas.common.plan_commit.progress.project`
        for why a route that doubles back needs one.
        """
        return project(self.world_xy, x, y, from_segment)

    def segment_heading(self, segment: int) -> Optional[float]:
        """Which way the route runs on ``segment``: world radians, or None.

        This is the route's own tangent, and it is deliberately not the bearing
        from the aircraft to some point ahead. The expert that trained the
        policy encodes exactly this as the label's yaw channel --
        ``to_navdp_label`` takes ``atan2`` of each resampled route *step* -- so
        an aircraft that holds the tangent is holding the heading the policy was
        taught to expect, and its camera is looking down the route it is about
        to fly. A chord bearing cuts the corner: on a turn it points inside the
        arc, so the nose lags the route and the next observation is of somewhere
        the aircraft is not going.

        Returns:
            The heading, or ``None`` where the route does not move -- a stopped
            prediction has no direction, and inventing one points the nose east.
        """
        if self.world_xy.shape[0] < 2:
            return None
        seg = max(0, min(int(segment), self.world_xy.shape[0] - 2))
        delta = self.world_xy[seg + 1] - self.world_xy[seg]
        if float(np.hypot(delta[0], delta[1])) <= DEGENERATE_M:
            return None
        return float(np.arctan2(delta[1], delta[0]))

    def carrot(self, x: float, y: float, lookahead_m: float,
               from_segment: int = 0) -> Tuple[float, float, Optional[float]]:
        """Where to fly and which way to look: ``(x, y, heading)``.

        Pure pursuit against the frozen route, not a fixed point re-picked every
        inference: as the aircraft advances the carrot advances with it, so the
        *shape* of the prediction gets flown instead of only its first segment.

        The lookahead is measured **along the route**, not as a radius around the
        aircraft. The two agree on anything gently curved and disagree exactly
        where it matters. A radius asks "which point on the route is
        ``lookahead_m`` away from me", and on a U-turn tighter than the
        lookahead the answer is a point on the *return* leg, beside the
        aircraft and pointing back the way it came -- so the aircraft is told to
        skip the turn and reverse. Measured on a 24-waypoint switchback, a
        radius carrot at 1.2 m returned a bearing of 168 degrees. Arc length
        cannot do that: the point 1.2 m further along a route is 1.2 m further
        along it, whatever shape it is.

        The carrot rides the **whole** prediction, not just the committed part.
        It has to: a lookahead that stopped at the commit point would shrink to
        nothing as the aircraft approached it and decelerate into every leg
        boundary. So the last ``lookahead_m`` of a commitment is steered by the
        speculative tail -- aimed at, never promised, and replaced by the next
        inference before the aircraft gets there.

        Beyond the end of the plan the last vertex is returned, which decelerates
        the aircraft onto it rather than overshooting.

        The third value is the route's heading where the carrot sits -- see
        :meth:`segment_heading`. Fly the point, look along the heading.
        """
        arc, _, _ = project(self.world_xy, float(x), float(y), from_segment)
        target = min(arc + float(lookahead_m), float(self.arc[-1]))
        # searchsorted 'right' then step back names the segment the target arc
        # falls inside, which is the one whose direction the aircraft should
        # hold. Clamped because a target exactly at the end lands past the last.
        segment = max(0, int(np.searchsorted(self.arc, target, side="right")) - 1)
        return (float(np.interp(target, self.arc, self.world_xy[:, 0])),
                float(np.interp(target, self.arc, self.world_xy[:, 1])),
                self.segment_heading(segment))


def commit_index_for(waypoints: int, fraction: float) -> int:
    """Which waypoint a ``fraction`` commitment ends at.

    Rounded, then clamped into ``[1, waypoints]``: a commitment of zero
    waypoints is a plan that is finished before it starts and would re-infer on
    the very next tick, which is the failure this whole package exists to stop.

    Args:
        waypoints: How many waypoints the prediction holds.
        fraction: Share of the prediction to commit to, ``0..1``.

    Returns:
        An index into a ``(waypoints + 1, 2)`` polyline whose first vertex is
        the anchor -- so ``8`` for 16 waypoints at ``0.5``.
    """
    if waypoints < 1:
        raise ValueError("a prediction with no waypoints cannot be committed to")
    index = int(round(float(fraction) * waypoints))
    return max(1, min(int(waypoints), index))


def anchor_plan(trajectory: np.ndarray, pose: Sequence[float], issued_s: float,
                fraction: float) -> CommittedPlan:
    """Freeze a body-frame prediction into a :class:`CommittedPlan`.

    Args:
        trajectory: ``(T, >=2)`` body-frame ``(forward, left)`` waypoints; extra
            columns (NavDP carries a yaw) are ignored.
        pose: ``(x, y, yaw)`` the prediction was made from, world FLU. Use the
            pose captured with the frame that produced it, not the live one --
            anything else bakes the inference latency into the route as a
            translation.
        issued_s: Clock at inference.
        fraction: Share of the prediction to commit to.

    Returns:
        The anchored plan.

    Raises:
        ValueError: The trajectory holds no waypoints.
    """
    body = np.atleast_2d(np.asarray(trajectory, dtype=np.float64))[:, :2]
    if body.shape[0] < 1:
        raise ValueError("a prediction with no waypoints cannot be committed to")
    cos, sin = np.cos(float(pose[2])), np.sin(float(pose[2]))
    world = np.stack([
        float(pose[0]) + body[:, 0] * cos - body[:, 1] * sin,
        float(pose[1]) + body[:, 0] * sin + body[:, 1] * cos,
    ], axis=1)
    anchored = np.concatenate(
        [np.array([[float(pose[0]), float(pose[1])]]), world], axis=0)
    return CommittedPlan(anchored, pose, issued_s,
                         commit_index_for(body.shape[0], fraction))
