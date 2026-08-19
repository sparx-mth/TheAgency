"""Turn a world-frame polyline into a constant-cruise :class:`Trajectory`.

The N1 policy node publishes the committed prediction as a bare ``nav_msgs/Path``
-- a sequence of world points with no timing, because a language-conditioned
policy has no notion of *when* to be anywhere, only *where* to go. The pure
pursuit tracker, on the other hand, consumes a
:class:`~sparx_agency.core.common.types.Trajectory` and samples it by time. This
module is the seam between the two: it time-parameterises the polyline at a fixed
cruise speed and hands back a
:class:`~sparx_agency.core.planning.smoothers.adapter.DiscreteTrajectory`, so the
follower stays a pure tracker and the "how fast" lives in one place.

It is deliberately ROS-free -- it takes plain ``(x, y, z)`` tuples, not
``PoseStamped`` -- so the timing law is unit-tested in the plain ``.venv`` with no
ROS 2 on the path, next to the sign and monotonicity bugs that live in it.
"""
from __future__ import annotations

from math import atan2, hypot
from typing import List, Sequence, Tuple

from sparx_agency.core.common.types import TrajectoryPoint
from sparx_agency.core.planning.smoothers.adapter import DiscreteTrajectory

#: Consecutive points closer than this collapse to one: a zero-length segment
#: carries no heading and would stall the arc clock without advancing the route.
MIN_SEGMENT_M = 1e-3


def _dedup(points, min_segment_m):
    # type: (Sequence[Sequence[float]], float) -> List[Tuple[float, float, float]]
    """Drop points that repeat their predecessor within ``min_segment_m``."""
    kept = []  # type: List[Tuple[float, float, float]]
    for p in points:
        xyz = (float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0)
        if not kept or hypot(xyz[0] - kept[-1][0], xyz[1] - kept[-1][1]) > min_segment_m:
            kept.append(xyz)
    return kept


def trajectory_from_points(points, cruise_speed, min_segment_m=MIN_SEGMENT_M):
    # type: (Sequence[Sequence[float]], float, float) -> DiscreteTrajectory
    """Time-parameterise a world polyline at a constant horizontal cruise speed.

    Args:
        points: ordered world ``(x, y[, z])`` vertices -- the committed N1 route.
        cruise_speed: horizontal speed to schedule the route at, m/s. The
            follower reads its own speed profile from the tracker; this only
            spaces the samples in time so ``sample_by_time`` returns them evenly.
        min_segment_m: collapse consecutive vertices closer than this.

    Returns:
        A :class:`DiscreteTrajectory` whose points carry position, the route
        tangent as ``yaw``, and cumulative arc length as ``s``. Velocities are
        left zero: the pure pursuit tracker derives speed from its own cruise
        parameter, not from the reference.

    Raises:
        ValueError: fewer than two distinct points, or a non-positive
            ``cruise_speed`` -- both of which mean "there is nothing to fly", and
            the follower answers that with a hold, not a guess.
    """
    if cruise_speed <= 0.0:
        raise ValueError("cruise_speed must be > 0, got %r" % (cruise_speed,))
    kept = _dedup(points, min_segment_m)
    if len(kept) < 2:
        raise ValueError("need at least two distinct points, got %d" % (len(kept),))

    samples = []  # type: List[TrajectoryPoint]
    arc = 0.0
    for i, (x, y, z) in enumerate(kept):
        if i < len(kept) - 1:
            nxt = kept[i + 1]
            yaw = atan2(nxt[1] - y, nxt[0] - x)
        else:
            yaw = samples[-1].yaw  # last vertex holds the heading that reached it
        if i > 0:
            prev = kept[i - 1]
            arc += hypot(x - prev[0], y - prev[1])
        samples.append(TrajectoryPoint(
            t=arc / cruise_speed, x=x, y=y, z=z, yaw=yaw, s=arc))
    return DiscreteTrajectory(samples)

