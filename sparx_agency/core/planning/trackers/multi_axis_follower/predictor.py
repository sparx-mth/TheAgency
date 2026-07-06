"""Forward-simulate the multi-axis follower to predict the trajectory it flies.

Rolls a throwaway :class:`MultiAxisFollower` forward against a first-order-lag
*holonomic* plant (the platform lags the commanded forward, lateral and yaw
rates and coasts when a command drops). Lets a caller see the approximate path
the drone will actually fly from its current pose and score it. Pure and
ROS-free; obstacle awareness is injected so the core never depends on a map.

Reuses :class:`~...waypoint_follower.predictor.PredictionResult`,
:class:`~...waypoint_follower.predictor.MotionModelParams` and
:func:`~...waypoint_follower.predictor.prediction_score` so a viewer can score the
output of either tracker the same way (the ``vx_tau_s`` time constant is applied
to both translation axes; ``yaw_tau_s`` to yaw).
"""
from __future__ import annotations

from math import ceil, cos, hypot, sin
from typing import Callable, List, Optional, Sequence

from sparx_agency.core.common.types import Pose2D, normalize_angle

from ..waypoint_follower.predictor import (  # noqa: F401 (PredictionResult re-export)
    MotionModelParams,
    PredictionResult,
    prediction_score,
)
from .follower import MultiAxisFollower
from .params import MultiAxisFollowerParams
from .types import MultiAxisState

OccupiedFn = Callable[[float, float], bool]


def _lag(actual: float, command: float, dt: float, tau: float) -> float:
    """One stable first-order-lag step toward ``command`` (alpha in (0, 1))."""
    if tau <= 0.0:
        return command
    alpha = dt / (tau + dt)
    return actual + (command - actual) * alpha


def predict_trajectory(
    params: MultiAxisFollowerParams,
    start: Pose2D,
    path: Sequence[Pose2D],
    dt: float,
    horizon_s: float,
    motion: Optional[MotionModelParams] = None,
    occupied_fn: Optional[OccupiedFn] = None,
) -> PredictionResult:
    """Roll the multi-axis follower forward from ``start`` along ``path``.

    A throwaway :class:`MultiAxisFollower` is stepped (axis always confirmed, no
    hold) while a first-order-lag holonomic plant integrates its body-frame
    commands into the world pose.

    Args:
        params: The live follower tuning (so the prediction matches behaviour).
        start: Current pose to roll out from.
        path: World waypoints (>= 2).
        dt: Control period used for the rollout (s).
        horizon_s: Max simulated time (s); the rollout also stops at HOLD.
        motion: Plant inertia model; defaults to :class:`MotionModelParams`.
        occupied_fn: Optional ``(x, y) -> bool`` obstacle test for ``collides``.

    Raises:
        ValueError: on a path shorter than 2 points, or non-positive dt/horizon.
    """
    if path is None or len(path) < 2:
        raise ValueError("predict_trajectory needs a path with >= 2 waypoints")
    if dt <= 0.0:
        raise ValueError("predict_trajectory needs dt > 0")
    if horizon_s <= 0.0:
        raise ValueError("predict_trajectory needs horizon_s > 0")

    motion = motion or MotionModelParams()
    follower = MultiAxisFollower(params)
    follower.set_path(list(path), start)

    x, y, yaw = start.x, start.y, start.yaw
    vx_act = vy_act = wz_act = 0.0
    poses: List[Pose2D] = [Pose2D(x, y, yaw)]
    total_yaw = 0.0
    n_stops = 0
    collides = bool(occupied_fn is not None and occupied_fn(x, y))
    reached_hold = False

    for _ in range(int(ceil(horizon_s / dt))):
        cmd = follower.step(Pose2D(x, y, yaw), dt, axis_confirmed=True)
        if follower.state == MultiAxisState.HOLD and not reached_hold:
            n_stops += 1                     # the single terminal station-keep
            reached_hold = True

        vx_act = _lag(vx_act, cmd.vx, dt, motion.vx_tau_s)
        vy_act = _lag(vy_act, cmd.vy, dt, motion.vx_tau_s)
        wz_act = _lag(wz_act, cmd.wz, dt, motion.yaw_tau_s)
        new_yaw = normalize_angle(yaw + wz_act * dt)
        total_yaw += abs(normalize_angle(new_yaw - yaw))
        yaw = new_yaw
        # Body (forward, left) -> world via the yaw rotation.
        x += (vx_act * cos(yaw) - vy_act * sin(yaw)) * dt
        y += (vx_act * sin(yaw) + vy_act * cos(yaw)) * dt
        poses.append(Pose2D(x, y, yaw))

        if occupied_fn is not None and occupied_fn(x, y):
            collides = True
        if follower.done:
            break

    goal = path[-1]
    end_gap = hypot(goal.x - poses[-1].x, goal.y - poses[-1].y)
    return PredictionResult(
        poses=poses,
        reaches_goal=follower.done,
        end_gap=end_gap,
        total_yaw=total_yaw,
        n_stops=n_stops,
        collides=collides,
    )
