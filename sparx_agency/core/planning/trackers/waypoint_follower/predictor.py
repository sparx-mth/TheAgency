"""Forward-simulate the waypoint follower to predict the trajectory it will fly.

The one-axis-at-a-time follower never tracks a path exactly: it advances in
straight segments, pauses to settle, and yaws in place — overshooting and
coasting on the way. This module rolls a throwaway :class:`WaypointFollower`
forward against a simple first-order-lag unicycle plant so a caller can *see*
the approximate path the drone will actually fly from its current pose, and
score how good that is (does it reach the goal, how much does it stop/turn, does
it hit anything). Pure and ROS-free; obstacle awareness is injected so the core
never depends on a map.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, cos, hypot, pi, sin
from typing import Callable, List, Optional, Sequence

from sparx_agency.core.common.types import Pose2D, normalize_angle

from .follower import WaypointFollower
from .params import WaypointFollowerParams
from .types import FollowerState

OccupiedFn = Callable[[float, float], bool]
_PAUSE_STATES = (FollowerState.BRAKE, FollowerState.YAW_SETTLE)


@dataclass(frozen=True)
class MotionModelParams:
    """First-order-lag unicycle plant emulating the platform's inertia.

    The follower emits a slew-shaped command; the real platform lags it (it
    cannot change yaw rate or speed instantly and coasts when the command drops).
    Modelling that lag makes the prediction show the real overshoot/coast rather
    than a perfect retrace of the path.

    Attributes:
        yaw_tau_s: Yaw-rate first-order time constant (s); larger = more inertia.
        vx_tau_s: Forward-speed first-order time constant (s).
    """

    yaw_tau_s: float = 0.5
    vx_tau_s: float = 0.3


@dataclass(frozen=True)
class PredictionResult:
    """Outcome of a follower rollout.

    Attributes:
        poses: Predicted poses (x, y, yaw), one per simulated tick (>= 1).
        reaches_goal: Whether the follower reached DONE within the horizon.
        end_gap: Distance from the last predicted pose to the final waypoint (m).
        total_yaw: Total absolute heading swept over the rollout (rad).
        n_stops: Number of pauses (entries into BRAKE/YAW_SETTLE).
        collides: True if any predicted pose was occupied (needs ``occupied_fn``).
    """

    poses: List[Pose2D]
    reaches_goal: bool
    end_gap: float
    total_yaw: float
    n_stops: int
    collides: bool


def _lag(actual: float, command: float, dt: float, tau: float) -> float:
    """One stable first-order-lag step toward ``command`` (alpha in (0, 1))."""
    if tau <= 0.0:
        return command
    alpha = dt / (tau + dt)
    return actual + (command - actual) * alpha


def predict_trajectory(
    params: WaypointFollowerParams,
    start: Pose2D,
    path: Sequence[Pose2D],
    dt: float,
    horizon_s: float,
    motion: Optional[MotionModelParams] = None,
    occupied_fn: Optional[OccupiedFn] = None,
) -> PredictionResult:
    """Roll the follower forward from ``start`` along ``path`` and report it.

    A throwaway :class:`WaypointFollower` is stepped (axis always confirmed, no
    hold) while a first-order-lag unicycle integrates its commands. Note this
    omits the platform's flight-mode handshake latency, so the wall-clock cost of
    each stop is an under-estimate.

    Args:
        params: The live follower tuning (so the prediction matches behaviour).
        start: Current pose to roll out from.
        path: World waypoints (>= 2).
        dt: Control period used for the rollout (s).
        horizon_s: Max simulated time (s); the rollout also stops at DONE.
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
    follower = WaypointFollower(params)
    follower.set_path(list(path), start)

    x, y, yaw = start.x, start.y, start.yaw
    vx_act = 0.0
    wz_act = 0.0
    poses: List[Pose2D] = [Pose2D(x, y, yaw)]
    total_yaw = 0.0
    n_stops = 0
    collides = bool(occupied_fn is not None and occupied_fn(x, y))
    prev_state = follower.state

    for _ in range(int(ceil(horizon_s / dt))):
        cmd = follower.step(Pose2D(x, y, yaw), dt, axis_confirmed=True)
        if follower.state in _PAUSE_STATES and prev_state not in _PAUSE_STATES:
            n_stops += 1
        prev_state = follower.state

        vx_act = _lag(vx_act, cmd.vx, dt, motion.vx_tau_s)
        wz_act = _lag(wz_act, cmd.wz, dt, motion.yaw_tau_s)
        new_yaw = normalize_angle(yaw + wz_act * dt)
        total_yaw += abs(normalize_angle(new_yaw - yaw))
        yaw = new_yaw
        x += vx_act * cos(yaw) * dt
        y += vx_act * sin(yaw) * dt
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


def prediction_score(result: PredictionResult, yaw_ref: float = pi) -> float:
    """Map a :class:`PredictionResult` to a 0..1 "how good" scalar (1 = best).

    A collision scores 0. Otherwise the score multiplies three soft factors:
    reaching the goal (penalised by the leftover gap), little total yaw, and few
    stops — so a clean straight run scores near 1 and a stop-and-spin path scores
    low. Intended for display, not control thresholds.
    """
    if result.collides:
        return 0.0
    reach = 1.0 if result.reaches_goal else max(0.0, 1.0 - result.end_gap)
    turn = 1.0 / (1.0 + result.total_yaw / max(yaw_ref, 1e-6))
    stop = 1.0 / (1.0 + 0.25 * result.n_stops)
    return max(0.0, min(1.0, reach * turn * stop))
