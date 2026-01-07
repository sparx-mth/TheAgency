from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.common.types.control import ControlCommand, KinematicLimits
from core.common.types.motion import State3D
from core.common.types.planning import Trajectory, TrajectoryPoint

from core.planning.interfaces.tracker import BaseTracker, TrackerRequest, TrackerResult
from core.planning.trackers.pure_pursuit.algorithm import (
    clearance_to_factor,
    compute_lookahead,
    compute_speed,
    compute_yaw_rate,
    distance_to_goal,
    find_closest_index,
    get_curvature,
    pick_lookahead_index,
    world_to_body_velocity,
)
from core.planning.trackers.pure_pursuit.params import PurePursuitParams


def _get_limits(req: TrackerRequest, fallback: KinematicLimits) -> KinematicLimits:
    return req.limits if req.limits is not None else fallback


def _sample_trajectory(traj: Trajectory, sample_dt: float) -> List[TrajectoryPoint]:
    if sample_dt <= 0.0:
        raise ValueError(f"sample_dt must be > 0, got {sample_dt}")
    pts = traj.sample_by_time(sample_dt)
    if len(pts) < 2:
        raise ValueError("Trajectory.sample_by_time() must return at least 2 points")
    return pts


@dataclass
class _InternalState:
    """
    Internal tracker memory.

    - progress_index: monotonically increasing index into sampled points
    - speed_cmd: filtered speed command
    - yaw_rate_cmd: filtered yaw rate command
    """
    progress_index: int = 0
    speed_cmd: float = 0.0
    yaw_rate_cmd: float = 0.0


class PurePursuitTracker(BaseTracker):
    """
    Pure Pursuit tracker implementing the project's BaseTracker interface.

    Inputs come via TrackerRequest:
    - state: State3D
    - trajectory: Trajectory (time-param)
    - t: time since trajectory start (seconds)

    Output:
    - ControlCommand.velocity() in BODY frame (vx, vy, vz, yaw_rate)
    """

    name: str = "pure_pursuit"

    def __init__(
        self,
        params: Optional[PurePursuitParams] = None,
        *,
        default_limits: Optional[KinematicLimits] = None,
        clearance_callback: Optional[Any] = None,
        min_clearance_for_full_speed: float = 1.0,
        min_clearance_threshold: float = 0.15,
    ) -> None:
        self.params = params or PurePursuitParams()
        self._default_limits = default_limits or KinematicLimits()

        # Optional clearance callback: (x, y) -> clearance_m
        self._clearance_cb = clearance_callback
        self._min_clearance_for_full_speed = float(min_clearance_for_full_speed)
        self._min_clearance_threshold = float(min_clearance_threshold)

        self._st = _InternalState()

    def reset(self) -> None:
        self._st = _InternalState()

    # ---------------------------------------------------------------------
    # Main step
    # ---------------------------------------------------------------------

    def step(self, request: TrackerRequest) -> TrackerResult:
        p = self.params
        limits = _get_limits(request, self._default_limits)

        state: State3D = request.state
        traj: Trajectory = request.trajectory

        # NOTE:
        # Pure Pursuit is spatial; our Trajectory is time-based.
        # We adapt by resampling the Trajectory into discrete points once per step.
        pts = _sample_trajectory(traj, p.sample_dt)

        x = float(state.pose.x)
        y = float(state.pose.y)
        z = float(state.pose.z)
        yaw = float(state.pose.yaw)

        # Goal check
        dist_goal = distance_to_goal(pts, x, y)
        if dist_goal <= p.goal_tolerance:
            cmd = ControlCommand.velocity(0.0, 0.0, 0.0, 0.0, reason="goal_reached", tracker=self.name)
            return TrackerResult(
                command=cmd,
                reference=pts[-1],
                metadata={"done": True, "reason": "goal_reached", "dist_to_goal": dist_goal},
            )

        # Closest-point search around progress index (monotonic)
        n = len(pts)
        self._st.progress_index = max(0, min(self._st.progress_index, n - 1))

        closest_i, cte = find_closest_index(
            pts,
            x,
            y,
            current_index=self._st.progress_index,
            search_back=p.closest_search_back,
            search_forward=p.closest_search_forward,
        )

        if closest_i > self._st.progress_index:
            self._st.progress_index = closest_i

        # Divergence check
        if cte > p.path_tolerance:
            cmd = ControlCommand.velocity(0.0, 0.0, 0.0, 0.0, reason="path_diverged", tracker=self.name)
            return TrackerResult(
                command=cmd,
                reference=pts[self._st.progress_index],
                metadata={
                    "done": False,
                    "failed": True,
                    "reason": "path_diverged",
                    "cross_track_error": cte,
                },
            )

        # Curvature at current progress point (if provided by smoother)
        curvature = get_curvature(pts[self._st.progress_index])

        # Clearance factor (optional)
        clearance_factor = 1.0
        if self._clearance_cb is not None:
            try:
                clearance_m = float(self._clearance_cb(x, y))
                clearance_factor = clearance_to_factor(
                    clearance_m,
                    min_clearance_for_full_speed=self._min_clearance_for_full_speed,
                    min_clearance_threshold=self._min_clearance_threshold,
                )
            except Exception:
                clearance_factor = 1.0

        # Speed target + smoothing
        speed_target = compute_speed(
            cruise_speed=p.cruise_speed,
            min_speed=p.min_speed,
            max_speed=min(p.max_speed, limits.max_speed_xy),
            slow_down_distance=p.slow_down_distance,
            curvature_speed_factor=p.curvature_speed_factor,
            dist_to_goal=dist_goal,
            curvature=curvature,
            clearance_factor=clearance_factor,
        )

        alpha = max(0.0, min(1.0, p.speed_smoothing_alpha))
        self._st.speed_cmd = alpha * speed_target + (1.0 - alpha) * self._st.speed_cmd

        # Lookahead
        lookahead_m = compute_lookahead(
            base_lookahead=p.base_lookahead,
            min_lookahead=p.min_lookahead,
            max_lookahead=p.max_lookahead,
            lookahead_speed_gain=p.lookahead_speed_gain,
            current_speed=self._st.speed_cmd,
            curvature=curvature,
        )

        look_i = pick_lookahead_index(pts, closest_index=self._st.progress_index, lookahead_m=lookahead_m)
        ref = pts[look_i]

        dx = float(ref.x) - x
        dy = float(ref.y) - y
        dist = math.hypot(dx, dy)

        if dist > 1e-6:
            vx_world = (dx / dist) * self._st.speed_cmd
            vy_world = (dy / dist) * self._st.speed_cmd
            desired_yaw = math.atan2(dy, dx)
        else:
            vx_world, vy_world = 0.0, 0.0
            desired_yaw = yaw

        # Yaw-rate smoothing + clamp by limits
        self._st.yaw_rate_cmd = compute_yaw_rate(
            current_yaw=yaw,
            desired_yaw=desired_yaw,
            current_speed=self._st.speed_cmd,
            yaw_kp=p.yaw_kp,
            max_yaw_rate=min(p.max_yaw_rate, limits.max_yaw_rate),
            yaw_deadband=p.yaw_deadband,
            yaw_speed_threshold=p.yaw_speed_threshold,
            yaw_rate_smoothing=p.yaw_rate_smoothing,
            prev_yaw_rate=self._st.yaw_rate_cmd,
        )

        # World -> body
        vx_body, vy_body = world_to_body_velocity(vx_world, vy_world, yaw)

        # Vertical control (optional via params option; default: use State3D target via options)
        target_altitude = request.options.get("target_altitude", None)
        vz = 0.0
        if target_altitude is not None:
            ez = float(target_altitude) - z
            vz = p.altitude_kp * ez
            vz = max(-p.max_vertical_speed, min(vz, p.max_vertical_speed))
            vz = max(-limits.max_speed_z, min(vz, limits.max_speed_z))

        # Build command
        cmd = ControlCommand.velocity(
            vx_body,
            vy_body,
            vz,
            self._st.yaw_rate_cmd,
            tracker=self.name,
        )

        return TrackerResult(
            command=cmd,
            reference=ref,
            metadata={
                "done": False,
                "failed": False,
                "reason": "ok",
                "progress_index": self._st.progress_index,
                "cross_track_error": cte,
                "dist_to_goal": dist_goal,
                "lookahead_m": lookahead_m,
                "speed_target": speed_target,
                "speed_cmd": self._st.speed_cmd,
                "curvature": curvature,
                "desired_yaw": desired_yaw,
                "clearance_factor": clearance_factor,
            },
        )
