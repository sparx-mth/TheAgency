"""
Pure Pursuit tracker.

Implements the classic Pure Pursuit algorithm (Coulter 1992) with
modern extensions for adaptive lookahead and speed profiling.

Supports both holonomic (omnidirectional) and non-holonomic
(differential drive) robots.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from sparx_agency.core.common.types import (
    ControlCommand,
    KinematicLimits,
    State3D,
    Trajectory,
    TrajectoryPoint,
)
from sparx_agency.core.planning.interfaces.tracker import TrackerRequest, TrackerResult

from .algorithm import (
    adaptive_lookahead,
    compute_body_velocity,
    compute_pure_pursuit_curvature,
    compute_steering_commands,
    compute_target_speed,
    find_closest_index,
    find_lookahead_point,
    get_curvature,
    trajectory_to_xy,
)
from .params import PurePursuitParams


@dataclass
class _State:
    """Tracker memory."""
    progress_idx: int = 0
    speed_cmd: float = 0.0
    yaw_rate_cmd: float = 0.0


class PurePursuitTracker:
    """
    Pure Pursuit trajectory tracker.

    Uses the classic geometric formula: κ = 2·sin(α) / L_d

    For holonomic robots: outputs (vx, vy, vz, yaw_rate) in body frame.
    For non-holonomic: outputs (v, 0, vz, yaw_rate) where yaw_rate = v·κ.
    """

    name: str = "pure_pursuit"

    def __init__(
        self,
        params: Optional[PurePursuitParams] = None,
        default_limits: Optional[KinematicLimits] = None,
        clearance_fn: Optional[Callable[[float, float], float]] = None,
    ) -> None:
        self.params = params or PurePursuitParams()
        self._limits = default_limits or KinematicLimits()
        self._clearance_fn = clearance_fn
        self._s = _State()

    def reset(self) -> None:
        self._s = _State()

    def step(self, req: TrackerRequest) -> TrackerResult:
        p = self.params
        lim = req.limits or self._limits
        pose = req.state.pose

        # Sample trajectory
        pts: List[TrajectoryPoint] = req.trajectory.sample_by_time(p.sample_dt)
        if len(pts) < 2:
            return self._stop("trajectory_empty")

        xy = trajectory_to_xy(pts)
        pos = np.array([pose.x, pose.y])

        # Distance to goal
        dist_goal = float(np.linalg.norm(xy[-1] - pos))
        if dist_goal <= p.goal_tolerance:
            return self._stop("goal_reached", done=True, ref=pts[-1])

        # Find closest point (bounded search window)
        closest_idx, cte = find_closest_index(
            xy, pos,
            current_idx=self._s.progress_idx,
            search_back=p.closest_search_back,
            search_forward=p.closest_search_forward,
        )

        # Update progress (monotonic forward)
        if closest_idx > self._s.progress_idx:
            self._s.progress_idx = closest_idx

        # Check path divergence
        if cte > p.path_tolerance:
            return self._stop("path_diverged", failed=True)

        # Path curvature at current point
        path_curvature = get_curvature(pts[self._s.progress_idx])

        # Compute target speed (use params limits, not KinematicLimits)
        speed_target = compute_target_speed(
            cruise=p.cruise_speed,
            dist_to_goal=dist_goal,
            curvature=path_curvature,
            slow_down_dist=p.slow_down_distance,
            curvature_factor=p.curvature_speed_factor,
            bounds=(p.min_speed, p.max_speed),
        )

        # Apply clearance factor if available
        if self._clearance_fn is not None:
            try:
                clearance = self._clearance_fn(pose.x, pose.y)
                clearance_factor = float(np.clip(clearance, 0.3, 1.0))
                speed_target *= clearance_factor
            except Exception:
                pass

        # Smooth speed command
        self._s.speed_cmd += p.speed_smoothing * (speed_target - self._s.speed_cmd)

        # Adaptive lookahead
        lookahead_dist = adaptive_lookahead(
            base=p.base_lookahead,
            speed=self._s.speed_cmd,
            curvature=path_curvature,
            speed_gain=p.lookahead_speed_gain,
            curvature_gain=p.curvature_lookahead_factor,
            bounds=(p.min_lookahead, p.max_lookahead),
        )

        # Find lookahead point
        look_idx, target = find_lookahead_point(xy, pos, self._s.progress_idx, lookahead_dist)

        # Compute curvature
        curvature, alpha, actual_L_d = compute_pure_pursuit_curvature(pos, pose.yaw, target)

        # Convert to steering commands
        max_yaw_rate = min(p.max_yaw_rate, lim.max_yaw_rate)
        yaw_rate, steering_angle = compute_steering_commands(
            curvature=curvature,
            speed=self._s.speed_cmd,
            max_yaw_rate=max_yaw_rate,
            wheelbase=p.wheelbase,
        )

        # Smooth yaw rate
        self._s.yaw_rate_cmd += p.yaw_rate_smoothing * (yaw_rate - self._s.yaw_rate_cmd)

        # Compute body-frame velocity
        vx, vy = compute_body_velocity(
            speed=self._s.speed_cmd,
            alpha=alpha,
            holonomic=p.holonomic,
        )

        # Altitude control
        vz = 0.0
        if (alt := req.options.get("target_altitude")) is not None:
            vz = float(np.clip(
                p.altitude_kp * (alt - pose.z),
                -min(p.max_vertical_speed, lim.max_speed_z),
                min(p.max_vertical_speed, lim.max_speed_z),
            ))

        return TrackerResult(
            command=ControlCommand.velocity(vx, vy, vz, self._s.yaw_rate_cmd, tracker=self.name),
            reference=pts[look_idx],
            metadata={
                "done": False,
                "failed": False,
                "progress_idx": self._s.progress_idx,
                "cross_track_error": cte,
                "dist_to_goal": dist_goal,
                "lookahead_dist": lookahead_dist,
                "actual_lookahead": actual_L_d,
                "alpha": alpha,
                "curvature": curvature,
                "path_curvature": path_curvature,
                "speed_target": speed_target,
                "speed_cmd": self._s.speed_cmd,
                "yaw_rate_cmd": self._s.yaw_rate_cmd,
                "steering_angle": steering_angle,
            },
        )

    def _stop(
        self,
        reason: str,
        done: bool = False,
        failed: bool = False,
        ref: Optional[TrajectoryPoint] = None,
    ) -> TrackerResult:
        """Return a stop command (zero velocity)."""
        return TrackerResult(
            command=ControlCommand.velocity(0.0, 0.0, 0.0, 0.0, tracker=self.name, reason=reason),
            reference=ref,
            metadata={"done": done, "failed": failed, "reason": reason},
        )