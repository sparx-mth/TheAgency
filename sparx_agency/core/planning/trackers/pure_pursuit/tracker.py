"""
Pure Pursuit tracker (2D and 3D).

2D: Classic curvature-based steering (Coulter 1992)
3D: Direct velocity toward lookahead point (holonomic drones)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from sparx_agency.core.common.types import (
    ControlCommand,
    KinematicLimits,
    TrajectoryPoint,
)
from sparx_agency.core.planning.interfaces.tracker import (
    TrackerRequest,
    TrackerResult,
)

from . import algorithm as alg
from .params import PurePursuitParams, PurePursuitParams3D


# =============================================================================
# 2D Pure Pursuit Tracker
# =============================================================================

@dataclass
class _State2D:
    """2D tracker memory."""
    progress_idx: int = 0
    speed_cmd: float = 0.0
    yaw_rate_cmd: float = 0.0


class PurePursuitTracker:
    """
    Pure Pursuit trajectory tracker (2D).

    Uses κ = 2·sin(α) / L_d for steering.
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
        self._s = _State2D()

    def reset(self) -> None:
        self._s = _State2D()

    def step(self, req: TrackerRequest) -> TrackerResult:
        p = self.params
        lim = req.limits or self._limits
        pose = req.state.pose

        pts: List[TrajectoryPoint] = req.trajectory.sample_by_time(p.sample_dt)
        if len(pts) < 2:
            return self._stop("trajectory_empty")

        xy = alg.trajectory_to_xy(pts)
        pos = np.array([pose.x, pose.y])

        dist_goal = float(np.linalg.norm(xy[-1] - pos))
        if dist_goal <= p.goal_tolerance:
            return self._stop("goal_reached", done=True, ref=pts[-1])

        closest_idx, cte = alg.find_closest_index(
            xy, pos, self._s.progress_idx,
            p.closest_search_back, p.closest_search_forward
        )

        if closest_idx > self._s.progress_idx:
            self._s.progress_idx = closest_idx

        if cte > p.path_tolerance:
            return self._stop("path_diverged", failed=True, cross_track_error=cte,
                              dist_to_goal=dist_goal)

        path_curvature = alg.get_curvature(pts[self._s.progress_idx])

        speed_target = alg.compute_target_speed(
            p.cruise_speed, dist_goal, path_curvature,
            p.slow_down_distance, p.curvature_speed_factor,
            (p.min_speed, p.max_speed)
        )

        if self._clearance_fn is not None:
            try:
                clearance = self._clearance_fn(pose.x, pose.y)
                speed_target *= float(np.clip(clearance, 0.3, 1.0))
            except Exception:
                pass

        self._s.speed_cmd += p.speed_smoothing * (speed_target - self._s.speed_cmd)

        lookahead_dist = alg.adaptive_lookahead(
            p.base_lookahead, self._s.speed_cmd, path_curvature,
            p.lookahead_speed_gain, p.curvature_lookahead_factor,
            (p.min_lookahead, p.max_lookahead)
        )

        look_idx, target = alg.find_lookahead_point(xy, pos, self._s.progress_idx, lookahead_dist)
        curvature, alpha, actual_L_d = alg.compute_pure_pursuit_curvature(pos, pose.yaw, target)

        max_yaw_rate = min(p.max_yaw_rate, lim.max_yaw_rate)
        yaw_rate, steering_angle = alg.compute_steering_commands(
            curvature, self._s.speed_cmd, max_yaw_rate, p.wheelbase
        )

        self._s.yaw_rate_cmd += p.yaw_rate_smoothing * (yaw_rate - self._s.yaw_rate_cmd)

        vx, vy = alg.compute_body_velocity(self._s.speed_cmd, alpha, p.holonomic)

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
                "done": False, "failed": False,
                "progress_idx": self._s.progress_idx,
                "cross_track_error": cte,
                "dist_to_goal": dist_goal,
                "lookahead_dist": lookahead_dist,
                "curvature": curvature,
                "speed_cmd": self._s.speed_cmd,
                "yaw_rate_cmd": self._s.yaw_rate_cmd,
            },
        )

    def _stop(self, reason: str, done: bool = False, failed: bool = False,
              ref: Optional[TrajectoryPoint] = None,
              cross_track_error: float = 0.0,
              dist_to_goal: float = 0.0) -> TrackerResult:
        """Halt, saying why.

        ``cross_track_error`` is carried through because ``path_diverged`` is
        the one stop a caller has to explain afterwards, and it is the number
        that explains it. Omitting it left every diverged flight reporting
        "0.0 m off the planned route" — a caller reading the metadata cannot
        tell a genuine divergence from an unpopulated field.
        """
        return TrackerResult(
            command=ControlCommand.velocity(0, 0, 0, 0, tracker=self.name, reason=reason),
            reference=ref,
            metadata={"done": done, "failed": failed, "reason": reason,
                      "cross_track_error": float(cross_track_error),
                      "dist_to_goal": float(dist_to_goal)},
        )


# =============================================================================
# 3D Pure Pursuit Tracker (new)
# =============================================================================

@dataclass
class _State3D:
    """3D tracker memory."""
    progress_idx: int = 0
    speed_cmd: float = 0.0
    yaw_rate_cmd: float = 0.0


class PurePursuitTracker3D:
    """
    Pure Pursuit trajectory tracker (3D).

    For holonomic 3D robots (drones): velocity points directly
    toward the lookahead point in 3D space.
    """
    name: str = "pure_pursuit_3d"

    def __init__(
        self,
        params: Optional[PurePursuitParams3D] = None,
        default_limits: Optional[KinematicLimits] = None,
        clearance_fn: Optional[Callable[[float, float, float], float]] = None,
    ) -> None:
        self.params = params or PurePursuitParams3D()
        self._limits = default_limits or KinematicLimits()
        self._clearance_fn = clearance_fn
        self._s = _State3D()

    def reset(self) -> None:
        self._s = _State3D()

    def step(self, req: TrackerRequest) -> TrackerResult:
        p = self.params
        lim = req.limits or self._limits
        pose = req.state.pose

        pts: List[TrajectoryPoint] = req.trajectory.sample_by_time(p.sample_dt)
        if len(pts) < 2:
            return self._stop("trajectory_empty")

        xyz = alg.trajectory_to_xyz(pts)
        pos = np.array([pose.x, pose.y, pose.z])

        # 3D distance to goal
        dist_goal = float(np.linalg.norm(xyz[-1] - pos))
        if dist_goal <= p.goal_tolerance:
            return self._stop("goal_reached", done=True, ref=pts[-1])

        # Find closest point (3D)
        closest_idx, cte = alg.find_closest_index_3d(
            xyz, pos, self._s.progress_idx,
            p.closest_search_back, p.closest_search_forward
        )

        if closest_idx > self._s.progress_idx:
            self._s.progress_idx = closest_idx

        if cte > p.path_tolerance:
            return self._stop("path_diverged", failed=True, cross_track_error=cte,
                              dist_to_goal=dist_goal)

        # Use xy curvature for speed adaptation
        path_curvature = alg.get_curvature(pts[self._s.progress_idx])

        speed_target = alg.compute_target_speed(
            p.cruise_speed, dist_goal, path_curvature,
            p.slow_down_distance, p.curvature_speed_factor,
            (p.min_speed, p.max_speed)
        )

        # 3D clearance check
        if self._clearance_fn is not None:
            try:
                clearance = self._clearance_fn(pose.x, pose.y, pose.z)
                speed_target *= float(np.clip(clearance, 0.3, 1.0))
            except Exception:
                pass

        self._s.speed_cmd += p.speed_smoothing * (speed_target - self._s.speed_cmd)

        # Adaptive lookahead (based on xy curvature)
        lookahead_dist = alg.adaptive_lookahead(
            p.base_lookahead, self._s.speed_cmd, path_curvature,
            p.lookahead_speed_gain, p.curvature_lookahead_factor,
            (p.min_lookahead, p.max_lookahead)
        )

        # Find lookahead point (3D)
        look_idx, target = alg.find_lookahead_point_3d(xyz, pos, self._s.progress_idx, lookahead_dist)

        # Compute 3D velocity toward target
        max_speed_z = min(p.max_speed_z, lim.max_speed_z)
        vx, vy, vz, target_yaw = alg.compute_velocity_3d(
            pos, target, pose.yaw, self._s.speed_cmd, max_speed_z)

        # Compute yaw rate
        max_yaw_rate = min(p.max_yaw_rate, lim.max_yaw_rate)
        yaw_rate = alg.compute_yaw_rate_3d(pose.yaw, target_yaw, max_yaw_rate, p.sample_dt)

        self._s.yaw_rate_cmd += p.yaw_rate_smoothing * (yaw_rate - self._s.yaw_rate_cmd)

        return TrackerResult(
            command=ControlCommand.velocity(vx, vy, vz, self._s.yaw_rate_cmd, tracker=self.name),
            reference=pts[look_idx],
            metadata={
                "done": False, "failed": False,
                "progress_idx": self._s.progress_idx,
                "cross_track_error": cte,
                "dist_to_goal": dist_goal,
                "lookahead_dist": lookahead_dist,
                "target_yaw": target_yaw,
                "speed_cmd": self._s.speed_cmd,
                "yaw_rate_cmd": self._s.yaw_rate_cmd,
                "vz": vz,
            },
        )

    def _stop(self, reason: str, done: bool = False, failed: bool = False,
              ref: Optional[TrajectoryPoint] = None,
              cross_track_error: float = 0.0,
              dist_to_goal: float = 0.0) -> TrackerResult:
        """Halt, saying why.

        ``cross_track_error`` is carried through because ``path_diverged`` is
        the one stop a caller has to explain afterwards, and it is the number
        that explains it. Omitting it left every diverged flight reporting
        "0.0 m off the planned route" — a caller reading the metadata cannot
        tell a genuine divergence from an unpopulated field.
        """
        return TrackerResult(
            command=ControlCommand.velocity(0, 0, 0, 0, tracker=self.name, reason=reason),
            reference=ref,
            metadata={"done": done, "failed": failed, "reason": reason,
                      "cross_track_error": float(cross_track_error),
                      "dist_to_goal": float(dist_to_goal)},
        )