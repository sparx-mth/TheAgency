"""Cross-track ROLL corrector: a stateful lateral-drift control law.

Given the current pose, the active path and *what the base follower is doing*
(advancing / turning / holding), this produces a small lateral (``+vy`` = left)
velocity that pulls the drone back onto its trajectory — and, while turning or
holding, an optional forward/back nudge for along-track drift. It commands
nothing about where to go; the waypoint follower owns that. See
:mod:`sparx_agency.core.planning.trackers.roll_assist_follower.params`.

The law is: project the pose onto the active trajectory segment, take the
body-frame offset to that closest point, scale it by a proportional gain (weaker
while turning/holding than while advancing), then saturate, slew-limit and
minimum-force-shape each axis so the published correction is either zero or a
command the platform can actually act on.
"""
from __future__ import annotations

from typing import Sequence, Tuple

from sparx_agency.core.common.types import Pose2D

from . import algorithm as alg
from .params import CrossTrackRollParams

XY = Tuple[float, float]


class CrossTrackRollCorrector:
    """Stateful cross-track ROLL (+ along-track) correction generator."""

    def __init__(self, params: CrossTrackRollParams = None) -> None:
        self.params = params or CrossTrackRollParams()
        self.reset()

    def reset(self) -> None:
        """Clear the slew memory (call on a fresh path)."""
        self._last_vy = 0.0
        self._last_vx = 0.0

    def relax(self, dt: float) -> Tuple[float, float]:
        """Decay both corrections toward zero (used while the base is held/done).

        Returns the shaped ``(vy, vx_extra)`` so the correction eases out smoothly
        rather than snapping to zero the instant the drone is asked to hold.
        """
        return self._shape_and_store(0.0, 0.0, dt)

    def correct(
        self,
        pose: Pose2D,
        path: Sequence[XY],
        wp_idx: int,
        *,
        advancing: bool,
        yaw_active: bool,
        dt: float,
    ) -> Tuple[float, float]:
        """Compute the ``(vy, vx_extra)`` correction for this tick.

        Args:
            pose: Current pose (x, y, yaw) in the path frame.
            path: The base follower's active (re-anchored) waypoints.
            wp_idx: Index of the waypoint the base is currently pursuing.
            advancing: True when the base is flying forward (ADVANCE).
            yaw_active: True when the base is rotating in place this tick.
            dt: Seconds since the previous call (drives slew shaping).

        Returns:
            ``(vy, vx_extra)`` — the lateral ROLL correction (``+`` = left) and the
            along-track forward/back correction to *add* to the base command.
        """
        p = self.params
        if len(path) < 2:
            return self.relax(dt)

        ax, ay, bx, by = alg.active_segment(path, wp_idx)
        qx, qy, _ = alg.project_point_on_segment(pose.x, pose.y, ax, ay, bx, by)
        e_fwd, e_lat = alg.body_offset_to_point(pose.x, pose.y, pose.yaw, qx, qy)

        if advancing:
            lat_frac, fwd_frac = p.advance_frac, 0.0
        elif yaw_active:
            lat_frac, fwd_frac = p.turn_frac, p.turn_fwd_frac
        else:
            lat_frac, fwd_frac = p.hold_frac, p.hold_fwd_frac

        vy_target = alg.saturate(
            p.kp_lat * alg.deadband(e_lat, p.deadband_m) * lat_frac,
            p.lateral_speed_max)
        vx_target = alg.saturate(
            p.kp_fwd * alg.deadband(e_fwd, p.forward_deadband_m) * fwd_frac,
            p.forward_speed_max)
        return self._shape_and_store(vy_target, vx_target, dt)

    def _shape_and_store(self, vy_target: float, vx_target: float,
                         dt: float) -> Tuple[float, float]:
        """Slew toward the targets (storing the continuous value) then shape.

        The slew memory keeps the un-shaped continuous signal so the ramp stays
        smooth; the returned command is minimum-force-shaped so it is always zero
        or a value the motors act on.
        """
        p = self.params
        step = p.accel_limit * dt
        vy = alg.slew(vy_target, self._last_vy, step)
        vx = alg.slew(vx_target, self._last_vx, step)
        self._last_vy, self._last_vx = vy, vx      # continuous (pre-shape) memory
        vy = alg.shape_axis(vy, p.min_vy, p.release_frac, p.cmd_zero_eps)
        vx = alg.shape_axis(vx, p.min_vx, p.release_frac, p.cmd_zero_eps)
        return vy, vx
