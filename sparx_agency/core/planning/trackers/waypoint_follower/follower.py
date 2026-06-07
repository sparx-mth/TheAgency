"""One-axis-at-a-time waypoint follower (ROS-free state machine).

This is a deliberately "stupid" path tracker for a platform that cannot
translate and rotate at the same time. Every command it produces is either a
pure forward advance (``vx``, ``wz = 0``) or a pure in-place rotation
(``wz``, ``vx = 0``) — never both. Lateral and vertical motion are always zero;
that invariant is the caller's responsibility to honour when actuating.

The follower owns ONLY the navigation phase: given the current 2D pose and a
2D path, decide the next planar command. Platform bring-up (takeoff, settle),
the control-axis handshake (rotating vs. advancing must be confirmed by the
flight controller) and all I/O live in the ROS adapter. The adapter tells the
follower, each tick, whether the axis it currently needs has been confirmed;
until then the follower holds zero and does not advance its state machine.

State machine::

    YAW_ALIGN  -> ADVANCE -> (BRAKE -> YAW_ALIGN -> ADVANCE)* -> DONE
"""
from __future__ import annotations

from math import atan2, copysign, hypot
from typing import List, Optional, Sequence

from sparx_agency.core.common.types import ControlCommand, Pose2D, normalize_angle

from . import algorithm as alg
from .params import WaypointFollowerParams
from .types import ControlAxis, FollowerCommand, FollowerState


class WaypointFollower:
    """Stateful planar path follower; pure X advance or pure yaw, never both."""

    name: str = "waypoint_follower"

    def __init__(self, params: Optional[WaypointFollowerParams] = None) -> None:
        self.params = params or WaypointFollowerParams()
        self.reset()

    # ─── Public API ──────────────────────────────────────────────
    def reset(self) -> None:
        """Clear path, state and slew memory."""
        self._path: List[alg.XY] = []
        self._wp_idx = 0
        self._state = FollowerState.YAW_ALIGN
        self._time_in_state = 0.0
        self._last_vx = 0.0
        self._last_wz = 0.0
        # Snapshots captured on state entry.
        self._yaw_align_lead = 0.0
        self._advance_yaw_at_entry = 0.0

    @property
    def state(self) -> FollowerState:
        return self._state

    @property
    def done(self) -> bool:
        return self._state == FollowerState.DONE

    def required_axis(self) -> Optional[ControlAxis]:
        """Axis the follower needs confirmed before it will command motion."""
        if self._state == FollowerState.YAW_ALIGN:
            return ControlAxis.YAW
        if self._state == FollowerState.ADVANCE:
            return ControlAxis.FORWARD
        return None

    def set_path(self, waypoints: Sequence[Pose2D], pose: Optional[Pose2D]) -> None:
        """Adopt a fresh path, dropping waypoints already passed.

        Re-anchors the path against ``pose`` then chooses the entry state based
        on whether the robot is currently moving and how far it must turn.
        """
        pts = [(float(p.x), float(p.y)) for p in waypoints]
        self._path = alg.reanchor_path(pts, pose, self.params.pos_radius)
        self._wp_idx = 0
        self._enter(self._entry_state(pose), pose)

    def step(
        self,
        pose: Pose2D,
        dt: float,
        *,
        axis_confirmed: bool = True,
        hold: bool = False,
    ) -> FollowerCommand:
        """Advance the state machine one tick and return the command.

        Args:
            pose: Current robot pose (x, y, yaw) in the path frame.
            dt: Seconds since the previous step (drives slew + timeouts).
            axis_confirmed: Whether :meth:`required_axis` has been confirmed by
                the platform. While False the follower holds zero and does not
                transition.
            hold: External request to suppress all motion (e.g. a startup hold).
                Same effect as an unconfirmed axis.
        """
        gating = hold or (self.required_axis() is not None and not axis_confirmed)
        if gating:
            return self._emit(self._finalize(0.0, 0.0, dt), freeze=None)

        self._time_in_state += dt
        if self._state == FollowerState.YAW_ALIGN:
            return self._step_yaw_align(pose, dt)
        if self._state == FollowerState.ADVANCE:
            return self._step_advance(pose, dt)
        if self._state == FollowerState.BRAKE:
            return self._step_brake(pose, dt)
        return self._emit(self._finalize(0.0, 0.0, dt), freeze=False)  # DONE

    # ─── State bodies ────────────────────────────────────────────
    def _step_yaw_align(self, pose: Pose2D, dt: float) -> FollowerCommand:
        p = self.params
        if not self._has_target():
            self._enter(FollowerState.ADVANCE, pose)
            return self._emit(self._finalize(0.0, 0.0, dt), freeze=False)

        eyaw = self._heading_error(pose)
        eyaw_lead = eyaw - copysign(self._yaw_align_lead, eyaw)
        if abs(eyaw_lead) <= p.yaw_settle:
            self._enter(FollowerState.ADVANCE, pose)
            return self._emit(self._finalize(0.0, 0.0, dt), freeze=False)

        wz_now = self._last_wz
        brake_d = alg.yaw_brake_distance(wz_now, p.yaw_accel_limit, dt)
        same_sign = (wz_now * eyaw_lead) >= 0.0
        if same_sign and abs(eyaw_lead) <= brake_d:
            wz_target = 0.0
        else:
            wz_target = copysign(p.yaw_rate, eyaw_lead)
        return self._emit(self._finalize(0.0, wz_target, dt), freeze=True)

    def _step_advance(self, pose: Pose2D, dt: float) -> FollowerCommand:
        p = self.params
        tx, ty = self._path[self._wp_idx]
        cx, cy = pose.x, pose.y
        d = hypot(tx - cx, ty - cy)
        bearing_err = normalize_angle(atan2(ty - cy, tx - cx) - pose.yaw)

        captured = d < p.pos_radius
        passed = abs(bearing_err) > p.passed_bearing_rad
        if captured or passed:
            self._wp_idx += 1
            if self._wp_idx >= len(self._path):
                self._enter(FollowerState.BRAKE, pose)
                return self._emit(self._finalize(0.0, 0.0, dt), freeze=False)
            ntx, nty = self._path[self._wp_idx]
            next_err = normalize_angle(atan2(nty - cy, ntx - cx) - pose.yaw)
            if abs(next_err) >= p.skip_yaw_thresh:
                # Sharp corner: brake and re-align toward the next waypoint.
                self._enter(FollowerState.BRAKE, pose)
                return self._emit(self._finalize(0.0, 0.0, dt), freeze=False)
            # Glide: keep advancing toward the next waypoint this same tick.

        actual_drift = normalize_angle(pose.yaw - self._advance_yaw_at_entry)
        if abs(actual_drift) > p.yaw_drift_thresh:
            self._enter(FollowerState.BRAKE, pose)
            return self._emit(self._finalize(0.0, 0.0, dt), freeze=False)

        return self._emit(self._finalize(p.vel_x, 0.0, dt), freeze=False)

    def _step_brake(self, pose: Pose2D, dt: float) -> FollowerCommand:
        p = self.params
        cmd = self._finalize(0.0, 0.0, dt)
        stopped = abs(self._last_vx) < p.vx_brake_thresh
        timed_out = self._time_in_state > p.brake_timeout_s
        if stopped or timed_out:
            nxt = (FollowerState.DONE if self._wp_idx >= len(self._path)
                   else FollowerState.YAW_ALIGN)
            self._enter(nxt, pose)
        return self._emit(cmd, freeze=False)

    # ─── Helpers ─────────────────────────────────────────────────
    def _has_target(self) -> bool:
        return bool(self._path) and self._wp_idx < len(self._path)

    def _heading_error(self, pose: Pose2D) -> float:
        tx, ty = self._path[self._wp_idx]
        yaw_des = atan2(ty - pose.y, tx - pose.x)
        return normalize_angle(yaw_des - pose.yaw)

    def _entry_state(self, pose: Optional[Pose2D]) -> FollowerState:
        """Pick the state to enter when a new path is adopted."""
        if not self._path or pose is None:
            return FollowerState.YAW_ALIGN
        tx, ty = self._path[0]
        if hypot(tx - pose.x, ty - pose.y) < 1e-3:
            return FollowerState.YAW_ALIGN
        bearing = atan2(ty - pose.y, tx - pose.x)
        aligned = abs(normalize_angle(bearing - pose.yaw)) < self.params.skip_yaw_thresh
        moving = abs(self._last_vx) > 0.05
        if moving and aligned:
            return FollowerState.ADVANCE
        return FollowerState.BRAKE if moving else FollowerState.YAW_ALIGN

    def _enter(self, new: FollowerState, pose: Optional[Pose2D]) -> None:
        # Forward-only mode never rotates in place: jump straight to ADVANCE.
        if new == FollowerState.YAW_ALIGN and self.params.forward_only:
            new = FollowerState.ADVANCE
        self._state = new
        self._time_in_state = 0.0
        if new == FollowerState.YAW_ALIGN and pose is not None and self._has_target():
            eyaw = self._heading_error(pose)
            self._yaw_align_lead = alg.yaw_lead_offset(eyaw, self.params.yaw_lead_pct)
        elif new == FollowerState.YAW_ALIGN:
            self._yaw_align_lead = 0.0
        if new == FollowerState.ADVANCE and pose is not None:
            self._advance_yaw_at_entry = pose.yaw

    def _finalize(self, vx: float, wz: float, dt: float):
        """Enforce the platform invariant, saturate and slew (vx=0 OR wz=0)."""
        p = self.params
        if abs(vx) > 1e-6 and abs(wz) > 1e-6:
            wz = 0.0  # invariant: never both axes at once
        vx = alg.saturate(vx, p.vel_xy_sat)
        wz = alg.saturate(wz, p.yaw_rate_sat)
        vx = alg.slew(vx, self._last_vx, p.accel_limit * dt)
        wz = alg.slew(wz, self._last_wz, p.yaw_accel_limit * dt)
        self._last_vx = vx
        self._last_wz = wz
        return ControlCommand.velocity(vx, 0.0, 0.0, wz, tracker=self.name)

    def _emit(self, command: ControlCommand, freeze: Optional[bool]) -> FollowerCommand:
        return FollowerCommand(
            command=command,
            state=self._state,
            required_axis=self.required_axis(),
            freeze=freeze,
            done=self._state == FollowerState.DONE,
            wp_idx=self._wp_idx,
            num_waypoints=len(self._path),
        )
