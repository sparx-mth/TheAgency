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

Yaw is a discrete *pulse -> settle -> re-measure* loop, not a continuous
controller: the platform turns slowly, coasts on yaw inertia, and its
localization jumps while rotating but is accurate when still. ``YAW_ALIGN``
sizes one open-loop burst from the last *settled* pose (ignoring the noisy live
pose); ``YAW_SETTLE`` coasts to a stop (sensors frozen) then dwells in place
(sensors live) so localization re-converges before the heading is re-measured.

State machine::

    YAW_ALIGN <-> YAW_SETTLE ... -> ADVANCE -> (BRAKE -> YAW_SETTLE ...)* -> DONE
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
        # YAW_ALIGN burst bookkeeping (one open-loop burst per YAW_ALIGN entry).
        self._burst_active = False
        self._burst_sign = 0.0
        self._burst_target = 0.0
        self._burst_swept = 0.0
        self._burst_ticks = 0
        # YAW_SETTLE dwell bookkeeping.
        self._settle_unfrozen_s = 0.0
        self._settle_yaws: List[float] = []
        # Last pose measured while settled; sizes the next burst.
        self._settled_pose: Optional[Pose2D] = None
        self._advance_yaw_at_entry = 0.0
        self._advance_ticks = 0          # forward ticks emitted this ADVANCE

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
        self._settled_pose = pose
        self._enter(self._entry_state(pose), pose)

    def step(
        self,
        pose: Pose2D,
        dt: float,
        *,
        axis_confirmed: bool = True,
        hold: bool = False,
        map_ready: bool = True,
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
            map_ready: Whether a fresh map/voxel update has been integrated since
                the current stop began. While False, YAW_SETTLE keeps dwelling
                (stopped, sensors live) and will NOT start the next rotation, so
                the map always reflects post-stop data before the drone moves on.
                The adapter supplies this (and a timeout); defaults True.
        """
        gating = hold or (self.required_axis() is not None and not axis_confirmed)
        if gating:
            return self._emit(self._finalize(0.0, 0.0, dt), freeze=None)

        self._time_in_state += dt
        if self._state == FollowerState.YAW_ALIGN:
            return self._step_yaw_align(pose, dt)
        if self._state == FollowerState.YAW_SETTLE:
            return self._step_yaw_settle(pose, dt, map_ready)
        if self._state == FollowerState.ADVANCE:
            return self._step_advance(pose, dt)
        if self._state == FollowerState.BRAKE:
            return self._step_brake(pose, dt)
        return self._emit(self._finalize(0.0, 0.0, dt), freeze=False)  # DONE

    # ─── State bodies ────────────────────────────────────────────
    def _step_yaw_align(self, pose: Pose2D, dt: float) -> FollowerCommand:
        if not self._has_target():
            self._enter(FollowerState.ADVANCE, pose)
            return self._emit(self._finalize(0.0, 0.0, dt), freeze=False)
        if not self._burst_active:
            return self._decide_yaw(pose, dt)
        return self._run_burst(dt)

    def _decide_yaw(self, pose: Pose2D, dt: float) -> FollowerCommand:
        """First YAW_ALIGN tick: advance if aligned enough (predictive gate, read
        from the last *settled* pose), else commit to one open-loop burst."""
        p = self.params
        meas = self._settled_pose or pose
        eyaw = self._heading_error(meas)
        tx, ty = self._path[self._wp_idx]
        dist = hypot(tx - meas.x, ty - meas.y)
        floor = alg.sweep_floor(p.yaw_rate, dt, p.min_motion_ticks)
        accept = alg.yaw_accept_floor(p.yaw_rate, dt, p.min_motion_ticks,
                                      p.yaw_coast_rad)
        if alg.advance_gate(eyaw, dist, p.yaw_capture_tol_m, accept,
                            p.yaw_acquire_max):
            self._enter(FollowerState.ADVANCE, pose)
            return self._emit(self._finalize(0.0, 0.0, dt), freeze=False)
        self._burst_active = True
        self._burst_sign = copysign(1.0, eyaw)
        self._burst_target = alg.burst_target_angle(
            eyaw, p.yaw_coast_rad, floor, p.yaw_burst_max_rad)
        self._burst_swept = 0.0
        self._burst_ticks = 0
        return self._run_burst(dt)

    def _run_burst(self, dt: float) -> FollowerCommand:
        """Execute one open-loop burst tick; ignores the noisy live pose.

        A burst always lasts at least ``min_motion_ticks`` ticks so it actually
        overcomes the yaw deadband (a lone tick does not turn the platform)."""
        p = self.params
        reached = (self._burst_swept >= self._burst_target
                   and self._burst_ticks >= p.min_motion_ticks)
        if reached or self._burst_ticks >= p.yaw_burst_max_ticks:
            self._enter(FollowerState.YAW_SETTLE, None)
            return self._emit(self._finalize(0.0, 0.0, dt), freeze=True)
        cmd = self._finalize(0.0, self._burst_sign * p.yaw_rate, dt)
        self._burst_swept += abs(self._last_wz) * dt
        self._burst_ticks += 1
        return self._emit(cmd, freeze=True)

    def _step_yaw_settle(self, pose: Pose2D, dt: float,
                         map_ready: bool) -> FollowerCommand:
        """Coast to a stop (frozen, dwell clock held), then dwell in place (live)
        collecting heading samples. Leaves only once the dwell has elapsed AND a
        fresh map update has landed (``map_ready``), so the map reflects
        post-stop data before the next rotation; hands a robust heading estimate
        back to YAW_ALIGN."""
        p = self.params
        cmd = self._finalize(0.0, 0.0, dt)  # keep slewing wz -> 0
        if abs(self._last_wz) >= p.yaw_settle_eps:
            self._settle_unfrozen_s = 0.0
            self._settle_yaws = []
            return self._emit(cmd, freeze=True)
        self._settle_unfrozen_s += dt
        self._settle_yaws.append(pose.yaw)
        if self._settle_unfrozen_s >= p.yaw_settle_dwell_s and map_ready:
            yaw = alg.circular_mean(self._settle_yaws)
            self._settled_pose = Pose2D(pose.x, pose.y, yaw)
            self._enter(FollowerState.YAW_ALIGN, self._settled_pose)
        return self._emit(cmd, freeze=False)

    def _step_advance(self, pose: Pose2D, dt: float) -> FollowerCommand:
        p = self.params
        # Minimum motion commitment: once advancing has started, keep advancing
        # for at least min_motion_ticks so the command overcomes the forward
        # deadband (a lone tick does not move the platform). The entry tick
        # (count 0) still runs the normal checks, so an already-captured waypoint
        # never triggers a pointless forward pulse.
        if 0 < self._advance_ticks < p.min_motion_ticks:
            return self._advance_forward(dt)

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

        return self._advance_forward(dt)

    def _advance_forward(self, dt: float) -> FollowerCommand:
        """Emit one forward tick, counting it toward the motion commitment."""
        self._advance_ticks += 1
        return self._emit(self._finalize(self.params.vel_x, 0.0, dt), freeze=False)

    def _step_brake(self, pose: Pose2D, dt: float) -> FollowerCommand:
        p = self.params
        cmd = self._finalize(0.0, 0.0, dt)
        stopped = abs(self._last_vx) < p.vx_brake_thresh
        timed_out = self._time_in_state > p.brake_timeout_s
        if stopped or timed_out:
            if self._wp_idx >= len(self._path):
                nxt = FollowerState.DONE
            elif p.forward_only:
                nxt = FollowerState.YAW_ALIGN   # _enter redirects to ADVANCE
            else:
                # Settle first so YAW_ALIGN measures a fresh, converged heading.
                nxt = FollowerState.YAW_SETTLE
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
        # If a path arrives mid-rotation, the live yaw is unreliable; settle
        # first (coast + dwell), then decide from a converged heading.
        if (not self.params.forward_only
                and abs(self._last_wz) > self.params.yaw_settle_eps):
            return FollowerState.YAW_SETTLE
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
        if self.params.forward_only and new in (
                FollowerState.YAW_ALIGN, FollowerState.YAW_SETTLE):
            new = FollowerState.ADVANCE
        self._state = new
        self._time_in_state = 0.0
        if new == FollowerState.YAW_ALIGN:
            self._burst_active = False   # re-measure and re-decide on entry
        elif new == FollowerState.YAW_SETTLE:
            self._settle_unfrozen_s = 0.0
            self._settle_yaws = []
        elif new == FollowerState.ADVANCE:
            self._advance_ticks = 0
            if pose is not None:
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
