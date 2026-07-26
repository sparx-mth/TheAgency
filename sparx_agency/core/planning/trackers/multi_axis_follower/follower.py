"""Multi-axis waypoint follower (ROS-free state machine).

A continuous path tracker for a platform that *can* translate and rotate at the
same time. Every tick it produces a combined body-frame command — forward
(``vx``), lateral (``vy``) and yaw (``wz``) — that drives the drone toward the
active waypoint, holding altitude (``vz`` is always 0). It is the multi-axis
counterpart of the one-axis ``WaypointFollower`` and exposes the same public API
(``set_path`` / ``step`` / ``reset`` / ``state`` / ``done`` / ``required_axis``)
so an adapter can swap between them.

Design, given that localization/depth are noisiest while yawing and while
standing still, and cleanest while flying forward:

  * **Reach the waypoint by translating** directly toward it in the body frame —
    forward plus lateral. This always works and needs no rotation, so small
    offsets are absorbed by crabbing (ROLL), keeping yaw — the noisy axis — idle.
  * **Yaw only to face the travel direction**, and only past a deadband with
    hysteresis, so the drone turns for large heading errors but not small ones.
    When it does yaw, it keeps translating — never a stop-and-spin.
  * **Never fly fast into a blind direction**: a travel-cone clamp bounds how far
    off forward the translation may point, so the forward camera always roughly
    sees where the drone is going; yaw rotates the cone onto the target.
  * **Decisive minimum forces**: each axis command is either zero or at least the
    platform's minimum effective command — a sub-threshold command that the
    motors would ignore is never emitted.
  * **Station-keep, don't chase noise**: once the final goal is captured the drone
    holds position with a generous deadband and gentle, decisive nudges, since
    fighting the standstill noise only adds jitter.

State machine::

    IDLE --(set_path)--> RUN --(final waypoint captured)--> HOLD
                          ^                                   |
                          +--------(drifted far away)---------+
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from sparx_agency.core.common.types import ControlCommand, Pose2D

from ..waypoint_follower.algorithm import reanchor_path
from . import allocation as alloc
from .params import MultiAxisFollowerParams
from .types import MultiAxisCommand, MultiAxisState

XY = Tuple[float, float]


class MultiAxisFollower:
    """Stateful continuous path follower commanding vx + vy + yaw together."""

    name: str = "multi_axis_follower"

    def __init__(self, params: Optional[MultiAxisFollowerParams] = None) -> None:
        self.params = params or MultiAxisFollowerParams()
        self.reset()

    # ─── Public API ──────────────────────────────────────────────
    def reset(self) -> None:
        """Clear path, state, yaw latch and slew memory."""
        self._path: List[XY] = []
        self._wp_idx = 0
        self._state = MultiAxisState.IDLE
        self._yawing = False          # yaw hysteresis latch
        self._last_vx = 0.0
        self._last_vy = 0.0
        self._last_wz = 0.0

    @property
    def state(self) -> MultiAxisState:
        return self._state

    @property
    def done(self) -> bool:
        """True once the final goal is reached (station-keeping)."""
        return self._state == MultiAxisState.HOLD

    @property
    def active_path(self) -> List[Tuple[float, float]]:
        """The current re-anchored path as ``(x, y)`` tuples (read-only copy).

        Indices line up with ``FollowerCommand.wp_idx``, so a caller can resolve
        the waypoint being pursued to a position. Matches
        :attr:`WaypointFollower.active_path`; the two are used interchangeably by
        the adapter, which picks a follower by rosparam.
        """
        return list(self._path)

    def required_axis(self) -> Optional[str]:
        """The multi-axis tracker needs no per-axis handshake; always ``None``."""
        return None

    def set_path(self, waypoints: Sequence[Pose2D], pose: Optional[Pose2D]) -> None:
        """Adopt a fresh path, dropping waypoints already passed."""
        pts = [(float(p.x), float(p.y)) for p in waypoints]
        self._path = reanchor_path(pts, pose, self.params.pos_radius)
        self._wp_idx = 0
        self._yawing = False
        self._state = MultiAxisState.RUN if self._path else MultiAxisState.IDLE

    def step(
        self,
        pose: Pose2D,
        dt: float,
        *,
        axis_confirmed: bool = True,
        hold: bool = False,
        map_ready: bool = True,
    ) -> MultiAxisCommand:
        """Advance one tick and return the combined command.

        Args:
            pose: Current robot pose (x, y, yaw) in the path frame.
            dt: Seconds since the previous step (drives slew limiting).
            axis_confirmed: Whether the platform has confirmed it will accept the
                command; while False the follower holds zero. Defaults True.
            hold: External request to suppress all motion. Defaults False.
            map_ready: Accepted for interface parity with the one-axis follower;
                unused here (this tracker never stops to wait for a map update).
        """
        del map_ready  # interface parity only
        if hold or not axis_confirmed:
            return self._emit(self._finalize(0.0, 0.0, 0.0, dt))
        if self._state == MultiAxisState.IDLE or not self._path:
            return self._emit(self._finalize(0.0, 0.0, 0.0, dt))
        if self._state == MultiAxisState.RUN:
            return self._step_run(pose, dt)
        return self._step_hold(pose, dt)

    # ─── State bodies ────────────────────────────────────────────
    def _step_run(self, pose: Pose2D, dt: float) -> MultiAxisCommand:
        """Pursue the active waypoint with combined translation + yaw."""
        p = self.params
        if not self._advance_to_active(pose):
            self._state = MultiAxisState.HOLD
            self._yawing = False
            return self._step_hold(pose, dt)

        tx, ty = self._path[self._wp_idx]
        _, _, dist, eyaw = alloc.body_error(pose.x, pose.y, pose.yaw, tx, ty)
        is_final = self._wp_idx == len(self._path) - 1

        # Translation: toward the waypoint, ramped near the final goal, throttled
        # while grossly mis-pointed, clamped to the forward-facing travel cone.
        slow = p.slow_radius if is_final else p.pos_radius
        speed = alloc.approach_speed(dist, p.pos_radius, slow,
                                     p.cruise_speed, p.arrive_speed_min)
        speed *= alloc.alignment_gate(eyaw, p.travel_cone_rad,
                                      p.translate_suppress_rad,
                                      p.translate_suppress_floor)
        travel = alloc.clamp_travel_angle(eyaw, p.travel_cone_rad)
        vx, vy = alloc.allocate_translation(speed, travel, p.lateral_speed_max)

        # Yaw: engaged only past the deadband (hysteresis), then proportional.
        self._yawing = alloc.yaw_engaged(self._yawing, eyaw,
                                         p.yaw_engage_rad, p.yaw_release_rad)
        wz = alloc.yaw_setpoint(eyaw, self._yawing, p.yaw_kp, p.yaw_rate)

        return self._emit(self._finalize(vx, vy, wz, dt))

    def _step_hold(self, pose: Pose2D, dt: float) -> MultiAxisCommand:
        """Station-keep at the final goal with a decisive deadband."""
        p = self.params
        tx, ty = self._path[-1]
        _, _, dist, eyaw = alloc.body_error(pose.x, pose.y, pose.yaw, tx, ty)

        # Drifted well outside the capture radius: resume full pursuit. Re-aim at
        # the goal (last waypoint) — wp_idx was left past the end on capture, so
        # reset it, otherwise pursuit would immediately fall back through to HOLD.
        if dist > p.pos_radius + p.hold_reacquire_margin:
            self._state = MultiAxisState.RUN
            self._yawing = False
            self._wp_idx = len(self._path) - 1
            return self._step_run(pose, dt)

        # Inside the deadband: ride out the noise, command nothing.
        if dist <= p.hold_deadband:
            return self._emit(self._finalize(0.0, 0.0, 0.0, dt))

        # Outside the deadband: a gentle but decisive nudge back to the point.
        # Never yaw while station-keeping (yaw is the noisiest axis). The travel
        # direction is the raw bearing (NOT clamped to the cone): the cone is a
        # RUN-only constraint that relies on yaw to rotate it onto the target, so
        # with yaw disabled a goal that has drifted behind the nose must be reached
        # by crabbing straight back to it, not pushed away at the cone edge.
        speed = min(p.hold_kp * dist, p.hold_speed_max)
        vx, vy = alloc.allocate_translation(speed, eyaw, p.lateral_speed_max)
        return self._emit(self._finalize(vx, vy, 0.0, dt))

    # ─── Helpers ─────────────────────────────────────────────────
    def _advance_to_active(self, pose: Pose2D) -> bool:
        """Skip captured / passed waypoints; return False if the path is exhausted.

        A waypoint is captured within ``pos_radius`` or — for non-final waypoints
        only — passed once its bearing swings past ``passed_bearing_rad`` (it is
        now behind). The final goal is never skipped on bearing; the drone turns
        back to it.
        """
        p = self.params
        n = len(self._path)
        while self._wp_idx < n:
            tx, ty = self._path[self._wp_idx]
            _, _, dist, eyaw = alloc.body_error(pose.x, pose.y, pose.yaw, tx, ty)
            captured = dist < p.pos_radius
            passed = (abs(eyaw) > p.passed_bearing_rad
                      and self._wp_idx < n - 1)
            if captured or passed:
                self._wp_idx += 1
                continue
            return True
        return False

    def _finalize(self, vx: float, vy: float, wz: float, dt: float) -> ControlCommand:
        """Saturate, slew-limit, then minimum-force-shape the command (vz=0).

        Minimum-force shaping is the FINAL stage — applied AFTER the slew limiter —
        so the published command is always either zero or at least the platform's
        minimum effective command, even on the first ticks of an acceleration ramp
        (where ``accel_limit*dt`` could otherwise leave a sub-threshold dribble the
        motors ignore). The slew memory keeps the un-shaped continuous signal so
        the ramp itself stays smooth.
        """
        p = self.params
        vx, vy = alloc.saturate_translation(vx, vy, p.vel_xy_sat)
        wz = alloc.saturate(wz, p.yaw_rate_sat)
        lin_step = p.accel_limit * dt
        vx = alloc.slew(vx, self._last_vx, lin_step)
        vy = alloc.slew(vy, self._last_vy, lin_step)
        wz = alloc.slew(wz, self._last_wz, p.yaw_accel_limit * dt)
        self._last_vx, self._last_vy, self._last_wz = vx, vy, wz   # continuous (pre-shape)
        vx = alloc.shape_axis(vx, p.min_vx, p.release_frac, p.cmd_zero_eps)
        vy = alloc.shape_axis(vy, p.min_vy, p.release_frac, p.cmd_zero_eps)
        wz = alloc.shape_axis(wz, p.min_wz, p.release_frac, p.cmd_zero_eps)
        return ControlCommand.velocity(vx, vy, 0.0, wz, tracker=self.name)

    def _emit(self, command: ControlCommand) -> MultiAxisCommand:
        return MultiAxisCommand(
            command=command,
            state=self._state,
            required_axis=None,
            freeze=None,
            done=self._state == MultiAxisState.HOLD,
            wp_idx=self._wp_idx,
            num_waypoints=len(self._path),
        )
