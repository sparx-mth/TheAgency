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
pose); ``YAW_SETTLE`` coasts to a stop then dwells in place (sensors live) so
localization re-converges before the heading is re-measured.

Map freeze is angle-gated. A turn larger than ``freeze_yaw_thresh_rad`` (a real
rotation, where depth lies and localization lags) FREEZES the map for the whole
burst-and-coast, then makes YAW_SETTLE re-observe the scene ``settle_map_updates``
times while stopped before moving on. A small heading correction at or below the
threshold is executed with the sensors LIVE — the map keeps updating and no
stationary re-observation is forced — because freezing a few degrees is not worth
the stop. The decision is latched per alignment episode (from the first,
largest, error) and clears on reaching ADVANCE. The follower only *emits* the
desired freeze/re-observe intent (``FollowerCommand.freeze`` and
:attr:`WaypointFollower.settle_map_updates_required`); the ROS adapter maps them
onto the platform's ``turning`` demo-mode and the depth-fusion gate.

State machine::

    YAW_ALIGN <-> YAW_SETTLE ... -> ADVANCE -> (BRAKE -> YAW_SETTLE ...)* -> DONE
"""
from __future__ import annotations

from math import atan2, copysign, cos, hypot
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
        # Whether the CURRENT alignment episode freezes the map. Latched True by
        # _decide_yaw once any burst's error exceeds freeze_yaw_thresh_rad, and
        # cleared on reaching ADVANCE. Drives the per-burst / coast freeze flag and
        # settle_map_updates_required (small live corrections leave it False).
        self._episode_freeze = False
        # YAW_ALIGN burst bookkeeping (one burst per YAW_ALIGN entry).
        self._burst_active = False
        self._burst_sign = 0.0
        self._burst_target = 0.0
        self._burst_swept = 0.0
        self._burst_ticks = 0
        self._burst_planned_ticks = 0    # graded-mode tick budget for this burst
        self._fb_over = 0                # consecutive mid-burst "reached" ticks
        # Anti-deadlock: per-alignment-episode reversal bookkeeping.
        self._reversals = 0
        self._last_burst_sign = 0.0
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

    @property
    def active_path(self) -> List[alg.XY]:
        """The current re-anchored path as ``(x, y)`` tuples (read-only copy).

        Exposed so a wrapping layer (e.g. the roll-assist cross-track corrector)
        can measure the drone's offset from the exact trajectory the follower is
        flying, without reaching into private state.
        """
        return list(self._path)

    def required_axis(self) -> Optional[ControlAxis]:
        """Axis the follower needs confirmed before it will command motion."""
        if self._state == FollowerState.YAW_ALIGN:
            return ControlAxis.YAW
        if self._state == FollowerState.ADVANCE:
            return ControlAxis.FORWARD
        return None

    @property
    def settle_map_updates_required(self) -> int:
        """Fresh map/voxel updates the caller should confirm before this stop ends.

        Non-zero only inside a *frozen* turn's YAW_SETTLE: the map was frozen for
        the rotation, so the drone must re-observe the scene from the new, settled
        heading ``settle_map_updates`` times before moving on. Zero everywhere else
        (advancing, or a small live correction that never froze the map), so the
        adapter never forces a stationary re-observation it does not need. The
        adapter counts real map updates and feeds the answer back as ``map_ready``
        to :meth:`step`; see the waypoint_follower ROS node.
        """
        if self._state == FollowerState.YAW_SETTLE and self._episode_freeze:
            return int(self.params.settle_map_updates)
        return 0

    def set_path(self, waypoints: Sequence[Pose2D], pose: Optional[Pose2D]) -> None:
        """Adopt a fresh path, dropping waypoints already passed.

        Re-anchors the path against ``pose`` then chooses the entry state based
        on whether the robot is currently moving and how far it must turn.
        """
        pts = [(float(p.x), float(p.y)) for p in waypoints]
        self._path = alg.reanchor_path(pts, pose, self.params.pos_radius)
        self._wp_idx = 0
        self._settled_pose = pose
        self._reversals = 0           # a fresh path starts a new alignment episode
        self._last_burst_sign = 0.0
        # A re-plan from (near) standstill starts a clean alignment episode: let
        # _decide_yaw re-decide the freeze from the new heading error, so a small
        # correction after an interrupted big turn is not stuck in the frozen
        # ritual. A re-plan that arrives WHILE STILL PHYSICALLY TURNING keeps the
        # latch (the coast must stay frozen); _entry_state routes that case to
        # YAW_SETTLE via _last_wz > yaw_settle_eps, and a genuinely large residual
        # re-latches True on the first _decide_yaw tick.
        if abs(self._last_wz) <= self.params.yaw_settle_eps:
            self._episode_freeze = False
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
        return self._run_burst(pose, dt)

    def _decide_yaw(self, pose: Pose2D, dt: float) -> FollowerCommand:
        """First YAW_ALIGN tick: advance if aligned enough (predictive gate, read
        from the last *settled* pose), else commit to one burst.

        Anti-deadlock (when enabled): a burst whose direction reverses the last
        one increments a per-episode counter that widens the accept band and, at
        ``yaw_max_reversals``, FORCES ADVANCE instead of firing the opposing
        burst — so the machine can never ping-pong forever (the classic
        10°R→10°L→10°R). The 10° case never even bursts: it is already inside the
        un-improvable accept floor, so it advances on the first tick.
        """
        p = self.params
        meas = self._settled_pose or pose
        eyaw = self._heading_error(meas)
        tx, ty = self._path[self._wp_idx]
        dist = hypot(tx - meas.x, ty - meas.y)
        floor = alg.sweep_floor(p.yaw_rate, dt, p.min_motion_ticks)
        base_accept = alg.yaw_accept_floor(p.yaw_rate, dt, p.min_motion_ticks,
                                           p.yaw_coast_rad)
        accept, locked = alg.accept_with_reversals(
            base_accept, self._reversals, p.yaw_accept_growth_rad, p.yaw_max_reversals)
        if ((locked and cos(eyaw) > 0.0)
                or alg.advance_gate(eyaw, dist, p.yaw_capture_tol_m, accept,
                                    p.yaw_acquire_max)):
            self._enter(FollowerState.ADVANCE, pose)
            return self._emit(self._finalize(0.0, 0.0, dt), freeze=False)
        sign = copysign(1.0, eyaw)
        if self._last_burst_sign != 0.0 and sign != self._last_burst_sign:
            self._reversals += 1                      # count the direction flip ...
            accept, locked = alg.accept_with_reversals(
                base_accept, self._reversals, p.yaw_accept_growth_rad,
                p.yaw_max_reversals)
            if locked and cos(eyaw) > 0.0:            # ... and never fire the burst
                self._enter(FollowerState.ADVANCE, pose)
                return self._emit(self._finalize(0.0, 0.0, dt), freeze=False)
        self._last_burst_sign = sign
        # Latch the freeze decision for this episode from the (largest) error that
        # first commits a burst: a turn past freeze_yaw_thresh_rad freezes the map
        # and forces the post-turn re-observation; a gentle correction stays live.
        # Once latched True it stays frozen for the rest of the episode (so the
        # small final burst of a big turn is still frozen), clearing at ADVANCE.
        self._episode_freeze = (p.freeze_on_rotation
                                and (self._episode_freeze
                                     or abs(eyaw) > p.freeze_yaw_thresh_rad))
        self._burst_sign = sign
        self._burst_active = True
        if p.yaw_graded_pulses:
            self._burst_planned_ticks = alg.burst_tick_count(
                eyaw, p.yaw_coast_rad, p.yaw_rate * dt, p.min_motion_ticks,
                p.yaw_burst_grade_max_ticks)
        else:
            self._burst_planned_ticks = 0
            self._burst_target = alg.burst_target_angle(
                eyaw, p.yaw_coast_rad, floor, p.yaw_burst_max_rad)
        self._burst_swept = 0.0
        self._burst_ticks = 0
        self._fb_over = 0
        return self._run_burst(pose, dt)

    def _run_burst(self, pose: Pose2D, dt: float) -> FollowerCommand:
        """Execute one burst tick.

        Graded mode sizes the burst by a tick budget (``burst_tick_count`` snaps
        to 2/4/6 ticks, cap 6) instead of a swept angle; legacy mode keeps the
        swept-angle test. A burst always lasts at least ``min_motion_ticks`` ticks
        so it overcomes the yaw deadband. With ``yaw_burst_live_feedback`` the
        (now-denoised, estimator-fed) live pose can CUT the burst short — but only
        one-way (stop early on a confirmed overshoot), never reverse in flight, so
        it cannot seed a ping-pong; the clean YAW_SETTLE re-measure decides the
        next pulse.
        """
        p = self.params
        if p.yaw_graded_pulses:
            reached = (self._burst_ticks >= self._burst_planned_ticks
                       and self._burst_ticks >= p.min_motion_ticks)
            capped = self._burst_ticks >= p.yaw_burst_grade_max_ticks
        else:
            reached = (self._burst_swept >= self._burst_target
                       and self._burst_ticks >= p.min_motion_ticks)
            capped = self._burst_ticks >= p.yaw_burst_max_ticks
        if reached or capped:
            self._enter(FollowerState.YAW_SETTLE, None)
            return self._emit(self._finalize(0.0, 0.0, dt),
                              freeze=self._episode_freeze)
        # Mid-burst one-way CUT: only after the deadband-clearing min ticks, and
        # only on yaw_fb_confirm_ticks consecutive "reached/overshot" readings so a
        # single noisy frame can't trigger it. Never reverses here.
        if p.yaw_burst_live_feedback and self._burst_ticks >= p.min_motion_ticks:
            remaining = self._burst_sign * self._heading_error(pose)
            self._fb_over = self._fb_over + 1 if remaining <= p.yaw_fb_reach_rad else 0
            if self._fb_over >= p.yaw_fb_confirm_ticks:
                self._enter(FollowerState.YAW_SETTLE, None)
                return self._emit(self._finalize(0.0, 0.0, dt),
                                  freeze=self._episode_freeze)
        cmd = self._finalize(0.0, self._burst_sign * p.yaw_rate, dt)
        self._burst_swept += abs(self._last_wz) * dt
        self._burst_ticks += 1
        return self._emit(cmd, freeze=self._episode_freeze)

    def _step_yaw_settle(self, pose: Pose2D, dt: float,
                         map_ready: bool) -> FollowerCommand:
        """Coast to a stop (map held at the episode's freeze while wz slews down),
        then dwell in place (live) collecting heading samples. Leaves only once the
        dwell has elapsed AND ``map_ready`` (the adapter's answer to
        :attr:`settle_map_updates_required`, so a frozen turn re-observes the scene
        the required number of times before moving on); hands a robust heading
        estimate back to YAW_ALIGN."""
        p = self.params
        cmd = self._finalize(0.0, 0.0, dt)  # keep slewing wz -> 0
        if abs(self._last_wz) >= p.yaw_settle_eps:
            # Still physically rotating (inertial coast): hold the episode's freeze
            # so a frozen turn keeps the map frozen through the coast, not just the
            # commanded burst -- the end-of-turn inertia frames are the worst.
            self._settle_unfrozen_s = 0.0
            self._settle_yaws = []
            return self._emit(cmd, freeze=self._episode_freeze)
        self._settle_unfrozen_s += dt
        self._settle_yaws.append(pose.yaw)
        # Inertia-proportional dwell: a longer burst built more momentum, so it
        # dwells longer before re-measuring (graded mode only; else fixed).
        dwell = (alg.settle_dwell(p.yaw_settle_dwell_s, p.yaw_settle_dwell_per_tick,
                                  self._burst_ticks, p.yaw_burst_grade_max_ticks)
                 if p.yaw_graded_pulses else p.yaw_settle_dwell_s)
        if self._settle_unfrozen_s >= dwell and map_ready:
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
            # Re-anchor against the WHOLE route rather than stepping one index.
            # The advance gate is predictive -- it starts flying as soon as going
            # straight would pass within yaw_capture_tol_m -- so a waypoint is
            # routinely retired without the drone ever entering pos_radius of it,
            # and on a tight corner the drone can already be level with, or past,
            # the one after it. Stepping by one would then hand it a waypoint
            # BEHIND it: next_err comes out huge, this brakes, and the drone turns
            # around and flies back for a point it has already been to. Projecting
            # onto the route instead picks the first waypoint genuinely ahead.
            self._wp_idx = alg.live_waypoint_index(self._path, pose, p.pos_radius,
                                                   self._wp_idx + 1)
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
            self._reversals = 0          # reaching alignment ends the episode
            self._last_burst_sign = 0.0
            self._episode_freeze = False  # next alignment re-decides freeze afresh
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
