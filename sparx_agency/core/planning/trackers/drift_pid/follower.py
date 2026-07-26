"""The drift-PID path follower: fly the leg, hold the line, get unstuck.

Two references, switched by regime, because the drift the drone suffers is not
the same drift in each:

  * **TRACK** — flying a leg. The reference is the *line*. A feed-forward cruise
    speed drives the forward axis; the cross-track PID owns the lateral axis and
    the heading PID owns yaw. This is where the sideways drift (the big one on
    this airframe) is cancelled: its integral learns the standing lateral push
    and feeds it forward, so the drone stops settling at an offset from the line.
  * **TURN / HOLD** — rotating in place, waiting, or parked on the goal. The
    reference is a fixed *anchor*, latched onto the trajectory rather than onto
    wherever the drone happened to be, so a turn also pulls it back on track.
    Both translation PIDs run against that anchor, which is what cancels the
    fore/aft drift that only shows up when the drone is not flying forward.
    The anchor pull is *coordinated with the rotation*: while the yaw axis is
    active the translation vector is confined to a forward cone (measured on
    this airframe, a backward pitch under a turn degrades and then INVERTS the
    delivered yaw), so a correction that needs reverse waits its turn.

On top of that sit two things the platform forces on any honest controller here:
localization that degrades gracefully rather than failing cleanly (see
:mod:`.confidence`), and walls the camera cannot see (see :mod:`.blockage` and
:mod:`.escape`).

What this module does NOT do, on purpose: choose a route, remember an obstacle,
or decide to go somewhere else. When its reflexes are spent it raises
``report_blocked`` once and keeps holding the line it was given. Routing is the
planner's job and stays there.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from sparx_agency.core.common.types import (
    ControlCommand,
    Pose2D,
    normalize_angle,
)
from sparx_agency.core.planning.trackers.multi_axis_follower.allocation import (
    alignment_gate,
    approach_speed,
    saturate,
    turn_coordination,
    yaw_engaged,
)

from math import cos, sin

from . import geometry as geo
from .blockage import BlockageMonitor
from .confidence import ConfidenceScheduler, LocalizationQuality
from .envelope import ForceEnvelope
from .escape import EscapeManeuver
from .params import DriftPidParams
from .pid import AxisPid
from .types import DriftPidCommand, DriftPidState, DriftTelemetry

XY = Tuple[float, float]


class DriftPidFollower:
    """Stateful continuous path follower with per-axis drift cancellation."""

    name = "drift_pid"

    def __init__(self, params=None):
        # type: (Optional[DriftPidParams]) -> None
        p = params or DriftPidParams()
        self.params = p
        self._lat = AxisPid(p.lateral_pid, leak_tau_s=p.drift_leak_s)
        self._fwd = AxisPid(p.forward_pid, leak_tau_s=p.drift_leak_s)
        self._yaw = AxisPid(p.yaw_pid, leak_tau_s=p.drift_leak_s)
        self._envelope = ForceEnvelope(p.envelope)
        self._scheduler = ConfidenceScheduler(p.confidence)
        self._blockage = BlockageMonitor(p.blockage)
        self._escape = EscapeManeuver(p.escape)
        self.reset()

    # ─── Lifecycle ───────────────────────────────────────────────
    def reset(self):
        # type: () -> None
        """Return to IDLE and clear every loop, including the learned drift."""
        self._path = []            # type: List[XY]
        self._idx = 0
        self._state = DriftPidState.IDLE
        self._done = False
        self._turning = False
        self._anchor = None        # type: Optional[XY]
        self._anchor_yaw = 0.0
        self._quality = LocalizationQuality()
        self._last = (0.0, 0.0, 0.0)
        self._reported = False
        self._prev_age = None      # type: Optional[float]
        self._yield_scale = 1.0
        for pid in (self._lat, self._fwd, self._yaw):
            pid.reset()
        self._envelope.reset()
        self._blockage.reset()
        self._escape.reset()

    @property
    def state(self):
        # type: () -> str
        """State the controller is in."""
        return self._state

    @property
    def done(self):
        # type: () -> bool
        """True once the final waypoint has been captured."""
        return self._done

    @property
    def active_path(self):
        # type: () -> List[XY]
        """The re-anchored waypoints currently being flown."""
        return list(self._path)

    @property
    def settle_map_updates_required(self):
        # type: () -> int
        """Fresh voxel updates the adapter should wait for while stopped."""
        return self.params.settle_map_updates if self._turning else 0

    def required_axis(self):
        # type: () -> Optional[str]
        """Always None: this controller drives every axis and needs no handshake."""
        return None

    def set_quality(self, quality):
        # type: (LocalizationQuality) -> None
        """Feed the latest localization quality snapshot (call before ``step``)."""
        self._quality = quality

    def set_path(self, waypoints, pose=None):
        # type: (Sequence[Pose2D], Optional[Pose2D]) -> None
        """Adopt a new route, re-anchored to where the drone actually is.

        The derivative memory of every loop is dropped (the setpoint just jumped,
        and differentiating that step would kick the axes) but the **integrals are
        kept**: the drift the controller learned belongs to the airframe and the
        room, and it did not change because the planner did.
        """
        self._path = [(float(w.x), float(w.y)) for w in waypoints]
        self._idx = 0
        self._done = False
        self._turning = False
        self._anchor = None
        for pid in (self._lat, self._fwd, self._yaw):
            pid.reset_derivative()
        if pose is not None and len(self._path) >= 2:
            self._retire(pose)

    # ─── Control ─────────────────────────────────────────────────
    def step(self, pose, dt, axis_confirmed=True, hold=False, map_ready=True,
             translation_scale=1.0):
        # type: (Pose2D, float, bool, bool, bool, float) -> DriftPidCommand
        """Advance one control tick.

        Args:
            pose: Current pose in the path frame.
            dt: Seconds since the previous call.
            axis_confirmed: Ignored (kept for interface parity: this controller
                needs no per-axis flight-mode handshake).
            hold: External stop request from the adapter.
            map_ready: False means the adapter is waiting for a fresh voxel
                update; treated exactly like ``hold``.
            translation_scale: Fraction of the horizontal translation an outside
                authority (the altitude hold, during a climb pulse) leaves to
                this controller this tick (0..1). Folded in BEFORE the envelope
                so slew, minimum-force shaping, effort, the blockage monitor and
                the published telemetry all see the command that actually
                flies -- a yield applied after the fact would leave the
                controller believing it commanded speed the drone never got.
                Yaw is untouched: rotating costs no lift.

        Returns:
            The command to publish, plus what the controller has learned.
        """
        del axis_confirmed
        if dt <= 0.0:
            raise ValueError("DriftPidFollower.step: dt must be > 0")
        if not 0.0 <= translation_scale <= 1.0:
            raise ValueError("DriftPidFollower.step: translation_scale must be "
                             "in [0, 1], got %r" % (translation_scale,))
        self._yield_scale = float(translation_scale)
        auth = self._scheduler.evaluate(self._quality)
        fresh = self._measurement_fresh()
        if len(self._path) < 2:
            return self._emit(0.0, 0.0, 0.0, DriftPidState.IDLE, dt, auth)

        # The blockage detector sees the RAW pose, always. The latency-led pose
        # below is advanced by the drone's own commands — feed that to a stuck
        # detector and a pinned drone would appear to obey every command.
        last_vx, _, last_wz = self._last
        blocked = self._blockage.update(pose, last_vx, last_wz, dt, self._quality)
        frozen = bool(hold) or not map_ready or auth.hold

        if not blocked.blocked and not self._escape.active:
            # Moving again: give the next obstacle a full set of attempts.
            self._escape.episode_over()
            self._reported = False

        escaped = self._run_escape(pose, blocked, frozen, dt, auth)
        if escaped is not None:
            return escaped

        report = False
        if (blocked.blocked and self._escape.exhausted and not self._reported
                and not frozen):
            # The reflexes have had their turn and the drone is still stuck AND it
            # is actually trying to fly. A frozen drone (held by the adapter, lost
            # localization, GO not given, or waiting on a map update) was told to
            # stop -- it is not "blocked", and reporting a blockage from a spot it
            # was never allowed to fly through only teaches the planner a phantom
            # obstacle. Say it once, when it is real; the planner owns the reroute.
            self._reported = True
            report = True

        if frozen:
            return self._hold_still(pose, dt, auth, blocked, report)

        steer = self._lead_pose(pose, auth.lead_s)
        self._retire(steer)
        return self._navigate(steer, dt, auth, blocked, report, fresh)

    # ─── Measurement bookkeeping ─────────────────────────────────
    def _measurement_fresh(self):
        # type: () -> bool
        """True when a new localization frame arrived since the previous tick.

        The pose stream is event-driven (~10 Hz with gaps) while the control loop
        is a timer, so some ticks re-see the held previous pose. Detected from
        the quality snapshot's age: a new frame RESETS the age, so an age that
        did not grow by a control period means a new measurement landed. Only the
        derivative terms consume this — differentiating a held value produces a
        zero then a spike, which is exactly the noise the D low-pass should not
        have to eat.
        """
        age = self._quality.age_s
        prev = self._prev_age
        self._prev_age = age
        if prev is None:
            return True
        return age <= prev

    def _lead_pose(self, pose, lead_s):
        # type: (Pose2D, float) -> Pose2D
        """The pose the drone is at NOW, estimated from the delayed measurement.

        Advances the measured pose along the last commanded body velocity for the
        vision loop's transport delay, so steering reacts to the present rather
        than to ``latency_s`` ago. The scheduler has already zeroed ``lead_s``
        while coasting (the provider's coast is command-propagated — leading it
        would double-count) and scaled it by proven effectiveness (commands that
        demonstrably do not move the drone must not move its pose estimate).
        """
        if lead_s <= 0.0:
            return pose
        vx, vy, wz = self._last
        cos_y = cos(pose.yaw)
        sin_y = sin(pose.yaw)
        return Pose2D(pose.x + (vx * cos_y - vy * sin_y) * lead_s,
                      pose.y + (vx * sin_y + vy * cos_y) * lead_s,
                      pose.yaw + wz * lead_s)

    # ─── Regimes ─────────────────────────────────────────────────
    def _run_escape(self, pose, blocked, frozen, dt, auth):
        # type: (Pose2D, object, bool, float, object) -> Optional[DriftPidCommand]
        """Drive or start an escape reflex; None when navigation should continue."""
        if self._escape.active:
            if frozen:
                # Never fly an open-loop manoeuvre on a pose that has gone cold.
                self._escape.abort()
                return None
            esc = self._escape.step(dt)
            if esc.active:
                self._anchor = (pose.x, pose.y)
                self._anchor_yaw = pose.yaw
                return self._emit(esc.vx, esc.vy, esc.wz, DriftPidState.ESCAPE,
                                  dt, auth, blocked, esc.state, esc.reason)
            return None
        if blocked.blocked and not frozen:
            carrot = geo.lookahead_point(self._path, self._idx, pose.x, pose.y,
                                         self.params.lookahead_m)
            prefer_left = geo.bearing_error(pose.x, pose.y, pose.yaw,
                                            carrot[0], carrot[1]) >= 0.0
            self._escape.trigger(blocked, prefer_left=prefer_left)
        return None

    def _hold_still(self, pose, dt, auth, blocked, report):
        # type: (Pose2D, float, object, object, bool) -> DriftPidCommand
        """Ramp to a stop and freeze every integrator.

        The drone was told to stop, so a position error is not something to
        correct — winding up on it would launch the drone the moment the hold
        lifts. The learned drift is kept, not cleared.
        """
        self._latch_anchor(pose)
        self._lat.update(0.0, dt, integrate=False, gain_scale=0.0)
        self._fwd.update(0.0, dt, integrate=False, gain_scale=0.0)
        self._yaw.update(0.0, dt, integrate=False, gain_scale=0.0)
        return self._emit(0.0, 0.0, 0.0, DriftPidState.HOLD, dt, auth, blocked,
                          report_blocked=report, reason=auth.reason)

    def _navigate(self, pose, dt, auth, blocked, report, fresh):
        # type: (Pose2D, float, object, object, bool, bool) -> DriftPidCommand
        """The normal control law: TRACK a leg, TURN onto it, or HOLD the goal.

        ``pose`` here is the latency-led steering pose; every error below is
        therefore measured against where the drone is now, not a frame ago.
        """
        p = self.params
        integrate = auth.integrate
        gain = auth.gain_scale

        if self._done:
            self._turning = False
            self._latch_anchor(pose, at_goal=True)
            vx, vy, wz, e_fwd, e_lat, e_yaw = self._station_keep(
                pose, dt, auth, fresh, self._anchor_yaw)
            return self._emit(vx, vy, wz, DriftPidState.HOLD, dt, auth, blocked,
                              report_blocked=report, e_fwd=e_fwd, e_lat=e_lat,
                              e_yaw=e_yaw)

        carrot = geo.lookahead_point(self._path, self._idx, pose.x, pose.y,
                                     p.lookahead_m)
        e_yaw = geo.bearing_error(pose.x, pose.y, pose.yaw, carrot[0], carrot[1])
        self._turning = yaw_engaged(self._turning, e_yaw, p.yaw_engage_rad,
                                    p.yaw_release_rad)
        # Two yaw regimes, two caps: TURNING owns the rotation and gets the
        # strong approach rate; TRACKING only trims it -- there the cross-track
        # ROLL owns the line, and an unthrottled yaw would swing the nose off
        # course on every pose wobble.
        wz = saturate(self._yaw.update(e_yaw, dt, integrate=integrate,
                                       gain_scale=gain,
                                       deadband_extra=auth.yaw_deadband_extra_rad,
                                       fresh=fresh),
                      p.approach_yaw_rate if self._turning
                      else p.track_yaw_rate)

        if self._turning:
            # Rotate onto the leg while station-keeping, so the drone comes out of
            # the turn where it went in rather than a drift-width downwind of it.
            self._latch_anchor(pose, on_track=True)
            vx, vy, e_fwd, e_lat = self._anchor_correction(
                pose, dt, auth, fresh, lateral_frac=p.lateral_turn_frac)
            vx *= alignment_gate(e_yaw, p.travel_cone_rad,
                                 p.translate_suppress_rad,
                                 p.translate_suppress_floor)
            # The turn owns the translation direction: forward at least
            # turn_pitch_bias (backward flips the delivered yaw on this
            # airframe), lateral confined to the sideslip cone.
            vx, vy = turn_coordination(vx, vy, wz, self._yaw_active_rad(),
                                       p.turn_pitch_bias, p.turn_side_cone_rad)
            return self._emit(vx, vy, wz, DriftPidState.TURN, dt, auth, blocked,
                              report_blocked=report, e_fwd=e_fwd, e_lat=e_lat,
                              e_yaw=e_yaw)

        # TRACK: the line is the reference and the cruise owns the forward axis.
        self._anchor = None
        e_fwd, e_lat, _ = geo.cross_track_error(self._path, self._idx,
                                                pose.x, pose.y, pose.yaw)
        vy = self._lat.update(e_lat, dt, integrate=integrate, gain_scale=gain,
                              deadband_extra=auth.deadband_extra_m, fresh=fresh)
        # Straight-flight bonus: while the yaw correction is quiet the drone may
        # cruise harder; as |wz| grows toward the track cap the speed blends back
        # to the base cruise -- never sprint and swing the nose at once.
        cruise = p.cruise_speed
        if p.cruise_speed_straight > 0.0:
            yaw_frac = min(1.0, abs(wz) / p.track_yaw_rate)
            cruise = (p.cruise_speed_straight
                      - (p.cruise_speed_straight - p.cruise_speed) * yaw_frac)
        vx = approach_speed(self._distance_to_goal(pose), p.pos_radius,
                            p.slow_radius, cruise, p.arrive_speed_min)
        vx *= alignment_gate(e_yaw, p.travel_cone_rad, p.translate_suppress_rad,
                             p.translate_suppress_floor)
        if p.forward_track_frac > 0.0:
            vx += p.forward_track_frac * self._fwd.update(
                e_fwd, dt, integrate=integrate, gain_scale=gain,
                deadband_extra=auth.deadband_extra_m, fresh=fresh)
        else:
            # Keep the loop's memory current so switching regimes does not start
            # it from a stale error, but let it contribute nothing here.
            self._fwd.update(e_fwd, dt, integrate=False, gain_scale=0.0,
                             fresh=fresh)
        # Mid-leg yaw trims obey the same coupling as turns: while the trim is
        # genuinely rotating the airframe, never let the translation point
        # backward or sideways-first (floor 0: the cruise already owns forward).
        vx, vy = turn_coordination(vx, vy, wz, self._yaw_active_rad(), 0.0,
                                   p.turn_side_cone_rad)
        return self._emit(vx, vy, wz, DriftPidState.TRACK, dt, auth, blocked,
                          report_blocked=report, e_fwd=e_fwd, e_lat=e_lat,
                          e_yaw=e_yaw)

    # ─── Helpers ─────────────────────────────────────────────────
    def _station_keep(self, pose, dt, auth, fresh, hold_yaw):
        # type: (Pose2D, float, object, bool, float) -> Tuple[float, ...]
        """Full 2-axis position hold on the anchor plus a heading hold.

        The heading hold uses the gentle TRACK cap: holding station is trimming,
        not turning onto a leg."""
        vx, vy, e_fwd, e_lat = self._anchor_correction(pose, dt, auth, fresh)
        e_yaw = normalize_angle(hold_yaw - pose.yaw)
        wz = saturate(self._yaw.update(e_yaw, dt, integrate=auth.integrate,
                                       gain_scale=auth.gain_scale,
                                       deadband_extra=auth.yaw_deadband_extra_rad,
                                       fresh=fresh),
                      self.params.track_yaw_rate)
        # Rotate first, translate after: while the heading trim is actively
        # rotating the airframe a backward (or roll-first) correction inverts
        # the delivered yaw, so it is deferred until the trim quietens. The
        # position error stays open and is closed then. Floor 0: a hold never
        # invents forward motion the anchor did not ask for.
        vx, vy = turn_coordination(vx, vy, wz, self._yaw_active_rad(), 0.0,
                                   self.params.turn_side_cone_rad)
        return vx, vy, wz, e_fwd, e_lat, e_yaw

    def _anchor_correction(self, pose, dt, auth, fresh, lateral_frac=1.0):
        # type: (Pose2D, float, object, bool, float) -> Tuple[float, float, float, float]
        """Body-frame correction pulling the drone back onto its latched anchor.

        The hold deadband widens with the provider's own error estimate: while
        the pose is only good to +-20 cm, a 5 cm "wander" around the anchor is
        the measurement moving, not the drone.
        """
        ax, ay = self._anchor if self._anchor else (pose.x, pose.y)
        e_fwd, e_lat = geo.body_offset_to_point(pose.x, pose.y, pose.yaw, ax, ay)
        db = self.params.hold_deadband_m + auth.deadband_extra_m
        vx = self._fwd.update(e_fwd if abs(e_fwd) > db else 0.0, dt,
                              integrate=auth.integrate,
                              gain_scale=auth.gain_scale, fresh=fresh)
        vy = lateral_frac * self._lat.update(e_lat if abs(e_lat) > db else 0.0,
                                             dt, integrate=auth.integrate,
                                             gain_scale=auth.gain_scale,
                                             fresh=fresh)
        return vx, vy, e_fwd, e_lat

    def _latch_anchor(self, pose, on_track=False, at_goal=False):
        # type: (Pose2D, bool, bool) -> None
        """Fix the station-keeping reference, once, on entering a holding regime.

        ``on_track`` puts the anchor on the trajectory rather than under the
        drone, so holding still also finishes closing whatever cross-track error
        was open when the hold began.
        """
        if self._anchor is not None:
            return
        if at_goal and self._path:
            self._anchor = self._path[-1]
        elif on_track:
            _, _, foot = geo.cross_track_error(self._path, self._idx,
                                               pose.x, pose.y, pose.yaw)
            self._anchor = foot
        else:
            self._anchor = (pose.x, pose.y)
        self._anchor_yaw = pose.yaw
        for pid in (self._lat, self._fwd, self._yaw):
            pid.reset_derivative()

    def _yaw_active_rad(self):
        # type: () -> float
        """Yaw command at/above which the airframe will actually rotate.

        The envelope's minimum-force shaping snaps anything above
        ``release_frac * min_wz`` up to ``min_wz``, so that release point — not
        ``min_wz`` itself — is where a raw yaw demand starts moving the drone,
        and with it the translation-direction coupling the turn coordination
        exists to respect.
        """
        env = self.params.envelope
        return env.release_frac * env.min_wz

    def _distance_to_goal(self, pose):
        # type: (Pose2D) -> float
        """Straight-line range to the FINAL waypoint (drives the arrival ramp)."""
        gx, gy = self._path[-1]
        return ((gx - pose.x) ** 2 + (gy - pose.y) ** 2) ** 0.5

    def _retire(self, pose):
        # type: (Pose2D) -> None
        """Advance past waypoints that are captured, or already behind us."""
        p = self.params
        n = len(self._path)
        while self._idx < n:
            wx, wy = self._path[self._idx]
            dist = ((wx - pose.x) ** 2 + (wy - pose.y) ** 2) ** 0.5
            final = self._idx == n - 1
            if dist <= p.pos_radius:
                if final:
                    self._done = True
                    return
                self._idx += 1
                self._anchor = None
                continue
            if not final:
                bearing = geo.bearing_error(pose.x, pose.y, pose.yaw, wx, wy)
                if abs(bearing) > p.passed_bearing_rad:
                    self._idx += 1
                    self._anchor = None
                    continue
            return
        self._idx = n - 1
        self._done = True

    def _emit(self, vx, vy, wz, state, dt, auth, blocked=None, escape_state="IDLE",
              reason="", report_blocked=False, e_fwd=0.0, e_lat=0.0, e_yaw=0.0):
        # type: (...) -> DriftPidCommand
        """Push the desired velocity through the force envelope and package it."""
        # An external yield (climb pulse) shrinks the horizontal translation
        # BEFORE the envelope, so the slew memory, minimum-force shaping and the
        # blockage monitor's notion of "what was commanded" all stay truthful.
        vx *= self._yield_scale
        vy *= self._yield_scale
        vx, vy, wz = self._envelope.apply(
            vx, vy, wz, dt, speed_scale=auth.speed_scale,
            yaw_speed_scale=getattr(auth, "yaw_speed_scale", None))
        self._last = (vx, vy, wz)
        self._state = state
        telemetry = DriftTelemetry(
            drift_vy=self._lat.drift, drift_vx=self._fwd.drift,
            drift_wz=self._yaw.drift, cross_track_m=e_lat, along_track_m=e_fwd,
            heading_err_rad=e_yaw, effort=self._envelope.effort(vx, vy, wz),
            speed_scale=auth.speed_scale, lead_s=auth.lead_s,
            deadband_extra_m=auth.deadband_extra_m,
            authority=reason or auth.reason,
            blocked_axis=blocked.axis if blocked is not None else "",
            escape_state=escape_state)
        return DriftPidCommand(
            command=ControlCommand.velocity(vx, vy, 0.0, wz, tracker=self.name),
            state=state, required_axis=None, freeze=None, done=self._done,
            wp_idx=min(self._idx, max(0, len(self._path) - 1)),
            num_waypoints=len(self._path), telemetry=telemetry,
            report_blocked=report_blocked)
