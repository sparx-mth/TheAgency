"""Parameters for the drift-PID controller.

The controller has two jobs and they want opposite things, so the tuning is split
to match:

  * **Get to the next waypoint.** Feed-forward: a cruise speed along the leg, a
    yaw rate when the heading error is big enough to be worth rotating for. These
    are the "how fast do we fly" dials, and they are the ones to raise once the
    drone flies cleanly.
  * **Stay exactly on the line while doing it.** Feedback: three PID loops whose
    integral terms learn the standing drift on each axis. These are the "how
    tightly do we hold" dials, and their corrections are deliberately capped far
    below the cruise speeds — a correction that can out-run the flight is a
    correction that flies the drone.

Nothing here decides *where* to go. The route comes from the planner as a list of
waypoints; the controller flies it and holds the line.

Sub-parameter groups live with the code that reads them
(:mod:`.pid`, :mod:`.envelope`, :mod:`.confidence`, :mod:`.blockage`,
:mod:`.escape`) so each module can be understood and tested on its own; this
module only composes them and adds the navigation-level dials.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import radians

from .blockage import BlockageParams
from .confidence import ConfidenceParams
from .envelope import EnvelopeParams
from .escape import EscapeParams
from .pid import PidGains


def _default_lateral_pid():
    # type: () -> PidGains
    """Cross-track (ROLL) loop — the main drift channel on this airframe.

    This is the axis the drone loses most: it slides sideways in forward flight
    and in turns alike. The gains are low and the integral limit is what actually
    does the work; ``i_limit`` is the largest standing sideways push the
    controller may hold on its own.
    """
    return PidGains(kp=0.55, ki=0.06, kd=0.12, i_limit=0.05, d_tau_s=0.4,
                    deadband=0.03, out_limit=0.10)


def _default_forward_pid():
    # type: () -> PidGains
    """Along-track loop — fore/aft station-keeping while turning or holding.

    Inactive during a straight leg (the cruise feed-forward owns the axis then);
    this loop exists for the drift that shows up while the drone is turning or
    standing still.
    """
    return PidGains(kp=0.50, ki=0.05, kd=0.10, i_limit=0.05, d_tau_s=0.4,
                    deadband=0.05, out_limit=0.10)


def _default_yaw_pid():
    # type: () -> PidGains
    """Heading loop. The least-drifting axis, so the smallest integral budget."""
    return PidGains(kp=0.90, ki=0.08, kd=0.15, i_limit=0.08, d_tau_s=0.35,
                    deadband=radians(2.0), out_limit=0.35)


@dataclass(frozen=True)
class DriftPidParams:
    """Tuning for :class:`~.follower.DriftPidFollower` (SI, body frame REP-103).

    Attributes:
        cruise_speed: Forward speed on a straight leg (m/s). The main "how fast"
            dial; the envelope's ``max_vx`` is the ceiling it may never exceed.
        approach_yaw_rate: Yaw rate cap while TURNING onto a leg (rad/s) --
            rotation is the mission then, so it gets the strong cap.
        track_yaw_rate: Yaw rate cap while TRACKING a leg or station-keeping
            (rad/s). Deliberately far below ``approach_yaw_rate``: mid-leg the
            heading error is small and the cross-track PID (ROLL) owns the line,
            so yaw only trims -- an unthrottled yaw here swings the nose on
            every pose wobble and steers the drone off course, which is exactly
            what the flight logs showed. Must not exceed ``approach_yaw_rate``.
        cruise_speed_straight: Forward speed when the yaw correction is QUIET
            (m/s). The straight-flight bonus: while the heading needs no work
            the drone may fly harder, and as the yaw command grows toward
            ``track_yaw_rate`` the speed blends smoothly back down to
            ``cruise_speed`` -- correcting the nose and sprinting at once is
            what smears the depth frames and overshoots the line. 0 disables
            (fly ``cruise_speed`` everywhere, the old behaviour); otherwise it
            must sit in [cruise_speed, envelope.max_vx].
        pos_radius: Waypoint capture radius (m). Inside it, the waypoint is
            retired.
        slow_radius: Distance to the FINAL goal at which the speed starts ramping
            down toward ``arrive_speed_min`` (m). Intermediate waypoints are
            glided through at cruise.
        arrive_speed_min: Floor on the approach speed while still outside
            ``pos_radius`` (m/s), so the drone does not crawl to a noisy stop
            short of the point.
        lookahead_m: How far along the leg the heading setpoint is taken from (m).
            Larger looks further ahead and flies smoother but cuts corners; smaller
            tracks the line more exactly and weaves more.
        yaw_engage_rad: Heading error above which the controller rotates. Below it
            the drone corrects with ROLL alone, which is the low-noise regime for
            both localization and the depth model.
        yaw_release_rad: Heading error below which rotation is released once
            engaged. Must be below ``yaw_engage_rad``; the gap is the hysteresis
            band that stops the yaw command chattering on pose noise.
        travel_cone_rad: Largest body-frame angle off straight-ahead the
            translation may point. Keeps the drone from flying fast into space its
            forward-facing camera and the BEV map have never seen.
        translate_suppress_rad: Heading error at/above which translation is
            throttled to ``translate_suppress_floor``, so a badly mis-pointed
            drone mostly turns before it flies. Must exceed ``travel_cone_rad``.
        translate_suppress_floor: Fraction of speed kept at/beyond that angle
            (0..1). Non-zero so the drone keeps creeping rather than stopping dead.
        passed_bearing_rad: A non-final waypoint whose bearing exceeds this is
            treated as passed and retired, instead of turning back for it.
        hold_deadband_m: Station-keeping position error left uncorrected (m).
        forward_track_frac: Fraction of the along-track PID applied during a
            straight leg (0..1). 0 by default — the cruise feed-forward owns the
            forward axis then, and a second authority on it just fights.
        lateral_turn_frac: Fraction of the cross-track PID applied while
            rotating (0..1). Reduced: a large ROLL during a turn perturbs the
            rotation itself.
        turn_pitch_bias: Forward speed ridden during a TURN whose translation
            would otherwise be zero (m/s). A pure yaw leaves this airframe flat,
            so the turn bites late and coasts -- measured on the deployed drone
            at ~11% yaw delivery in place versus 30-68% while translating. The
            same trick the one-axis follower's ``yaw_pitch_bias`` applies at
            publish time, done here inside the control law so the envelope,
            telemetry and logs all see the command that actually flies. Only
            replaces a smaller |vx|; a real station-keeping correction that is
            already larger wins. 0 disables (a pure yaw stays pure).
        settle_map_updates: Fresh voxel updates the controller asks the adapter to
            wait for while stopped. Exposed for interface parity with the one-axis
            follower's map-settle gate.
        lateral_pid: Cross-track (ROLL) loop gains.
        forward_pid: Along-track (PITCH) loop gains.
        yaw_pid: Heading (YAW) loop gains.
        drift_leak_s: Time constant over which every learned drift bleeds back
            toward zero (s). Long: the drift belongs to the airframe and the room,
            not to this minute.
        envelope: Per-axis force limits and the combined-speed budget.
        confidence: How localization quality maps onto control authority.
        blockage: How "commanded but not moving" is detected.
        escape: The reflexes run when it is.
    """

    # ── Feed-forward: how fast we fly ──
    cruise_speed: float = 0.18
    cruise_speed_straight: float = 0.0
    approach_yaw_rate: float = 0.35
    track_yaw_rate: float = 0.35
    pos_radius: float = 0.30
    slow_radius: float = 0.80
    arrive_speed_min: float = 0.08
    lookahead_m: float = 0.60

    # ── Roll-versus-yaw allocation ──
    yaw_engage_rad: float = radians(22.0)
    yaw_release_rad: float = radians(9.0)
    travel_cone_rad: float = radians(70.0)
    translate_suppress_rad: float = radians(110.0)
    translate_suppress_floor: float = 0.15
    passed_bearing_rad: float = radians(110.0)

    # ── Station-keeping / regime scaling ──
    hold_deadband_m: float = 0.06
    forward_track_frac: float = 0.0
    lateral_turn_frac: float = 0.40
    turn_pitch_bias: float = 0.0
    settle_map_updates: int = 0

    # ── Feedback: how tightly we hold ──
    lateral_pid: PidGains = field(default_factory=_default_lateral_pid)
    forward_pid: PidGains = field(default_factory=_default_forward_pid)
    yaw_pid: PidGains = field(default_factory=_default_yaw_pid)
    drift_leak_s: float = 180.0

    # ── Platform limits and the world's opinion of us ──
    envelope: EnvelopeParams = field(default_factory=EnvelopeParams)
    confidence: ConfidenceParams = field(default_factory=ConfidenceParams)
    blockage: BlockageParams = field(default_factory=BlockageParams)
    escape: EscapeParams = field(default_factory=EscapeParams)

    def __post_init__(self):
        # type: () -> None
        """Validate the invariants the control law relies on."""
        for name in ("cruise_speed", "approach_yaw_rate", "pos_radius",
                     "lookahead_m", "yaw_engage_rad", "travel_cone_rad",
                     "drift_leak_s"):
            if getattr(self, name) <= 0.0:
                raise ValueError("DriftPidParams." + name + " must be > 0")
        if self.yaw_release_rad >= self.yaw_engage_rad:
            raise ValueError("DriftPidParams.yaw_release_rad must be below "
                             "yaw_engage_rad -- without the gap the yaw command "
                             "chatters on pose noise")
        if self.slow_radius < self.pos_radius:
            raise ValueError("DriftPidParams.slow_radius must be >= pos_radius")
        if self.translate_suppress_rad <= self.travel_cone_rad:
            raise ValueError("DriftPidParams.translate_suppress_rad must exceed "
                             "travel_cone_rad")
        for name in ("translate_suppress_floor", "forward_track_frac",
                     "lateral_turn_frac"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError("DriftPidParams." + name + " must be in [0, 1]")
        if self.turn_pitch_bias < 0.0:
            raise ValueError("DriftPidParams.turn_pitch_bias must be >= 0")
        if self.turn_pitch_bias > self.envelope.max_vx:
            raise ValueError(
                "DriftPidParams.turn_pitch_bias (%.2f) exceeds envelope.max_vx "
                "(%.2f) -- the bias is a nudge that makes a turn bite, not a "
                "flight speed" % (self.turn_pitch_bias, self.envelope.max_vx))
        if self.cruise_speed > self.envelope.max_vx:
            raise ValueError(
                "DriftPidParams.cruise_speed (%.2f) exceeds envelope.max_vx "
                "(%.2f) -- the cruise would be clamped away every tick"
                % (self.cruise_speed, self.envelope.max_vx))
        if self.arrive_speed_min < self.envelope.release_frac * self.envelope.min_vx:
            raise ValueError(
                "DriftPidParams.arrive_speed_min (%.3f) is below the "
                "minimum-force release floor (%.3f) -- the drone would stall just "
                "outside pos_radius, commanding a speed the motors ignore"
                % (self.arrive_speed_min,
                   self.envelope.release_frac * self.envelope.min_vx))
        if self.lateral_pid.out_limit > self.envelope.max_vy:
            raise ValueError(
                "DriftPidParams.lateral_pid.out_limit (%.2f) exceeds "
                "envelope.max_vy (%.2f) -- corrections must stay inside the axis"
                % (self.lateral_pid.out_limit, self.envelope.max_vy))
        if self.forward_pid.out_limit > self.envelope.max_vx_back:
            raise ValueError(
                "DriftPidParams.forward_pid.out_limit (%.2f) exceeds "
                "envelope.max_vx_back (%.2f) -- a station-keeping correction can "
                "push BACKWARD, so it must fit inside the blind-reverse cap"
                % (self.forward_pid.out_limit, self.envelope.max_vx_back))
        if self.yaw_pid.out_limit > self.envelope.max_wz:
            raise ValueError(
                "DriftPidParams.yaw_pid.out_limit (%.2f) exceeds envelope.max_wz "
                "(%.2f) -- corrections must stay inside the axis"
                % (self.yaw_pid.out_limit, self.envelope.max_wz))
        if self.approach_yaw_rate > self.envelope.max_wz:
            raise ValueError(
                "DriftPidParams.approach_yaw_rate (%.2f) exceeds envelope.max_wz "
                "(%.2f) -- the turn rate would be clamped away every tick"
                % (self.approach_yaw_rate, self.envelope.max_wz))
        if not 0.0 < self.track_yaw_rate <= self.approach_yaw_rate:
            raise ValueError(
                "DriftPidParams.track_yaw_rate (%.2f) must be in (0, "
                "approach_yaw_rate=%.2f] -- tracking trims the heading, turning "
                "owns it; a track cap above the turn cap is a contradiction"
                % (self.track_yaw_rate, self.approach_yaw_rate))
        if self.cruise_speed_straight != 0.0 and not (
                self.cruise_speed <= self.cruise_speed_straight
                <= self.envelope.max_vx):
            raise ValueError(
                "DriftPidParams.cruise_speed_straight (%.2f) must be 0 (off) or "
                "in [cruise_speed=%.2f, max_vx=%.2f] -- the straight-flight "
                "bonus can only ever ADD speed, inside the axis"
                % (self.cruise_speed_straight, self.cruise_speed,
                   self.envelope.max_vx))
