#!/usr/bin/env python3
"""altitude_hold.py -- a cautious altitude loop for a platform with no native hold.

Nothing on this stack commanded ``linear.z`` and the logs show the price: the
platform's own height drifts (it sank at hover with zero commands), turns bleed
height through the roll/pitch tilt, and one run overshot to 1.25 m then sagged to
0.33 m. The AprilTags all sit at ~1.0 m, so height errors directly degrade the
localization the whole stack flies on.

This is a deliberately timid P loop on the SOLVED altitude (the AprilTag pose z):

* **Climbing is the guarded direction.** The XTEND vertical axis is scaled by
  the FORWARD calibration and climbs far harder than the nominal m/s, and the
  ceiling is real: the operator marks 1.5 m as dangerous and does not want 1.3 m
  reached at all. So a climb needs a TRUSTED pose (confidence at/above
  ``conf_min_climb``, never coasting), is rate-capped tiny (``climb_max``), and
  is refused outright at/above ``ceiling_m`` no matter what the error says.
* **Descending is the safe direction** (toward the tag plane, away from the
  ceiling) but still requires a live, believed pose: the z being corrected is
  itself the measurement being distrusted.
* **A deadband absorbs solve noise** -- the z std of a fix is several
  centimetres, and chasing it would turn the loop into throttle jitter.
* **On the ground it does nothing.** Below ``min_z_m`` the drone is either not
  airborne yet or the z solve is broken; both mean hands off.

Pure and ROS-free so it can be unit-tested; the follower node feeds it the pose
z + localization quality each tick and rides the returned vz on the published
twist. Python 3.8 compatible (runs under the Noetic adapter).
"""
from __future__ import annotations

from typing import NamedTuple


class AltitudeCommand(NamedTuple):
    """One tick's vertical decision, plus how much to yield the horizontal axes.

    Attributes:
        vz: Vertical velocity to command (m/s, + up).
        translation_scale: Multiplier the follower applies to horizontal
            TRANSLATION (vx, vy) this tick, 0..1. 1.0 normally; low during a
            climb PULSE, because pitching/rolling tilts the thrust vector and
            steals the very lift a climb needs -- so a real climb briefly stops
            translating, gains height, and hands the drone back. Yaw is left
            alone (it costs no lift, and sweeping tags in helps the pose a climb
            needs to be trusted).
        reason: Short human-readable account, for narration/logging.
    """

    vz: float
    translation_scale: float
    reason: str


class AltitudeHoldParams(object):
    """Tuning for :class:`AltitudeHold` (SI units).

    Attributes:
        target_z: Altitude to hold (m). On this stack, the tag-plane height.
        deadband_m: |error| below which no correction is made (m). Keep above
            the pose solve's own z noise.
        kp: Proportional gain, metres of error -> m/s of vertical command.
        climb_max: Cap on the UP command (m/s). The guarded direction -- the
            vertical axis borrows the forward calibration and climbs far harder
            than nominal, so keep this small.
        descend_max: Cap on the DOWN command (m/s).
        ceiling_m: At/above this measured altitude a climb is NEVER commanded,
            whatever the error; only descent is allowed. The operator's hard
            line: 1.3 m must not be reached, 1.5 m is dangerous.
        conf_min_climb: Pose confidence required to command UP. Climbing on a
            vague pose is how a drone meets a ceiling.
        conf_min_descend: Pose confidence required to command DOWN. Lower than
            the climb floor -- descending toward the tag plane is the recovering
            direction -- but not zero: the z being corrected must be believed.
        min_z_m: Below this measured altitude the loop does nothing (not
            airborne, or the solve is broken).
        pulse_trigger_m: Sag below target (m) at which a CLIMB PULSE begins:
            the drone stops translating and dedicates thrust to the (tapered)
            climb. The pulse releases at HALF this sag, so the final stretch is
            flown as a gentle trim and the platform's momentum dies before the
            target rather than past it. Must exceed ``deadband_m`` -- a pulse is
            for a real height loss, not the trim wander the deadband absorbs.
            Set huge (e.g. 10) to disable pulsing and only ever trim.
        pulse_translation_scale: The (vx, vy) multiplier held during a pulse
            (0..1). Low so the platform's thrust goes to lift, not to tilting.
            0 stops horizontal translation dead while climbing.
    """

    def __init__(self, target_z=1.0, deadband_m=0.10, kp=0.5, climb_max=0.15,
                 descend_max=0.10, ceiling_m=1.2, conf_min_climb=0.35,
                 conf_min_descend=0.10, min_z_m=0.2, pulse_trigger_m=0.20,
                 pulse_translation_scale=0.2):
        self.target_z = float(target_z)
        self.deadband_m = float(deadband_m)
        self.kp = float(kp)
        self.climb_max = float(climb_max)
        self.descend_max = float(descend_max)
        self.ceiling_m = float(ceiling_m)
        self.conf_min_climb = float(conf_min_climb)
        self.conf_min_descend = float(conf_min_descend)
        self.min_z_m = float(min_z_m)
        self.pulse_trigger_m = float(pulse_trigger_m)
        self.pulse_translation_scale = float(pulse_translation_scale)
        if self.target_z <= 0.0:
            raise ValueError("AltitudeHoldParams.target_z must be > 0")
        for name in ("deadband_m", "kp", "climb_max", "descend_max"):
            if getattr(self, name) <= 0.0:
                raise ValueError("AltitudeHoldParams." + name + " must be > 0")
        if self.pulse_trigger_m <= self.deadband_m:
            raise ValueError(
                "AltitudeHoldParams.pulse_trigger_m (%.2f) must exceed deadband "
                "(%.2f) -- a climb pulse is for a real sag, not deadband wander"
                % (self.pulse_trigger_m, self.deadband_m))
        if not 0.0 <= self.pulse_translation_scale <= 1.0:
            raise ValueError("AltitudeHoldParams.pulse_translation_scale must be "
                             "in [0, 1]")
        if self.ceiling_m <= self.target_z + self.deadband_m:
            raise ValueError(
                "AltitudeHoldParams.ceiling_m (%.2f) must exceed target_z + "
                "deadband (%.2f) -- otherwise the hold chatters against its own "
                "ceiling" % (self.ceiling_m, self.target_z + self.deadband_m))
        for name in ("conf_min_climb", "conf_min_descend"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError("AltitudeHoldParams." + name +
                                 " must be in [0, 1]")
        if self.conf_min_descend > self.conf_min_climb:
            raise ValueError(
                "AltitudeHoldParams.conf_min_descend must not exceed "
                "conf_min_climb -- descending is the safe direction and must "
                "never need MORE trust than climbing")


class AltitudeHold(object):
    """The loop. One bit of state -- whether a climb PULSE is in progress."""

    def __init__(self, params=None):
        self.params = params or AltitudeHoldParams()
        #: Human-readable account of the last decision, for narration/logging.
        self.last_reason = "idle"
        #: True while a dedicated climb pulse is running (hysteresis: it latches
        #: on at pulse_trigger_m of sag and off at half that, so it does not
        #: chatter, and the last half is flown as a trim, killing the platform's
        #: climb momentum before the target instead of past it).
        self._pulse = False

    @property
    def pulsing(self):
        # type: () -> bool
        """True while a climb pulse is suppressing horizontal translation."""
        return self._pulse

    def _hold(self, reason):
        # type: (str) -> AltitudeCommand
        """No vertical command, full horizontal authority, pulse cleared."""
        self._pulse = False
        self.last_reason = reason
        return AltitudeCommand(0.0, 1.0, reason)

    def update(self, z, confidence, coasting, pose_valid, flying):
        """Decide this tick's vertical command and horizontal yield.

        Args:
            z: Measured altitude (m, solved pose z), or None if unknown.
            confidence: Pose confidence 0..1.
            coasting: True while the pose is dead reckoning. A coasted z was
                propagated from commands that contain no vertical truth, so the
                loop never acts on one.
            pose_valid: False when localization is absent or stale.
            flying: False while the follower is held (GO not given, lost pose,
                startup) -- a held drone must not be nudged vertically either.

        Returns:
            An :class:`AltitudeCommand`. ``vz`` is 0 and ``translation_scale``
            is 1 whenever any gate says no.
        """
        p = self.params
        if not flying:
            return self._hold("held -- no vertical authority")
        if z is None or not pose_valid or coasting:
            return self._hold("pose not trustworthy -- holding throttle")
        z = float(z)
        if z < p.min_z_m:
            return self._hold("below %.2fm -- not airborne, hands off" % p.min_z_m)

        err = p.target_z - z
        if abs(err) <= p.deadband_m:
            return self._hold("on altitude (%.2fm)" % z)

        if err > 0.0:
            # CLIMB: the guarded direction.
            if z >= p.ceiling_m:
                return self._hold("at the %.2fm ceiling -- climb refused"
                                  % p.ceiling_m)
            if confidence < p.conf_min_climb:
                return self._hold("confidence %.2f below %.2f -- not climbing on "
                                  "a vague pose" % (confidence, p.conf_min_climb))
            # The vertical demand always TAPERS toward the target (kp * err,
            # capped). Never hold climb_max flat until arrival: this platform
            # climbs far harder than the nominal m/s, and a flat-out climb
            # released only at the deadband coasted to 1.5-1.6 m on the logs --
            # straight through the ceiling into the operator's danger zone.
            vz = min(p.kp * err, p.climb_max)
            # A big sag latches a dedicated PULSE (yield translation, climb);
            # it releases at HALF the trigger -- early, so the last stretch is
            # flown as a gentle trim and the platform's momentum has room to
            # die before the target, not after it.
            if err >= p.pulse_trigger_m:
                self._pulse = True
            elif err <= 0.5 * p.pulse_trigger_m:
                self._pulse = False
            if self._pulse:
                self.last_reason = ("climb pulse: holding still to regain "
                                    "%.2fm (at %.2fm, %.2fm/s)"
                                    % (p.target_z, z, vz))
                return AltitudeCommand(vz, p.pulse_translation_scale,
                                       self.last_reason)
            # Small sag: trim up gently while still flying the route.
            self.last_reason = "trimming up %.2f -> %.2fm at %.2fm/s" % (
                z, p.target_z, vz)
            return AltitudeCommand(vz, 1.0, self.last_reason)

        # DESCEND: the safe direction (away from the ceiling, toward the tags),
        # and it does not steal lift, so it never suppresses translation.
        self._pulse = False
        if confidence < p.conf_min_descend:
            return self._hold("confidence %.2f below %.2f -- not trusting the "
                              "altitude enough to descend on it"
                              % (confidence, p.conf_min_descend))
        vz = max(p.kp * err, -p.descend_max)
        self.last_reason = "descending %.2f -> %.2fm at %.2fm/s" % (
            z, p.target_z, -vz)
        return AltitudeCommand(vz, 1.0, self.last_reason)
