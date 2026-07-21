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
    """

    def __init__(self, target_z=1.0, deadband_m=0.10, kp=0.5, climb_max=0.06,
                 descend_max=0.10, ceiling_m=1.2, conf_min_climb=0.35,
                 conf_min_descend=0.10, min_z_m=0.2):
        self.target_z = float(target_z)
        self.deadband_m = float(deadband_m)
        self.kp = float(kp)
        self.climb_max = float(climb_max)
        self.descend_max = float(descend_max)
        self.ceiling_m = float(ceiling_m)
        self.conf_min_climb = float(conf_min_climb)
        self.conf_min_descend = float(conf_min_descend)
        self.min_z_m = float(min_z_m)
        if self.target_z <= 0.0:
            raise ValueError("AltitudeHoldParams.target_z must be > 0")
        for name in ("deadband_m", "kp", "climb_max", "descend_max"):
            if getattr(self, name) <= 0.0:
                raise ValueError("AltitudeHoldParams." + name + " must be > 0")
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
    """The loop. Stateless between ticks: every decision is this tick's alone."""

    def __init__(self, params=None):
        self.params = params or AltitudeHoldParams()
        #: Human-readable account of the last decision, for narration/logging.
        self.last_reason = "idle"

    def update(self, z, confidence, coasting, pose_valid, flying):
        """Decide this tick's vertical command.

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
            vz in m/s, + up. 0.0 whenever any gate says no.
        """
        p = self.params
        if not flying:
            self.last_reason = "held -- no vertical authority"
            return 0.0
        if z is None or not pose_valid or coasting:
            self.last_reason = "pose not trustworthy -- holding throttle"
            return 0.0
        z = float(z)
        if z < p.min_z_m:
            self.last_reason = "below %.2fm -- not airborne, hands off" % p.min_z_m
            return 0.0

        err = p.target_z - z
        if abs(err) <= p.deadband_m:
            self.last_reason = "on altitude (%.2fm)" % z
            return 0.0

        if err > 0.0:
            # CLIMB: the guarded direction.
            if z >= p.ceiling_m:
                self.last_reason = ("at the %.2fm ceiling -- climb refused"
                                    % p.ceiling_m)
                return 0.0
            if confidence < p.conf_min_climb:
                self.last_reason = ("confidence %.2f below %.2f -- not climbing "
                                    "on a vague pose" % (confidence,
                                                         p.conf_min_climb))
                return 0.0
            vz = min(p.kp * err, p.climb_max)
            self.last_reason = "climbing %.2f -> %.2fm at %.2fm/s" % (
                z, p.target_z, vz)
            return vz

        # DESCEND: the safe direction, but the z must still be believed.
        if confidence < p.conf_min_descend:
            self.last_reason = ("confidence %.2f below %.2f -- not trusting the "
                                "altitude enough to descend on it"
                                % (confidence, p.conf_min_descend))
            return 0.0
        vz = max(p.kp * err, -p.descend_max)
        self.last_reason = "descending %.2f -> %.2fm at %.2fm/s" % (
            z, p.target_z, -vz)
        return vz
