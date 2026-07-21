"""Localization quality -> how hard the controller is allowed to fly.

The pose this controller closes its loops on is an AprilTag fix, and it lies in
three distinct ways that need three distinct answers:

  * **It gets vague.** Fewer tags, a worse viewing geometry, a higher reprojection
    error — the pose is still real, just imprecise. Answer: fly slower and lean on
    the fast (P/D) terms less, so the drone does not chase noise.
  * **It stops being a measurement.** When no tag is in view the provider *coasts*:
    it keeps publishing a pose propagated by the commanded motion. That pose is a
    guess dressed as a fix. Answer: keep flying briefly on what was already
    learned, but freeze the integrators — learning drift from dead reckoning means
    learning from your own commands, which teaches nothing and can diverge.
  * **It stops entirely.** The coast budget runs out and the topic goes silent.
    Answer: hold, and let the recovery own the drone.

Calibration note, because the numbers are not intuitive: pose confidence is
*multiplicatively* degraded and a single visible tag is hard-capped near 0.21 in
practice, while a coasted pose is capped at 0.25. So a threshold like "0.5 = good"
would ground the drone, and confidence ALONE cannot tell a coasted pose from a
genuine one-tag fix. That is why :class:`LocalizationQuality` carries the
``coasting`` flag separately — it comes from the provider's source string, and it
is the only honest answer to "is this a measurement?".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _ramp(value, lo, hi):
    # type: (float, float, float) -> float
    """Map ``value`` onto 0..1 across ``[lo, hi]``, clamped outside it."""
    if hi <= lo:
        return 1.0 if value >= hi else 0.0
    if value <= lo:
        return 0.0
    if value >= hi:
        return 1.0
    return (value - lo) / (hi - lo)


@dataclass(frozen=True)
class LocalizationQuality:
    """One snapshot of how much the pose can be trusted.

    Assembled by the ROS layer from the localization topics; the controller never
    talks to ROS itself.

    Attributes:
        confidence: Pose confidence, 0..1. Remember the ceilings: ~0.21 for a lone
            tag, 0.25 for a coasted pose.
        pos_std_m: The provider's own estimate of this pose's position error (m).
        age_s: Seconds since the pose arrived. Silence is a signal on this stack —
            rejected frames and an exhausted coast budget both publish nothing —
            so age is checked independently of confidence.
        coasting: True when the pose is dead reckoning (source ``apriltag_coast``),
            not a measurement.
        cmd_effectiveness: Fraction of commanded translation the world actually
            delivered, 0..1, learned as an EMA by the localization provider. Only
            meaningful while genuinely commanding motion — it freezes during a
            hover and starts low (0.3) before it has learned anything.
        valid: False when no localization has ever been seen, so a caller can tell
            "not started" from "bad".
    """

    confidence: float = 0.0
    pos_std_m: float = 1.0
    age_s: float = 999.0
    coasting: bool = False
    cmd_effectiveness: float = 1.0
    valid: bool = False


@dataclass(frozen=True)
class ConfidenceParams:
    """Thresholds mapping :class:`LocalizationQuality` onto control authority.

    Attributes:
        conf_full: Confidence at/above which the controller flies at full speed.
            Keep it reachable: a two-or-three-tag view is what "good" looks like
            here, not 0.9.
        conf_min: Confidence at/below which the controller flies at
            ``speed_floor``. Below ``conf_hold`` it stops instead.
        speed_floor: Speed scale held at/below ``conf_min`` (0..1). Not zero —
            creeping while unsure beats stopping dead in a corridor, and motion is
            what brings a tag back into view.
        gain_floor: Scale held on the P/D terms at/below ``conf_min`` (0..1).
            Lower than ``speed_floor`` on purpose: when the pose is vague the
            planned motion is still roughly right, but reacting to the *error* in
            that pose is what injects the noise.
        conf_integrate: Confidence below which the integrators freeze. Drift must
            only ever be learned from poses worth learning from.
        conf_hold: Confidence below which the controller stops and holds.
        max_age_s: Pose age beyond which the controller stops and holds (s).
        coast_speed_scale: Extra speed multiplier applied while the pose is
            coasting (0..1). Coasting is bounded (~0.5 s) and the drone should
            spend it slowing down, not committing further.
        eff_floor: ``cmd_effectiveness`` at/below which the commands are
            considered not to be reaching the world at all. The localization tests
            pin a genuinely stuck drone below 0.15.
        eff_full: ``cmd_effectiveness`` at/above which commands are considered
            fully effective.
        eff_speed_floor: Speed multiplier held while the commands are UNPROVEN
            (0..1). The provider's effectiveness starts at 0.3 and must be earned,
            so a fresh flight begins at roughly this fraction of cruise and speeds
            up over the first metres as the world confirms the commands work —
            caution exactly while the calibration is still a guess. 1.0 disables.
        latency_s: Transport delay from the camera exposing a frame to this
            controller acting on its pose (detection + solve + publish + bridge),
            in seconds. The controller advances the pose it steers by this much
            along the LAST COMMANDED velocity, so the fast (P/D) terms react to
            where the drone is, not where it was a frame ago — the cheap trick
            that lets a ~10 Hz vision loop carry respectable gains without
            oscillating. Scaled by proven effectiveness (an unproven or stuck
            drone gets no lead — its commands demonstrably do not move it) and
            forced to zero while coasting (a coasted pose is ALREADY propagated
            by the commands; leading it would double-count them). 0 disables.
        std_ref_m: The provider's ``pos_std_m`` at/below which the pose counts as
            crisp (m). A healthy multi-tag fix sits near 0.01–0.05.
        std_deadband_gain: Extra tracking deadband per metre of ``pos_std_m``
            above ``std_ref_m`` (m/m). This is the honesty dial: when the provider
            itself says this pose is only good to +-20 cm, a 3 cm cross-track
            "error" is measurement noise, and correcting it — or letting the
            integrators learn drift from it — is chasing fiction. The deadband
            widens with the reported error and tightens back as the fix sharpens.
        deadband_extra_max_m: Cap on that extra deadband (m), so a coasting pose
            (std up to 0.5) cannot open the deadband so far the drone stops
            correcting at all.
        yaw_scale_floor: Floor on the speed scale as applied to the YAW axis
            (0..1). The confidence ramp exists because flying fast on a vague
            pose grows the position error -- but rotating in place carries no
            position risk, and on this platform it is the cure: sweeping the
            camera is what brings tags back into view and sharpens the pose.
            Measured on the deployed airframe, halving the yaw counts (the
            default scaling at conf ~0.3) left a 90-degree turn undeliverable
            for 20 s. 0 keeps the old behaviour (yaw scaled like translation);
            1 gives yaw full authority whenever the drone may move at all.
            A hold still zeroes every axis, yaw included.
    """

    conf_full: float = 0.35
    conf_min: float = 0.10
    speed_floor: float = 0.35
    gain_floor: float = 0.25
    conf_integrate: float = 0.18
    conf_hold: float = 0.05
    max_age_s: float = 0.6
    coast_speed_scale: float = 0.5
    eff_floor: float = 0.15
    eff_full: float = 0.60
    eff_speed_floor: float = 0.5
    latency_s: float = 0.12
    std_ref_m: float = 0.05
    std_deadband_gain: float = 0.6
    deadband_extra_max_m: float = 0.15
    yaw_scale_floor: float = 0.0

    def __post_init__(self):
        # type: () -> None
        """Validate the ordering the schedule depends on."""
        if self.conf_min >= self.conf_full:
            raise ValueError("ConfidenceParams.conf_min must be below conf_full")
        if self.conf_hold > self.conf_integrate:
            raise ValueError("ConfidenceParams.conf_hold must not exceed "
                             "conf_integrate (the controller would hold while "
                             "still being told to learn drift)")
        if self.eff_floor >= self.eff_full:
            raise ValueError("ConfidenceParams.eff_floor must be below eff_full")
        for name in ("speed_floor", "gain_floor", "coast_speed_scale",
                     "eff_speed_floor", "yaw_scale_floor"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError("ConfidenceParams." + name + " must be in [0, 1]")
        if self.max_age_s <= 0.0:
            raise ValueError("ConfidenceParams.max_age_s must be > 0")
        for name in ("latency_s", "std_ref_m", "std_deadband_gain",
                     "deadband_extra_max_m"):
            if getattr(self, name) < 0.0:
                raise ValueError("ConfidenceParams." + name + " must be >= 0")
        if self.latency_s >= self.max_age_s:
            raise ValueError(
                "ConfidenceParams.latency_s (%.2f) must stay below max_age_s "
                "(%.2f) -- a lead longer than the staleness limit would steer on "
                "pure extrapolation" % (self.latency_s, self.max_age_s))


@dataclass(frozen=True)
class ControlAuthority:
    """What the controller may do this tick, given the pose it has.

    Attributes:
        speed_scale: Multiplier on every per-axis speed cap (0..1). Folds
            together the confidence ramp, the coast slow-down and the
            earned-speed (proven effectiveness) factor.
        yaw_speed_scale: Multiplier applied to the YAW cap specifically (0..1).
            ``max(speed_scale, yaw_scale_floor)`` — never below the translation
            scale, optionally pinned high so a vague pose slows the flying but
            not the turning (turning is what un-vagues the pose). 0 on a hold.
        gain_scale: Multiplier on the P and D terms (0..1). The integral is never
            scaled — see :class:`~.pid.AxisPid`.
        integrate: False freezes every integrator this tick.
        hold: True means stop: the pose is not good enough to fly on at all.
        lead_s: Seconds to advance the steering pose along the last commanded
            velocity, compensating the vision loop's transport delay. 0 while
            coasting or while the commands are unproven. NEVER applied to the
            blockage detector, which must see the raw pose — a stuck drone whose
            pose is advanced by its own commands would look like it is moving.
        deadband_extra_m: Extra tracking deadband this tick (m), from the
            provider's own ``pos_std_m``: errors smaller than the pose's stated
            accuracy are noise, not facts.
        yaw_deadband_extra_rad: Extra heading deadband this tick (rad), derived
            from confidence via the provider's own yaw-std law
            (``0.02 + 0.20*(1-conf)^2``) — yaw std is not published, but its
            formula is known, so the heading loop gets the same honesty.
        reason: Short human-readable explanation, for narration and logs.
    """

    speed_scale: float = 1.0
    yaw_speed_scale: float = 1.0
    gain_scale: float = 1.0
    integrate: bool = True
    hold: bool = False
    lead_s: float = 0.0
    deadband_extra_m: float = 0.0
    yaw_deadband_extra_rad: float = 0.0
    reason: str = "localization healthy"


class ConfidenceScheduler:
    """Turns a :class:`LocalizationQuality` into a :class:`ControlAuthority`.

    Stateless by design: every tick's decision depends only on that tick's
    quality snapshot, so there is no hysteresis to reason about here. The one
    place hysteresis genuinely belongs — deciding the drone is lost and handing it
    to the recovery — already lives in ``lost_localization_node``, and duplicating
    it would give the drone two authorities fighting over the same decision.
    """

    def __init__(self, params=None):
        # type: (Optional[ConfidenceParams]) -> None
        self.params = params or ConfidenceParams()

    def evaluate(self, quality):
        # type: (LocalizationQuality) -> ControlAuthority
        """Decide this tick's control authority.

        Args:
            quality: The latest localization quality snapshot.

        Returns:
            The speed/gain scaling, the latency lead, the noise-honest deadband
            widening, whether to learn drift, and whether to hold.
        """
        p = self.params
        if not quality.valid:
            return ControlAuthority(speed_scale=0.0, yaw_speed_scale=0.0,
                                    gain_scale=0.0, integrate=False, hold=True,
                                    reason="no localization yet")
        if quality.age_s > p.max_age_s:
            return ControlAuthority(
                speed_scale=0.0, yaw_speed_scale=0.0, gain_scale=0.0,
                integrate=False, hold=True,
                reason="pose is %.2fs old (limit %.2fs)" % (quality.age_s,
                                                            p.max_age_s))
        if quality.confidence < p.conf_hold:
            return ControlAuthority(
                speed_scale=0.0, yaw_speed_scale=0.0, gain_scale=0.0,
                integrate=False, hold=True,
                reason="localization confidence %.2f below the hold floor %.2f"
                       % (quality.confidence, p.conf_hold))

        frac = _ramp(quality.confidence, p.conf_min, p.conf_full)
        speed = p.speed_floor + (1.0 - p.speed_floor) * frac
        gain = p.gain_floor + (1.0 - p.gain_floor) * frac

        # Earned speed: the effectiveness EMA starts unproven (0.3) and is only
        # meaningful while commanding, so this throttles the first metres of a
        # flight and any stretch where the world stops honouring the commands.
        eff_frac = _ramp(quality.cmd_effectiveness, p.eff_floor, p.eff_full)
        speed *= p.eff_speed_floor + (1.0 - p.eff_speed_floor) * eff_frac

        # Latency lead: only to the extent the commands provably move the drone,
        # and never while coasting (a coasted pose is already command-propagated;
        # leading it would count the same commands twice).
        lead = 0.0 if quality.coasting else p.latency_s * eff_frac

        # Noise-honest deadbands, from the provider's own error estimates.
        extra = p.std_deadband_gain * (quality.pos_std_m - p.std_ref_m)
        extra = min(p.deadband_extra_max_m, max(0.0, extra))
        yaw_std = 0.02 + 0.20 * (1.0 - quality.confidence) ** 2
        yaw_extra = min(0.10, p.std_deadband_gain * max(0.0, yaw_std - 0.05))

        integrate = (quality.confidence >= p.conf_integrate
                     and not quality.coasting)
        if quality.coasting:
            speed *= p.coast_speed_scale
            reason = "coasting on dead reckoning -- slowed, drift learning frozen"
        elif not integrate:
            reason = ("confidence %.2f below the learning floor %.2f -- drift "
                      "learning frozen" % (quality.confidence, p.conf_integrate))
        elif eff_frac < 1.0 and quality.cmd_effectiveness < p.eff_full:
            reason = ("commands %d%% proven -- flying at %d%% speed until the "
                      "world confirms them" % (
                          int(round(quality.cmd_effectiveness * 100.0)),
                          int(round(speed * 100.0))))
        elif frac < 1.0:
            reason = "localization confidence %.2f -- flying at %d%% speed" % (
                quality.confidence, int(round(speed * 100.0)))
        else:
            reason = "localization healthy"
        # Yaw is scheduled separately: a vague pose is a reason to FLY slower,
        # not to TURN slower -- turning is what brings tags back into view.
        yaw_speed = max(speed, p.yaw_scale_floor)
        return ControlAuthority(speed_scale=speed, yaw_speed_scale=yaw_speed,
                                gain_scale=gain,
                                integrate=integrate, hold=False, lead_s=lead,
                                deadband_extra_m=extra,
                                yaw_deadband_extra_rad=yaw_extra, reason=reason)
