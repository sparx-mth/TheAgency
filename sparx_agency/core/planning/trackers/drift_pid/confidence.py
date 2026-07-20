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
        for name in ("speed_floor", "gain_floor", "coast_speed_scale"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError("ConfidenceParams." + name + " must be in [0, 1]")
        if self.max_age_s <= 0.0:
            raise ValueError("ConfidenceParams.max_age_s must be > 0")


@dataclass(frozen=True)
class ControlAuthority:
    """What the controller may do this tick, given the pose it has.

    Attributes:
        speed_scale: Multiplier on every per-axis speed cap (0..1).
        gain_scale: Multiplier on the P and D terms (0..1). The integral is never
            scaled — see :class:`~.pid.AxisPid`.
        integrate: False freezes every integrator this tick.
        hold: True means stop: the pose is not good enough to fly on at all.
        reason: Short human-readable explanation, for narration and logs.
    """

    speed_scale: float = 1.0
    gain_scale: float = 1.0
    integrate: bool = True
    hold: bool = False
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
            The speed/gain scaling, whether to learn drift, and whether to hold.
        """
        p = self.params
        if not quality.valid:
            return ControlAuthority(0.0, 0.0, False, True,
                                    "no localization yet")
        if quality.age_s > p.max_age_s:
            return ControlAuthority(
                0.0, 0.0, False, True,
                "pose is %.2fs old (limit %.2fs)" % (quality.age_s, p.max_age_s))
        if quality.confidence < p.conf_hold:
            return ControlAuthority(
                0.0, 0.0, False, True,
                "localization confidence %.2f below the hold floor %.2f"
                % (quality.confidence, p.conf_hold))

        frac = _ramp(quality.confidence, p.conf_min, p.conf_full)
        speed = p.speed_floor + (1.0 - p.speed_floor) * frac
        gain = p.gain_floor + (1.0 - p.gain_floor) * frac

        integrate = (quality.confidence >= p.conf_integrate
                     and not quality.coasting)
        if quality.coasting:
            speed *= p.coast_speed_scale
            reason = "coasting on dead reckoning -- slowed, drift learning frozen"
        elif not integrate:
            reason = ("confidence %.2f below the learning floor %.2f -- drift "
                      "learning frozen" % (quality.confidence, p.conf_integrate))
        elif frac < 1.0:
            reason = "localization confidence %.2f -- flying at %d%% speed" % (
                quality.confidence, int(round(speed * 100.0)))
        else:
            reason = "localization healthy"
        return ControlAuthority(speed, gain, integrate, False, reason)
