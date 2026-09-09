"""Turn the nose toward the demand before flying at it.

ROS-free so it can be unit-tested without a Noetic container, and Python 3.8
compatible because that is what the container runs. Imported by
``falcon_exploration_follower_node.py``; nothing here knows about ROS.

**The defect this exists for.** The tracker emits a *world* velocity. The
follower rotates it into the body frame and publishes it, so a demand pointing
behind the aircraft becomes a negative ``linear.x`` -- the aircraft flies
backward. On this airframe that is blind flight: the only camera faces forward
with a 135 deg horizontal field of view, so nothing behind the nose is in the
map the planner just used. Measured over one flight's 4598 airborne driving
ticks, backward flight occupied 9.5% of them and put the aircraft within 0.30 m
of mapped geometry on 33% of those ticks against 9% otherwise -- p10 clearance
0.00 m against 0.30 m. Backward flight is where this aircraft meets walls.

**Why a gate already existed and did not fire.** The follower has an align gate,
a cos-fade and a turn-creep floor, and all three live inside a branch guarded by
``not use_lateral``. The flight under analysis ran with ``use_lateral`` true, so
the entire block was skipped and the full velocity vector went to the wire
verbatim. A safety property must not be a side effect of a performance flag,
which is why this gate is unconditional.

**Why the threshold is 90 deg and not the camera's half-field.** Past 90 deg the
sign of the useful forward component flips: "forward" is genuinely the wrong
direction, and no amount of speed shaping fixes it. That makes 90 deg the one
threshold that needs no tuning argument. A tighter gate at the camera's 67.5 deg
half-field would also stop the aircraft translating into unsensed space, and is
available by parameter -- but on the recorded flight it would have engaged on
41% of ticks against 28%, and this repo has already measured a too-eager gate
(40 deg) stopping the aircraft dead at ordinary 45-57 deg mid-turn errors. That
measurement was taken on the *broken* system, where the nose was being aimed at
FALCON's camera yaw rather than along travel, so the honest position is that the
tighter value needs a flight to justify it and the sign change does not.

**Hysteresis is not optional.** A gate with one threshold chatters, and this
codebase has measured the cost twice: a tilt reflex without hysteresis fired
56-196 times in a single run, once every 3-7 seconds, each time zeroing
translation. The release threshold is well inside the engage threshold so a
turn completes rather than being re-triggered on its own overshoot.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

#: Flying the plan: translation is allowed.
TRACKING = "tracking"

#: The demand is behind the nose. Rotate toward it, translate afterwards.
TURNING = "turning"


@dataclass(frozen=True)
class HeadingGateParams:
    """When to stop translating and turn instead.

    Attributes:
        enabled: False restores the pre-gate behaviour completely -- no turn,
            and no reverse clamp either, so a demand pointing backward is
            published as backward flight. One knob, one behaviour: a half-off
            gate would be a configuration nobody reasoned about. Kept for
            comparison flights only; it is not a supported configuration.
        engage_deg: Demand angle above which translation stops and the aircraft
            turns. 90 is the sign change, past which forward thrust moves the
            aircraft away from where the plan wants it.
        resume_deg: Demand angle the turn must reach before translation is
            allowed again. Must be below ``engage_deg``; the gap is what stops
            the gate chattering on its own overshoot.
        min_speed_mps: Commanded speed below which the demand *direction* is
            meaningless and the gate does not act. The direction of a small
            vector is ill-conditioned -- measured, below 0.25 m/s half of all
            direction changes exceed a 45 deg/s slew ceiling -- and gating on
            noise would stop the aircraft at random.
        max_reverse_mps: Hard ceiling on commanded backward body speed, applied
            at the publish boundary as a backstop behind the gate. 0 forbids
            backward flight outright. Two motions are exempt, both because the
            aircraft has just been there and the ground is therefore known-clear:
            the escape reflex backing out of a nose-in contact, and a small
            settle onto a finished plan's endpoint (:meth:`settled`).
    """

    enabled: bool = True
    engage_deg: float = 90.0
    resume_deg: float = 60.0
    min_speed_mps: float = 0.25
    max_reverse_mps: float = 0.0

    def __post_init__(self):
        # type: () -> None
        """Validate the invariants the gate relies on."""
        if not 0.0 < self.engage_deg <= 180.0:
            raise ValueError("engage_deg must be in (0, 180], got %r" % (self.engage_deg,))
        if not 0.0 <= self.resume_deg < self.engage_deg:
            raise ValueError(
                "resume_deg must be in [0, engage_deg) -- without a gap the gate "
                "chatters on its own overshoot; got resume_deg=%r engage_deg=%r"
                % (self.resume_deg, self.engage_deg))
        if self.min_speed_mps < 0.0:
            raise ValueError("min_speed_mps must be >= 0, got %r" % (self.min_speed_mps,))
        if self.max_reverse_mps < 0.0:
            raise ValueError("max_reverse_mps must be >= 0, got %r"
                             % (self.max_reverse_mps,))


@dataclass(frozen=True)
class HeadingDecision:
    """What the gate decided this tick.

    Attributes:
        state: :data:`TRACKING` or :data:`TURNING`.
        allow_translation: Whether the caller may command body translation. False
            while turning -- past the engage angle there is no component of
            forward thrust that helps, and a creep would carry momentum the wrong
            way.
        demand_angle_rad: Signed angle from the nose to the commanded velocity,
            positive to the left. The quantity the gate acts on, reported so a
            replay can see why.
        engaged_now: True on the tick the turn started, for logging.
    """

    state: str
    allow_translation: bool
    demand_angle_rad: float
    engaged_now: bool = False

    @property
    def turning(self):
        # type: () -> bool
        """Whether the aircraft is turning in place this tick."""
        return self.state == TURNING


class HeadingGate:
    """Hysteretic turn-in-place gate on the angle from the nose to the demand.

    Stateful: which side of the hysteresis band the aircraft is on depends on
    what it was doing last tick, so step it once per control tick.

    Args:
        params: Thresholds. Defaults gate at the sign change with a 30 deg band.
    """

    def __init__(self, params=None):
        # type: (HeadingGateParams) -> None
        self.params = params or HeadingGateParams()
        self._turning = False

    def reset(self):
        # type: () -> None
        """Forget whether a turn was in progress."""
        self._turning = False

    def update(self, body_vx, body_vy):
        # type: (float, float) -> HeadingDecision
        """Decide whether to translate or turn, from the body-frame demand.

        Args:
            body_vx: Commanded forward velocity, m/s, before gating.
            body_vy: Commanded lateral velocity, m/s (positive left), before
                gating.

        Returns:
            The decision for this tick.
        """
        angle = math.atan2(float(body_vy), float(body_vx))
        speed = math.hypot(float(body_vx), float(body_vy))
        if not self.params.enabled or speed < self.params.min_speed_mps:
            # Too slow for the direction to mean anything, or deliberately off.
            # An in-progress turn is abandoned rather than latched on noise.
            self._turning = False
            return HeadingDecision(TRACKING, True, angle)

        threshold = self.params.resume_deg if self._turning else self.params.engage_deg
        was_turning = self._turning
        self._turning = abs(math.degrees(angle)) > threshold
        return HeadingDecision(
            TURNING if self._turning else TRACKING,
            not self._turning,
            angle,
            engaged_now=self._turning and not was_turning,
        )

    def settled(self, body_vx, body_vy):
        # type: (float, float) -> HeadingDecision
        """Pass a station-keeping nudge through ungated.

        For the one case the gate must not act on: a small correction onto the
        endpoint of a *finished* plan. The aircraft has just flown through that
        point, so the correction is not a travel demand and turning to face it
        would spend a 180 degree spin reclaiming centimetres. Any in-progress
        turn is abandoned -- there is nothing left to turn toward.

        Args:
            body_vx: Commanded forward velocity, m/s.
            body_vy: Commanded lateral velocity, m/s (positive left).

        Returns:
            A :data:`TRACKING` decision that allows translation.
        """
        self._turning = False
        return HeadingDecision(TRACKING, True,
                               math.atan2(float(body_vy), float(body_vx)))

    def limit_reverse(self, body_vx, allow_reverse=False):
        # type: (float, bool) -> float
        """Clamp commanded backward speed, as a backstop behind the gate.

        The gate stops translation while the demand is behind the nose, but the
        gate can be disabled, the demand can sit just inside the threshold, and
        the smoothed command can disagree with the raw one for a tick. This is
        the property -- "this airframe does not fly backward" -- enforced where
        the command leaves the node, so no path upstream can violate it.

        Args:
            body_vx: Forward velocity about to be published, m/s.
            allow_reverse: True for the two motions that are known-clear
                because the aircraft has just been there: the escape reflex
                backing out of a contact, and a small settle onto the endpoint
                of a finished plan (see :meth:`settled`).

        Returns:
            The forward velocity to publish.
        """
        if allow_reverse or not self.params.enabled:
            return float(body_vx)
        return max(-self.params.max_reverse_mps, float(body_vx))
