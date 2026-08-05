"""Lead the nose into the turn while the body keeps flying the leg.

The manoeuvre this module exists to produce, in the words of the pilot who flew
it: down a corridor, look forward and fly forward. Coming up on the right turn
at the end, start easing the nose right *early* — and because the nose is no
longer pointing where the drone is going, hold left ROLL to keep the body on the
corridor. The closer the corner, the further round the nose is and the more of
the progress vector is roll rather than pitch. At the corner the nose is fully
into the new corridor and the last stretch is flown on roll alone. Then the
corner retires, the new leg *is* where the nose already points, and the drone
simply flies forward out of the turn.

What that replaces: arriving at the corner pointing the old way, stopping, and
rotating in place — which is the single worst thing this airframe can be asked
to do. A yaw with no translation under it delivers about 11% of the commanded
rate (measured 2026-07-21) and standing still is where the drone drifts most.

This module owns only the *heading schedule*: how far round the nose should be
right now. Three separate mechanisms keep that schedule honest:

  * **A blend over distance, not time.** The lead is a function of how far the
    corner still is along the path, so it is unchanged by flying slower, being
    held, or losing a pose frame — all of which a time ramp would corrupt.
  * **A world-frame target, slewed.** The state is the absolute heading the nose
    is being walked toward, not the offset from the leg. When the corner retires
    the leg heading jumps by the turn and the desired lead drops by the same
    amount, so the absolute target does not move at all and the drone comes out
    of the turn without a kick.
  * **A catch-up guard.** The lead only advances while the nose is actually
    keeping up with it (within ``catchup_rad``). A drone whose yaw is saturated,
    throttled by a poor pose, or fighting a wall cannot be walked further and
    further off its heading by a schedule that does not know it — and, because
    the lead can never open a heading error larger than the guard, the
    anticipation can never trip the controller's own "I am badly mis-pointed,
    stop and turn" latch. The manoeuvre degrades into the old behaviour instead
    of into a stall.

Angles are in the path frame, REP-103 (``+`` = left / counter-clockwise).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import degrees, pi, radians
from typing import Optional, Sequence, Tuple

from sparx_agency.core.common.types import normalize_angle

from .corners import Corner, find_corner

XY = Tuple[float, float]

#: Returned by :func:`approach_limit` when nothing is capping the approach.
_NO_LIMIT = 1e9


def _clamp(value, limit):
    # type: (float, float) -> float
    """Clamp ``value`` to ``[-limit, limit]``."""
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


def _clamp_to_nose(offset, leg_heading, yaw, band):
    # type: (float, float, float, float) -> float
    """Shrink a carried lead until it sits no further than ``band`` ahead of the nose.

    **Shrink only, and never past zero.** The guard exists to stop a lead the
    schedule is *carrying* from opening a large heading error when the reference
    moves under it; it must never work the other way and manufacture a lead to
    make an error look small. Without that restriction a drone knocked 90
    degrees off its heading inside the corner window — a gust, a pose jump —
    has the whole error rewritten as "schedule lead": the yaw loop is shown 12
    degrees instead of 90, the stop-and-turn latch never engages, the rotation
    is capped at the tracking rate instead of the approach rate, and the drone
    keeps translating while pointed at a wall. Shrinking cannot do that, because
    a lead of zero is always an available answer and is the one a drone with no
    anticipation running would have.
    """
    residual = normalize_angle(leg_heading + offset - yaw)
    if abs(residual) <= band:
        return offset
    wanted = normalize_angle(offset - residual
                             + (band if residual > 0.0 else -band))
    if offset >= 0.0:
        return min(offset, max(0.0, wanted))
    return max(offset, min(0.0, wanted))


@dataclass(frozen=True)
class YawLookaheadParams:
    """Tuning for :class:`YawLookahead` (SI, angles in radians).

    Attributes:
        enabled: Master switch. False (the default) leaves the controller
            flying exactly as it did before this module existed — no corner
            search, no lead, no crab, and the classic body-frame allocation.
            This is a change to how the drone behaves at every corner, so it is
            opt-in until it has been flown.
        start_m: Arc distance to the corner at which the nose starts easing
            round (m). The headline dial, and the one real trade in the
            manoeuvre: bigger anticipates earlier and more gently, but flies
            more of the leg crabbed — and a crab is capped by the weak lateral
            axis, so it is slower than a cruise. Smaller keeps the drone
            pointed where it is going for longer and demands a brisker
            rotation at the end. Too small and it is not achievable at all;
            :func:`approach_limit` then slows the drone until it is, which
            costs the time back. Scaled by how sharp the corner is — a
            90-degree turn gets the full distance, a 30-degree one
            proportionally less, because a gentle bend needs no run-up.
        align_m: Arc distance at which the nose is fully round (m). The last
            stretch into the corner is then flown with the nose on the new leg
            and the body still on the old one — pure roll, the crab that ends
            the manoeuvre. Keep at or a little above the waypoint capture
            radius: a value below it is never reached, because the corner
            retires first.
        corner_rad: Smallest heading change that counts as a turn rather than
            as route noise (rad). Below it the route is "straight enough" and
            nothing is anticipated. A grid A* route weaves by ~10 degrees on a
            straight corridor, so this must sit clear of that.
        confirm_m: How much path past the corner establishes the outgoing
            heading (m). Also the guard against looking too far: the run stops
            at the next direction change, so a turn-then-turn is anticipated
            one turn at a time.
        max_offset_rad: Hard cap on how far the nose may lead the direction of
            travel (rad). 90 degrees is the absolute ceiling — at it the drone
            flies exactly sideways, and past it backwards and blind — but the
            *useful* limit is lower and it is set by the airframe, not by
            geometry: a crab at 90 degrees has no forward speed left, and this
            drone barely rotates without one (~11% yaw delivery standing still
            against 30-68% while translating). Lead the nose all the way and
            the manoeuvre eats itself — the last of the rotation is the part
            the drone can no longer perform. The default (70 degrees) still
            leaves a third of the progress vector pointing forward, which is
            what holds the yaw authority that brings the nose round; whatever
            is left of the turn is finished at the corner *while moving onto
            the new leg*, which is the strong regime, not the standing one. It
            was chosen by sweeping it against `start_m` in
            ``tasks/planning/turn_anticipation_rig``, not by reasoning.
        catchup_rad: The lead only advances while the heading error is inside
            this (rad). See the module docstring: this is what keeps the
            schedule from walking away from a drone that cannot follow it, and
            what guarantees the anticipation never trips the stop-and-turn
            latch. Keep it comfortably below ``DriftPidParams.yaw_engage_rad``.
        rate: Fastest the schedule may rotate the heading target (rad/s). It
            bounds the discontinuities geometry does not produce (a replanned
            route, a corner appearing mid-leg), and it is the rate
            :func:`approach_limit` budgets the approach against, so it must
            not exceed ``DriftPidParams.track_yaw_rate`` — a schedule that
            asks for more rotation than the controller is allowed to command
            mid-leg would fall behind by construction, every time.
        side_cone_rad: Sideslip cone allowed while the crab is running (rad),
            or 0 for none. This deliberately *overrides*
            ``DriftPidParams.turn_side_cone_rad``, which caps lateral speed at
            ``tan(cone) * vx`` while the yaw is active and would therefore
            forbid exactly the manoeuvre this module exists to fly (at a full
            90-degree lead ``vx`` is zero, so any cone at all cancels the
            roll). The cone was measured for an unplanned station-keeping
            correction riding a turn; here the roll IS the manoeuvre and the
            yaw is the slow, planned part. Backward travel stays forbidden
            either way.
        feedforward: Fraction of the target's own rate of change fed straight
            to the yaw axis (0..1). Without it the heading PID must hold a
            standing error to produce the rotation rate the schedule asks for,
            so the nose sits permanently a few degrees behind the plan. 0
            disables it and leaves the loop purely feedback.
    """

    enabled: bool = False
    start_m: float = 2.50
    align_m: float = 0.35
    corner_rad: float = radians(25.0)
    confirm_m: float = 1.00
    max_offset_rad: float = radians(70.0)
    catchup_rad: float = radians(12.0)
    rate: float = 0.25
    side_cone_rad: float = 0.0
    feedforward: float = 1.0

    def __post_init__(self):
        # type: () -> None
        """Validate the invariants the schedule relies on."""
        for name in ("start_m", "corner_rad", "confirm_m", "max_offset_rad",
                     "catchup_rad", "rate"):
            if getattr(self, name) <= 0.0:
                raise ValueError("YawLookaheadParams." + name + " must be > 0")
        if self.align_m < 0.0:
            raise ValueError("YawLookaheadParams.align_m must be >= 0")
        if self.align_m >= self.start_m:
            raise ValueError(
                "YawLookaheadParams.align_m (%.2f) must be below start_m "
                "(%.2f) -- the lead ramps from start_m in to align_m, and an "
                "empty span is a step change in the heading target"
                % (self.align_m, self.start_m))
        if self.corner_rad >= pi:
            raise ValueError("YawLookaheadParams.corner_rad must be below 180 "
                             "degrees")
        if self.max_offset_rad > pi / 2.0:
            raise ValueError(
                "YawLookaheadParams.max_offset_rad (%.1f deg) exceeds 90 "
                "degrees -- past that the drone would be crabbing BACKWARD, "
                "into space no camera on this airframe has seen"
                % degrees(self.max_offset_rad))
        if self.side_cone_rad < 0.0 or self.side_cone_rad >= pi / 2.0:
            raise ValueError(
                "YawLookaheadParams.side_cone_rad must be 0 (no cap) or in "
                "(0, 90deg) -- it is the half-angle of a sideslip cone")
        if not 0.0 <= self.feedforward <= 1.0:
            raise ValueError("YawLookaheadParams.feedforward must be in [0, 1]")


@dataclass(frozen=True)
class YawLead:
    """How far the nose is being led this tick, and what it costs the loop.

    Attributes:
        offset_rad: Signed lead of the nose over the direction of the leg (rad,
            + = nose to the left of the leg). 0 on a straight run, growing to
            the whole turn at the corner.
        rate_hint: Feed-forward yaw rate the schedule is asking for (rad/s).
        corner_distance_m: Arc distance to the corner being anticipated (m), or
            -1 when there is none.
        corner_index: Index of that corner in the active path, or -1. The
            heading carrot is clamped there.
        turn_rad: Signed total turn at that corner (rad), for narration.
    """

    offset_rad: float = 0.0
    rate_hint: float = 0.0
    corner_distance_m: float = -1.0
    corner_index: int = -1
    turn_rad: float = 0.0


def blend_fraction(distance_m, turn_rad, params):
    # type: (float, float, YawLookaheadParams) -> float
    """How much of the turn the nose should already have taken (0..1).

    A smoothstep from 0 at the start of the anticipation to 1 at ``align_m``,
    so the rotation eases in and eases out rather than switching on. The start
    distance is scaled by the sharpness of the corner (full at 90 degrees and
    beyond): a gentle bend does not need a long run-up, and giving it one only
    means flying more of a straight corridor sideways.

    Args:
        distance_m: Arc distance from the drone to the corner (m).
        turn_rad: Signed turn at that corner (rad).
        params: The schedule's tuning.

    Returns:
        The blend fraction, clamped to [0, 1].
    """
    sharpness = min(1.0, abs(turn_rad) / (pi / 2.0))
    start = params.align_m + (params.start_m - params.align_m) * sharpness
    if distance_m <= params.align_m:
        return 1.0
    if distance_m >= start:
        return 0.0
    span = start - params.align_m
    if span <= 1e-9:
        return 1.0
    x = (start - distance_m) / span
    return x * x * (3.0 - 2.0 * x)


def approach_limit(lead, params, floor):
    # type: (YawLead, YawLookaheadParams, float) -> float
    """Fastest approach that still gets the nose round in the distance left.

    A pilot slows into a turn, and this is why: the nose can only be brought
    round at a bounded rate, so arriving at a corner faster than
    ``rate * distance_left / turn_left`` means arriving pointed the wrong way —
    and then stopping to rotate, which is the whole thing the anticipation
    exists to avoid. Expressing the limit on what is *left* rather than on the
    nominal schedule closes the loop: a nose that has fallen behind (a throttled
    yaw, a vague pose, a short leg between two corners) slows the drone until it
    catches up, and one that is on schedule costs no speed at all.

    Args:
        lead: The lead in force this tick.
        params: The schedule's tuning.
        floor: Speed the limit may never fall below (m/s) — a drone crawling to
            a noisy stop just short of a corner is worse than one that arrives
            a few degrees under-aligned and finishes the turn there.

    Returns:
        The speed cap (m/s), or a very large number when there is nothing left
        to turn for.
    """
    if lead.corner_index < 0:
        return _NO_LIMIT
    remaining_turn = abs(normalize_angle(lead.turn_rad - lead.offset_rad))
    if remaining_turn < 1e-3:
        return _NO_LIMIT
    remaining_dist = lead.corner_distance_m - params.align_m
    if remaining_dist <= 0.0:
        return floor
    return max(floor, params.rate * remaining_dist / remaining_turn)


class YawLookahead:
    """Stateful schedule that walks the heading target into the next corner."""

    def __init__(self, params=None):
        # type: (Optional[YawLookaheadParams]) -> None
        self.params = params or YawLookaheadParams()
        self.reset()

    def reset(self):
        # type: () -> None
        """Forget the heading target (call on a fresh path or a full reset)."""
        self._target = None        # type: Optional[float]

    @property
    def enabled(self):
        # type: () -> bool
        """True when the manoeuvre may run at all."""
        return self.params.enabled

    def find(self, path, wp_idx, px, py):
        # type: (Sequence[XY], int, float, float) -> Optional[Corner]
        """The corner being anticipated, or None. Pure; safe to call every tick.

        Separate from :meth:`update` because the controller needs the corner
        *before* it can work out the direction of travel: the heading carrot is
        clamped at the corner, and the carrot is what the lead is measured
        from.
        """
        if not self.params.enabled:
            return None
        # A run shorter than align_m is not a leg to line up with: align_m IS
        # the distance the crab is flown over, so a "corner" the route holds for
        # less than that is a jog, and turning the nose into it would only have
        # to be undone.
        return find_corner(path, wp_idx, px, py, self.params.start_m,
                           self.params.corner_rad, self.params.confirm_m,
                           self.params.align_m)

    def update(self, corner, leg_heading, yaw, dt):
        # type: (Optional[Corner], float, float, float) -> YawLead
        """Advance the heading schedule one tick.

        Args:
            corner: The corner from :meth:`find`, or None when the route runs
                straight for as far as the schedule looks.
            leg_heading: Direction of the leg being flown, in the path frame
                (rad). The lead is measured from the LEG rather than from the
                carrot: the carrot swings with the cross-track error, and a
                heading schedule that swung with it would turn every metre of
                drift into a rotation.
            yaw: Current heading in the path frame (rad).
            dt: Seconds since the previous call.

        Returns:
            The lead to apply this tick.
        """
        if not self.params.enabled:
            self._target = None
            return YawLead()
        if dt <= 0.0:
            raise ValueError("YawLookahead.update: dt must be > 0")
        p = self.params

        if corner is None:
            # Nothing to anticipate. The lead is given back at once, with no
            # rate limit and no guard: releasing the manoeuvre must never be
            # slower than the drone, and a heading error the schedule is not
            # asking for is the honest one the yaw loop -- and the stop-and-turn
            # latch behind it -- has to be allowed to see at full authority.
            self._target = normalize_angle(leg_heading)
            return YawLead()

        desired = blend_fraction(corner.distance_m, corner.turn_rad,
                                 p) * corner.turn_rad
        if desired > p.max_offset_rad:
            desired = p.max_offset_rad
        elif desired < -p.max_offset_rad:
            desired = -p.max_offset_rad

        # The lead is carried as an absolute heading and re-read against the leg
        # every tick. Two things fall out of that: a republished route (the
        # FALCON planner sends one several times a second) changes nothing,
        # because the nose is still where the nose is; and the tick a corner
        # retires -- where the leg heading jumps by the whole turn and the
        # schedule's answer drops by exactly the same amount -- moves the nose's
        # actual setpoint not at all.
        offset = (0.0 if self._target is None
                  else normalize_angle(self._target - leg_heading))
        # Re-anchoring can hand back an offset the schedule would never have
        # chosen, because the reference itself can move under it: a waypoint
        # index flickering on its capture radius, or a replanned route, swings
        # the leg heading by the whole turn between one tick and the next. Both
        # bounds are therefore reasserted on the carried value before it is used
        # -- never lead further than the cap, and (the load-bearing one) never
        # sit further ahead of the nose than the catch-up band. That second
        # clamp is what makes "the anticipation cannot open a heading error big
        # enough to trip the stop-and-turn latch" true absolutely, and not just
        # true of the tick-by-tick growth below.
        offset = _clamp(offset, p.max_offset_rad)
        offset = _clamp_to_nose(offset, leg_heading, yaw, p.catchup_rad)

        delta = normalize_angle(desired - offset)
        step = p.rate * dt
        if delta > step:
            delta = step
        elif delta < -step:
            delta = -step
        # Catch-up guard: the schedule may not walk away from the nose it is
        # steering, so the lead may only move as far as leaves the heading error
        # inside the band. Note this gates the LEAD, never the heading error
        # itself -- a drone that is genuinely mis-pointed still shows the yaw
        # loop the full error and still gets the full turn authority for it.
        residual = normalize_angle(leg_heading + offset - yaw)
        if delta > 0.0:
            delta = min(delta, max(0.0, p.catchup_rad - residual))
        elif delta < 0.0:
            delta = max(delta, min(0.0, -p.catchup_rad - residual))
        offset = _clamp(normalize_angle(offset + delta), p.max_offset_rad)
        self._target = normalize_angle(leg_heading + offset)
        return YawLead(offset_rad=offset,
                       rate_hint=p.feedforward * delta / dt,
                       corner_distance_m=corner.distance_m,
                       corner_index=corner.index, turn_rad=corner.turn_rad)
