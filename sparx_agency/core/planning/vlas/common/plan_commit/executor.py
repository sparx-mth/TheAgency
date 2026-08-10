"""Fly the prediction you have before asking for another one.

A navigation policy will answer as fast as it is asked, and asking it every
frame is the obvious way to use it -- it is also how it was trained and
evaluated, one frame at a time, each prediction scored on its own. In the air
that turns into a pathology: at 3 Hz and 1 m/s the aircraft covers a third of a
metre before the plan it is flying is thrown away and replaced, so it only ever
executes the first segment of anything. The policy's route shape never happens.
Whatever bias lives in that first segment is the only thing that ever compounds.

The fix is a commitment. One inference, anchored where it was made
(:mod:`~sparx_agency.core.planning.vlas.common.plan_commit.committed_plan`),
flown as a route until roughly half of it is behind the aircraft, and only then
replaced. Half, because a learned trajectory is most trustworthy near the
observation it came from: the near end is metres the camera has actually seen,
the far end is extrapolation, and by the time the aircraft gets there the world
has moved on. Committing to the near half buys real progress at the horizon
where the prediction is worth the most.

Three things end a commitment early, so it cannot become a trap: a prediction
that barely moves is not flown at all, one the aircraft has stopped tracking is
abandoned, and one that is taking too long is abandoned.
:attr:`CommitSpec.min_period_s` is the fourth knob and works the other way -- it
holds a reason back rather than raising one, so a fast server cannot reintroduce
per-frame inference through any of the three.

ROS-free, numpy-only, Python 3.8 idioms -- the FALCON Noetic adapter imports it.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite
from typing import Optional, Sequence, Tuple

import numpy as np

from sparx_agency.core.planning.vlas.common.plan_commit.committed_plan import (
    CommittedPlan,
    anchor_plan,
)

NO_PLAN = "no plan"
FLOWN = "commitment flown"
TOO_SHORT = "prediction too short to fly"
OFF_ROUTE = "aircraft is off the committed route"
EXPIRED = "commitment took too long"


@dataclass
class CommitSpec:
    """How much of a prediction to fly, and when to give up on it.

    Attributes:
        fraction: Share of the predicted waypoints to commit to. ``0.5`` on 16
            waypoints commits through waypoint 8.
        lookahead_m: Pure-pursuit lookahead along the committed route.
        arrive_radius_m: Close enough to the commit point to call it reached.
            Catches the corner-cutting case, where the aircraft passes inside
            the commit point and its projected arc never quite reaches it.
            Proximity alone is not enough: arrival also requires the commitment
            to be all but flown in arc terms, within this same distance. A long
            route whose commit *point* merely happens to lie near the aircraft
            -- a loop, or a corridor entered and reversed out of within the
            committed half -- would otherwise be "arrived at" from a standing
            start, which is the per-frame inference this package exists to stop.
            This doubles as the arc-length slack, so it must stay larger than
            the shortfall a genuine corner cut leaves behind.
        min_commit_m: A commitment shorter than this is not worth flying -- the
            policy has predicted a near-stop -- so ask again instead of crawling
            to a halt. Do not set it to zero: a degenerate prediction would then
            be "flown" instantly and every tick would re-infer.
        max_commit_s: Give up on a commitment that has taken this long. The
            aircraft may be yawing in place, blocked, or fighting wind; whatever
            the reason, a stale observation is no longer worth flying.
        max_deviation_m: Give up on a commitment the aircraft is no longer
            tracking. Being this far off the route means the route is not what
            is being flown, so it should not be what decides when to re-plan.
        min_period_s: Floor between inferences, seconds. This is a rate
            *ceiling* on the policy, not a schedule: the commitment decides
            when, this only stops "when" from being "every tick".
    """

    fraction: float = 0.5
    lookahead_m: float = 1.2
    arrive_radius_m: float = 0.30
    min_commit_m: float = 0.40
    max_commit_s: float = 8.0
    max_deviation_m: float = 2.0
    min_period_s: float = 0.33

    def __post_init__(self):
        """Reject a spec that silently restores per-frame inference.

        ``fraction <= 0`` clamps to a one-waypoint commitment about 0.2 m long,
        which is below ``min_commit_m`` -- so every plan is TOO_SHORT, every tick
        re-infers at the rate floor, and the package quietly does nothing. That
        is the exact behaviour it exists to prevent, so it is worth a crash
        rather than a puzzling flight.
        """
        if not 0.0 < self.fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]; got %r" % (self.fraction,))
        for name in ("lookahead_m", "min_commit_m", "max_commit_s", "min_period_s"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError("%s must be positive; got %r"
                                 % (name, getattr(self, name)))
        for name in ("arrive_radius_m", "max_deviation_m"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError("%s cannot be negative; got %r"
                                 % (name, getattr(self, name)))
        # An arrival radius at or above the shortest legal commitment means the
        # shortest commitments are "arrived at" before they are flown, which is
        # per-frame inference again -- the same class of misconfiguration that
        # `fraction <= 0` is rejected for, and just as quiet in the air.
        if float(self.arrive_radius_m) >= float(self.min_commit_m):
            raise ValueError(
                "arrive_radius_m (%r) must be smaller than min_commit_m (%r), or "
                "the shortest legal commitment is reached before it is flown"
                % (self.arrive_radius_m, self.min_commit_m))


@dataclass
class CommitTick:
    """What the executor says on one control step.

    Attributes:
        target: The point to fly at, world frame, or ``None`` when nothing is
            committed yet and the caller should hold station.
        heading: Which way to point while flying there: the route's own tangent
            at the target, world radians. ``None`` where the route does not move
            (a stopped prediction has no direction) -- hold the last heading
            rather than inventing one. **Fly the target, look along this.** The
            bearing from the aircraft to the target is not the same thing and is
            the wrong answer on a turn: it cuts inside the arc, so the nose lags
            the route and the camera frames somewhere the aircraft is not going
            -- which then becomes the observation the next inference is made
            from. It is also the heading the expert labels encode (see
            ``to_navdp_label``), so it is what the policy was trained against.
        replan_reason: Why a new inference is due, or ``None`` to keep flying.
            A short phrase, meant to be logged or narrated as-is.
        arc_m: How far along the committed route the aircraft has got.
        commit_arc_m: How long the commitment is.
        lateral_m: How far the aircraft is from the route.
        fraction: ``arc_m / commit_arc_m``, clamped to ``0..1``.
    """

    target: Optional[Tuple[float, float]]
    heading: Optional[float]
    replan_reason: Optional[str]
    arc_m: float
    commit_arc_m: float
    lateral_m: float
    fraction: float


class PlanCommitExecutor(object):
    """Holds one commitment at a time and says when it has been flown.

    Typical use, once per control step::

        tick = executor.tick(x, y, now)
        if tick.replan_reason is not None:
            executor.mark_attempt(now)
            result = policy.step(...)           # may fail; the plan is kept
            if result is not None:
                executor.commit(trajectory, pose_at_capture, now)
                tick = executor.tick(x, y, now)   # the NEW plan's carrot
        if tick.target is not None:
            fly_at(tick.target)

    ``mark_attempt`` is deliberately separate from ``commit``: a dropped
    inference must still cost a period, or a server that is down is asked again
    every tick.
    """

    def __init__(self, spec: Optional[CommitSpec] = None) -> None:
        self.spec = spec or CommitSpec()
        self.plan = None                    # type: Optional[CommittedPlan]
        self.commitments = 0
        self._last_attempt_s = None         # type: Optional[float]
        self._peak_arc_m = 0.0
        self._segment = 0
        self._held = None                   # type: Optional[str]
        self._travelled_m = 0.0
        self._last_xy = None                # type: Optional[Tuple[float, float]]

    # ── lifecycle ────────────────────────────────────────────────────
    def reset(self) -> None:
        """Forget the commitment and the rate limit. Use between missions."""
        self.plan = None
        self.commitments = 0
        self._last_attempt_s = None
        self._peak_arc_m = 0.0
        self._segment = 0
        self._held = None
        self._travelled_m = 0.0
        self._last_xy = None

    def mark_attempt(self, now_s: float) -> None:
        """Record that the policy was asked, successfully or not.

        This is also what discharges a held replan reason -- being *asked* is
        acting on it, whatever the server then does.
        """
        self._last_attempt_s = float(now_s)
        self._held = None

    def commit(self, trajectory: np.ndarray, pose: Sequence[float],
               now_s: float) -> CommittedPlan:
        """Adopt a fresh prediction as the commitment.

        Args:
            trajectory: ``(T, >=2)`` body-frame waypoints from the policy.
            pose: ``(x, y, yaw)`` the frame behind that prediction was captured
                at -- not the live pose, which is one inference latency later.
            now_s: Clock at inference.

        Returns:
            The plan now being flown.
        """
        self.plan = anchor_plan(trajectory, pose, now_s, self.spec.fraction)
        self._peak_arc_m = 0.0
        self._segment = 0
        self._held = None
        # Distance is measured per commitment, not per mission: the new route is
        # anchored where this prediction was made, so what the aircraft covered
        # flying the previous one buys no credit against this one.
        self._travelled_m = 0.0
        self._last_xy = None
        self.commitments += 1
        return self.plan

    # ── per control step ─────────────────────────────────────────────
    def tick(self, x: float, y: float, now_s: float) -> CommitTick:
        """Advance progress along the commitment and decide what happens next.

        Progress only ever moves forward, twice over: the projection is refused
        any segment behind a cursor this advances, and the arc it returns is
        kept as a high-water mark. A route that comes back near itself would
        otherwise read as finished from a standing start, and one the aircraft
        is blown backwards along would un-finish itself.
        """
        # The plan is validated once per inference; the aircraft pose arrives on
        # every control step and is the likelier source of a NaN -- a diverged
        # estimator, a bad TF, an uninitialised Pose message. Unchecked, it
        # projects to a nan arc, argmin picks index 0, and the carrot comes back
        # (nan, nan) with replan_reason None: a non-finite setpoint the follower
        # is told to keep flying. Refuse it here rather than downstream.
        if not (isfinite(float(x)) and isfinite(float(y)) and isfinite(float(now_s))):
            raise ValueError("aircraft pose and clock must be finite; got "
                             "x=%r y=%r now_s=%r" % (x, y, now_s))
        if self.plan is None:
            return CommitTick(None, None, self._gate(NO_PLAN, now_s),
                              0.0, 0.0, 0.0, 0.0)

        if self._last_xy is not None:
            self._travelled_m += hypot(x - self._last_xy[0], y - self._last_xy[1])
        self._last_xy = (float(x), float(y))

        arc, lateral, segment = self.plan.progress(x, y, self._segment)
        self._segment = max(self._segment, segment)
        # Arc credit is capped by the distance actually covered. Projection is a
        # nearest-point search over a window of CURSOR_WINDOW segments, so on a
        # route that folds back within that window the aircraft can be credited
        # with arc it never flew: with a small `fraction` the commit point lands
        # inside the window, and a single projection from a few centimetres off
        # the anchor jumps straight to it -- FLOWN after 4 cm of a 1.03 m
        # commitment. Walking the cursor is only a defence when the commitment is
        # longer than the window. Distance flown cannot be manufactured that way,
        # and it bounds arc from above however the route is shaped. The arrival
        # radius is the tolerance: a corner cut covers slightly less ground than
        # the route it is tracking, and must still be able to complete.
        credit = min(arc, self._travelled_m + self.spec.arrive_radius_m)
        self._peak_arc_m = max(self._peak_arc_m, credit)
        commit_arc = self.plan.commit_arc_m
        cx, cy, heading = self.plan.carrot(x, y, self.spec.lookahead_m, self._segment)
        fraction = min(1.0, self._peak_arc_m / commit_arc) if commit_arc > 0 else 1.0
        return CommitTick((cx, cy), heading,
                          self._gate(self._reason(x, y, lateral, now_s), now_s),
                          self._peak_arc_m, commit_arc, lateral, fraction)

    # ── the decision ─────────────────────────────────────────────────
    def _reason(self, x: float, y: float, lateral: float,
                now_s: float) -> Optional[str]:
        """Why the commitment is over, or ``None`` while it still stands."""
        plan = self.plan
        commit_arc = plan.commit_arc_m
        if commit_arc < self.spec.min_commit_m:
            return TOO_SHORT
        if self._peak_arc_m >= commit_arc:
            return FLOWN
        commit_x, commit_y = plan.commit_point
        # Catches corner-cutting: the aircraft passes inside the commit point and
        # its projected arc never quite reaches the end.
        #
        # Proximity to the commit point alone is not enough, and the arc-length
        # bound below is what makes it safe. `commit_arc > arrive_radius_m` only
        # rules out a commitment *shorter* than the radius; a long route whose
        # commit **point** happens to sit near the anchor -- a loop, or a
        # corridor entered and reversed out of within the committed half -- is
        # within the radius from a standing start, and was declared flown having
        # flown nothing. That is the original re-infer-every-frame bug wearing a
        # different hat, so arrival also requires the commitment to be all but
        # complete in arc terms, which corner-cutting is and a loop is not.
        #
        # The arc allowance is the smaller of the radius and a quarter of the
        # commitment. A flat allowance is most of a short commitment: at the
        # shipped defaults a legal 0.40 m commitment (min_commit_m) with a
        # 0.30 m radius was declared FLOWN after 0.105 m -- 26 % of it -- because
        # `commit_arc - arrive_radius_m` left only 0.10 m to cover. Scaling the
        # allowance keeps corner-cutting working on the long commitments it was
        # written for while refusing to hand back three quarters of a short one.
        slack = min(self.spec.arrive_radius_m, 0.25 * commit_arc)
        if (commit_arc > self.spec.arrive_radius_m
                and self._peak_arc_m >= commit_arc - slack
                and hypot(x - commit_x, y - commit_y) <= self.spec.arrive_radius_m):
            return FLOWN
        if lateral > self.spec.max_deviation_m:
            return OFF_ROUTE
        if float(now_s) - plan.issued_s > self.spec.max_commit_s:
            return EXPIRED
        return None

    def _gate(self, reason: Optional[str], now_s: float) -> Optional[str]:
        """Hold back a reason that would ask the policy faster than the ceiling.

        Held back, not thrown away. FLOWN, TOO_SHORT and EXPIRED all re-derive
        the same answer on the next tick, but OFF_ROUTE does not: the aircraft
        can be blown wide of the route and back inside ``max_deviation_m``
        within one period, and a reason that only ever existed during the
        suppressed window would vanish with it -- leaving the aircraft flying a
        commitment it has already been observed to have lost.

        Cleared by :meth:`mark_attempt`, not by being returned: a caller that
        can only act on some ticks -- ``fly_navdp`` asks the policy on render
        steps but ticks at the physics rate -- would otherwise be told once, on
        a step where it cannot do anything, and never again.
        """
        if reason is not None:
            self._held = reason
        if self._held is None:
            return None
        if (self._last_attempt_s is not None
                and float(now_s) - self._last_attempt_s < self.spec.min_period_s):
            return None
        return self._held
