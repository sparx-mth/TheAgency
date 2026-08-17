"""Judge a follower's proximity reflexes against the plan's own clearance.

A trajectory follower carries reflexes the planner does not: a personal-space
bubble, a speed governor keyed on the nearest obstacle, a depth veto. Every one
of them is a constant distance, and a constant distance is the wrong shape for
this problem, because the amount of room a correctly flown aircraft has is a
property of the CORRIDOR, not of the airframe.

The consequence is not subtle. The same follower, unchanged, flies a 1.4 m
warehouse aisle with 0.70 m of half width -- comfortably outside every reflex --
and a 0.90 m hospital doorway with 0.45 m, which is inside all of them. In the
doorway the aircraft is vetoed, throttled to a crawl and eventually reversed
back out, not because anything is wrong but because it is exactly where a
planner holding the corridor's full available clearance put it. Tuning the
constants down fixes the hospital and re-opens the warehouse contacts the
constants were raised to stop; tuning them up does the reverse. There is no
single constant, which is the tell that a constant is the wrong object.

This class replaces the constant with a comparison. The planner's curve is
itself a clearance measurement -- the reference point's distance to the nearest
occupied voxel is the room the planner believed it had and chose to use -- so
the follower can ask the only question that transfers between worlds:

    **Is the aircraft closer to something than the plan is?**

If it is not, no proximity reflex may fire, whatever the absolute distance. The
planner had the whole map and an explicit clearance objective; the follower has
one depth frame and a bubble, and overruling the better-informed party for being
right is how a doorway becomes unflyable. If it IS closer, the reflexes fire in
proportion to the DEFICIT -- how much of the plan's own margin has been spent --
which is the same number in both worlds and needs no per-world tuning.

Underneath sits one absolute: ``hard_floor_m``, the distance at which the
airframe is about to touch something. Physics does not care what the planner
intended, so that floor is not relative to anything and is never relaxed.

Pure stdlib and Python 3.8: this runs inside the Noetic FALCON container.
"""
from __future__ import annotations


class ClearanceEnvelopeConfig(object):
    """Tunables for :class:`ClearanceEnvelope`.

    Attributes:
        hard_floor_m: Distance to the nearest occupied voxel CENTRE at which
            the aircraft must stop, whatever the plan wanted. The airframe
            radius plus half a voxel plus a little: below this the next command
            is a contact, and no planner intent overrides that.
        tolerance_m: How much closer than the reference the aircraft may be and
            still count as flying the plan. This is tracking noise, not margin:
            a follower holding a curve to a few centimetres crosses zero
            deficit constantly, and a reflex chattering at that boundary is
            worse than one that never fires.
        deficit_span_m: The deficit at which the speed allowance reaches
            ``floor_speed``. Sets how sharply the aircraft is slowed as it eats
            into the plan's margin. Not a clearance and not world specific: it
            is the size of the error being corrected.
        breach_deficit_m: Deficit past which the aircraft counts as having
            breached its budget -- a near contact the follower should retreat
            from rather than merely slow for.
        floor_speed: The slowest the governor will ask for rather than
            stopping. A crawl still makes progress and still maps; a stop in a
            doorway hands the aircraft to a retreat that drives it back out.
        open_clearance_m: The clearance assumed when the plan's own is unknown
            -- no reference yet, or nothing occupied within the search radius.
            Open space: the envelope collapses to the absolute floor and the
            follower behaves as it always did.
    """

    def __init__(self,
                 hard_floor_m=0.30,
                 tolerance_m=0.10,
                 deficit_span_m=0.25,
                 breach_deficit_m=0.20,
                 floor_speed=0.10,
                 open_clearance_m=0.90):
        # type: (float, float, float, float, float, float) -> None
        self.hard_floor_m = float(hard_floor_m)
        self.tolerance_m = float(tolerance_m)
        self.deficit_span_m = max(1e-3, float(deficit_span_m))
        self.breach_deficit_m = float(breach_deficit_m)
        self.floor_speed = float(floor_speed)
        self.open_clearance_m = float(open_clearance_m)


class ClearanceBudget(object):
    """What the aircraft's current clearance entitles it to.

    Attributes:
        plan_clearance_m: Room the reference point has, metres. The planner's
            own measurement of the corridor.
        actual_clearance_m: Room the aircraft has, metres.
        deficit_m: How much closer than the plan the aircraft is, past the
            tolerance. Zero whenever the aircraft is no closer than the plan,
            which is the normal state on a well flown curve.
        speed_scale: Fraction of the planned speed the envelope allows, in
            ``[0, 1]``. 1.0 whenever there is no deficit.
        hard_stop: The absolute floor has been reached. Nothing overrides this.
        breached: The aircraft has spent more of the plan's margin than
            ``breach_deficit_m``, and this is a near contact rather than a
            tight-but-intended pass.
        reason: Short phrase naming what bound this tick, for the follower's
            limiter attribution and its logs.
    """

    def __init__(self, plan_clearance_m, actual_clearance_m, deficit_m,
                 speed_scale, hard_stop, breached, reason):
        # type: (float, object, float, float, bool, bool, str) -> None
        self.plan_clearance_m = float(plan_clearance_m)
        self.actual_clearance_m = actual_clearance_m
        self.deficit_m = float(deficit_m)
        self.speed_scale = float(speed_scale)
        self.hard_stop = bool(hard_stop)
        self.breached = bool(breached)
        self.reason = reason

    def __repr__(self):
        # type: () -> str
        return ("ClearanceBudget(plan=%.2f, actual=%s, deficit=%.2f, scale=%.2f,"
                " %s)" % (self.plan_clearance_m,
                          "none" if self.actual_clearance_m is None
                          else "%.2f" % self.actual_clearance_m,
                          self.deficit_m, self.speed_scale, self.reason))


class ClearanceEnvelope(object):
    """Turn a pair of clearances into what the follower is allowed to do."""

    def __init__(self, config=None):
        # type: (object) -> None
        self._cfg = config or ClearanceEnvelopeConfig()

    @property
    def config(self):
        # type: () -> ClearanceEnvelopeConfig
        """The thresholds in force, for logging and for the launch audit."""
        return self._cfg

    def budget(self, plan_clearance_m, actual_clearance_m):
        # type: (object, object) -> ClearanceBudget
        """Rule on one tick.

        Args:
            plan_clearance_m: Distance from the current reference point to the
                nearest occupied voxel, metres, or ``None`` when nothing is
                within the search radius (which means "plenty").
            actual_clearance_m: The same distance measured at the aircraft, or
                ``None`` for the same reason.

        Returns:
            The :class:`ClearanceBudget` for this tick.
        """
        cfg = self._cfg
        # Nothing near the aircraft: no proximity reflex has anything to act
        # on, whatever the plan's clearance was.
        if actual_clearance_m is None:
            return ClearanceBudget(
                plan_clearance_m if plan_clearance_m is not None
                else cfg.open_clearance_m,
                None, 0.0, 1.0, False, False, "clear")

        actual = float(actual_clearance_m)
        # An unknown plan clearance is treated as open space rather than as
        # zero: the alternative reads "the planner has no margin" and brakes
        # hardest exactly where there is most room.
        plan = (cfg.open_clearance_m if plan_clearance_m is None
                else float(plan_clearance_m))
        # A plan closer to the wall than the absolute floor is not a licence to
        # fly there. Clamp UP, so the deficit is measured against a corridor the
        # airframe could actually survive.
        plan = max(plan, cfg.hard_floor_m)

        deficit = max(0.0, (plan - cfg.tolerance_m) - actual)
        if actual <= cfg.hard_floor_m:
            # A hard stop is NOT automatically a breach, and keeping them apart
            # is what stops a doorway becoming a loop. In a 0.90 m opening the
            # plan holds 0.45 m, so the floor is reached after only 0.15 m of
            # drift -- routinely, on a pass the aircraft is entitled to make. If
            # that counted as a breach the follower would retreat 1.3 m back out
            # of the door, replan the identical curve, and do it again forever
            # (measured: 60 retreats in one hospital run). Stopping is the
            # correct response; reversing out of the building is not. The
            # breach test below stays absolute in the deficit, so in a wide
            # aisle -- where reaching the floor means the aircraft has thrown
            # away 0.40 m of margin -- it still fires.
            return ClearanceBudget(plan, actual, deficit, 0.0, True,
                                   deficit > cfg.breach_deficit_m, "hard_floor")
        if deficit <= 0.0:
            # The aircraft is no closer than the plan is. It is flying what it
            # was given, in a corridor the planner measured, and there is
            # nothing here for a reflex to correct.
            return ClearanceBudget(plan, actual, 0.0, 1.0, False, False,
                                   "on_clearance")

        scale = max(0.0, 1.0 - deficit / cfg.deficit_span_m)
        breached = deficit > cfg.breach_deficit_m
        return ClearanceBudget(plan, actual, deficit, scale, False, breached,
                               "breach" if breached else "deficit")

    def speed_cap(self, budget, planned_speed):
        # type: (object, float) -> float
        """The speed this budget allows, given what the plan asked for.

        The floor is a speed, not a fraction: scaling a plan that already asked
        for 0.12 m/s by 0.3 produces a command the airframe's own deadband
        swallows, and an aircraft that is not moving cannot leave the situation
        the governor is objecting to.
        """
        cfg = self._cfg
        if budget.hard_stop:
            return 0.0
        allowed = budget.speed_scale * float(planned_speed)
        if allowed >= float(planned_speed):
            return float(planned_speed)
        return max(min(cfg.floor_speed, float(planned_speed)), allowed)
