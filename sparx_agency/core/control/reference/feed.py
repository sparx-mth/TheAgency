"""Hold the trajectory being flown and answer "what should the aircraft be doing".

Everything about *the plan* lives here: which curve is live, when a newly arrived
one takes over, where on it the aircraft currently is, and how far off it is.
Nothing about the airframe does -- no gains, no limits, no command. That split is
what lets an acceleration backend and a velocity backend fly the same plan
without either one owning the trajectory or the two drifting apart on what the
plan says.

The one piece of real subtlety is **promotion**. FALCON deliberately starts each
curve a planning-time in the future so it joins smoothly onto the one still being
flown, so switching the instant a message arrives would jump the reference
forward to a point the aircraft has not reached yet. A new curve is therefore
queued and adopted when its own start time comes up.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from sparx_agency.core.control.reference.diagnosis import decompose_error
from sparx_agency.core.control.reference.params import ReferenceParams
from sparx_agency.core.control.reference.types import ReferenceSample
from sparx_agency.core.planning.trajectories.bspline.projection import TrajectoryProjector


class TrajectoryFeed:
    """The live trajectory, and the reference point taken from it each tick.

    One instance per aircraft. Stateful -- it holds the curve being flown, the
    curve queued behind it, and where on the curve the aircraft was last tick --
    so :meth:`reset` whenever the aircraft stops flying the plan.

    Args:
        params: How the reference point is chosen. The defaults suit FALCON's
            replan cadence.
    """

    def __init__(self, params=None):
        # type: (Optional[ReferenceParams]) -> None
        self.params = params or ReferenceParams()
        self._projector = TrajectoryProjector(self.params.projection)
        self._current = None        # type: Optional[object]
        self._pending = None        # type: Optional[object]

    def reset(self):
        # type: () -> None
        """Drop the held and queued curves and forget where on them we were.

        Call whenever the aircraft stops flying the plan. Dropping the curves
        rather than merely resetting the projector is deliberate: a caller
        saying "stop flying the plan" that left a usable trajectory loaded would
        resume it on the very next tick.
        """
        self._projector.reset()
        self._current = None
        self._pending = None

    def set_trajectory(self, trajectory):
        # type: (object) -> bool
        """Queue a newly planned trajectory.

        It is *queued*, not adopted; see the module docstring.

        Args:
            trajectory: A
                :class:`~sparx_agency.core.planning.trajectories.bspline.BsplineTrajectory`.

        Returns:
            True if it was queued. False rejects a trajectory whose id is not
            newer than what is already held -- a re-send or a misordered
            message, either of which would restart a curve mid-flight.
        """
        newest = self._pending or self._current
        if newest is not None and trajectory.traj_id <= newest.traj_id:
            return False
        self._pending = trajectory
        return True

    @property
    def trajectory(self):
        # type: () -> Optional[object]
        """The curve being flown, or None when there is none."""
        return self._current

    @property
    def trajectory_id(self):
        # type: () -> int
        """Id of the trajectory being flown, or -1 when none is."""
        return -1 if self._current is None else int(self._current.traj_id)

    @property
    def last_reference_time(self):
        # type: () -> float
        """Where on the curve the reference was taken last tick, seconds."""
        return self._projector.last_t

    def promote(self, now_s):
        # type: (float) -> bool
        """Swap in the queued trajectory once its own start time has arrived.

        Args:
            now_s: Current time, on the clock the trajectories are stamped on.

        Returns:
            True if a swap happened this tick. Callers use it to drop the
            derivative state that a new parameterisation invalidates.
        """
        if self._pending is None:
            return False
        if self._current is None or self._pending.start_time_s <= float(now_s):
            self._current = self._pending
            self._pending = None
            # The new curve is a different parameterisation, so last tick's
            # position along the old one means nothing on it.
            self._projector.reset()
            return True
        return False

    def usable(self, now_s):
        # type: (float) -> bool
        """Whether the held trajectory is worth following at ``now_s``."""
        if self._current is None:
            return False
        elapsed = self._current.elapsed(now_s)
        if elapsed < 0.0:
            return False
        return elapsed < self._current.duration + self.params.max_trajectory_age_s

    def resolve(self, measured_position, now_s, lookahead_s=0.0):
        # type: (object, float, float) -> Optional[ReferenceSample]
        """The plan's state at the reference point, with the error against it.

        Args:
            measured_position: Measured world ``(x, y, z)``, metres.
            now_s: Current time, on the trajectories' clock.
            lookahead_s: Seconds to advance the reference beyond the projected
                point. Leave at zero unless you have measured that it helps:
                a lookahead settles at a constant position error of
                ``lookahead * speed``, which the position gain turns into a
                standing forward push.

        Returns:
            The reference sample, or None when no trajectory is usable.
        """
        if not self.usable(now_s):
            return None
        current = self._current
        measured = np.asarray(measured_position, dtype=float).reshape(3)
        elapsed = current.elapsed(now_s)

        projected = self._project(measured, elapsed)
        # Once the schedule has run out the reference is pinned to the stopped
        # endpoint, whatever the projection says. This is not a tidying detail:
        # the search returns a time marginally *inside* the curve, and sampling
        # there hands back the plan's full cruise velocity as a feedforward. An
        # aircraft given that keeps flying at cruise past the end of a
        # trajectory -- measured, a metre and a half into space FALCON never
        # checked -- while the position term alone gently disagrees.
        past_end = elapsed >= current.duration
        reference_time = (current.duration if past_end
                          else min(projected + float(lookahead_s), current.duration))
        point = current.sample(reference_time)

        on_schedule_time = min(max(0.0, elapsed), current.duration)
        offset = current.position_at(on_schedule_time) - measured
        gap, along, cross = decompose_error(offset, self._tangent(on_schedule_time))
        return ReferenceSample(
            point=point,
            reference_time_s=reference_time,
            elapsed_s=elapsed,
            gap_m=gap,
            along_track_lag_m=along,
            cross_track_error_m=cross,
            trajectory_id=int(current.traj_id),
            past_end=past_end,
            duration_s=current.duration)

    def _project(self, measured, elapsed):
        # type: (np.ndarray, float) -> float
        """Where on the curve the aircraft is, in seconds from its start.

        With projection off this is the elapsed time, and the projector is still
        re-anchored to that answer so ``last_reference_time`` stays truthful
        either way -- otherwise it reports a stale 0.0 for the whole flight.
        """
        if not self.params.use_projection:
            clamped = min(max(0.0, elapsed), self._current.duration)
            self._projector.reset(clamped)
            return clamped
        return self._projector.project(self._current, measured)

    def _tangent(self, t):
        # type: (float) -> np.ndarray
        """The curve's direction of travel at ``t``, defined even past the end.

        Taken from the velocity *curve* rather than from a sampled point:
        ``BsplineTrajectory.sample`` zeroes every derivative outside the
        trajectory's span, which would leave a trailing aircraft with no
        direction of travel and its entire along-track lag misreported as
        cross-track.
        """
        return self._current.velocity.evaluate_at_time(
            min(max(0.0, float(t)), self._current.duration))
