"""Find where on a trajectory an aircraft actually is.

A tracking controller has to choose *which* point of the plan to compare itself
against. The obvious choice is the point at the current time; the alternative is
the point the aircraft is nearest to. This module provides the second.

**What it is actually for, measured.** The usual argument for projection is that
it stops a lagging aircraft cutting corners, and on this stack that argument
does not survive contact: flown against FALCON-shaped trajectories with a
realistic inner-loop lag, tracking the nearest point is *not* better than
tracking the point at time *t* through a bend -- it is marginally worse. The
reason is that FALCON's optimiser has already made the curve dynamically
feasible, so its curvature is bounded by roughly the same limits the airframe
has, and there is no runaway reference to cut away from.

Where it does win, clearly, is when the aircraft has been displaced in **time**
from the plan rather than in space. FALCON condemns its own live trajectory
whenever it finds an obstacle on it, and the aircraft holds until a replacement
arrives -- but the plan's clock keeps running through the hold. On resuming, a
time-indexed reference sits seconds further down the route, around the corner
and through whatever lies between, and the aircraft is pulled straight at it.
Projection resumes from where the aircraft is and flies the part of the route it
had not flown yet. Measured over a two-second hold on an L-shaped route, that is
the difference between a worst departure of 0.32 m and one of 0.60 m.

It also makes the diagnostics exact rather than approximate: the distance to the
curve is measured rather than inferred from a decomposition, and the schedule
lag comes out as a number in its own right.

The search is windowed around the previous result rather than global, because an
exploration route crosses itself: the globally nearest point on a curve that
loops back through the same room can be a leg the aircraft flew a minute ago,
and snapping to it would fly it backwards.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProjectionParams:
    """Bounds and resolution for the nearest-point search.

    Attributes:
        search_back_s: How far *before* the previous result the window reaches.
            Non-zero so an aircraft pushed off the path can rejoin behind where
            it was, but small: this is the only direction in which the search
            can undo progress.
        search_ahead_s: How far *after* the previous result the window reaches.
            Must exceed the largest jump one control tick can produce, or the
            projection falls progressively behind the aircraft and the window
            drags it backwards for the rest of the flight.
        coarse_step_s: Sampling interval of the first pass. The curve's distance
            profile is not unimodal, so a scan has to bracket the minimum before
            a refinement can find it; this must be finer than the smallest
            feature of the route.
        refine_iterations: Golden-section steps inside the bracketing interval.
            Each roughly halves the remaining interval, so ~12 takes a 0.2 s
            bracket below a millisecond, which is far finer than it needs to be
            and still costs nothing.
        lookahead_s: How far past the projected point the reference is taken.
            **Zero by default, and it should usually stay there.** Leaning the
            reference forward looks like a cheap way to keep a little along-track
            pull, and it is not: on a straight run it settles at a *constant*
            position error of ``lookahead_s * speed``, which the position gain
            turns into a standing forward push, which the damping term balances
            only by flying faster than the plan. Measured, 0.15 s of lookahead
            flew 14% over the planned speed and overshot the end of the
            trajectory by a third of a metre. Schedule is recovered by the
            tracker's along-track catch-up term instead, which cannot cut a
            corner because it only ever acts along the tangent.
    """

    search_back_s: float = 0.5
    search_ahead_s: float = 1.5
    coarse_step_s: float = 0.05
    refine_iterations: int = 12
    lookahead_s: float = 0.0

    def __post_init__(self):
        # type: () -> None
        """Validate the bounds the search relies on."""
        if self.search_back_s < 0.0:
            raise ValueError("search_back_s must be >= 0, got %r" % (self.search_back_s,))
        if self.search_ahead_s <= 0.0:
            raise ValueError("search_ahead_s must be > 0, got %r" % (self.search_ahead_s,))
        if self.coarse_step_s <= 0.0:
            raise ValueError("coarse_step_s must be > 0, got %r" % (self.coarse_step_s,))
        if self.refine_iterations < 0:
            raise ValueError("refine_iterations must be >= 0, got %r"
                             % (self.refine_iterations,))
        if self.lookahead_s < 0.0:
            raise ValueError("lookahead_s must be >= 0, got %r" % (self.lookahead_s,))


class TrajectoryProjector:
    """Tracks where an aircraft sits along a trajectory, tick to tick.

    Stateful: each search starts from the previous answer, which is what keeps
    it cheap and what keeps it from jumping to a different leg of a route that
    crosses itself. :meth:`reset` whenever the trajectory is replaced.

    Args:
        params: Search window and resolution.
    """

    def __init__(self, params=None):
        # type: (ProjectionParams) -> None
        self.params = params or ProjectionParams()
        self._last_t = 0.0

    def reset(self, t=0.0):
        # type: (float) -> None
        """Re-anchor the search window, in seconds along the new trajectory."""
        self._last_t = max(0.0, float(t))

    @property
    def last_t(self):
        # type: () -> float
        """The most recent projection, seconds along the trajectory."""
        return self._last_t

    def project(self, trajectory, position):
        # type: (object, object) -> float
        """Find the point of ``trajectory`` nearest ``position``.

        Args:
            trajectory: A :class:`~.trajectory.BsplineTrajectory`.
            position: The aircraft's world ``(x, y, z)``.

        Returns:
            Seconds along the trajectory. Also stored, to anchor the next call.
        """
        measured = np.asarray(position, dtype=float).reshape(3)
        duration = trajectory.duration
        low = max(0.0, self._last_t - self.params.search_back_s)
        high = min(duration, self._last_t + self.params.search_ahead_s)
        if high <= low:
            self._last_t = min(max(0.0, low), duration)
            return self._last_t

        best_t, bracket = self._coarse_scan(trajectory, measured, low, high)
        self._last_t = self._refine(trajectory, measured,
                                    max(low, best_t - bracket),
                                    min(high, best_t + bracket))
        return self._last_t

    def reference_time(self, trajectory, position):
        # type: (object, object) -> float
        """Project, then lean forward by ``lookahead_s``.

        This is what a controller should sample the trajectory at: the nearest
        point tells it where it *is* on the plan, and the lookahead is what
        keeps it moving along it.
        """
        projected = self.project(trajectory, position)
        return min(projected + self.params.lookahead_s, trajectory.duration)

    def _coarse_scan(self, trajectory, measured, low, high):
        # type: (object, np.ndarray, float, float) -> tuple
        """Sample the window and return the best sample and half its spacing."""
        count = max(2, int(np.ceil((high - low) / self.params.coarse_step_s)) + 1)
        samples = np.linspace(low, high, count)
        best_t = low
        best_distance = float("inf")
        for t in samples:
            offset = trajectory.position_at(float(t)) - measured
            distance = float(np.dot(offset, offset))
            if distance < best_distance:
                best_distance = distance
                best_t = float(t)
        spacing = (high - low) / float(count - 1)
        return best_t, spacing

    def _refine(self, trajectory, measured, low, high):
        # type: (object, np.ndarray, float, float) -> float
        """Golden-section search for the minimum inside a bracketing interval."""
        if high <= low:
            return low
        ratio = 0.6180339887498949
        left = high - ratio * (high - low)
        right = low + ratio * (high - low)
        left_distance = self._distance(trajectory, measured, left)
        right_distance = self._distance(trajectory, measured, right)
        for _ in range(self.params.refine_iterations):
            if left_distance < right_distance:
                high, right, right_distance = right, left, left_distance
                left = high - ratio * (high - low)
                left_distance = self._distance(trajectory, measured, left)
            else:
                low, left, left_distance = left, right, right_distance
                right = low + ratio * (high - low)
                right_distance = self._distance(trajectory, measured, right)
        return 0.5 * (low + high)

    @staticmethod
    def _distance(trajectory, measured, t):
        # type: (object, np.ndarray, float) -> float
        """Squared distance from the aircraft to the curve at ``t``."""
        offset = trajectory.position_at(float(t)) - measured
        return float(np.dot(offset, offset))
