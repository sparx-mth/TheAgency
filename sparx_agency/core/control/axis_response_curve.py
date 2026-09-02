"""Measured stick-response curve for a remote-control axis, and its inverse.

Some airframes answer a stick deflection with a transmitter-style expo curve
rather than a dead band plus a linear ramp. A model with the wrong shape cannot
be fixed by tuning: every straight line fitted to a different part of an expo
curve yields a different "dead band", which is exactly what happened on the
ROBOTICAN Rooster (six mutually contradictory dead-band values over one week).

This module makes no shape assumption at all: it holds the measured
(counts, speed) points and interpolates between them, both ways. The points
come from calibration flights and live with the robot that was measured
(e.g. ``robots/ROBOTICAN``); this class is drone-agnostic.
"""

from __future__ import annotations


class AxisResponseCurve(object):
    """Piecewise-linear map between axis counts and steady-state speed.

    Args:
        points: Measured ``(counts, speed)`` pairs for one axis direction,
            magnitudes only. Must start at ``(0, 0)``, be strictly increasing
            in both coordinates, and end at the largest counts value that may
            ever be commanded -- the curve doubles as the axis ceiling, so
            ``axis_for`` never returns more counts than the last point's.

    Raises:
        ValueError: If fewer than two points are given, the first is not
            ``(0, 0)``, or either coordinate is not strictly increasing.
    """

    def __init__(self, points):
        # type: (list) -> None
        pts = [(float(c), float(v)) for c, v in points]
        if len(pts) < 2:
            raise ValueError("need at least two points, got %d" % len(pts))
        if pts[0] != (0.0, 0.0):
            raise ValueError("first point must be (0, 0), got %r" % (pts[0],))
        for (c0, v0), (c1, v1) in zip(pts, pts[1:]):
            if c1 <= c0 or v1 <= v0:
                raise ValueError(
                    "points must be strictly increasing, got %r -> %r"
                    % ((c0, v0), (c1, v1)))
        self._points = pts

    @property
    def max_counts(self):
        # type: () -> float
        """Largest counts value the curve covers -- the axis ceiling."""
        return self._points[-1][0]

    @property
    def max_speed(self):
        # type: () -> float
        """Speed at ``max_counts`` -- the fastest this axis may be asked for."""
        return self._points[-1][1]

    def speed_at(self, counts):
        # type: (float) -> float
        """Steady-state speed produced by ``counts``, sign carried through.

        Magnitudes beyond ``max_counts`` clamp to ``max_speed`` -- the caller
        should never command past the last measured point, but reading back a
        clipped log row must not extrapolate.
        """
        magnitude = self._interp(abs(float(counts)),
                                 key=0, out=1, hi=self.max_speed)
        return magnitude if counts >= 0.0 else -magnitude

    def axis_for(self, speed):
        # type: (float) -> float
        """Axis counts that hold ``speed`` in steady state, sign carried through.

        Requests beyond ``max_speed`` clamp to ``max_counts``: the plant cannot
        go faster, and asking for more stick buys tilt instead of speed.
        """
        magnitude = self._interp(abs(float(speed)),
                                 key=1, out=0, hi=self.max_counts)
        return magnitude if speed >= 0.0 else -magnitude

    def _interp(self, value, key, out, hi):
        # type: (float, int, int, float) -> float
        """Linear interpolation along one coordinate, clamped past the ends."""
        pts = self._points
        if value >= pts[-1][key]:
            return hi
        for lo_pt, hi_pt in zip(pts, pts[1:]):
            if value <= hi_pt[key]:
                span = hi_pt[key] - lo_pt[key]
                frac = (value - lo_pt[key]) / span
                return lo_pt[out] + frac * (hi_pt[out] - lo_pt[out])
        return hi
