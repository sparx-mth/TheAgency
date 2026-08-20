"""Steer toward the middle of a tight opening while flying through it.

A planner asked for clearance produces a curve that is *near* the middle of a
doorway; it does not produce one that is *on* the middle, and the difference is
the whole margin when the opening is 0.90 m wide and the airframe is 0.50 m
across. FALCON's distance cost is soft -- ``pow(dist - safe_distance, 2)``,
weighed against smoothness and dynamic feasibility -- so the optimiser spends
centimetres of clearance to buy a flyable curve whenever the two disagree, and
a corner just before a door is exactly where they disagree. The follower then
adds its own tracking error on top, in whichever direction the last correction
happened to be pushing.

Neither error is a fault. Together they are a wall strike.

This class closes the gap with the one measurement the follower can make that
the planner's curve does not carry: the shape of the free space AROUND the
aircraft, right now. It probes the clearance field laterally, fits the peak, and
returns a small sideways velocity toward it. Three properties make that safe to
add to a tracked trajectory rather than a fight with it:

* **It only engages in tight places.** Above ``engage_clearance_m`` there is
  nothing to centre in and the bias is exactly zero, so open-space flight --
  every metre of the warehouse -- is untouched.
* **It is bounded, and small.** ``max_speed`` is a fraction of cruise. It biases
  a trajectory; it cannot fly one.
* **It acts across the path, never along it.** The component is purely lateral
  to the direction of travel, so it changes where the aircraft passes through
  an opening and never when, and the along-track schedule the servo is holding
  is undisturbed.

The clearance field is injected as a callable rather than a map type, so this is
testable against analytic corridors -- which is how the doorway numbers below
were checked -- and works against any occupancy source the caller has.

Pure stdlib and Python 3.8: this runs inside the Noetic FALCON container.
"""
from __future__ import annotations

import math


class CorridorCenteringConfig(object):
    """Tunables for :class:`CorridorCentering`.

    Attributes:
        engage_clearance_m: Above this much room the aircraft is not in a tight
            place and the bias is zero. Sized just above the tightest opening
            worth centring in: a 0.90 m doorway offers 0.45 m, a 1.4 m warehouse
            aisle offers 0.70 m and does not need help.
        probe_m: How far to either side the clearance field is sampled. Too
            small and the three samples are the same number; too large and the
            probe lands beyond the opening, where clearance says nothing about
            passing through it. Comparable to the airframe radius.
        gain: Fraction of the estimated offset to the corridor centre that is
            commanded per second. Under 1 so the aircraft converges on the
            middle rather than oscillating across it.
        max_speed: Hard cap on the lateral command, m/s.
        min_asymmetry_m: Clearance difference between the two sides below which
            the aircraft is treated as already centred. Stops the bias
            chattering on grid quantisation: an occupancy field sampled on
            0.1 m voxels cannot resolve a finer asymmetry than this anyway.
    """

    def __init__(self,
                 engage_clearance_m=0.85,
                 probe_m=0.25,
                 gain=0.6,
                 max_speed=0.15,
                 min_asymmetry_m=0.05):
        # type: (float, float, float, float, float) -> None
        self.engage_clearance_m = float(engage_clearance_m)
        self.probe_m = float(probe_m)
        self.gain = float(gain)
        self.max_speed = float(max_speed)
        self.min_asymmetry_m = float(min_asymmetry_m)


class CenteringBias(object):
    """A lateral nudge toward the middle of the local free space.

    Attributes:
        world_vx: World-frame x component of the bias, m/s.
        world_vy: World-frame y component, m/s.
        offset_m: Estimated signed distance from the aircraft to the corridor
            centre, positive to the left of travel. Diagnostic: this is the
            number to plot when asking whether the aircraft is threading a door
            or grazing its jamb.
        engaged: Whether the bias is non-zero.
    """

    def __init__(self, world_vx=0.0, world_vy=0.0, offset_m=0.0, engaged=False):
        # type: (float, float, float, bool) -> None
        self.world_vx = float(world_vx)
        self.world_vy = float(world_vy)
        self.offset_m = float(offset_m)
        self.engaged = bool(engaged)

    @property
    def speed(self):
        # type: () -> float
        """Magnitude of the bias, m/s."""
        return math.hypot(self.world_vx, self.world_vy)

    def __repr__(self):
        # type: () -> str
        return ("CenteringBias(%.3f, %.3f, offset=%+.3f m, %s)"
                % (self.world_vx, self.world_vy, self.offset_m,
                   "engaged" if self.engaged else "idle"))


class CorridorCentering(object):
    """Bias a tracked trajectory toward the centre of a tight opening."""

    IDLE = CenteringBias()

    def __init__(self, config=None):
        # type: (object) -> None
        self._cfg = config or CorridorCenteringConfig()

    def bias(self, clearance_at, position, direction_xy):
        # type: (object, tuple, tuple) -> CenteringBias
        """Compute the lateral bias for one tick.

        Args:
            clearance_at: ``f(x, y, z) -> float or None``. Distance to the
                nearest obstacle at a world point; ``None`` means nothing was
                found within the caller's search radius, i.e. open space.
            position: ``(x, y, z)`` world position of the aircraft.
            direction_xy: ``(dx, dy)`` direction of travel; magnitude ignored.
                The bias is perpendicular to this.

        Returns:
            The bias, which is :attr:`IDLE` whenever the aircraft is not in a
            tight place or the field gives no usable asymmetry.
        """
        cfg = self._cfg
        norm = math.hypot(direction_xy[0], direction_xy[1])
        if norm < 1e-6:
            # Standing still has no "across", and centring an aircraft that is
            # not going anywhere would push it sideways into a wall it is not
            # facing.
            return self.IDLE
        ux, uy = direction_xy[0] / norm, direction_xy[1] / norm
        px, py = -uy, ux                       # lateral unit, positive to the left
        x, y, z = float(position[0]), float(position[1]), float(position[2])

        here = clearance_at(x, y, z)
        if here is None or here > cfg.engage_clearance_m:
            return self.IDLE                   # open space: nothing to centre in

        d = cfg.probe_m
        left = clearance_at(x + px * d, y + py * d, z)
        right = clearance_at(x - px * d, y - py * d, z)
        # A probe that finds nothing is at least as clear as the search radius
        # allowed. Substituting the engage threshold keeps the comparison
        # meaningful and one-sided in the right direction: that side is roomier.
        left = cfg.engage_clearance_m if left is None else float(left)
        right = cfg.engage_clearance_m if right is None else float(right)

        if abs(left - right) < cfg.min_asymmetry_m:
            return self.IDLE                   # already centred, within the grid

        offset = self._peak_offset(left, right, d)
        if offset == 0.0:
            return self.IDLE
        speed = max(-cfg.max_speed, min(cfg.max_speed, cfg.gain * offset))
        return CenteringBias(world_vx=px * speed, world_vy=py * speed,
                             offset_m=offset, engaged=True)

    def across_width(self, first_block, position, direction_xy, max_m=1.5):
        # type: (object, tuple, tuple, float) -> object
        """Free width across the direction of travel, metres, or ``None``.

        Args:
            first_block: ``f(position, (dx, dy), max_dist) -> float or None``.
                Distance to the first obstacle along a RAY. ``VoxelBrakeGate``'s
                ``blocked_distance`` has exactly this signature.
            position: ``(x, y, z)`` world position of the aircraft.
            direction_xy: direction of travel; the width is measured across it.
            max_m: how far to look to each side.

        Returns:
            The free width through the aircraft, or ``None`` when either side
            is open past ``max_m`` — i.e. this is not a passage.

        The distinction this exists to draw is between an aircraft **in** a
        corridor and one that has **drifted into a wall** in open space. Both
        report the same small clearance, and they want opposite responses: in a
        doorway, backing out 1.3 m only replays the approach, while beside a
        wall in a room it is exactly right.

        It takes a RAY function rather than the clearance field the rest of this
        class uses, and that is the whole correctness of it. Clearance is
        direction-agnostic: probed 0.25 m either side of an aircraft standing
        0.30 m from a single wall, *both* probes return distances to that same
        wall (0.05 m and 0.55 m), and their sum reads as a 1.10 m corridor that
        does not exist. Only a ray can answer "is there something on the OTHER
        side", which is the question being asked.

        Measuring it at the aircraft rather than from the plan is also
        deliberate. The clearance at the reference point is the planner's own
        statement about the corridor, but it is a statement about where the
        REFERENCE is, and the case that matters most is exactly the one where
        the aircraft is not there. Measured in the hospital: the aircraft
        wedged in a 0.93 m doorway while its reference sat in the open room
        beyond holding 0.90 m, so a plan-side test read "not a passage" at the
        moment the aircraft was inside one.
        """
        norm = math.hypot(direction_xy[0], direction_xy[1])
        if norm < 1e-6:
            return None
        px, py = -direction_xy[1] / norm, direction_xy[0] / norm
        left = first_block(position, (px, py), max_m)
        right = first_block(position, (-px, -py), max_m)
        if left is None or right is None:
            return None
        return float(left) + float(right)

    @staticmethod
    def _peak_offset(left, right, d):
        # type: (float, float, float) -> float
        """Signed lateral distance to the clearance peak, positive to the left.

        Half the difference between the two probes, which is EXACT rather than
        approximate, and the reason is worth stating because the obvious
        alternative is wrong. A clearance field is a distance field, so it is
        1-Lipschitz and, across a straight corridor, falls 1:1 with lateral
        offset from the centre line. For an aircraft sitting ``e`` to the left
        of centre in a corridor of half width ``h``, the left probe reads
        ``h - e - d`` and the right one ``h - d + e``, so their half difference
        is exactly ``-e``: the move that centres the aircraft, with no fitting
        and no gain to choose.

        The obvious alternative -- a parabolic sub-sample peak fit, the standard
        trick for finding a maximum between three samples -- is BIASED here, and
        biased in the dangerous direction. A parabola through three points of a
        V-shaped field returns ``-d*e/(d - e)``, which overshoots by ``d/(d - e)``
        and diverges as the aircraft approaches one probe distance from centre:
        at ``d`` = 0.15 m and ``e`` = 0.10 m it asks for 0.30 m of correction to
        fix a 0.10 m error. Parabolic fitting assumes a smooth quadratic peak;
        a corridor's clearance field has a CORNER at its centre line.

        Being 1-Lipschitz also bounds the result: the two probes cannot differ by
        more than ``2 * d``, so the estimate cannot exceed ``d`` and needs no
        clamp to stay inside the geometry it sampled. The clamp below is kept
        only against a caller whose clearance function is not a true distance.
        """
        return max(-d, min(d, 0.5 * (left - right)))
