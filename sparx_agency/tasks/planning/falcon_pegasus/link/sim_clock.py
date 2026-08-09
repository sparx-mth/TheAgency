"""The map between FALCON's clock and the simulator's.

FALCON runs in its own container, on the wall clock, and stamps every
trajectory it plans with ``ros::Time::now()``. The aircraft it is flying lives
in Isaac Sim, whose time advances at whatever fraction of real time the GPU can
manage -- **0.66 on this machine**, measured over a whole office exploration.

Those two facts do not compose. A trajectory that says "be 1.0 m along in one
second" means one *wall* second, and in one wall second the aircraft has only
been given 0.66 s of physics to move in. It arrives 34 cm short, every second,
on every trajectory. The tracker sees a schedule it can never meet, reports a
standing along-track lag, and spends its speed margin chasing a deadline that
recedes as fast as it closes -- while FALCON, replanning four times a second
from the aircraft's true position, keeps issuing fresh impossible schedules.

Measured on the stub, changing nothing but this factor:

======================  =========  =========  =========
airframe                mean err   mean lag   mean xte
======================  =========  =========  =========
real time               0.26 m     -0.08 m    0.11 m
0.66x real time         0.92 m     +0.69 m    0.31 m
======================  =========  =========  =========

The lag flips from *ahead of the plan* to two thirds of a metre behind it, and
the aircraft's sideways deviation -- the component that hits walls -- triples.

The fix is to count each trajectory's schedule from the clock the aircraft
actually experiences. The geometry is untouched: the aircraft flies the same
curve, at the same speed *relative to the map*, and simply covers less ground
per wall-clock second, which is the honest consequence of a slow simulator.

The principled alternative is ROS's own ``use_sim_time`` with Isaac publishing
``/clock`` into the container, which would additionally slow FALCON's own
planning cadence to match. That is a larger change to a stack that took a while
to stabilise, and it is the right thing to do next; this is contained to the
boundary where a wall-stamped message enters a simulated world.
"""
from __future__ import annotations

import math

DEFAULT_WINDOW_S = 2.0
"""Wall-clock time constant of the real-time-factor estimate, seconds.

Long enough to span many render cycles, short enough to follow a genuine
slowdown when the aircraft turns to face an expensive corner of the office.

**Two seconds and not two hundred milliseconds** because the thing being
averaged is extremely lumpy. ``SimLoop.step`` renders one tick in twenty-one at
250 Hz against a 12 Hz frame rate, and the render tick costs an order of
magnitude more wall clock than the physics-only ticks around it. An average
shorter than several render periods just measures whichever kind of tick it
happened to land on.
"""

MIN_WALL_STEP_S = 1e-6
"""Below this a wall step is treated as no information rather than as speed.

Two ticks that read the same wall clock would otherwise divide by ~0.
"""

MIN_FACTOR = 0.05
MAX_FACTOR = 2.0
"""Bounds on the estimate.

A simulator that has genuinely stopped would otherwise drive the factor to zero
and freeze the reference on the curve's first point, which looks exactly like a
planner that has died. Clamping keeps the failure legible.
"""


class SimClock:
    """Tracks how fast simulated time runs against the wall clock.

    Fed both clocks every tick, it estimates the real-time factor and converts
    instants stamped on the wall clock into the simulator's.

    Args:
        window_s: Wall-clock time constant of the estimate.

    Attributes:
        real_time_factor: Simulated seconds per wall-clock second. Starts at
            1.0 and converges within a few window lengths.
    """

    def __init__(self, window_s=DEFAULT_WINDOW_S):
        # type: (float) -> None
        if float(window_s) <= 0.0:
            raise ValueError("window_s must be > 0, got %r" % (window_s,))
        self.window_s = float(window_s)
        self.real_time_factor = 1.0
        self._wall = None       # type: float
        self._sim = None        # type: float
        self._sim_sum = 0.0
        self._wall_sum = 0.0

    def update(self, wall_s, sim_s):
        # type: (float, float) -> None
        """Record both clocks, and refine the factor from how far each moved.

        The estimate is the ratio of ACCUMULATED simulated time to accumulated
        wall time over the window -- not an average of per-tick ratios. That
        distinction is the whole correctness of this class. Isaac's ticks are
        wildly uneven: twenty cheap physics-only steps, then one step that also
        renders and costs an order of magnitude more wall clock. Averaging
        per-tick ratios weights the cheap ticks and the expensive one equally,
        so the twenty fast ratios swamp the one slow ratio and the estimate
        comes out far too high -- measured on a real flight, 0.75 reported
        against a true 0.61, and peaks above 1.0 on a simulator that never once
        ran at real time. Weighting each sample by its own wall step is exactly
        what summing before dividing does.

        Only forward motion on both clocks contributes. A tick that advanced
        neither -- the mission polls faster than the physics steps -- carries no
        information about the rate and must not be read as a stall.

        Args:
            wall_s: The wall clock, seconds. The clock FALCON stamps on.
            sim_s: The simulator's clock, seconds.
        """
        wall_s = float(wall_s)
        sim_s = float(sim_s)
        if self._wall is not None:
            wall_step = wall_s - self._wall
            sim_step = sim_s - self._sim
            if wall_step > MIN_WALL_STEP_S and sim_step >= 0.0:
                decay = math.exp(-wall_step / self.window_s)
                self._sim_sum = self._sim_sum * decay + sim_step
                self._wall_sum = self._wall_sum * decay + wall_step
                if self._wall_sum > MIN_WALL_STEP_S:
                    observed = self._sim_sum / self._wall_sum
                    self.real_time_factor = min(MAX_FACTOR,
                                                max(MIN_FACTOR, observed))
        self._wall = wall_s
        self._sim = sim_s

    def to_sim(self, wall_s):
        # type: (float) -> float
        """Convert an instant on the wall clock to the simulator's clock.

        Anchored on the most recent tick rather than on some origin, so the
        conversion stays exact at "now" and only the *offset* from now is
        scaled. Trajectory start times sit a planning horizon either side of
        now -- a tenth of a second -- so even a badly wrong factor moves them
        by milliseconds, while the anchor carries all the accumulated drift
        exactly.

        Args:
            wall_s: An instant on the wall clock.

        Returns:
            The same instant on the simulator's clock.

        Raises:
            RuntimeError: If called before the first :meth:`update`. There is no
                sensible fallback: the two clocks do not share an epoch -- the
                simulator's starts at zero and the wall clock is a Unix time --
                so "assume they agree" would be off by decades, and the tracker
                would silently read every trajectory as long finished.
        """
        if self._wall is None:
            raise RuntimeError(
                "SimClock.to_sim called before update(); the wall and simulated "
                "clocks share no epoch, so there is nothing to convert from")
        return self._sim + (float(wall_s) - self._wall) * self.real_time_factor
