"""Decide when an autonomous exploration has stopped being worth its wall-clock.

An exploration mission has exactly one honest success signal -- the map grew --
and several dishonest ones. The aircraft is moving, so the position watchdog is
happy. The planner is publishing trajectories, so the liveness watchdog is
happy. The FSM says ``EXEC_TRAJ``, so the state machine is happy. And the map
has not gained a voxel in four minutes because the aircraft is flying a tight
orbit inside a room it has already seen, unable to get through the doorway it
keeps being routed at.

Nothing in the stack notices that, because every component's local view of it is
success. This class is the global view: it watches WHERE the aircraft has been
and HOW MUCH MAP that bought, and calls the mission when the second stops
following from the first.

The two signals, deliberately combined rather than used alone:

* **Confinement radius** -- the radius of the smallest disc containing the whole
  trailing window of positions. A drone doing useful work sweeps a building; a
  drone stuck in a doorway loop, or walled into a room whose exit it cannot
  thread, orbits inside a few metres. Net displacement (the usual watchdog) says
  nothing here: an aircraft that flies a 3 m circle every 15 s has plenty of net
  displacement over any short window and none at all over a long one.
* **Coverage growth** -- explored volume per minute over the same window.
  Confinement alone would fire on a legitimately slow, thorough sweep of one
  crowded room; growth alone would fire during a long transit through already-
  mapped corridors on the way to a far frontier. Neither is a failure. Both at
  once is: the aircraft is somewhere it has already been, learning nothing.

Escalation is two-stage on purpose. A confined-and-barren window is first a
NUDGE -- something the mission can still act on, by re-surveying, retiring the
frontier it is fixated on, or picking a different region -- and only becomes an
ABORT if the nudges do not restore growth. A watchdog that can only kill wastes
every run it fires on; one that can also poke turns half of them into finishes.

Pure stdlib and Python 3.8: this runs inside the Noetic FALCON container beside
the planner it is judging.
"""
from __future__ import annotations

import math

RUNNING = "running"
"""The mission is making progress, or is still inside its grace period."""

NUDGE_CONFINED = "nudge_confined"
"""Confined and barren: recoverable, act on it before it becomes an abort."""

ABORT_CONFINED = "abort_confined"
"""Confined and barren for long enough, and the nudges did not help."""

ABORT_NO_GROWTH = "abort_no_growth"
"""The map has not grown at all for the whole barren cap, wherever it flew."""

ABORT_NO_MOVEMENT = "abort_no_movement"
"""The aircraft has not gone anywhere at all: pinned, not merely orbiting."""

ABORT_TIME_CAP = "abort_time_cap"
"""The hard wall-clock ceiling. Nothing is wrong; the mission is simply over."""

_ABORTS = (ABORT_CONFINED, ABORT_NO_GROWTH, ABORT_NO_MOVEMENT, ABORT_TIME_CAP)


class ExplorationProgressConfig(object):
    """Thresholds for :class:`ExplorationProgressMonitor`.

    Every default is expressed in the mission's own clock (seconds of the clock
    the samples carry -- sim time, if that is what is fed in), so a simulator
    running at half real time is judged on the flight it flew rather than on the
    wall-clock it burned.

    Attributes:
        time_cap_s: Hard mission ceiling. The one threshold that is not a
            diagnosis: reaching it means the mission ran out of time, not that
            anything failed.
        grace_s: Nothing is judged before this. Takeoff, the survey turn and
            FALCON's first plan take tens of seconds during which the aircraft
            is legitimately parked on the spot with a coverage number that has
            barely moved.
        window_s: The trailing window both signals are measured over. Must be
            comfortably longer than one plan-execute-replan cycle (FALCON's
            cadence here is ~2.5 s) and longer than one recovery maneuver
            (retreat + face + dwell is ~13 s), or a normal recovery reads as
            confinement.
        confine_radius_m: The aircraft is "confined" when every position in the
            window fits inside a disc of this radius. Size it above the largest
            legitimate stationary maneuver -- a survey turn is a point, a
            retreat is ~1.3 m -- and below the smallest useful exploration leg.
        growth_m3_per_min: Explored volume per minute below which the window is
            "barren".
        confine_cap_s: How long confined-and-barren must persist before the
            nudge becomes an abort.
        nudge_every_s: Minimum spacing between nudges, so a confined mission
            gets a handful of distinct interventions rather than one per sample.
        barren_cap_s: Abort after this long with no coverage growth AT ALL,
            regardless of where the aircraft went. The backstop for the case
            confinement misses: an aircraft touring a large already-mapped
            region because the planner cannot reach any real frontier.
        no_move_m: Net displacement below which the aircraft counts as pinned.
        no_move_cap_s: How long pinned before that is an abort.
        min_growth_m3: Coverage growth smaller than this is treated as noise
            rather than progress, so a mapper dithering by fractions of a cubic
            metre cannot hold a doomed mission open forever.
    """

    def __init__(self,
                 time_cap_s=2400.0,
                 grace_s=90.0,
                 window_s=120.0,
                 confine_radius_m=3.0,
                 growth_m3_per_min=3.0,
                 confine_cap_s=240.0,
                 nudge_every_s=45.0,
                 barren_cap_s=300.0,
                 no_move_m=0.6,
                 no_move_cap_s=120.0,
                 min_growth_m3=0.5):
        # type: (float, float, float, float, float, float, float, float, float, float, float) -> None
        self.time_cap_s = float(time_cap_s)
        self.grace_s = float(grace_s)
        self.window_s = float(window_s)
        self.confine_radius_m = float(confine_radius_m)
        self.growth_m3_per_min = float(growth_m3_per_min)
        self.confine_cap_s = float(confine_cap_s)
        self.nudge_every_s = float(nudge_every_s)
        self.barren_cap_s = float(barren_cap_s)
        self.no_move_m = float(no_move_m)
        self.no_move_cap_s = float(no_move_cap_s)
        self.min_growth_m3 = float(min_growth_m3)


class ProgressVerdict(object):
    """One judgement, with the numbers that produced it.

    Attributes:
        state: One of the module-level state constants.
        reason: Human-readable sentence naming the numbers, for a log line and
            for the run's verdict file. Empty while simply running.
        elapsed_s: Mission time at this sample.
        confinement_radius_m: Radius of the smallest disc containing the
            trailing window, or ``None`` before the window is full.
        growth_m3_per_min: Coverage growth rate over the window, or ``None``
            before the window is full.
        coverage_m3: The latest explored volume.
        confined_for_s: How long the mission has been continuously
            confined-and-barren.
        barren_for_s: How long since coverage last grew by ``min_growth_m3``.
    """

    def __init__(self, state, reason, elapsed_s, confinement_radius_m,
                 growth_m3_per_min, coverage_m3, confined_for_s, barren_for_s):
        # type: (str, str, float, object, object, float, float, float) -> None
        self.state = state
        self.reason = reason
        self.elapsed_s = float(elapsed_s)
        self.confinement_radius_m = confinement_radius_m
        self.growth_m3_per_min = growth_m3_per_min
        self.coverage_m3 = float(coverage_m3)
        self.confined_for_s = float(confined_for_s)
        self.barren_for_s = float(barren_for_s)

    @property
    def is_abort(self):
        # type: () -> bool
        """Whether this verdict ends the mission."""
        return self.state in _ABORTS

    @property
    def is_nudge(self):
        # type: () -> bool
        """Whether this verdict asks the mission to try something different."""
        return self.state == NUDGE_CONFINED

    def as_dict(self):
        # type: () -> dict
        """A JSON-serialisable record, for the run's progress trace."""
        return {
            "state": self.state,
            "reason": self.reason,
            "elapsed_s": round(self.elapsed_s, 2),
            "confinement_radius_m": (None if self.confinement_radius_m is None
                                     else round(self.confinement_radius_m, 3)),
            "growth_m3_per_min": (None if self.growth_m3_per_min is None
                                  else round(self.growth_m3_per_min, 3)),
            "coverage_m3": round(self.coverage_m3, 2),
            "confined_for_s": round(self.confined_for_s, 1),
            "barren_for_s": round(self.barren_for_s, 1),
        }

    def __repr__(self):
        # type: () -> str
        return "ProgressVerdict(%s, %s)" % (self.state, self.reason)


class ExplorationProgressMonitor(object):
    """Judge an exploration from its position track and its explored volume.

    Feed it one sample per second or so with :meth:`update`; it returns a
    :class:`ProgressVerdict` every time. The monitor is a pure function of the
    samples it has been given -- no clock of its own, no I/O -- so a whole
    mission can be replayed through it from a recorded trace, which is how its
    thresholds were checked against runs that had already happened.
    """

    def __init__(self, config=None):
        # type: (object) -> None
        self._cfg = config or ExplorationProgressConfig()
        self._samples = []          # type: list  # (t, x, y, z, coverage)
        self._t0 = None             # type: object
        self._confined_since = None  # type: object
        self._best_coverage = None  # type: object
        self._best_coverage_at = None  # type: object
        self._move_anchor = None    # type: object  # (t, x, y, z)
        self._last_nudge_at = None  # type: object
        self.nudges = 0

    @property
    def config(self):
        # type: () -> ExplorationProgressConfig
        """The thresholds in force, for the caller to log at startup.

        A watchdog whose thresholds are not in the run's own log is a watchdog
        nobody can argue with after the fact.
        """
        return self._cfg

    # ── ingest ───────────────────────────────────────────────────────────

    def update(self, t_s, position, coverage_m3):
        # type: (float, tuple, float) -> ProgressVerdict
        """Add one sample and re-judge the mission.

        Args:
            t_s: Mission clock, seconds. Monotonic; samples that go backwards
                (a simulator reset) restart the mission rather than corrupting
                the windows.
            position: ``(x, y, z)`` world position of the aircraft, metres.
            coverage_m3: Explored volume so far, cubic metres. Monotonic in
                principle; a decrease (a planner respawn wiping the map) is
                tracked as a new low rather than treated as negative growth.

        Returns:
            The verdict for this sample.
        """
        t = float(t_s)
        # Backwards against the LAST sample, not against the mission start: a
        # simulator restarted at t=0 lands exactly on t0, so comparing with the
        # start leaves the old mission's windows and nudge count in place and
        # the fresh run inherits a verdict it did not earn.
        if self._t0 is None or (self._samples and t < self._samples[-1][0]):
            self._reset(t, position, coverage_m3)
        x, y, z = (float(position[0]), float(position[1]), float(position[2]))
        cov = float(coverage_m3)
        self._samples.append((t, x, y, z, cov))
        cutoff = t - self._cfg.window_s
        while len(self._samples) > 1 and self._samples[0][0] < cutoff:
            self._samples.pop(0)

        self._track_coverage(t, cov)
        self._track_movement(t, x, y, z)
        return self._judge(t, x, y, z, cov)

    def _reset(self, t, position, coverage_m3):
        # type: (float, tuple, float) -> None
        self._samples = []
        self._t0 = t
        self._confined_since = None
        self._best_coverage = float(coverage_m3)
        self._best_coverage_at = t
        self._move_anchor = (t, float(position[0]), float(position[1]),
                             float(position[2]))
        self._last_nudge_at = None
        self.nudges = 0

    def _track_coverage(self, t, cov):
        # type: (float, float) -> None
        """Advance the barren clock only on a real gain.

        ``_best_coverage`` is a MILESTONE, not a running maximum, and the
        difference is the whole mechanism. A running maximum follows the signal
        exactly, so ``cov > best + min_growth`` is never true and the clock can
        never advance -- which is a watchdog that aborts every healthy mission
        at the barren cap. The milestone only moves when the map has genuinely
        gained ``min_growth_m3`` since it last moved, so steady growth restamps
        it continually while a mapper dithering by fractions of a cubic metre
        never does.

        A material DROP restamps too: FALCON's exploration node respawns with an
        empty map, and re-earning ground already counted is real work being done
        by an aircraft that has not failed at anything.
        """
        cfg = self._cfg
        if self._best_coverage is None:
            self._best_coverage = cov
            self._best_coverage_at = t
        elif cov >= self._best_coverage + cfg.min_growth_m3:
            self._best_coverage = cov
            self._best_coverage_at = t
        elif cov < self._best_coverage - cfg.min_growth_m3:
            self._best_coverage = cov
            self._best_coverage_at = t

    def _track_movement(self, t, x, y, z):
        # type: (float, float, float, float) -> None
        """Advance the pinned clock only once the aircraft has actually gone."""
        if self._move_anchor is None:
            self._move_anchor = (t, x, y, z)
            return
        _, ax, ay, az = self._move_anchor
        if math.sqrt((x - ax) ** 2 + (y - ay) ** 2 + (z - az) ** 2) > self._cfg.no_move_m:
            self._move_anchor = (t, x, y, z)

    # ── judgement ────────────────────────────────────────────────────────

    def _window_metrics(self, t):
        # type: (float) -> tuple
        """``(radius, growth_per_min)`` over the trailing window, or ``(None, None)``.

        The radius is measured from the window's centroid rather than from its
        first sample: an aircraft drifting steadily along a corridor would show
        a large first-sample radius and a small centroid radius only if it
        doubled back, which is exactly the distinction being drawn.
        """
        if len(self._samples) < 3:
            return None, None
        span = self._samples[-1][0] - self._samples[0][0]
        if span < self._cfg.window_s * 0.9:
            return None, None
        n = float(len(self._samples))
        cx = sum(s[1] for s in self._samples) / n
        cy = sum(s[2] for s in self._samples) / n
        cz = sum(s[3] for s in self._samples) / n
        radius = max(math.sqrt((s[1] - cx) ** 2 + (s[2] - cy) ** 2
                               + (s[3] - cz) ** 2) for s in self._samples)
        grew = self._samples[-1][4] - self._samples[0][4]
        return radius, 60.0 * grew / span

    def _judge(self, t, x, y, z, cov):
        # type: (float, float, float, float, float) -> ProgressVerdict
        cfg = self._cfg
        elapsed = t - self._t0
        barren_for = t - (self._best_coverage_at
                          if self._best_coverage_at is not None else t)
        radius, growth = self._window_metrics(t)
        confined_for = (0.0 if self._confined_since is None
                        else t - self._confined_since)

        def verdict(state, reason):
            # type: (str, str) -> ProgressVerdict
            return ProgressVerdict(state, reason, elapsed, radius, growth, cov,
                                   confined_for, barren_for)

        if elapsed >= cfg.time_cap_s:
            return verdict(ABORT_TIME_CAP,
                           "mission time cap: %.0fs of %.0fs, coverage %.1f m3"
                           % (elapsed, cfg.time_cap_s, cov))
        if elapsed < cfg.grace_s:
            self._confined_since = None
            return verdict(RUNNING, "")

        pinned_for = t - self._move_anchor[0] if self._move_anchor else 0.0
        if pinned_for > cfg.no_move_cap_s:
            return verdict(ABORT_NO_MOVEMENT,
                           "pinned: net travel under %.1f m for %.0fs"
                           % (cfg.no_move_m, pinned_for))

        confined = (radius is not None
                    and radius < cfg.confine_radius_m
                    and growth < cfg.growth_m3_per_min)
        if not confined:
            self._confined_since = None
        elif self._confined_since is None:
            self._confined_since = t
        confined_for = (0.0 if self._confined_since is None
                        else t - self._confined_since)
        detail = ("confined to a %.1f m radius gaining %.1f m3/min for %.0fs "
                  "(coverage %.1f m3)"
                  % (radius or -1.0, growth or 0.0, confined_for, cov))

        # Confinement outranks bare barrenness even though it takes longer to
        # establish: a stuck mission is both, and "orbiting the same 2 m for
        # four minutes" is a diagnosis somebody can act on, where "the map
        # stopped growing" only says the run is over. The barren cap stays
        # underneath as the backstop for the case confinement misses -- a wide
        # tour of already-mapped space, which is not confined and is just as
        # worthless.
        if confined and confined_for > cfg.confine_cap_s:
            return verdict(ABORT_CONFINED, detail)
        if barren_for > cfg.barren_cap_s:
            return verdict(ABORT_NO_GROWTH,
                           "map has not grown %.1f m3 in %.0fs (coverage %.1f m3)"
                           % (cfg.min_growth_m3, barren_for, cov))
        if confined and (self._last_nudge_at is None
                         or t - self._last_nudge_at >= cfg.nudge_every_s):
            self._last_nudge_at = t
            self.nudges += 1
            return verdict(NUDGE_CONFINED, detail)
        return verdict(RUNNING, "")
