"""Detecting that a command is not reaching the world.

Indoors, close to a wall, this platform has a failure mode the map cannot see:
the drone is told to go and does not go. Sometimes the obstacle is genuinely
invisible to the camera (glass, a thin pole, a chair leg below the height band);
sometimes it is visible but the drone is already touching it and the airframe is
pinned. Either way the symptom is the same and it is *observable*: a command is
being issued and the pose is not changing.

Two independent witnesses are used, because each alone is fragile:

  1. **Our own progress measurement.** Over a trailing window, the distance the
     pose actually travelled *along the direction that was commanded*, compared
     with the distance that was commanded. Projecting rather than taking the raw
     displacement matters: a drone that is being pushed sideways by drift while
     its forward axis is blocked would otherwise look like it is moving fine.
  2. **The localization provider's own verdict** (``cmd_effectiveness``), which
     runs the same idea on raw measurements inside the provider and collapses
     within about two seconds of hitting a wall.

Translation and yaw are tracked separately and deliberately, because they fail
separately: a drone wedged against a wall can usually still rotate, and a drone
that cannot rotate (its side pinned) can usually still back off.

**A coasted pose is never used as evidence.** While coasting, the provider
propagates the pose *by the commanded motion* — so a coasted pose always appears
to obey commands perfectly. Feeding that into a stuck detector would guarantee it
never fires exactly when the drone is most likely stuck.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import cos, sin
from typing import Optional

from sparx_agency.core.common.types import Pose2D, normalize_angle

from .confidence import LocalizationQuality

#: Axis names a :class:`Blockage` can name.
AXIS_NONE = ""
AXIS_FORWARD = "forward"
AXIS_YAW = "yaw"


@dataclass(frozen=True)
class BlockageParams:
    """Tuning for :class:`BlockageMonitor`.

    Attributes:
        window_s: Length of the trailing window progress is measured over (s).
            Long enough to out-live pose noise, short enough to react before the
            drone has been grinding against a wall for a while.
        min_cmd_vx: Forward speed below which the tick does not count as "trying
            to translate" (m/s). Must sit above the minimum-force floor, or ticks
            the motors ignored would be counted as attempts.
        min_cmd_wz: Yaw rate below which the tick does not count as "trying to
            rotate" (rad/s). Keep it above the envelope's yaw minimum-force
            floor, or ticks the motors ignored by design would count as failed
            attempts.
        min_cmd_distance_m: Commanded distance that must accumulate inside the
            window before the ratio means anything (m). Guards against declaring
            a blockage from a couple of ticks of rounding.
        min_cmd_yaw_rad: Same guard for the yaw axis (rad).
        progress_frac: Achieved/commanded ratio below which the axis counts as
            making no progress (0..1). Not zero: the calibration is approximate
            and a heavy drone always under-delivers, so ~0.3 means "barely moved",
            not "moved slightly less than asked".
        confirm_ticks: Consecutive bad ticks needed to declare a blockage. This is
            a tick count, never a wall-clock timer: the controller must react to
            evidence, not to a clock that keeps running while it holds still.
        clear_ticks: Consecutive good ticks needed to clear one.
        stale_clear_s: Seconds of NOT pushing an axis after which a standing
            blockage on it is dropped as stale (0 disables). A blockage claims
            "pushing here does nothing"; once the follower/planner has stopped
            driving into it -- rerouted away, boxed in, or held -- that claim is
            no longer being tested, and left latched it freezes the drone against
            a spot nothing is re-probing. This must comfortably exceed the idle
            stretch of an escape manoeuvre (its brake/probe/settle phases, ~2 s),
            so a live escape can never clear the very blockage it is escaping.
        use_effectiveness: Whether to also believe the localization provider's
            ``cmd_effectiveness``.
        eff_floor: ``cmd_effectiveness`` at/below which the provider is taken to
            be reporting a stuck drone.
        enabled: False disables the detector outright: it never confirms a
            blockage, so no escape reflex ever runs and nothing is reported to
            the planner. The kill switch for flights where the platform itself
            under-delivers on every axis (weak battery, payload, trim) -- there
            the detector reads honest weakness as walls and the escapes consume
            the flight. Re-enable once the airframe demonstrably responds again.
    """

    enabled: bool = True
    window_s: float = 1.2
    min_cmd_vx: float = 0.07
    min_cmd_wz: float = 0.21
    min_cmd_distance_m: float = 0.06
    min_cmd_yaw_rad: float = 0.12
    progress_frac: float = 0.30
    confirm_ticks: int = 5
    clear_ticks: int = 3
    stale_clear_s: float = 4.0
    use_effectiveness: bool = True
    eff_floor: float = 0.15

    def __post_init__(self):
        # type: () -> None
        """Validate the invariants the detector relies on."""
        for name in ("window_s", "min_cmd_vx", "min_cmd_wz",
                     "min_cmd_distance_m", "min_cmd_yaw_rad"):
            if getattr(self, name) <= 0.0:
                raise ValueError("BlockageParams." + name + " must be > 0")
        if not 0.0 < self.progress_frac < 1.0:
            raise ValueError("BlockageParams.progress_frac must be in (0, 1)")
        for name in ("confirm_ticks", "clear_ticks"):
            if getattr(self, name) < 1:
                raise ValueError("BlockageParams." + name + " must be >= 1")
        if self.stale_clear_s < 0.0:
            raise ValueError("BlockageParams.stale_clear_s must be >= 0")


@dataclass(frozen=True)
class Blockage:
    """What the monitor believes about the drone's ability to move.

    Attributes:
        axis: ``""`` when nothing is blocked, else ``"forward"`` or ``"yaw"``.
        sign: Sign of the command that is not getting through (+1/-1); 0 when
            nothing is blocked. For ``forward`` this is +1 for a forward command;
            for ``yaw``, +1 for a left (CCW) turn.
        ratio: The achieved/commanded ratio that produced the verdict.
        by_provider: True when the localization's ``cmd_effectiveness`` agreed.
    """

    axis: str = AXIS_NONE
    sign: int = 0
    ratio: float = 1.0
    by_provider: bool = False

    @property
    def blocked(self):
        # type: () -> bool
        """True when any axis is confirmed blocked."""
        return self.axis != AXIS_NONE


class BlockageMonitor:
    """Watches commanded vs achieved motion and reports a confirmed blockage."""

    def __init__(self, params=None):
        # type: (Optional[BlockageParams]) -> None
        self.params = params or BlockageParams()
        self.reset()

    def reset(self):
        # type: () -> None
        """Forget the window and every confirmation streak."""
        self._samples = deque()   # (elapsed, x, y, yaw, cum_dist, cum_yaw)
        self._t = 0.0
        self._cum_dist = 0.0
        self._cum_yaw = 0.0
        self._bad_fwd = 0
        self._bad_yaw = 0
        self._good_fwd = 0
        self._good_yaw = 0
        self._idle_fwd_s = 0.0    # seconds the forward axis has gone un-pushed
        self._idle_yaw_s = 0.0    # seconds the yaw axis has gone un-pushed
        self._verdict = Blockage()

    @property
    def verdict(self):
        # type: () -> Blockage
        """The current (debounced) verdict, without advancing the monitor."""
        return self._verdict

    def update(self, pose, cmd_vx, cmd_wz, dt, quality):
        # type: (Pose2D, float, float, float, LocalizationQuality) -> Blockage
        """Advance the monitor one tick and return the current verdict.

        Args:
            pose: Pose this tick, in the path frame.
            cmd_vx: Forward speed commanded on the PREVIOUS tick (m/s) — that is
                the command whose effect this tick's pose reflects.
            cmd_wz: Yaw rate commanded on the previous tick (rad/s).
            dt: Seconds since the previous call.
            quality: Localization quality, used both to discard coasted evidence
                and to read the provider's own effectiveness verdict.

        Returns:
            The debounced :class:`Blockage`.
        """
        if dt <= 0.0:
            raise ValueError("BlockageMonitor.update: dt must be > 0")
        p = self.params
        if not p.enabled:
            # Killed by config: never confirm, and drop any standing verdict so
            # a blockage latched before the switch flipped cannot linger.
            self._verdict = Blockage()
            return self._verdict

        # A coasted (or absent) pose is propagated BY the command, so it can only
        # ever agree with it. Drop the window rather than poison it.
        if quality.coasting or not quality.valid:
            self._samples.clear()
            self._cum_dist = 0.0
            self._cum_yaw = 0.0
            return self._verdict

        self._t += dt
        self._cum_dist += abs(cmd_vx) * dt
        self._cum_yaw += abs(cmd_wz) * dt
        self._samples.append((self._t, pose.x, pose.y, pose.yaw,
                              self._cum_dist, self._cum_yaw))
        while len(self._samples) > 1 and self._t - self._samples[0][0] > p.window_s:
            self._samples.popleft()
        if len(self._samples) < 2:
            return self._verdict

        t0, x0, y0, yaw0, d0, a0 = self._samples[0]
        want_dist = self._cum_dist - d0
        want_yaw = self._cum_yaw - a0

        eff_says_stuck = (p.use_effectiveness
                          and quality.cmd_effectiveness <= p.eff_floor)

        # ── forward axis ──
        fwd_ratio = 1.0
        if abs(cmd_vx) >= p.min_cmd_vx and want_dist >= p.min_cmd_distance_m:
            self._idle_fwd_s = 0.0
            # Progress along the direction that was commanded at window start,
            # not raw displacement: sideways drift is not forward progress.
            moved = ((pose.x - x0) * cos(yaw0) + (pose.y - y0) * sin(yaw0))
            if cmd_vx < 0.0:
                moved = -moved
            fwd_ratio = max(0.0, moved) / want_dist
            if fwd_ratio < p.progress_frac or eff_says_stuck:
                self._bad_fwd += 1
                self._good_fwd = 0
            else:
                self._good_fwd += 1
                self._bad_fwd = 0
        else:
            # Not pushing this axis. Break the bad streak -- confirmation needs
            # CONSECUTIVE bad ticks and an un-pushed tick is not evidence, so a
            # stale streak must not keep re-confirming the verdict -- and start
            # timing the idle. A confirmed blockage cannot GROW here, and a standing
            # one goes stale: nobody is driving into it, so "pushing here does
            # nothing" is no longer under test. Without this a block found once
            # latches the drone forever the moment the follower stops pushing (an
            # escape probe, a hold, or a boxed-in reroute) -- a phantom obstacle
            # turned into a permanent freeze. _decide drops it after stale_clear_s.
            self._bad_fwd = 0
            self._idle_fwd_s += dt

        # ── yaw axis ──
        yaw_ratio = 1.0
        if abs(cmd_wz) >= p.min_cmd_wz and want_yaw >= p.min_cmd_yaw_rad:
            self._idle_yaw_s = 0.0
            turned = normalize_angle(pose.yaw - yaw0)
            if cmd_wz < 0.0:
                turned = -turned
            yaw_ratio = max(0.0, turned) / want_yaw
            if yaw_ratio < p.progress_frac:
                self._bad_yaw += 1
                self._good_yaw = 0
            else:
                self._good_yaw += 1
                self._bad_yaw = 0
        else:
            # Not turning: break the bad streak and time the idle (see forward).
            self._bad_yaw = 0
            self._idle_yaw_s += dt

        return self._decide(cmd_vx, cmd_wz, fwd_ratio, yaw_ratio, eff_says_stuck)

    def _decide(self, cmd_vx, cmd_wz, fwd_ratio, yaw_ratio, eff_says_stuck):
        # type: (float, float, float, float, bool) -> Blockage
        """Apply the confirm/clear streaks and update the standing verdict.

        Forward wins a tie: it is both the more common blockage and the more
        dangerous one to keep pushing on.
        """
        p = self.params
        if self._bad_fwd >= p.confirm_ticks:
            self._verdict = Blockage(AXIS_FORWARD, 1 if cmd_vx >= 0.0 else -1,
                                     fwd_ratio, eff_says_stuck)
        elif self._bad_yaw >= p.confirm_ticks:
            self._verdict = Blockage(AXIS_YAW, 1 if cmd_wz >= 0.0 else -1,
                                     yaw_ratio, eff_says_stuck)
        elif self._verdict.blocked:
            if self._verdict.axis == AXIS_FORWARD:
                good, idle = self._good_fwd, self._idle_fwd_s
            else:
                good, idle = self._good_yaw, self._idle_yaw_s
            # Cleared by motion (the drone got through) OR by going stale (the
            # drone stopped pushing into it long enough that the claim no longer
            # stands).
            stale = p.stale_clear_s > 0.0 and idle >= p.stale_clear_s
            if good >= p.clear_ticks or stale:
                self._verdict = Blockage()
        return self._verdict
