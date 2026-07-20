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
        use_effectiveness: Whether to also believe the localization provider's
            ``cmd_effectiveness``.
        eff_floor: ``cmd_effectiveness`` at/below which the provider is taken to
            be reporting a stuck drone.
    """

    window_s: float = 1.2
    min_cmd_vx: float = 0.07
    min_cmd_wz: float = 0.21
    min_cmd_distance_m: float = 0.06
    min_cmd_yaw_rad: float = 0.12
    progress_frac: float = 0.30
    confirm_ticks: int = 5
    clear_ticks: int = 3
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
            # Not trying: neither confirm nor clear. A blockage found while
            # pushing stays believed until motion disproves it.
            pass

        # ── yaw axis ──
        yaw_ratio = 1.0
        if abs(cmd_wz) >= p.min_cmd_wz and want_yaw >= p.min_cmd_yaw_rad:
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
            cleared = (self._good_fwd >= p.clear_ticks
                       if self._verdict.axis == AXIS_FORWARD
                       else self._good_yaw >= p.clear_ticks)
            if cleared:
                self._verdict = Blockage()
        return self._verdict
