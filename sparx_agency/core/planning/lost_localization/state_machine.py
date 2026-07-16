"""Lost-localization recovery: stop, then climb a ladder of blind manoeuvres.

AprilTag localization only produces a pose while a tag is in view. When the last
tag leaves the frame the pose topic simply goes SILENT -- nothing republishes the
last pose, but every consumer caches it, so the drone flies on a frozen pose and
believes it is somewhere it left seconds ago. That is the failure this recovers
from, and it is why the input here is an AGE rather than a pose: the question is
never "where am I" but "how long since anyone knew".

Two thresholds, because they answer different questions:

  * ``stale_s`` (~2 localization periods) -- "my pose is cold": STOP. Cheap,
    instantly reversible, and correct even for a routine one-frame dropout.
  * ``ladder_s`` -- "it is not coming back on its own": run the ladder. By then
    the drone has been standing still since ``stale_s``, so every rung starts
    from a standstill.

The ladder (see :mod:`.ladder`) retreats the way we came, then climbs, then
sweeps -- cheapest and most-likely-to-work first, each rung followed by a settle
because a stationary camera is what actually re-acquires a tag. Any fresh pose at
any point abandons the ladder and restarts it from the top next time: a pose
means we relocalized, and the next dropout is a new problem, not a continuation.

Everything is open loop. With no pose there is no odometry, so "back a little"
is speed x duration and nothing verifies it. The one exception is the sweep,
which closes on ``yaw`` when the caller has a localization-INDEPENDENT heading
(the platform's own compass/IMU bearing) and otherwise falls back to its timeout
-- which is why the timeout is validated to exceed the nominal sweep time.

ROS-free and clock-free: fed an age, a ``dt`` and optionally a heading, it owns
no I/O and reads no clock. Every rung is time-capped and the ladder is finite, so
recovery always terminates -- in :data:`GIVE_UP`, which is sticky: once we have
decided to land, a late tag must not bounce us back into flying.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Optional

from sparx_agency.core.common.types import ControlCommand, normalize_angle

from .ladder import BACK, CLIMB, STOP, TURN, Rung, build_ladder
from .params import LostLocalizationParams

#: Recovery states (also carried on the decision for logging).
NOMINAL = "NOMINAL"    # localization is fresh (or never seen); we own nothing
HOLD = "HOLD"          # pose is cold: stopped, waiting to see if it returns
LADDER = "LADDER"      # running a rung of the ladder
GIVE_UP = "GIVE_UP"    # ladder exhausted; stopped and asking to land (terminal)
DISABLED = "DISABLED"  # feature turned off

_ZERO = ControlCommand.velocity(0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class RecoveryDecision:
    """One tick's decision.

    Attributes:
        active: Recovery owns cmd_vel this tick. When False the caller must
            publish NOTHING (not zeros) and leave the follower in charge.
        command: The velocity to publish, or None when not active. A zero command
            is meaningful and must be published: the platform holds its last
            command until told otherwise, so silence would let it coast.
        state: One of the module-level state constants.
        rung: Index into the ladder, or -1 when not on it.
        rung_label: The current rung's name (``""`` when not on one).
        give_up: The ladder is exhausted -- the caller should ask to land.
        pose_age_s: The age this decision was made on (diagnostics).
        elapsed_s: Seconds since recovery was entered (0 when NOMINAL).
        sweep_rad: Rotation accumulated by the sweep so far (diagnostics).
    """

    active: bool
    command: Optional[ControlCommand]
    state: str
    rung: int = -1
    rung_label: str = ""
    give_up: bool = False
    pose_age_s: float = 0.0
    elapsed_s: float = 0.0
    sweep_rad: float = 0.0


class LostLocalizationRecovery(object):
    """Stop on a cold pose; escalate through a blind ladder; land if it fails."""

    def __init__(self, params: Optional[LostLocalizationParams] = None) -> None:
        self.p = params or LostLocalizationParams()
        self.ladder = build_ladder(self.p)
        # Deliberately NOT in reset(): a commitment to land must outlive it.
        self._gave_up = False
        self.reset()

    def reset(self) -> None:
        """Clear all state back to NOMINAL.

        Does NOT clear a GIVE_UP: once the ladder has run out and the drone has
        been committed to landing, that decision is final for the life of this
        object (see the module docstring). A caller resetting recovery for a new
        mission leg must not silently un-commit a land already under way; build a
        new instance if that is genuinely what you want.
        """
        self._state = NOMINAL
        self._rung = 0
        self._rung_s = 0.0
        self._elapsed_s = 0.0
        self._fresh_poses = 0
        self._prev_count = None    # type: Optional[int]
        self._sweep_rad = 0.0      # |net rotation| swept (diagnostics + the target)
        self._sweep_net_rad = 0.0  # SIGNED sum; see _accumulate_sweep
        self._prev_yaw = None      # type: Optional[float]

    @property
    def state(self) -> str:
        return self._state

    def update(self, pose_age_s: float, dt: float, pose_count: int,
               yaw: Optional[float] = None) -> RecoveryDecision:
        """Advance one tick and return this tick's decision.

        Args:
            pose_age_s: Seconds since the last localization message ARRIVED. Pass
                ``float("inf")`` when none has ever arrived -- that means the
                localization was never wired up, not that it was lost, and must
                never fly a blind manoeuvre. (The distinction is load-bearing: a
                mis-wired bridge would otherwise put the drone straight into the
                ladder on boot.)
            dt: Seconds since the previous call.
            pose_count: Monotonic count of localization messages received so far.
                The age alone CANNOT debounce the exit: after a single lone
                detection the age sits below ``stale_s`` for ``stale_s/dt`` ticks
                all on its own, so counting fresh ticks would let one flickering
                frame end the recovery. Diffing this count is what distinguishes
                "a pose is still arriving" from "a pose arrived once".
            yaw: A localization-INDEPENDENT heading (rad), e.g. the platform's own
                bearing, used to close the sweep on angle. None => the sweep runs
                open loop and ends on its timeout.

        Returns:
            The decision; see :class:`RecoveryDecision`.
        """
        dt = max(1e-6, float(dt))
        arrived = self._prev_count is not None and int(pose_count) > self._prev_count
        self._prev_count = int(pose_count)

        if not self.p.enabled:
            return RecoveryDecision(False, None, DISABLED)

        # Sticky: landing is irreversible, so a tag re-acquired mid-land must not
        # hand the drone back to the follower halfway down.
        if self._gave_up:
            return self._decide(GIVE_UP, _ZERO, give_up=True,
                                pose_age_s=pose_age_s)

        # Never bootstrapped: no localization has EVER arrived. Stay out of the
        # way entirely -- this is a wiring fault, and a ladder cannot fix it.
        if not isfinite(pose_age_s):
            return RecoveryDecision(False, None, NOMINAL, pose_age_s=pose_age_s)

        if pose_age_s < self.p.stale_s:
            return self._fresh(pose_age_s, arrived)

        if pose_age_s >= self.p.ladder_s:
            # Dead again, not merely slow: void the credit built toward handing
            # back. The threshold is ladder_s, NOT stale_s -- voiding it at
            # stale_s would demand exit_confirm_poses arrivals with no stale tick
            # between them, i.e. a sustained rate above 1/stale_s (3.3 Hz at the
            # defaults). A localization limping back at 1-3 Hz -- an AprilTag seen
            # obliquely, far away or half-occluded, which is precisely the regime
            # this recovery exists for -- would then never satisfy the exit, and
            # the drone would sit in HOLD for ever with the follower passive.
            self._fresh_poses = 0
        if self._state == NOMINAL:
            self._enter(HOLD)
        self._elapsed_s += dt

        if self._state == HOLD:
            return self._hold(pose_age_s, dt)
        return self._ladder(pose_age_s, dt, yaw)

    # ── states ──────────────────────────────────────────────────────
    def _fresh(self, pose_age_s: float, arrived: bool) -> RecoveryDecision:
        """The pose is fresh. Hand back once enough NEW ones have landed."""
        if self._state == NOMINAL:
            return RecoveryDecision(False, None, NOMINAL, pose_age_s=pose_age_s)
        if arrived:
            self._fresh_poses += 1
            if self._state == LADDER:
                # A pose landed, so we are no longer totally lost -- and a blind
                # manoeuvre is only justified while we are. Abandon the ladder and
                # fall back to a plain stop. Not just cosmetic: without this the
                # state stays LADDER, so the NEXT stale tick routes straight back
                # into a rung at stale_s (0.3s) instead of ladder_s (1.0s), and
                # the drone resumes reversing mid-rung on a 0.3s dropout. The rung
                # resets too -- the next escalation starts from the top.
                self._rung = 0
                self._enter(HOLD)
        if self._fresh_poses < self.p.exit_confirm_poses:
            # Not convinced yet: keep the drone stopped rather than resume onto a
            # pose that may be a single flickering detection.
            return self._decide(self._state, _ZERO, pose_age_s=pose_age_s)
        self.reset()
        return RecoveryDecision(False, None, NOMINAL, pose_age_s=pose_age_s)

    def _hold(self, pose_age_s: float, dt: float) -> RecoveryDecision:
        """Cold pose: stop. Escalate to the ladder only if it stays cold."""
        if pose_age_s >= self.p.ladder_s:
            if not self.ladder:
                return self._give_up(pose_age_s)
            self._enter(LADDER)
            return self._ladder(pose_age_s, dt, None)
        return self._decide(HOLD, _ZERO, pose_age_s=pose_age_s)

    def _ladder(self, pose_age_s: float, dt: float,
                yaw: Optional[float]) -> RecoveryDecision:
        """Run the current rung; advance when it is done; give up at the end."""
        if self._rung >= len(self.ladder):
            return self._give_up(pose_age_s)

        rung = self.ladder[self._rung]
        self._rung_s += dt
        if rung.kind == TURN:
            self._accumulate_sweep(yaw)

        if self._rung_done(rung):
            self._rung += 1
            self._rung_s = 0.0
            self._sweep_rad = 0.0
            self._sweep_net_rad = 0.0
            self._prev_yaw = None
            if self._rung >= len(self.ladder):
                return self._give_up(pose_age_s)
            rung = self.ladder[self._rung]

        return self._decide(LADDER, self._command(rung), rung=self._rung,
                            rung_label=rung.label, pose_age_s=pose_age_s)

    # ── helpers ─────────────────────────────────────────────────────
    def _rung_done(self, rung: Rung) -> bool:
        """A rung ends on its duration; a sweep may end early, on angle."""
        if rung.kind == TURN and self._sweep_rad >= self.p.turn_target_rad:
            return True
        return self._rung_s >= rung.duration_s

    def _accumulate_sweep(self, yaw: Optional[float]) -> None:
        """Integrate the heading delta into NET rotation, if the caller has one.

        Sum the SIGNED deltas and take the magnitude of the total -- never sum
        |delta|. Summing magnitudes turns heading noise into phantom progress: the
        deltas of a noisy signal are a random walk whose absolute values only ever
        add up, so at 20 Hz over a ~21 s sweep even ~1 deg of compass noise
        accumulates enough imaginary rotation to declare a full turn after barely
        200 deg of real one -- ending the search early, pointed nowhere useful.
        Summed signed, zero-mean noise cancels instead.

        The magnitude is taken at the END, not per-sample, which also makes this
        agnostic to the platform's sign convention: a compass that counts clockwise
        and one that counts counter-clockwise both accumulate a total whose SIZE is
        the rotation swept. (Per-sample deltas stay far below pi at any sane sweep
        rate, so the unwrap is unambiguous.)
        """
        if yaw is None:
            return
        if self._prev_yaw is not None:
            self._sweep_net_rad += normalize_angle(yaw - self._prev_yaw)
            self._sweep_rad = abs(self._sweep_net_rad)
        self._prev_yaw = yaw

    def _command(self, rung: Rung) -> ControlCommand:
        """The single-axis velocity a rung drives (see :mod:`.ladder`)."""
        if rung.kind == BACK:
            return ControlCommand.velocity(-self.p.back_speed, 0.0, 0.0, 0.0)
        if rung.kind == CLIMB:
            return ControlCommand.velocity(0.0, 0.0, self.p.climb_speed, 0.0)
        if rung.kind == TURN:
            return ControlCommand.velocity(
                0.0, 0.0, 0.0, self.p.turn_rate * self.p.turn_dir)
        return _ZERO   # STOP

    def _give_up(self, pose_age_s: float) -> RecoveryDecision:
        self._gave_up = True
        self._enter(GIVE_UP)
        return self._decide(GIVE_UP, _ZERO, give_up=True, pose_age_s=pose_age_s)

    def _enter(self, state: str) -> None:
        if state == self._state:
            return
        self._state = state
        self._rung_s = 0.0
        self._sweep_rad = 0.0
        self._sweep_net_rad = 0.0
        self._prev_yaw = None

    def _decide(self, state, command, rung=-1, rung_label="", give_up=False,
                pose_age_s=0.0) -> RecoveryDecision:
        return RecoveryDecision(True, command, state, rung=rung,
                                rung_label=rung_label, give_up=give_up,
                                pose_age_s=pose_age_s, elapsed_s=self._elapsed_s,
                                sweep_rad=self._sweep_rad)
