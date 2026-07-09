"""Search <-> approach mission state machine (pure, ROS-free).

The "switch onto a specific object" logic: it decides, per control tick, which
mode the mission is in and — crucially — whether this node should be driving
``/cmd_vel`` at all. In SEARCH the node stays passive so the existing A*/NavDP
follower flies the coordinate route while the detector scans; once the target is
confirmed and the tracker is locked, the machine hands control to the visual servo
and never lets the follower fight it.

States (string labels, à la
:class:`~sparx_agency.core.planning.trackers.rotation_supervisor.RotationReobserveSupervisor`):

  * ``SEARCH``      — planner flies the route; node passive (``drive_cmd_vel=False``).
  * ``SCAN``        — the route reached its goal without a confirmation, so the node
    drives a slow rotate-with-stops sweep of the room (see the scan-search policy)
    to look for the object. Still "searching", but the node now owns ``/cmd_vel``.
  * ``APPROACH``    — servo drives toward the target.
  * ``HOVER_LOCK``  — target centred and close: success, hold and keep tracking.
  * ``RECOVER``     — track lost; node sweeps to re-acquire (see the recovery policy).

Transitions are driven only by booleans the node already has (target confirmed,
track valid, servo at-target, arrived-at-goal), plus a lost-timer for the recovery
timeout — no clock, no I/O. A short hysteresis on the HOVER_LOCK exit prevents
chatter at the success boundary. On a recovery timeout the machine returns to
SEARCH and flags ``reset_acquisition`` so the node clears the confirmation gate and
tracker (from SEARCH it re-enters SCAN on the next tick if still at the goal).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

SEARCH = "SEARCH"
SCAN = "SCAN"
APPROACH = "APPROACH"
HOVER_LOCK = "HOVER_LOCK"
RECOVER = "RECOVER"


@dataclass(frozen=True)
class ApproachFSMConfig:
    """Tuning for :class:`VisualApproachStateMachine`.

    Attributes:
        recover_timeout_s: Time lost in RECOVER before giving up to SEARCH. This
            timer accumulates across the whole recovery episode and is NOT reset by
            a lone valid tick, so intermittent flicker cannot starve it forever.
        recover_confirm_ticks: Consecutive valid-track ticks required in RECOVER to
            credit a true re-acquisition (RECOVER->APPROACH). >1 prevents a single
            spurious re-detection from resetting the episode and ping-ponging.
        hover_release_ticks: Consecutive not-at-target ticks required to fall from
            HOVER_LOCK back to APPROACH (hysteresis; absorbs a moving target's jitter).
    """

    recover_timeout_s: float = 6.0
    recover_confirm_ticks: int = 2
    hover_release_ticks: int = 5

    def __post_init__(self) -> None:
        if self.recover_timeout_s <= 0.0:
            raise ValueError("recover_timeout_s must be > 0.")
        if self.recover_confirm_ticks < 1:
            raise ValueError("recover_confirm_ticks must be >= 1.")
        if self.hover_release_ticks < 1:
            raise ValueError("hover_release_ticks must be >= 1.")


@dataclass(frozen=True)
class ApproachDecision:
    """One tick's mission decision.

    Attributes:
        mode: Current state label (one of SEARCH/SCAN/APPROACH/HOVER_LOCK/RECOVER).
        drive_cmd_vel: True when this node owns ``/cmd_vel`` this tick (everything
            except SEARCH — SCAN drives the sweep, so it owns it too). The node
            requests the ``visual_servoing`` hand-off while this is True and
            releases the follower when it goes False.
        reset_acquisition: True on the RECOVER->SEARCH give-up edge — the node
            clears the confirmation gate and tracker to re-acquire from scratch.
        lost_for_s: Seconds the track has been lost (0 outside RECOVER); pass to
            the recovery policy to size the sweep / give-up.
    """

    mode: str
    drive_cmd_vel: bool
    reset_acquisition: bool
    lost_for_s: float


class VisualApproachStateMachine:
    """Decide SEARCH / APPROACH / HOVER_LOCK / RECOVER each control tick."""

    def __init__(self, config: Optional[ApproachFSMConfig] = None) -> None:
        self.cfg = config or ApproachFSMConfig()
        self.reset()

    def reset(self) -> None:
        """Return to SEARCH and clear timers."""
        self._state = SEARCH
        self._lost_for = 0.0
        self._release_count = 0
        self._recover_valid = 0

    @property
    def state(self) -> str:
        return self._state

    def update(self, confirmed: bool, track_valid: bool, at_target: bool,
               dt: float, arrived_at_goal: bool = False) -> ApproachDecision:
        """Advance one tick.

        Args:
            confirmed: Target confirmed by the acquisition gate (used in SEARCH/SCAN).
            track_valid: The tracker holds a (measured or predicted) box this frame.
            at_target: The servo reports the target centred and close enough.
            dt: Seconds since the previous update (for the recovery timer).
            arrived_at_goal: The coordinate route has reached its goal. While True
                and the target is not yet confirmed, the machine sits in SCAN and
                the node sweeps the room; when it goes False (goal changed) SCAN
                falls back to SEARCH so the planner flies the new route. Defaults
                False — an offline replay with no planner never scans.
        """
        dt = max(0.0, float(dt))

        if self._state == SEARCH:
            # Only leave SEARCH once BOTH confirmed and the tracker is actually
            # locked (the node seeds the tracker on confirmation). Otherwise, once
            # the route has reached its goal, switch to a room sweep (SCAN).
            if confirmed and track_valid:
                self._enter(APPROACH)
            elif arrived_at_goal:
                self._enter(SCAN)
            return self._decide()

        if self._state == SCAN:
            # Sweeping the room at the goal. Confirm+lock -> approach; if the goal
            # moves out from under us (arrived_at_goal cleared) hand back to the
            # planner via SEARCH.
            if confirmed and track_valid:
                self._enter(APPROACH)
            elif not arrived_at_goal:
                self._enter(SEARCH)
            return self._decide()

        if self._state == APPROACH:
            if not track_valid:
                self._enter(RECOVER)
            elif at_target:
                self._enter(HOVER_LOCK)
            return self._decide()

        if self._state == HOVER_LOCK:
            if not track_valid:
                self._enter(RECOVER)
            elif not at_target:
                self._release_count += 1
                if self._release_count >= self.cfg.hover_release_ticks:
                    self._enter(APPROACH)
            else:
                self._release_count = 0
            return self._decide()

        # RECOVER. The lost timer accumulates across the whole episode (a lone
        # valid tick does NOT reset it); a true re-acquisition needs
        # recover_confirm_ticks consecutive valid ticks, so a flickering spurious
        # re-detection cannot hold the drone in RECOVER forever.
        self._lost_for += dt
        reset = False
        if track_valid:
            self._recover_valid += 1
            if self._recover_valid >= self.cfg.recover_confirm_ticks:
                self._enter(APPROACH)
        else:
            self._recover_valid = 0
        if self._state == RECOVER and self._lost_for >= self.cfg.recover_timeout_s:
            self._enter(SEARCH)
            reset = True
        return self._decide(reset_acquisition=reset)

    # ── helpers ──────────────────────────────────────────────────────
    def _enter(self, new: str) -> None:
        if new == self._state:
            return
        self._state = new
        self._recover_valid = 0
        if new != RECOVER:
            self._lost_for = 0.0
        if new != HOVER_LOCK:
            self._release_count = 0

    def _decide(self, reset_acquisition: bool = False) -> ApproachDecision:
        return ApproachDecision(
            mode=self._state,
            drive_cmd_vel=(self._state != SEARCH),
            reset_acquisition=reset_acquisition,
            lost_for_s=self._lost_for if self._state == RECOVER else 0.0,
        )
