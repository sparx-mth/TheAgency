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
    Also the pre-arm hold: while ``arm_ready`` is False the machine cannot leave SEARCH
    at all (no confirmation, arrival or aim takes it out), so the obstacle-aware follower
    keeps control until the drone has flown the hard part of the route into the room.
  * ``AIM``         — the route reached a *staging* goal (a vantage point, not the
    object itself) without a confirmation, and the object's catalogued position IS
    known: turn the nose onto its bearing and hold still, looking (see the
    aim-bearing policy). Entered only when the caller reports ``aim_ready``; it
    takes precedence over both SCAN and the arrival-land, because arriving at a
    staging point is not arriving at the object. Ends one of two ways — the
    detector confirms the object (→ acquire, the whole point of aiming), or the
    caller reports ``aim_done``. What ``aim_done`` does depends on
    ``aim_fail_to_scan``: by default it returns to SEARCH with ``escalate_goal``
    raised so the mission re-targets the object's own coordinate, but with
    ``aim_fail_to_scan`` it hands off to SCAN instead — sweep the room in place
    rather than ever fly onto the object's imprecise catalogued position.
  * ``SCAN``        — the route reached its goal without a confirmation, so the node
    drives a slow rotate-with-stops sweep of the room (see the scan-search policy)
    to look for the object. Still "searching", but the node now owns ``/cmd_vel``.
    With ``scan_land_revolutions`` set, the sweep is bounded: the caller feeds the
    per-tick swept angle and the machine commits to the terminal LAND once the sweep
    has turned that many full revolutions without a confirmation (the give-up land).
  * ``ACQUIRE_STOP`` — the target was just confirmed+locked: hold a brief
    stop-in-place (``acquire_stop_s``) before approaching, so the drone arrests any
    inherited route (A*/NavDP) motion and no stale follower command lingers into the
    visual approach. The node owns ``/cmd_vel`` here and publishes a real zero-stop.
    Skipped (straight to APPROACH) when ``acquire_stop_s <= 0``.
  * ``APPROACH``    — servo drives toward the target.
  * ``HOVER_LOCK``  — target centred and close: success, hold and keep tracking.
  * ``RECOVER``     — track lost; node sweeps to re-acquire (see the recovery policy).
  * ``LAND``        — reached the object: the machine never leaves this **terminal**
    state, ``drive_cmd_vel`` goes False, and the ``land`` flag is raised so the node
    stops sending motion commands and triggers the platform's land sequence. Three
    independent triggers can enter it:
      - **visual reach** (from APPROACH / HOVER_LOCK): the metric range to the target
        has stayed ``<= land_range_m`` for ``land_confirm_ticks`` consecutive closure
        ticks. Disabled (never entered) when ``land_range_m is None`` (the default),
        preserving the hover-lock-forever behaviour for missions that only hold in
        front of the object.
      - **coordinate reach** (from SEARCH, ``land_at_goal`` on): the coordinate route
        reached its goal (``arrived_at_goal``) still unconfirmed for
        ``arrive_land_confirm_ticks`` consecutive ticks — the object was flown to "by
        A* alone" (never seen by the detector), so the goal *is* the object and the
        mission lands there instead of sweeping the room. Needs only the pose-based
        ``arrived_at_goal`` flag, no depth. Off by default (``land_at_goal`` False),
        which keeps the legacy arrival→SCAN sweep.
      - **sweep exhausted** (from SCAN, ``scan_land_revolutions`` > 0): the room sweep
        turned its configured number of full revolutions without a confirmation, so
        the object is nowhere in view — give up and land in place. Off by default
        (``scan_land_revolutions`` 0), which keeps the sweep-forever behaviour.
    The triggers never contend: the coordinate one lives only in SEARCH, the visual
    one only in APPROACH/HOVER_LOCK, and the sweep one only in SCAN.

Transitions are driven only by booleans the node already has (target confirmed,
track valid, servo at-target, arrived-at-goal) plus the optional metric ``range_m``
for the LAND trigger, and a lost-timer for the recovery timeout — no clock, no I/O.
A short hysteresis on the HOVER_LOCK exit prevents chatter at the success boundary,
and the LAND trigger requires ``land_confirm_ticks`` consecutive in-range ticks so a
single depth glitch cannot land the drone. On a recovery timeout the machine returns
to SEARCH and flags ``reset_acquisition`` so the node clears the confirmation gate and
tracker (from SEARCH it re-enters SCAN on the next tick if still at the goal).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

SEARCH = "SEARCH"
AIM = "AIM"
SCAN = "SCAN"
ACQUIRE_STOP = "ACQUIRE_STOP"
APPROACH = "APPROACH"
HOVER_LOCK = "HOVER_LOCK"
RECOVER = "RECOVER"
LAND = "LAND"


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
        acquire_stop_s: Seconds to hold a stop-in-place in ACQUIRE_STOP right after
            the target is confirmed+locked, before APPROACH begins — long enough for
            the drone to brake off inherited route motion and for the follower's last
            command to lapse. ``0`` disables the settle (confirm+lock -> APPROACH
            directly, the legacy behaviour).
        land_range_m: Metric range (m) at or below which the mission is "reached" and
            the machine commits to the terminal LAND state, abandoning hover-lock so
            the node can stop and land the drone. Needs depth (``range_m`` fed to
            :meth:`update`); ``None`` disables the LAND trigger entirely (the default),
            keeping HOVER_LOCK as the terminal behaviour. Usually set larger than the
            servo's ``target_range_m`` hover standoff so the drone lands on approach
            rather than after settling into the closer hover-lock.
        land_confirm_ticks: Consecutive closure ticks with ``range_m <= land_range_m``
            required before entering LAND, so a single spurious depth reading cannot
            land the drone. The streak resets on any tick that is out of range or has
            no range, and on any state change out of the closure states.
        land_at_goal: When True, arriving at the coordinate goal (``arrived_at_goal``)
            still unconfirmed is itself a terminal-LAND trigger — the object was reached
            "by A* alone" (the goal is its location), so the mission lands there rather
            than sweeping the room. Pose-based: needs no depth / ``range_m``. ``False``
            (default) keeps the legacy arrival→SCAN room sweep. Independent of
            ``land_range_m`` (either, both, or neither may be enabled).
        arrive_land_confirm_ticks: Consecutive SEARCH ticks with ``arrived_at_goal``
            required before the ``land_at_goal`` trigger commits to LAND, so a brief
            pose excursion into the arrival radius cannot land the drone. The streak
            resets on any tick that is not arrived and on any state change. Only used
            when ``land_at_goal`` is True.
        aim_fail_to_scan: When True, an AIM that finishes unseen (``aim_done``) hands
            off to SCAN — sweep the room in place looking for the object — instead of
            returning to SEARCH with ``escalate_goal``. It is how a mission looks for
            the object from the vantage point without EVER flying onto its imprecise
            catalogued coordinate. ``False`` (default) keeps the legacy escalate.
        scan_land_revolutions: Number of full in-place revolutions the room SCAN may
            turn before giving up and committing to the terminal LAND (the object was
            never found). The caller feeds the per-tick swept angle as
            ``scan_swept_delta`` and the machine accumulates it while in SCAN, landing
            once it reaches this many turns. ``0`` (default) disables the cap, so SCAN
            sweeps until the target is confirmed or the goal changes.
    """

    recover_timeout_s: float = 6.0
    recover_confirm_ticks: int = 2
    hover_release_ticks: int = 5
    acquire_stop_s: float = 0.0
    land_range_m: Optional[float] = None
    land_confirm_ticks: int = 3
    land_at_goal: bool = False
    arrive_land_confirm_ticks: int = 5
    aim_fail_to_scan: bool = False
    scan_land_revolutions: float = 0.0

    def __post_init__(self) -> None:
        if self.recover_timeout_s <= 0.0:
            raise ValueError("recover_timeout_s must be > 0.")
        if self.recover_confirm_ticks < 1:
            raise ValueError("recover_confirm_ticks must be >= 1.")
        if self.hover_release_ticks < 1:
            raise ValueError("hover_release_ticks must be >= 1.")
        if self.acquire_stop_s < 0.0:
            raise ValueError("acquire_stop_s must be >= 0.")
        if self.land_range_m is not None and self.land_range_m <= 0.0:
            raise ValueError("land_range_m must be > 0 when set (None disables LAND).")
        if self.land_confirm_ticks < 1:
            raise ValueError("land_confirm_ticks must be >= 1.")
        if self.arrive_land_confirm_ticks < 1:
            raise ValueError("arrive_land_confirm_ticks must be >= 1.")
        if self.scan_land_revolutions < 0.0:
            raise ValueError("scan_land_revolutions must be >= 0 (0 disables the cap).")


@dataclass(frozen=True)
class ApproachDecision:
    """One tick's mission decision.

    Attributes:
        mode: Current state label (one of
            SEARCH/AIM/SCAN/ACQUIRE_STOP/APPROACH/HOVER_LOCK/RECOVER/LAND).
        drive_cmd_vel: True when this node owns ``/cmd_vel`` this tick (everything
            except SEARCH — SCAN drives the sweep, AIM drives the turn and
            ACQUIRE_STOP publishes the settle stop, so they own it too). The node
            requests the ``visual_servoing`` hand-off while this is True and releases
            the follower when it goes False.
        reset_acquisition: True on the RECOVER->SEARCH give-up edge — the node
            clears the confirmation gate and tracker to re-acquire from scratch.
        lost_for_s: Seconds the track has been lost (0 outside RECOVER); pass to
            the recovery policy to size the sweep / give-up.
        land: True only in the terminal LAND state — the mission reached the object
            and the node must stop driving ``/cmd_vel`` and land the drone. Once True
            it stays True (LAND never exits), so the node can latch its land sequence.
        escalate_goal: True on the single AIM->SEARCH edge where aiming finished
            without a confirmation — the node re-targets the coordinate goal at the
            object's own catalogued position, so the planner flies the last leg it
            had deliberately been held back from.
    """

    mode: str
    drive_cmd_vel: bool
    reset_acquisition: bool
    lost_for_s: float
    land: bool = False
    escalate_goal: bool = False


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
        self._settle_for = 0.0
        self._land_confirm = 0
        self._arrive_confirm = 0
        self._scan_swept = 0.0

    @property
    def state(self) -> str:
        return self._state

    def update(self, confirmed: bool, track_valid: bool, at_target: bool,
               dt: float, arrived_at_goal: bool = False,
               range_m: Optional[float] = None, aim_ready: bool = False,
               aim_done: bool = False,
               scan_swept_delta: float = 0.0,
               arm_ready: bool = True) -> ApproachDecision:
        """Advance one tick.

        Args:
            confirmed: Target confirmed by the acquisition gate (used in SEARCH/SCAN).
            track_valid: The tracker holds a (measured or predicted) box this frame.
            at_target: The servo reports the target centred and close enough.
            dt: Seconds since the previous update (for the recovery timer).
            arrived_at_goal: The coordinate route has reached its goal. While True
                and the target is not yet confirmed, the machine sits in AIM or SCAN
                and the node turns / sweeps; when it goes False (goal changed) both
                fall back to SEARCH so the planner flies the new route. Defaults
                False — an offline replay with no planner never scans.
            range_m: Metric range (m) to the tracked target this frame, or None when
                unavailable (no depth). Only used for the LAND trigger; ignored when
                ``land_range_m is None``. Defaults None — a caller that never lands
                need not supply it.
            aim_ready: The goal just reached is a *staging* point and the object's own
                position is known, so arriving there should AIM at the object rather
                than land on / sweep around the staging point. Defaults False, which
                keeps the unstaged behaviour (arrival -> land-at-goal or SCAN).
            aim_done: The aim policy has finished (looked, or timed out) without the
                object being confirmed. Only read in AIM. Defaults False.
            scan_swept_delta: Radians the drone has turned since the previous tick,
                accumulated only while in SCAN toward the ``scan_land_revolutions``
                give-up cap. Ignored outside SCAN and when the cap is disabled, so a
                caller that never bounds the sweep need not supply it. Defaults 0.
            arm_ready: Whether the mission may leave SEARCH at all. ``False`` pins the
                machine in passive SEARCH regardless of confirmation or arrival, so the
                obstacle-aware route follower keeps control -- used to forbid the BLIND
                visual take-over until the drone has flown the hard part of the route
                into the room (the caller latches it True on reaching an entry point).
                Defaults True (no gate), so callers that never gate are unaffected.
        """
        dt = max(0.0, float(dt))

        # LAND is terminal: once reached the machine holds it regardless of inputs,
        # so the node's land sequence is never interrupted by a late track/detection.
        if self._state == LAND:
            return self._decide()

        if self._state == SEARCH:
            # The BLIND visual take-over must not engage until the drone has flown the
            # hard part of the route into the room: while not armed, stay passive so the
            # obstacle-aware follower keeps control, whatever the detector/arrival say.
            if not arm_ready:
                self._arrive_confirm = 0
                return self._decide()
            # Only leave SEARCH once BOTH confirmed and the tracker is actually
            # locked (the node seeds the tracker on confirmation). Otherwise, once
            # the route has reached its goal: AIM first when the goal was only a
            # staging point (aim_ready) -- arriving there is NOT arriving at the
            # object, so neither the arrival-land nor the room sweep may fire yet.
            # Failing that, with land_at_goal a sustained arrival is itself a terminal
            # LAND (the goal IS the object, reached by A* alone); without it, switch
            # to a room sweep (SCAN), the legacy behaviour.
            if confirmed and track_valid:
                self._acquire()
            elif arrived_at_goal and aim_ready and not confirmed:
                self._enter(AIM)
            elif arrived_at_goal and self.cfg.land_at_goal and not confirmed:
                # Reached the goal and the detector has NEVER seen the object
                # ("by A* alone"): accumulate consecutive arrived+unconfirmed ticks
                # in place (SEARCH stays passive, so the follower holds the drone at
                # the goal) and land once it is sustained long enough to rule out a
                # brief pose excursion into the radius. The ``not confirmed`` guard is
                # essential: a target the detector HAS confirmed (but whose tracker
                # cannot hold a box this tick) must NOT be blind-landed on the
                # coordinate -- we keep holding for the tracker to re-acquire and
                # visually approach it, never arrival-land on a seen object.
                self._arrive_confirm += 1
                if self._arrive_confirm >= self.cfg.arrive_land_confirm_ticks:
                    self._enter(LAND)
            elif arrived_at_goal and not self.cfg.land_at_goal:
                self._enter(SCAN)
            else:
                # Not arrived, or arrived-but-confirmed (object seen): reset the
                # arrival-land streak so it only counts sustained unconfirmed arrival.
                self._arrive_confirm = 0
            return self._decide()

        if self._state == AIM:
            # Standing at the staging point with the nose swinging onto the object's
            # bearing. Confirm+lock is the outcome we are aiming FOR -> acquire. If the
            # goal moves out from under us (a retarget) just hand back so the planner
            # flies it. If the aim finishes unseen, either sweep the room in place for
            # it (aim_fail_to_scan -- never fly onto the imprecise coordinate) or hand
            # back to SEARCH and tell the node to escalate the goal to the object's own
            # coordinate (the legacy staged fallback).
            if confirmed and track_valid:
                self._acquire()
            elif not arrived_at_goal:
                self._enter(SEARCH)
            elif aim_done:
                if self.cfg.aim_fail_to_scan:
                    self._enter(SCAN)
                else:
                    self._enter(SEARCH)
                    return self._decide(escalate_goal=True)
            return self._decide()

        if self._state == SCAN:
            # Sweeping the room at the goal. Confirm+lock -> acquire (settle then
            # approach); if the goal moves out from under us (arrived_at_goal
            # cleared) hand back to the planner via SEARCH. Failing both, keep
            # sweeping -- but bound it: once the sweep has turned scan_land_revolutions
            # full turns without finding the object, give up and land in place.
            if confirmed and track_valid:
                self._acquire()
            elif not arrived_at_goal:
                self._enter(SEARCH)
            elif self._sweep_exhausted(scan_swept_delta):
                self._enter(LAND)
            return self._decide()

        if self._state == ACQUIRE_STOP:
            # Just confirmed+locked: hold a stop-in-place for acquire_stop_s before
            # approaching. Time-boxed only (the node publishes the real zero-stop);
            # once elapsed, hand to APPROACH, which routes to RECOVER itself if the
            # brief settle happened to lose the track.
            self._settle_for += dt
            if self._settle_for >= self.cfg.acquire_stop_s:
                self._enter(APPROACH)
            return self._decide()

        if self._state == APPROACH:
            if not track_valid:
                self._enter(RECOVER)
            elif self._land_reached(range_m):
                self._enter(LAND)
            elif at_target:
                self._enter(HOVER_LOCK)
            return self._decide()

        if self._state == HOVER_LOCK:
            if not track_valid:
                self._enter(RECOVER)
            elif self._land_reached(range_m):
                self._enter(LAND)
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
    def _acquire(self) -> None:
        """Target confirmed+locked: settle in place first when configured, else go
        straight to APPROACH (``acquire_stop_s <= 0``, the legacy behaviour)."""
        self._enter(ACQUIRE_STOP if self.cfg.acquire_stop_s > 0.0 else APPROACH)

    def _land_reached(self, range_m: Optional[float]) -> bool:
        """Advance the land-confirm streak and report whether LAND is due.

        Returns True once ``land_confirm_ticks`` consecutive closure ticks have had a
        fresh in-range measurement (``range_m <= land_range_m``). Any tick without one
        (out of range, or no range this frame) resets the streak, so only a sustained
        close reading — not a single depth glitch — commits the drone to landing.
        Always False (and a no-op) when the LAND trigger is disabled.
        """
        if self.cfg.land_range_m is None or range_m is None or range_m > self.cfg.land_range_m:
            self._land_confirm = 0
            return False
        self._land_confirm += 1
        return self._land_confirm >= self.cfg.land_confirm_ticks

    def _sweep_exhausted(self, scan_swept_delta: float) -> bool:
        """Accumulate the room sweep's swept angle and report whether it has run its
        configured number of revolutions.

        Adds ``scan_swept_delta`` (radians turned this tick) to the running total and
        returns True once it reaches ``scan_land_revolutions`` full turns, so the SCAN
        gives up and lands. Always False (and a no-op) when the cap is disabled
        (``scan_land_revolutions`` 0). The accumulator is reset on entering SCAN, so
        each sweep episode counts its revolutions from zero.
        """
        if self.cfg.scan_land_revolutions <= 0.0:
            return False
        self._scan_swept += max(0.0, float(scan_swept_delta))
        return self._scan_swept >= self.cfg.scan_land_revolutions * 2.0 * math.pi

    def _enter(self, new: str) -> None:
        if new == self._state:
            return
        self._state = new
        self._recover_valid = 0
        self._settle_for = 0.0
        # A state change breaks the closure: require a fresh consecutive in-range
        # streak (so e.g. a RECOVER excursion cannot carry stale land progress).
        self._land_confirm = 0
        # Likewise reset the arrival-land streak whenever we leave (or re-enter)
        # SEARCH, so a goal that was cleared and re-reached must re-confirm.
        self._arrive_confirm = 0
        # A fresh sweep counts its revolutions from zero (a re-entered SCAN after an
        # excursion must not carry a stale angle that lands the drone early).
        if new == SCAN:
            self._scan_swept = 0.0
        if new != RECOVER:
            self._lost_for = 0.0
        if new != HOVER_LOCK:
            self._release_count = 0

    def _decide(self, reset_acquisition: bool = False,
                escalate_goal: bool = False) -> ApproachDecision:
        return ApproachDecision(
            mode=self._state,
            # LAND owns /cmd_vel only to stop it; the node handles LAND before the
            # drive/release branch, so False here is a safe default (never release).
            drive_cmd_vel=(self._state not in (SEARCH, LAND)),
            reset_acquisition=reset_acquisition,
            lost_for_s=self._lost_for if self._state == RECOVER else 0.0,
            land=(self._state == LAND),
            escalate_goal=escalate_goal,
        )
