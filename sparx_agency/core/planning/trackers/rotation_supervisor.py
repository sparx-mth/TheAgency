"""Rotation-freeze + periodic re-observation supervisor for CONTINUOUS trackers.

The one-axis :class:`WaypointFollower` freezes the voxel map during its discrete
turns and re-observes the scene while stopped between ~25 deg bursts. The
holonomic trackers (Pure Pursuit, multi-axis) instead crab and yaw continuously
along a spline, so they have no discrete "turn" to hang that discipline on. This
supervisor wraps such a tracker: watching the yaw rate it commands and the
drone's heading, it decides, per tick, whether to FREEZE the map (the adapter
maps this to the platform's ``turning`` demo-mode, which the mode-authoritative
depth gate keys on) and whether to HOLD the tracker (stop it) for a stationary
re-observation. A continuous turn therefore becomes "freeze throughout the turn,
and stop every ``reobserve_every_rad`` of rotation to re-observe the scene from a
standstill" — the same map guarantee the one-axis follower gives, for any tracker.

Design (matches the user's choices for the holonomic controllers):
  * freeze is RATE-based: the map is frozen for the WHOLE turn, from the moment
    the tracker commands a yaw rate above ``wz_turn_on`` until the turn ends
    (commanded rate stays below ``wz_turn_off`` for ``turn_off_ticks``). A gentle
    path-tracking yaw below ``wz_turn_on`` stays live.
  * every ``reobserve_every_rad`` of accumulated rotation the supervisor STOPS the
    tracker, waits out the yaw coast (still frozen), then re-observes ``settle_map_updates``
    fresh voxel updates while stopped (sensors live) before resuming the turn.
  * when the turn ends it does one final stop + re-observation, then resumes cruise.

ROS-free and clock-free: fed the current yaw, the tracker's last commanded yaw
rate, ``dt`` and a monotonically increasing voxel-update count; it owns no I/O.
Everything is timed out (``max_coast_s`` / ``map_wait_timeout_s``) so a mapping or
localization stall can never hang the drone. ``enabled=False`` makes every
decision ``freeze=False, hold=False`` — the master off switch.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import radians
from typing import Optional

from sparx_agency.core.common.types import normalize_angle

#: Supervisor states (also returned on the decision for logging).
CRUISE = "CRUISE"          # not turning; map live, tracker free
TURNING = "TURNING"        # actively turning; map frozen, tracker free
STOP_COAST = "STOP_COAST"  # stopping to re-observe; still coasting (frozen)
STOP_DWELL = "STOP_DWELL"  # stopped, sensors live, counting re-observations
DISABLED = "DISABLED"      # feature turned off


@dataclass(frozen=True)
class RotationSupervisorParams:
    """Tuning for :class:`RotationReobserveSupervisor`.

    Attributes:
        enabled: Master on/off. False => never freeze / never stop (map stays live
            through turns); use it when depth is trusted during rotation.
        wz_turn_on: Commanded yaw rate (rad/s) at or above which the tracker is
            treated as actively turning (freeze arms). Above path-tracking jitter.
        wz_turn_off: Commanded yaw rate (rad/s) below which the turn is ending
            (hysteresis vs ``wz_turn_on``).
        turn_off_ticks: Consecutive below-``wz_turn_off`` ticks to declare the turn
            over (debounce; also absorbs the yaw spin-up after a mid-turn stop).
        reobserve_every_rad: Accumulated rotation (rad) between mid-turn stops. A
            long turn is chopped into segments of at most this much rotation, each
            followed by a stationary re-observation.
        settle_eps: Measured yaw rate (rad/s) below which the post-stop coast is
            considered finished and the (unfrozen) dwell begins.
        settle_dwell_s: Minimum stationary dwell (s) before a stop is done.
        settle_map_updates: Fresh voxel updates required at each stop before
            resuming (>=2: never continue on a map built during the turn).
        max_coast_s: Cap (s) on the post-stop coast wait, so a noisy yaw that never
            reads below ``settle_eps`` cannot hang the stop.
        map_wait_timeout_s: Cap (s) on the re-observation wait, so a mapping stall
            cannot hang flight (the stop proceeds after this, logged upstream).
    """

    enabled: bool = True
    wz_turn_on: float = 0.20
    wz_turn_off: float = 0.10
    turn_off_ticks: int = 3
    reobserve_every_rad: float = radians(25.0)
    settle_eps: float = 0.05
    settle_dwell_s: float = 0.8
    settle_map_updates: int = 2
    max_coast_s: float = 2.0
    map_wait_timeout_s: float = 3.0


@dataclass(frozen=True)
class SupervisorDecision:
    """One tick's decision.

    Attributes:
        freeze: Freeze the voxel map this tick (adapter -> request ``turning``).
        hold: Stop the tracker this tick (adapter -> pass ``hold=True`` to it).
        state: Supervisor state label (diagnostics).
        reobserving: True while stopped and actively counting re-observations.
    """

    freeze: bool
    hold: bool
    state: str
    reobserving: bool = False


class RotationReobserveSupervisor:
    """Impose freeze-throughout + periodic stop-and-re-observe on a continuous tracker."""

    def __init__(self, params: Optional[RotationSupervisorParams] = None) -> None:
        self.p = params or RotationSupervisorParams()
        self.reset()

    def reset(self) -> None:
        """Clear all state to CRUISE."""
        self._state = CRUISE
        self._prev_yaw = None            # type: Optional[float]
        self._seg_accum = 0.0            # |rotation| since the last stop (rad)
        self._off_ticks = 0             # consecutive below-wz_turn_off ticks
        self._coast_s = 0.0
        self._dwell_s = 0.0
        self._wait_s = 0.0
        self._map_baseline = 0
        self._stop_final = False        # the stop we are in ends the turn

    @property
    def state(self) -> str:
        return self._state

    def update(self, yaw: float, cmd_wz: float, dt: float,
               map_update_count: int) -> SupervisorDecision:
        """Advance one tick and return the freeze/hold decision.

        Args:
            yaw: Current heading (rad); measured yaw rate is derived from its delta.
            cmd_wz: The yaw rate the tracker commanded last tick (its turn intent).
            dt: Seconds since the previous call.
            map_update_count: Monotonic count of voxel/BEV updates fused so far.
        """
        if not self.p.enabled:
            self._prev_yaw = yaw
            return SupervisorDecision(freeze=False, hold=False, state=DISABLED)

        dt = max(1e-6, float(dt))
        meas_wz = (0.0 if self._prev_yaw is None
                   else normalize_angle(yaw - self._prev_yaw) / dt)
        self._prev_yaw = yaw

        if self._state == CRUISE:
            return self._cruise(cmd_wz)
        if self._state == TURNING:
            return self._turning(cmd_wz, meas_wz, dt)
        if self._state == STOP_COAST:
            return self._stop_coast(meas_wz, dt)
        return self._stop_dwell(dt, map_update_count)   # STOP_DWELL

    # ── states ──────────────────────────────────────────────────────
    def _cruise(self, cmd_wz: float) -> SupervisorDecision:
        if abs(cmd_wz) >= self.p.wz_turn_on:
            self._state = TURNING
            self._seg_accum = 0.0
            self._off_ticks = 0
            return SupervisorDecision(freeze=True, hold=False, state=TURNING)
        return SupervisorDecision(freeze=False, hold=False, state=CRUISE)

    def _turning(self, cmd_wz: float, meas_wz: float, dt: float) -> SupervisorDecision:
        self._seg_accum += abs(meas_wz) * dt
        self._off_ticks = (self._off_ticks + 1
                           if abs(cmd_wz) < self.p.wz_turn_off else 0)
        turn_over = self._off_ticks >= self.p.turn_off_ticks
        seg_full = self._seg_accum >= self.p.reobserve_every_rad
        if turn_over or seg_full:
            # Begin a stationary re-observation. If the turn is over this is the
            # final stop (then cruise); otherwise it is a mid-turn checkpoint
            # (re-observe, then keep turning).
            self._enter_stop(final=turn_over)
            return SupervisorDecision(freeze=True, hold=True, state=STOP_COAST)
        return SupervisorDecision(freeze=True, hold=False, state=TURNING)

    def _stop_coast(self, meas_wz: float, dt: float) -> SupervisorDecision:
        # Still physically rotating: keep the map frozen so end-of-turn inertia
        # frames are discarded. Once the yaw has settled (or the coast caps out),
        # begin the unfrozen dwell and snapshot the re-observation baseline.
        self._coast_s += dt
        if abs(meas_wz) < self.p.settle_eps or self._coast_s >= self.p.max_coast_s:
            self._state = STOP_DWELL
            self._dwell_s = 0.0
            self._wait_s = 0.0
            self._map_baseline = None   # snapshot on the first dwell tick
            return SupervisorDecision(freeze=False, hold=True, state=STOP_DWELL,
                                      reobserving=True)
        return SupervisorDecision(freeze=True, hold=True, state=STOP_COAST)

    def _stop_dwell(self, dt: float, map_update_count: int) -> SupervisorDecision:
        if self._map_baseline is None:
            self._map_baseline = int(map_update_count)   # first dwell tick
        self._dwell_s += dt
        self._wait_s += dt
        got = int(map_update_count) - self._map_baseline
        reobserved = self._dwell_s >= self.p.settle_dwell_s and got >= self.p.settle_map_updates
        timed_out = self._wait_s >= self.p.map_wait_timeout_s
        if reobserved or timed_out:
            if self._stop_final:
                self._state = CRUISE
                return SupervisorDecision(freeze=False, hold=False, state=CRUISE)
            # Mid-turn checkpoint done -> resume turning (still frozen).
            self._state = TURNING
            self._seg_accum = 0.0
            self._off_ticks = 0
            return SupervisorDecision(freeze=True, hold=False, state=TURNING)
        return SupervisorDecision(freeze=False, hold=True, state=STOP_DWELL,
                                  reobserving=True)

    # ── helpers ─────────────────────────────────────────────────────
    def _enter_stop(self, final: bool) -> None:
        self._stop_final = final
        self._state = STOP_COAST
        self._coast_s = 0.0
