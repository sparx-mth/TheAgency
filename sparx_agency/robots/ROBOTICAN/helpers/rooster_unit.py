# rooster_unit.py
"""Single command-and-control owner for one Rooster drone.

RoosterUnit is the only object that talks to a given Rooster's FCU (manual
axes, keep-alive, force_arm). It is wrapped by
adapters/rooster_command_unit.py, whose job is to accept commands from
any source - the manual Tkinter UI today, a planner node in the future -
over a single ROS topic and call into this same RoosterUnit instance. That
way there is exactly one place per drone that can arm/disarm/move it.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from rclpy.node import Node
from std_srvs.srv import SetBool
from fcu_driver_interfaces.msg import ManualControl
from rooster_handler_interfaces.msg import KeepAlive
from rooster_manager_interfaces.msg import RoosterState

from sparx_agency.robots.common.math_utils import clamp_symmetric

# From RoosterState.msg / KeepAlive.msg (rooster_manager_interfaces,
# rooster_handler_interfaces). Only POSITION and ALTITUDE are currently
# used/tested by this repo -- see the requested_flight_mode docstring above.
FLIGHT_MODE_NONE = 0
FLIGHT_MODE_GROUND_ROLL = 1
FLIGHT_MODE_MANUAL = 2
FLIGHT_MODE_POSITION = 3
FLIGHT_MODE_ALTITUDE = 4
FLIGHT_MODE_ACRO = 5
FLIGHT_MODE_STABILIZED = 6
MAX_AXIS = 1000.0

# Arming: let zeroed axes reach the FCU before arming, then confirm via
# telemetry rather than trusting the force_arm service response alone -
# the FCU can accept the request and still reject arming afterwards
# (e.g. preflight throttle-centering check).
ARM_SETTLE_SEC = 0.3
ARM_CONFIRM_TIMEOUT_SEC = 2.0

# Landing must never free-fall: ranger (downward rangefinder, meters) is the
# ground-truth signal for "has it actually landed", not the throttle value.
# The commanded throttle only ever ramps down to a fraction of hover thrust,
# so a landing that hasn't reached the ground yet keeps sinking gently
# instead of cutting power mid-air (confirmed live: starting a land from an
# abnormal altitude with the old "disarm once throttle hits 0" logic caused
# a real free-fall crash, because throttle bottomed out long before the
# drone had actually descended).
LAND_MIN_THROTTLE_FRACTION = 0.3
GROUND_RANGER_M = 0.3

# Lowest altitude-hold setpoint a nudge may reach, metres AGL. Below this the
# rangefinder is reading floor clutter as often as floor, and the hold loop
# starts chasing furniture. See nudge_altitude_target().
MIN_ALTITUDE_TARGET_M = 0.6


class AxisModel:
    """Manual-control axes (x/y/z/r), each clamped to [-1000, 1000]."""

    def __init__(self):
        self.x = self.y = self.z = self.r = 0.0

    def set(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, r: float = 0.0):
        self.x = clamp_symmetric(x, MAX_AXIS)
        self.y = clamp_symmetric(y, MAX_AXIS)
        self.z = clamp_symmetric(z, MAX_AXIS)
        self.r = clamp_symmetric(r, MAX_AXIS)

    def reset(self):
        self.set()

    def as_tuple(self):
        return self.x, self.y, self.z, self.r


class RoosterUnit:
    """Owns arm/disarm/takeoff/land/manual-move for one Rooster ID.

    Callers (the command-unit node's cmd_nav dispatch, a future planner)
    only ever call these public methods - none of them touch ROS I/O
    directly, so behavior stays identical regardless of who is driving.
    """

    def __init__(
        self,
        node: Node,
        rooster_id: str,
        # 600 was too weak to escape PX4's landing-detector confirm window
        # (LNDMC_TRIG_TIME) before it re-triggers "landed" and disarms; 1000
        # is the value confirmed live to actually climb. See LESSONS.md.
        climb_z: float = 1000.0,
        hover_z: float = 550.0,
        land_step: float = 75.0,
        land_step_interval_sec: float = 1.0,
        land_timeout_sec: float = 30.0,
        # Was 3.0s -- at climb_z=1000 that built up enough velocity to fly
        # into the ceiling before altitude hold could arrest it. Shortened.
        climb_duration_sec: float = 1.0,
        # Coast at hover_z before _enable_altitude_hold() samples its target
        # -- see _climb(). Long enough for climb momentum to mostly settle.
        climb_settle_sec: float = 1.0,
        altitude_hold_kp: float = 500.0,
        # Downward-error gain; <=0 means "same as altitude_hold_kp". See
        # _altitude_hold_tick for why descent needs its own, larger gain.
        #
        # 900 was not enough. Measured over several 600 s flights, the loop
        # parked a steady ~0.22 m ABOVE its own setpoint (holding 1.54-1.57 m
        # against a 1.35 m target) and never converged. Solving the loop for the
        # observed thresholds: descent needs z <= ~400 and hover_z is 700, so a
        # -0.22 m error only reaches the descend zone once the gain exceeds
        # 300/0.22 = 1364. 1500 gives margin, and the per-tick slew limit plus
        # altitude_hold_max_correction still bound what one tick can do.
        altitude_hold_kp_down: float = 1500.0,
        altitude_hold_kd: float = 600.0,
        # Was 200 -- confirmed live (2026-08-13) that's too weak to recover
        # once drifted past max_ranger_m: pinned at hover_z-200 continuously
        # for 155/162s produced a flat-to-slightly-climbing trend (+0.18
        # m/min), not a descent. In the same flight, land()'s throttle ramp
        # (down to hover_z-490) descended cleanly and monotonically from a
        # similar height in ~11s -- so the vehicle *can* respond, -200 just
        # isn't enough push. Raised toward that demonstrated-working range;
        # re-verify against a live drift before trusting it further.
        # 2026-08-17: measured live (open-loop, altitude_hold_kp/kd=0, from a
        # ground start) that the real z-axis response is NOT a smooth thrust
        # curve -- z<=690 produced literally zero climb (ranger frozen at
        # ground level for 16-20s straight, repeated at 550/625/670/690) and
        # z=700 produced a fast, sustained ~1.6 m/s climb that didn't level
        # off on its own -- it climbed straight into the room's real physical
        # ceiling (~3.4-3.5m, confirmed against sphera_jail.yaml's own
        # documented ceiling height) and got physically pinned there. The
        # effective control band is a narrow (<=10 unit) step near 700, not a
        # gradual curve across the full 380-unit correction range below --
        # every prior flight's "rising and falling out of control" was this
        # loop's correction regularly saturating past that narrow band in
        # EITHER direction in one tick (>=700 -> full climb-to-ceiling,
        # <700 -> zero lift, i.e. an uncommanded sink). 380 was 5-40x wider
        # than the band it needs to move within. See LESSONS.md.
        altitude_hold_max_correction: float = 380.0,
        # The actual fix: bound how much z may change in ONE tick, separate
        # from (and tighter than) the correction magnitude bound above. This
        # is what stops the loop from ever jumping straight from "zero lift"
        # to "full climb" (or back) in a single 0.1s tick regardless of how
        # large the computed correction is -- it can still reach the same
        # eventual value, just over several ticks, which is exactly what
        # lets it find and hold the narrow effective band instead of
        # overshooting through it. UNVALIDATED live past a short bench test;
        # tune down further if altitude still swings, not up.
        altitude_hold_max_step: float = 15.0,
        # 2026-08-17: the velocity term feeding kd was a raw single-sample
        # finite difference on the ranger reading -- exactly the kind of
        # signal this file's own docstring already calls "fairly noise-
        # sensitive". Observed live in FLIGHT_MODE_ALTITUDE (see
        # requested_flight_mode): with roll/pitch finally stable, altitude
        # itself settled into a bounded ~0.2m limit-cycle around the
        # target rather than converging -- a noisy kd term is the textbook
        # cause of exactly that symptom (it injects high-frequency
        # correction that a P-only loop wouldn't have). An exponential
        # low-pass filter (time constant, seconds) smooths the estimate
        # before it multiplies kd. <=0.0 disables filtering (raw velocity,
        # the previous behavior).
        altitude_hold_velocity_filter_tau_s: float = 0.5,
        # Was 1.0s -- confirmed live (2026-08-13) that /R1/state (ranger's
        # source) actually updates at ~10Hz, so a 1Hz loop was reacting to
        # only 1 in 10 fresh readings and holding a stale throttle for up to
        # a full second between corrections. Matches this doesn't fix the
        # -380/380 correction bound above, but a loop this slow on a
        # double-integrator plant (throttle -> accel -> velocity ->
        # position) is a likely cause of the documented "10-15s oscillation,
        # never fully settles" behavior on its own. Re-verify live.
        altitude_hold_interval_sec: float = 0.1,
        # <=0.0 disables this entirely -- no ceiling behavior change for any
        # existing caller. Added 2026-08-10 after climb_duration_sec:=5.0
        # (a mission_control.py misconfiguration, since fixed) reintroduced
        # the exact "climbs into the ceiling" failure this file already
        # documents at climb_duration_sec=3.0 above. That fix removes the
        # immediate cause; this is the durable safety net so a future bad
        # config/drift can't do the same thing silently.
        max_ranger_m: float = 0.0,
        # <=0.0 disables -- altitude hold then keeps its original behavior of
        # locking onto whatever ranger the open-loop climb happened to reach
        # (see _enable_altitude_hold). Set this to actually choose a flight
        # height instead of however climb_z/climb_duration_sec/hover_z's
        # open-loop throttle burst happens to land. Added 2026-08-11.
        target_ranger_m: float = 0.0,
        # 2026-08-17: was hardcoded to FLIGHT_MODE_POSITION everywhere below.
        # Made selectable to test the user's own hypothesis after two
        # in-flight tilt events in POSITION mode with zero x/y/r commanded:
        # POSITION holds horizontal POSITION as well as attitude/altitude,
        # so with the stick centered it is actively correcting against
        # whatever the position/velocity estimate says, even though nothing
        # asked it to move. ALTITUDE mode drops that horizontal loop
        # entirely (self-level attitude + altitude hold only, no position
        # estimate in the loop at all) -- if POSITION's own position-hold is
        # fighting a noisy/wrong estimate, ALTITUDE mode should be visibly
        # more stable under the exact same z-axis control. See LESSONS.md.
        requested_flight_mode: int = FLIGHT_MODE_POSITION,
        # Sphera itself keeps one dormant bare-DDS publisher on
        # manual_control that cannot be removed, so 1 "other" is normal here.
        # Anything above this is a real second altitude authority.
        expected_other_manual_publishers: int = 1,
    ):
        self.id = rooster_id
        self.node = node
        self.requested_flight_mode = int(requested_flight_mode)
        self.expected_other_manual_publishers = int(expected_other_manual_publishers)
        self.climb_z = float(climb_z)
        self.hover_z = float(hover_z)
        self.land_step = float(land_step)
        self.land_step_interval_sec = float(land_step_interval_sec)
        self.land_timeout_sec = float(land_timeout_sec)
        self.climb_duration_sec = float(climb_duration_sec)
        self.climb_settle_sec = float(climb_settle_sec)
        self.altitude_hold_kp = float(altitude_hold_kp)
        self.altitude_hold_kp_down = float(altitude_hold_kp_down)
        self.altitude_hold_kd = float(altitude_hold_kd)
        self.altitude_hold_max_correction = float(altitude_hold_max_correction)
        self.altitude_hold_max_step = float(altitude_hold_max_step)
        self.altitude_hold_velocity_filter_tau_s = float(altitude_hold_velocity_filter_tau_s)
        self.altitude_hold_interval_sec = float(altitude_hold_interval_sec)
        self.max_ranger_m = float(max_ranger_m)
        self.target_ranger_m = float(target_ranger_m)

        self.axes = AxisModel()

        self.armed = False
        self.airborne = False
        self.ranger = float("inf")
        self.arm_pending = False
        self.busy_action: Optional[str] = None  # "takeoff" | "land" | None
        self._cancel_event = threading.Event()

        # hover_z is a single open-loop throttle constant (see _climb) with
        # no feedback of its own -- even a near-perfect value drifts slowly,
        # and a draining battery shifts the real thrust curve underneath it
        # mid-flight (confirmed live: hover_z=560 held for ~1-2 min then
        # drifted toward the ceiling). This nulls that drift by nudging z
        # toward whatever ranger reading was captured the moment hover
        # began, rather than holding a fixed throttle forever.
        #
        # PD, not P-only: throttle maps to vertical ACCELERATION here (a
        # double integrator to position), so a proportional-only correction
        # is slow to arrest velocity already built up during the climb phase
        # -- confirmed live, error kept growing for ~25s despite z visibly
        # dropping in response, before finally turning around. The kd term
        # damps the climb/sink RATE directly instead of waiting for enough
        # position error to accumulate.
        #
        # Tuning status (live-tested 2026-07-27, hover_z=560, kp=500/kd=600):
        # converges from a ~0.09m error down to ~0.01-0.02m within about
        # 6-8 ticks, but does NOT fully settle -- it drifts back out to
        # ~0.06m and re-converges in a slow (~10-15s period) bounded
        # oscillation rather than a flat hold. That's a large improvement
        # over the old open-loop behavior (unbounded drift into the
        # ceiling), but it is not a fully damped controller. If tighter
        # hold is needed later: try raising kd relative to kp further, or
        # low-pass filtering ranger before differentiating it (a single
        # 1-sample finite difference at these gains is fairly noise-
        # sensitive) -- don't assume the current gains are final.
        self._holding_altitude = False
        self._hold_ranger_target: Optional[float] = None
        self._hold_prev_ranger: Optional[float] = None
        self._hold_prev_ranger_time: Optional[float] = None
        self._hold_filtered_velocity: Optional[float] = None

        self.manual_pub = node.create_publisher(
            ManualControl, f"/{rooster_id}/manual_control", 10)
        self.keep_alive_pub = node.create_publisher(
            KeepAlive, f"/{rooster_id}/keep_alive", 10)
        self.state_sub = node.create_subscription(
            RoosterState, f"/{rooster_id}/state", self._state_cb, 10)
        self.force_arm_client = node.create_client(
            SetBool, f"/{rooster_id}/fcu/command/force_arm")
        node.create_timer(float(altitude_hold_interval_sec), self._altitude_hold_tick)
        # Altitude authority must be EXCLUSIVE. This class is documented as the
        # single owner of manual_control, but nothing enforced it -- and on
        # 2026-08-17 a stray examples/position_fly_controller.py silently
        # co-published for hours, fighting this loop for the z axis and making
        # every flight test unreproducible. Watch the real publisher count and
        # say so loudly instead of trusting convention.
        self._manual_topic = f"/{rooster_id}/manual_control"
        node.create_timer(5.0, self._check_exclusive_authority)

        node.get_logger().info(f"RoosterUnit ready for {rooster_id}")

    def _check_exclusive_authority(self) -> None:
        """Warn if anything else is also publishing manual_control."""
        try:
            others = self.node.count_publishers(self._manual_topic) - 1
        except Exception:            # count_publishers is rclpy-version dependent
            return
        if others > self.expected_other_manual_publishers:
            self.node.get_logger().warn(
                f"[{self.id}] ALTITUDE AUTHORITY CONFLICT: {others} other "
                f"(expected {self.expected_other_manual_publishers}) "
                f"publisher(s) on {self._manual_topic}. Altitude hold will fight "
                f"them and flights will not be reproducible -- find and stop them "
                f"(e.g. position_fly_controller.py, rooster_manual_control, "
                f"PathRunnerNode, keyboard tools).")

    # ---- telemetry ----

    def _state_cb(self, msg: RoosterState):
        self.armed = msg.armed
        self.airborne = msg.airborne
        self.ranger = msg.ranger

    # ---- publish (called by the owning node's timers) ----

    def publish_manual(self):
        x, y, z, r = self.axes.as_tuple()
        msg = ManualControl()
        msg.x, msg.y, msg.z, msg.r, msg.buttons = x, y, z, r, 0
        self.manual_pub.publish(msg)

    def publish_keep_alive(self):
        msg = KeepAlive()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.is_active = True
        msg.requested_flight_mode = self.requested_flight_mode
        msg.command_reboot = False
        self.keep_alive_pub.publish(msg)

    def _altitude_hold_tick(self):
        """PD correction toward the ranger reading captured when hover
        began. A no-op unless a takeoff has actually engaged hold (see
        _enable_altitude_hold) -- land() disables it immediately so it
        never fights a commanded descent."""
        if not self._holding_altitude or self._hold_ranger_target is None:
            return
        if self.ranger == float("inf"):
            return
        # Loop runs at ~10Hz, about the same as /R1/state's own update rate,
        # so about half the ticks see no new ranger sample at all. Treating
        # that as velocity=0 (as before) alternates full-PD and P-only
        # correction every other tick -- confirmed live 2026-08-13, a real
        # oscillation this caused. Skip entirely until a genuinely new
        # sample arrives instead of guessing a false zero velocity.
        if self._hold_prev_ranger is not None and self.ranger == self._hold_prev_ranger:
            return
        now = time.monotonic()
        error = self._hold_ranger_target - self.ranger  # +: sunk low, -: drifted high
        raw_velocity = 0.0
        dt = 0.0
        if self._hold_prev_ranger is not None and self._hold_prev_ranger_time is not None:
            dt = now - self._hold_prev_ranger_time
            if dt > 0.0:
                raw_velocity = (self.ranger - self._hold_prev_ranger) / dt
        self._hold_prev_ranger = self.ranger
        self._hold_prev_ranger_time = now
        # Exponential low-pass on the raw single-sample finite difference --
        # see altitude_hold_velocity_filter_tau_s's docstring. First real
        # sample seeds the filter directly rather than smoothing from zero.
        tau = self.altitude_hold_velocity_filter_tau_s
        if tau <= 0.0 or dt <= 0.0:
            velocity = raw_velocity
        elif self._hold_filtered_velocity is None:
            velocity = raw_velocity
        else:
            alpha = dt / (tau + dt)
            velocity = self._hold_filtered_velocity + alpha * (raw_velocity - self._hold_filtered_velocity)
        self._hold_filtered_velocity = velocity
        # Asymmetric gain: the z axis is a THREE-zone actuator, not a curve --
        # >=700 climbs hard, ~400-690 does nothing, <=400 descends weakly
        # (measured 2026-08-18 across nine 700s runs). At a symmetric kp=500 the
        # loop only reaches the descend zone at error <= -0.6 m, so every flight
        # overshot its setpoint by 0.35-0.5 m and simply stayed there. A larger
        # downward gain reaches that zone on a realistic error without making a
        # small upward error trigger the 1.6 m/s climb.
        kp = (self.altitude_hold_kp_down if error < 0.0 and self.altitude_hold_kp_down > 0.0
              else self.altitude_hold_kp)
        correction = clamp_symmetric(
            kp * error - self.altitude_hold_kd * velocity,
            self.altitude_hold_max_correction)
        at_ceiling = self.max_ranger_m > 0.0 and self.ranger >= self.max_ranger_m
        if at_ceiling:
            # Never push higher once at/above the ceiling -- correction can
            # still go negative (descend back under it), just not positive.
            # altitude_hold_max_correction alone can't do this: it only
            # bounds the correction's MAGNITUDE per tick, not the resulting
            # absolute altitude, so a sustained drift (see the hover_z drift
            # entry in LESSONS.md) could climb past any per-tick bound.
            correction = min(correction, 0.0)
        wanted_z = self.hover_z + correction
        # Slew-limit the actual axis, not just the correction term: the
        # 700 vs. <=690 measurement above means the wanted_z the formula
        # above computes can legitimately be on the wrong side of the
        # narrow effective band even after the correction clamp, and jumping
        # straight to it in one tick is exactly the "shoots to the ceiling
        # or drops like a stone" failure this fixes. See the constructor's
        # altitude_hold_max_step docstring.
        step = clamp_symmetric(wanted_z - self.axes.z, self.altitude_hold_max_step)
        new_z = self.axes.z + step
        self.axes.set(x=self.axes.x, y=self.axes.y, z=new_z, r=self.axes.r)
        self.node.get_logger().info(
            f"[{self.id}] altitude hold: ranger={self.ranger:.3f}m "
            f"target={self._hold_ranger_target:.3f}m error={error:+.3f}m "
            f"vel={velocity:+.4f}m/s wanted_z={wanted_z:.0f} z={new_z:.0f}"
            + (" [AT CEILING]" if at_ceiling else ""))

    def _enable_altitude_hold(self):
        if self.ranger == float("inf"):
            self.node.get_logger().warn(
                f"[{self.id}] No ranger telemetry yet -- altitude hold not engaged, "
                f"holding raw throttle only.")
            return
        # A configured target overrides the default "hold wherever the climb
        # happened to leave you" behavior -- the PD loop then climbs/descends
        # toward it like any other altitude error, no different code path.
        self._hold_ranger_target = (
            self.target_ranger_m if self.target_ranger_m > 0.0 else self.ranger)
        self._hold_prev_ranger = None
        self._hold_prev_ranger_time = None
        self._hold_filtered_velocity = None
        self._holding_altitude = True
        self.node.get_logger().info(
            f"[{self.id}] Altitude hold engaged at ranger={self.ranger:.3f}m, "
            f"target={self._hold_ranger_target:.3f}m.")

    @property
    def holding_altitude(self) -> bool:
        return self._holding_altitude

    def nudge_altitude_target(self, delta_m: float) -> None:
        """Move the altitude-hold setpoint by ``delta_m`` (signed, metres).

        2026-08-17: added because the ``up``/``down`` cmd_nav actions set a
        raw z pulse via set_axes() that _altitude_hold_tick() then
        overwrites within one tick (<=altitude_hold_interval_sec later) --
        while holding is engaged, "up"/"down" were nearly a no-op. This
        moves the actual setpoint the PD loop is chasing, so the same
        slew-limited, filtered controller already tuned for hover carries
        the climb/descent instead of a competing raw pulse. No-op (logs a
        warning) if altitude hold isn't engaged yet -- there is no sensible
        target to nudge before the first hold point is sampled.
        """
        if not self._holding_altitude or self._hold_ranger_target is None:
            self.node.get_logger().warn(
                f"[{self.id}] nudge_altitude_target({delta_m:+.2f}m) ignored -- "
                f"altitude hold not engaged yet.")
            return
        # Clamped, because the nudge caller has no idea what altitude is safe
        # here. Confirmed live 2026-08-18: FALCON's own climb demand walked the
        # target 1.20 -> 1.80 m within a minute of takeoff, straight past
        # max_ranger_m -- and max_ranger_m only caps the hold loop's CORRECTION,
        # it never bounded the setpoint the loop was chasing.
        ceiling = self.max_ranger_m if self.max_ranger_m > 0.0 else float("inf")
        wanted = self._hold_ranger_target + float(delta_m)
        clamped = max(MIN_ALTITUDE_TARGET_M, min(ceiling, wanted))
        if clamped != wanted:
            self.node.get_logger().warn(
                f"[{self.id}] altitude nudge {delta_m:+.2f}m clamped: "
                f"{wanted:.2f}m is outside "
                f"[{MIN_ALTITUDE_TARGET_M:.2f}, {ceiling:.2f}]m")
        self._hold_ranger_target = clamped
        self.node.get_logger().info(
            f"[{self.id}] altitude target nudged by {delta_m:+.2f}m -> "
            f"{self._hold_ranger_target:.2f}m")

    # ---- manual movement ----

    def set_axes(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, r: float = 0.0):
        """Set the manual-control axes. Like the XTEND UI's twist buttons,
        a value persists until something else changes it (another move
        command, stop(), or a takeoff/land sequence) - there is no
        auto-expiry, matching every other ROBOTICAN control surface in this
        codebase (position_fly_controller.py's w/s/j/l/i/k behave the same
        way; only its keyboard yaw is intentionally momentary)."""
        self.axes.set(x, y, z, r)

    def stop(self):
        """Zero horizontal/yaw movement but hold z (throttle/altitude-hold) -
        zeroing z here would cut hover power and drop the drone. Also
        cancels an in-progress takeoff/land so its background thread
        doesn't wake up and override this hold with climb_z/hover_z or
        the next descent step."""
        if self.busy_action in ("takeoff", "land"):
            self._cancel_event.set()
        self.set_axes(z=self.axes.z)

    # ---- arm / disarm ----

    def arm(self, on_confirmed: Optional[Callable[[], None]] = None,
            on_failed: Optional[Callable[[str], None]] = None):
        if self.arm_pending:
            self.node.get_logger().warn(f"[{self.id}] Arm already in progress.")
            # Bugfix: this branch used to return without invoking either
            # callback. takeoff() sets busy_action="takeoff" BEFORE calling
            # arm(), relying on arm()'s callback to eventually clear it -- if
            # arm() hit exactly this branch (e.g. ARM clicked, then TAKEOFF
            # clicked before the first arm confirmed), busy_action was left
            # stuck forever, silently no-oping every future land()/takeoff()
            # call. Confirmed live: repeated "Arm already in progress." /
            # "Busy with 'takeoff', ignoring takeoff." log spam with no ARM/
            # TAKEOFF ever actually happening.
            if on_failed:
                on_failed("arm already in progress")
            return
        if self.armed:
            self.node.get_logger().warn(f"[{self.id}] Already armed.")
            if on_confirmed:
                on_confirmed()
            return
        self._holding_altitude = False
        self.axes.reset()
        self.arm_pending = True
        threading.Thread(target=self._do_arm, args=(on_confirmed, on_failed), daemon=True).start()

    def _do_arm(self, on_confirmed, on_failed):
        # Give the FCU a moment to actually receive the zeroed
        # manual_control before arming - otherwise it can still see the
        # stale pre-reset throttle value and reject the arm request.
        time.sleep(ARM_SETTLE_SEC)
        if not self.force_arm_client.service_is_ready():
            self.node.get_logger().warn(f"[{self.id}] force_arm service not ready, waiting 2s...")
            time.sleep(2.0)
        req = SetBool.Request()
        req.data = True
        future = self.force_arm_client.call_async(req)

        def _done(fut):
            try:
                resp = fut.result()
            except Exception as e:
                self.node.get_logger().error(f"[{self.id}] force_arm error: {e}")
                self.arm_pending = False
                if on_failed:
                    on_failed(str(e))
                return
            if not resp.success:
                self.node.get_logger().warn(f"[{self.id}] Arm refused: {resp.message}")
                self.arm_pending = False
                if on_failed:
                    on_failed(resp.message)
                return
            threading.Thread(
                target=self._confirm_armed, args=(on_confirmed, on_failed), daemon=True,
            ).start()

        future.add_done_callback(_done)

    def _confirm_armed(self, on_confirmed, on_failed):
        """force_arm succeeding only means the FCU accepted the request - the
        FCU (e.g. preflight checks) can still refuse to actually arm. Poll
        real telemetry before trusting the drone is armed."""
        deadline = time.time() + ARM_CONFIRM_TIMEOUT_SEC
        while time.time() < deadline:
            if self.armed:
                self.arm_pending = False
                self.node.get_logger().info(f"[{self.id}] Armed.")
                if on_confirmed:
                    on_confirmed()
                return
            time.sleep(0.1)
        self.arm_pending = False
        msg = "force_arm accepted but FCU never confirmed armed - check FCU preflight checks."
        self.node.get_logger().error(f"[{self.id}] {msg}")
        if on_failed:
            on_failed(msg)

    def disarm(self, on_done: Optional[Callable[[], None]] = None):
        self._holding_altitude = False
        self.axes.reset()
        if not self.force_arm_client.service_is_ready():
            self.node.get_logger().warn(f"[{self.id}] force_arm service not ready for disarm.")
            return
        req = SetBool.Request()
        req.data = False
        future = self.force_arm_client.call_async(req)

        def _done(fut):
            try:
                resp = fut.result()
            except Exception as e:
                self.node.get_logger().error(f"[{self.id}] disarm error: {e}")
                return
            if resp.success:
                self.node.get_logger().info(f"[{self.id}] Disarmed.")
            else:
                self.node.get_logger().warn(f"[{self.id}] Disarm refused: {resp.message}")
            if on_done:
                on_done()

        future.add_done_callback(_done)

    # ---- takeoff / land scenarios ----

    def takeoff(self, on_hover: Optional[Callable[[], None]] = None,
                on_failed: Optional[Callable[[str], None]] = None):
        """Arm, confirm via telemetry, climb, then hold at hover_z."""
        if self.busy_action:
            self.node.get_logger().warn(f"[{self.id}] Busy with '{self.busy_action}', ignoring takeoff.")
            return
        self.busy_action = "takeoff"
        self._cancel_event.clear()

        def _armed():
            threading.Thread(target=self._climb, args=(on_hover,), daemon=True).start()

        def _failed(reason):
            self.busy_action = None
            if on_failed:
                on_failed(reason)

        self.arm(on_confirmed=_armed, on_failed=_failed)

    def _climb(self, on_hover):
        self.axes.set(z=self.climb_z)
        deadline = time.time() + self.climb_duration_sec
        while time.time() < deadline:
            if self._cancel_event.is_set():
                self.busy_action = None
                self._enable_altitude_hold()
                self.node.get_logger().info(f"[{self.id}] Climb cancelled - holding z={self.axes.z:.0f}.")
                if on_hover:
                    on_hover()
                return
            # Climb is open-loop (fixed throttle, no ranger feedback) -- see
            # the climb_duration_sec comment above for why that alone already
            # overshot into the ceiling once. This is the same early-exit as
            # a cancel, just triggered by the ranger instead of the user.
            if (self.max_ranger_m > 0.0 and self.ranger != float("inf")
                    and self.ranger >= self.max_ranger_m):
                self.busy_action = None
                self._enable_altitude_hold()
                self.node.get_logger().warn(
                    f"[{self.id}] Climb hit ceiling (ranger={self.ranger:.3f}m >= "
                    f"max_ranger_m={self.max_ranger_m:.2f}m) - holding early.")
                if on_hover:
                    on_hover()
                return
            time.sleep(0.1)
        self.axes.set(z=self.hover_z)
        # Coast before sampling the hold target: _enable_altitude_hold()
        # locks onto whatever ranger reads the instant it's called, but the
        # climb burst leaves real upward momentum that keeps carrying the
        # drone past that reading. Without this it locked onto a too-low
        # target mid-climb and sank back to the ground trying to return to
        # it. See LESSONS.md.
        time.sleep(self.climb_settle_sec)
        self._enable_altitude_hold()
        self.busy_action = None
        self.node.get_logger().info(f"[{self.id}] Climb done - hovering at z={self.hover_z}.")
        if on_hover:
            on_hover()

    def land(self, on_landed: Optional[Callable[[], None]] = None):
        """Step z down by land_step every land_step_interval_sec, starting
        from whatever altitude it's currently holding, until telemetry
        reports not-airborne or ranger confirms ground, then disarm."""
        if self.busy_action:
            self.node.get_logger().warn(f"[{self.id}] Busy with '{self.busy_action}', ignoring land.")
            return
        self.busy_action = "land"
        self._cancel_event.clear()
        threading.Thread(target=self._do_land, args=(on_landed,), daemon=True).start()

    def _do_land(self, on_landed):
        self._holding_altitude = False
        deadline = time.time() + self.land_timeout_sec
        min_throttle = self.hover_z * LAND_MIN_THROTTLE_FRACTION
        reason = "timeout"
        while time.time() < deadline:
            if self._cancel_event.is_set():
                self.busy_action = None
                self._enable_altitude_hold()
                self.node.get_logger().info(f"[{self.id}] Land cancelled - holding z={self.axes.z:.0f}.")
                if on_landed:
                    on_landed()
                return
            if not self.airborne or self.ranger <= GROUND_RANGER_M:
                reason = f"airborne={self.airborne}, ranger={self.ranger:.2f}m"
                break
            time.sleep(self.land_step_interval_sec)
            self.axes.set(z=max(min_throttle, self.axes.z - self.land_step))
            self.node.get_logger().info(
                f"[{self.id}] Landing: z={self.axes.z:.0f}, ranger={self.ranger:.2f}m")
        self.axes.reset()
        time.sleep(ARM_SETTLE_SEC)
        self.disarm()
        self.busy_action = None
        self.node.get_logger().info(f"[{self.id}] Land sequence complete ({reason}).")
        if on_landed:
            on_landed()
