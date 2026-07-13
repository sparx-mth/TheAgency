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

FLIGHT_MODE_POSITION = 3
MAX_AXIS = 1000.0

# Arming: let zeroed axes reach the FCU before arming, then confirm via
# telemetry rather than trusting the force_arm service response alone -
# the FCU can accept the request and still reject arming afterwards
# (e.g. preflight throttle-centering check).
ARM_SETTLE_SEC = 0.3
ARM_CONFIRM_TIMEOUT_SEC = 2.0


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
        climb_z: float = 600.0,
        hover_z: float = 550.0,
        land_step: float = 75.0,
        land_step_interval_sec: float = 1.0,
        land_timeout_sec: float = 30.0,
        climb_duration_sec: float = 3.0,
    ):
        self.id = rooster_id
        self.node = node
        self.climb_z = float(climb_z)
        self.hover_z = float(hover_z)
        self.land_step = float(land_step)
        self.land_step_interval_sec = float(land_step_interval_sec)
        self.land_timeout_sec = float(land_timeout_sec)
        self.climb_duration_sec = float(climb_duration_sec)

        self.axes = AxisModel()

        self.armed = False
        self.airborne = False
        self.arm_pending = False
        self.busy_action: Optional[str] = None  # "takeoff" | "land" | None
        self._cancel_event = threading.Event()

        self.manual_pub = node.create_publisher(
            ManualControl, f"/{rooster_id}/manual_control", 10)
        self.keep_alive_pub = node.create_publisher(
            KeepAlive, f"/{rooster_id}/keep_alive", 10)
        self.state_sub = node.create_subscription(
            RoosterState, f"/{rooster_id}/state", self._state_cb, 10)
        self.force_arm_client = node.create_client(
            SetBool, f"/{rooster_id}/fcu/command/force_arm")

        node.get_logger().info(f"RoosterUnit ready for {rooster_id}")

    # ---- telemetry ----

    def _state_cb(self, msg: RoosterState):
        self.armed = msg.armed
        self.airborne = msg.airborne

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
        msg.requested_flight_mode = FLIGHT_MODE_POSITION
        msg.command_reboot = False
        self.keep_alive_pub.publish(msg)

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
                self.node.get_logger().info(f"[{self.id}] Climb cancelled - holding z={self.axes.z:.0f}.")
                if on_hover:
                    on_hover()
                return
            time.sleep(0.1)
        self.axes.set(z=self.hover_z)
        self.busy_action = None
        self.node.get_logger().info(f"[{self.id}] Climb done - hovering at z={self.hover_z}.")
        if on_hover:
            on_hover()

    def land(self, on_landed: Optional[Callable[[], None]] = None):
        """Step z down by land_step every land_step_interval_sec, starting
        from whatever altitude it's currently holding, until telemetry
        reports not-airborne or throttle bottoms out, then disarm."""
        if self.busy_action:
            self.node.get_logger().warn(f"[{self.id}] Busy with '{self.busy_action}', ignoring land.")
            return
        self.busy_action = "land"
        self._cancel_event.clear()
        threading.Thread(target=self._do_land, args=(on_landed,), daemon=True).start()

    def _do_land(self, on_landed):
        deadline = time.time() + self.land_timeout_sec
        reason = "timeout"
        while time.time() < deadline:
            if self._cancel_event.is_set():
                self.busy_action = None
                self.node.get_logger().info(f"[{self.id}] Land cancelled - holding z={self.axes.z:.0f}.")
                if on_landed:
                    on_landed()
                return
            if not self.airborne:
                reason = "airborne false"
                break
            if self.axes.z <= 0.0:
                reason = "throttle at 0"
                break
            time.sleep(self.land_step_interval_sec)
            self.axes.set(z=max(0.0, self.axes.z - self.land_step))
            self.node.get_logger().info(f"[{self.id}] Landing: z={self.axes.z:.0f}")
        self.axes.reset()
        time.sleep(ARM_SETTLE_SEC)
        self.disarm()
        self.busy_action = None
        self.node.get_logger().info(f"[{self.id}] Land sequence complete ({reason}).")
        if on_landed:
            on_landed()
