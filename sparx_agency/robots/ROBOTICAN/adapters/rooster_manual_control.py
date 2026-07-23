#!/usr/bin/env python3
"""Rooster R1 ManualControl actuation, shared by every VLA bridge.

This is the *platform* half of running a learned policy on a Robotican Rooster R1
(the drone inside the Sphera container). It owns everything a policy must not
know about: the optional Rooster message types, arming, the KeepAlive heartbeat,
and the hold-then-stop latch that ManualControl requires.

Why it exists
-------------
Before the VLAs consolidation, `nomad`, `omnivla` and `internvla_n1` each carried
their own copy of this: the same ``try: from fcu_driver_interfaces.msg import
ManualControl ... except ImportError`` guard, the same ``/{id}/fcu/command/force_arm``
SetBool client, the same UAVState arm-state subscription, the same KeepAlive tick
and the same "publish the held frame at N Hz until it expires, then publish a stop
frame" loop. Four of those methods were byte-identical across bridges. One robot,
one adapter.

What it deliberately does NOT own
---------------------------------
The *meaning* of the ManualControl axes differs by flight mode and by policy --
NoMaD drives ``z`` as thrust (cruise/turn), OmniVLA drives it as tilt with a
-1000 brake, InternVLA-N1 maps discrete VLN actions through a YAML table. Those
mappings stay in each bridge, fed by the per-robot scale tables in
``robots/ROBOTICAN/config/vla/rooster_r1_<policy>.yaml``. This class takes an
already-computed :class:`ManualAxes` and gets it onto the wire correctly.

Axis convention (Rooster ManualControl, all in -1000 .. 1000)
--------------------------------------------------------------
``x``  forward / backward · ``y`` lateral (unused in ground-roll)
``z``  thrust *or* tilt, depending on flight mode · ``r`` yaw rotation

Availability
------------
``fcu_driver_interfaces`` / ``rooster_handler_interfaces`` exist only inside the
Sphera container. :data:`HAS_ROOSTER` is False elsewhere and
:meth:`RoosterManualControl.attach` refuses to wire anything up, so a bridge can
still run in ``handheld`` mode (inference only, no actuation) on a dev box.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from std_msgs.msg import Header

try:
    from fcu_driver_interfaces.msg import ManualControl, UAVState
    from rooster_handler_interfaces.msg import KeepAlive
    HAS_ROOSTER = True
except ImportError:                                  # not inside the Sphera container
    ManualControl = UAVState = KeepAlive = None
    HAS_ROOSTER = False

try:
    from std_srvs.srv import SetBool
except ImportError:                                  # pragma: no cover - ROS2 always has it
    SetBool = None

#: ManualControl axes saturate here; the FCU rejects anything wider.
AXIS_LIMIT = 1000.0

#: Rooster flight mode 1 == GROUND_ROLL (drive on the ground rather than fly).
FLIGHT_MODE_GROUND_ROLL = 1


def clamp_axis(value):
    """Clamp one ManualControl axis into ``[-AXIS_LIMIT, AXIS_LIMIT]``.

    Args:
        value: raw axis value, already scaled by the caller's gain table.

    Returns:
        The clamped value as a ``float`` (the message fields are float32).
    """
    return float(max(-AXIS_LIMIT, min(AXIS_LIMIT, float(value))))


@dataclass(frozen=True)
class ManualAxes:
    """One ManualControl command, in Rooster axis units.

    Attributes:
        x: forward / backward.
        y: lateral; unused in ground-roll mode.
        z: thrust or tilt, depending on the configured flight mode.
        r: yaw rotation.
        buttons: Rooster button bitfield; 0 for autonomous control.
        hold_s: how long this command stays live. ManualControl is a *held*
            command: :class:`RoosterManualControl` republishes it at
            ``publish_rate_hz`` until it expires, then publishes a zero frame.
            Without that the drone keeps executing the last command forever.
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    r: float = 0.0
    buttons: int = 0
    hold_s: float = 0.3

    @staticmethod
    def stop():
        """The all-zero frame published when no command is live."""
        return ManualAxes()


class RoosterManualControl:
    """Rooster R1 actuation attached to any rclpy node.

    Composition, not inheritance: a VLA bridge keeps being its own ``Node`` and
    holds one of these.

    Args:
        node: the ``rclpy.node.Node`` that owns the publishers and timers.
        rooster_id: namespace segment, e.g. ``"R1"`` for ``/R1/manual_control``.
        manual_control_topic: full topic; defaults to ``/<rooster_id>/manual_control``.
        keep_alive_topic: full topic; defaults to ``/<rooster_id>/keep_alive``.
        publish_rate_hz: ManualControl republish rate. The FCU expects a steady
            stream, not one message per decision.
        keep_alive_rate_hz: heartbeat rate.
        flight_mode: requested Rooster flight mode (see
            :data:`FLIGHT_MODE_GROUND_ROLL`).
        idle_axes: what to publish once the held command expires. Defaults to the
            all-zero frame, but **it is not always all-zero**: OmniVLA idles at
            ``z = stop_tilt = -1000``, which actively brakes rather than coasts.
            Getting this wrong changes how the robot stops, so each bridge passes
            its own.
        stabilize_s: settling window after the FCU reports armed, during which
            :meth:`send` refuses commands. Arming spins the rotors up; driving
            immediately makes the first command fight the spin-up. ``0`` disables.
        arm_retry_s: minimum gap between force-arm requests, so a bridge calling
            :meth:`send` at inference rate does not spam the service.
        callback_group: optional rclpy callback group for the timers.
    """

    def __init__(self, node, rooster_id="R1", manual_control_topic=None,
                 keep_alive_topic=None, publish_rate_hz=80.0,
                 keep_alive_rate_hz=10.0, flight_mode=FLIGHT_MODE_GROUND_ROLL,
                 idle_axes=None, stabilize_s=0.0, arm_retry_s=2.0,
                 callback_group=None):
        self.node = node
        self.rooster_id = rooster_id
        self.flight_mode = int(flight_mode)
        self.idle_axes = idle_axes if idle_axes is not None else ManualAxes.stop()
        self.stabilize_s = float(stabilize_s)
        self.arm_retry_s = float(arm_retry_s)
        self._armed_at = 0.0
        self._last_arm_request = 0.0
        self._mc_topic = manual_control_topic or "/%s/manual_control" % rooster_id
        self._ka_topic = keep_alive_topic or "/%s/keep_alive" % rooster_id
        self._publish_rate_hz = float(publish_rate_hz)
        self._keep_alive_rate_hz = float(keep_alive_rate_hz)
        self._callback_group = callback_group

        self._lock = threading.Lock()
        self._held = None            # the live ManualAxes, or None when idle
        self._expires_at = 0.0
        self._armed = False
        self._mc_pub = None
        self._ka_pub = None
        self._arm_client = None
        self._attached = False

    # ── lifecycle ────────────────────────────────────────────────────
    @property
    def available(self):
        """True when the Rooster message packages are importable."""
        return HAS_ROOSTER

    @property
    def armed(self):
        """Last armed state reported by the FCU on ``/<id>/fcu/state``."""
        with self._lock:
            return self._armed

    @property
    def stabilized(self):
        """True once the post-arm settling window has elapsed.

        Always True when ``stabilize_s`` is 0 and the drone is armed.
        """
        with self._lock:
            if not self._armed:
                return False
            return (time.time() - self._armed_at) >= self.stabilize_s

    @property
    def ready(self):
        """True when a command sent now would actually be executed."""
        return self._attached and self.stabilized

    def attach(self):
        """Create the publishers, timers, arm client and state subscription.

        Returns:
            ``True`` if actuation is wired up, ``False`` when the Rooster message
            packages are unavailable (dev box / handheld mode) -- in which case
            this object stays inert and :meth:`send` is a no-op.
        """
        if not HAS_ROOSTER:
            self.node.get_logger().warn(
                "Rooster interfaces unavailable (fcu_driver_interfaces / "
                "rooster_handler_interfaces not importable); running without "
                "actuation. This is expected outside the Sphera container.")
            return False

        kw = {"callback_group": self._callback_group} if self._callback_group else {}
        self._mc_pub = self.node.create_publisher(ManualControl, self._mc_topic, 10)
        self._ka_pub = self.node.create_publisher(KeepAlive, self._ka_topic, 10)
        self.node.create_timer(1.0 / self._publish_rate_hz, self._publish_held, **kw)
        self.node.create_timer(1.0 / self._keep_alive_rate_hz, self._publish_keep_alive, **kw)

        self.node.create_subscription(
            UAVState, "/%s/fcu/state" % self.rooster_id, self._on_uav_state, 10, **kw)
        if SetBool is not None:
            self._arm_client = self.node.create_client(
                SetBool, "/%s/fcu/command/force_arm" % self.rooster_id)
        self._attached = True
        self.node.get_logger().info(
            "Rooster actuation attached: %s @ %.0f Hz, keep_alive %s @ %.0f Hz, mode %d"
            % (self._mc_topic, self._publish_rate_hz, self._ka_topic,
               self._keep_alive_rate_hz, self.flight_mode))
        return True

    # ── commanding ───────────────────────────────────────────────────
    def send(self, axes):
        """Hold ``axes`` live for ``axes.hold_s``, arming first if needed.

        Args:
            axes: the :class:`ManualAxes` to execute.

        Returns:
            ``True`` if the command was accepted. ``False`` if actuation is not
            attached, the drone is not armed yet (an arm request is sent; the
            caller retries next tick), or it is armed but still inside the
            post-arm settling window.
        """
        if not self._attached:
            return False
        if not self.armed:
            self.request_arm()
            return False
        if not self.stabilized:
            return False
        with self._lock:
            self._held = axes
            self._expires_at = time.time() + float(axes.hold_s)
        return True

    def stop(self):
        """Drop the held command so the next tick publishes a zero frame."""
        with self._lock:
            self._held = None
            self._expires_at = 0.0

    def request_arm(self, arm=True):
        """Ask the FCU to force-arm. Non-blocking; safe to call every tick.

        Rate-limited to one request per ``arm_retry_s`` so a bridge calling
        :meth:`send` at inference rate does not flood the service.

        Args:
            arm: ``True`` to arm, ``False`` to disarm.

        Returns:
            ``True`` if a request was actually sent this call.
        """
        if self._arm_client is None or not self._arm_client.service_is_ready():
            return False
        now = time.time()
        with self._lock:
            if now - self._last_arm_request < self.arm_retry_s:
                return False
            self._last_arm_request = now
        request = SetBool.Request()
        request.data = bool(arm)
        self._arm_client.call_async(request)
        return True

    # ── timers ───────────────────────────────────────────────────────
    def _publish_held(self):
        """Republish the live command, or :attr:`idle_axes` once it has expired."""
        if self._mc_pub is None:
            return
        with self._lock:
            if self._held is not None and time.time() < self._expires_at:
                axes = self._held
            else:
                axes = self.idle_axes
                self._held = None
        self._mc_pub.publish(self._to_msg(axes))

    def _publish_keep_alive(self):
        """Publish the heartbeat that stops the Rooster handler disarming us."""
        if self._ka_pub is None:
            return
        msg = KeepAlive()
        msg.header = Header()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.is_active = True
        msg.requested_flight_mode = self.flight_mode
        msg.command_reboot = False
        self._ka_pub.publish(msg)

    def _on_uav_state(self, msg):
        armed = bool(msg.armed)
        with self._lock:
            if armed and not self._armed:
                # Rising edge: start the settling window.
                self._armed_at = time.time()
            self._armed = armed

    def _to_msg(self, axes):
        """:class:`ManualAxes` -> a stamped ``ManualControl`` message."""
        msg = ManualControl()
        msg.header = Header()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.x = clamp_axis(axes.x)
        msg.y = clamp_axis(axes.y)
        msg.z = clamp_axis(axes.z)
        msg.r = clamp_axis(axes.r)
        msg.buttons = int(axes.buttons)
        return msg
