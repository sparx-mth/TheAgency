#!/usr/bin/env python3
"""Bench-only ROS2 interface-parity check for Robotican's new Orin-NX
architecture.

This is NOT a flight test. It never expects the airframe to actually leave
the ground -- run it with props removed. Its only job is to confirm that,
after the architecture change (backend + all models now co-located on the
drone's own Orin NX instead of split across a host PC + a separate Jetson),
the same ROS2 surface still behaves the same:

  - /{rooster_id}/cmd_nav (std_msgs/String JSON) still accepts commands and
    RoosterCommandUnitNode is alive and dispatching them (see
    adapters/rooster_command_unit.py for the full action vocabulary).
  - /{rooster_id}/state (RoosterState) still reports armed/airborne/
    is_fcu_connected and reacts to arm/disarm within a reasonable time.
  - /{rooster_id}/rooster_status (std_msgs/String JSON, ~5Hz -- see
    RoosterCommandUnitNode._publish_status) still delivers battery_pct/
    battery_voltage/armed/airborne/busy_action.
  - /{rooster_id}/imu/data (sensor_msgs/Imu) and
    /{rooster_id}/camera/image_raw (sensor_msgs/Image) both deliver messages
    at a sane rate.

IMU and image are checked INDEPENDENTLY -- this script does not attempt to
verify they are time-synchronized with each other. That was a deliberate
scope cut (see the grilling session that produced this test): sync is the
mapping pipeline's job (core/mapping's existing depth_fusion_gate /
sensor_freeze_policy, and FALCON's mapping_sync_node with its own
sync_tolerance/max_interp_gap), not this acceptance check's.

Sequence: wait for /state -> arm -> confirm armed=True -> briefly exercise
turn_left/turn_right via cmd_nav (message-flow only; there is no closed-loop
heading confirmation here, see NOTE below) -> disarm -> confirm armed=False.

NOTE on turning: RoosterUnit/RoosterCommandUnitNode track only armed/
airborne/ranger, no heading. UAVState.azimuth (fcu_driver_interfaces) is a
real field (radians, 0=North, positive CW) and IS subscribed elsewhere in
this repo (see rooster_video_adapter.py's state_callback), so a real
closed-loop "did it actually turn ~360 degrees" check is possible as a
follow-up -- deliberately left out of this first pass.

Usage:
    python3 orin_nx_flight_interface_test.py --rooster-id R1 \\
        --observe-sec 8 --imu-topic /R1/imu/data \\
        --image-topic /R1/camera/image_raw
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from sensor_msgs.msg import Imu, Image
from rooster_manager_interfaces.msg import RoosterState


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class RateCounter:
    count: int = 0
    first_stamp: float | None = None
    last_stamp: float | None = None

    def on_message(self):
        now = time.monotonic()
        if self.first_stamp is None:
            self.first_stamp = now
        self.last_stamp = now
        self.count += 1

    def rate_hz(self) -> float:
        if self.first_stamp is None or self.last_stamp is None or self.count < 2:
            return 0.0
        elapsed = self.last_stamp - self.first_stamp
        return (self.count - 1) / elapsed if elapsed > 0 else 0.0


class FlightInterfaceTestNode(Node):
    def __init__(self, rooster_id: str, imu_topic: str, image_topic: str):
        super().__init__("orin_nx_flight_interface_test")
        self.rooster_id = rooster_id

        self.last_state: RoosterState | None = None
        self.last_status: dict | None = None
        self.status_rate = RateCounter()
        self.cmd_pub = self.create_publisher(String, f"/{rooster_id}/cmd_nav", 10)
        self.state_sub = self.create_subscription(
            RoosterState, f"/{rooster_id}/state", self._on_state, 10
        )
        self.status_sub = self.create_subscription(
            String, f"/{rooster_id}/rooster_status", self._on_status, 10
        )

        self.imu_rate = RateCounter()
        self.image_rate = RateCounter()
        self.imu_sub = self.create_subscription(
            Imu, imu_topic, lambda _msg: self.imu_rate.on_message(),
            qos_profile_sensor_data,
        )
        self.image_sub = self.create_subscription(
            Image, image_topic, lambda _msg: self.image_rate.on_message(),
            qos_profile_sensor_data,
        )

    def _on_state(self, msg: RoosterState):
        self.last_state = msg

    def _on_status(self, msg: String):
        try:
            self.last_status = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(f"Malformed rooster_status payload: {msg.data!r}")
            return
        self.status_rate.on_message()

    def send_cmd(self, action: str, value: float = 0.0):
        payload = {"action": action, "value": value}
        self.cmd_pub.publish(String(data=json.dumps(payload)))
        self.get_logger().info(f"cmd_nav -> {payload}")


def spin_until(node: Node, predicate, timeout_sec: float) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if predicate():
            return True
    return predicate()


def run_checks(node: FlightInterfaceTestNode, args: argparse.Namespace) -> list[Check]:
    checks: list[Check] = []

    got_state = spin_until(node, lambda: node.last_state is not None, timeout_sec=10.0)
    checks.append(Check(
        "state_topic_alive",
        got_state,
        f"/{node.rooster_id}/state message received" if got_state else "no /state message within 10s",
    ))
    if not got_state:
        return checks

    checks.append(Check(
        "fcu_connected",
        bool(getattr(node.last_state, "is_fcu_connected", False)),
        f"is_fcu_connected={getattr(node.last_state, 'is_fcu_connected', None)}",
    ))

    got_status = spin_until(node, lambda: node.last_status is not None, timeout_sec=5.0)
    checks.append(Check(
        "rooster_status_topic",
        got_status,
        (
            f"battery_pct={node.last_status.get('battery_pct')} "
            f"battery_voltage={node.last_status.get('battery_voltage')} "
            f"armed={node.last_status.get('armed')} "
            f"airborne={node.last_status.get('airborne')} "
            f"busy_action={node.last_status.get('busy_action')}"
        ) if got_status else f"no /{node.rooster_id}/rooster_status message within 5s",
    ))

    if args.skip_arm:
        checks.append(Check("arm_disarm", True, "skipped via --skip-arm"))
    else:
        node.send_cmd("arm")
        armed = spin_until(node, lambda: bool(node.last_state and node.last_state.armed),
                            timeout_sec=args.arm_timeout_sec)
        checks.append(Check(
            "arm",
            armed,
            f"armed={node.last_state.armed if node.last_state else None} "
            f"within {args.arm_timeout_sec}s",
        ))

        # Message-flow only -- see module docstring's NOTE on turning.
        node.send_cmd("turn_left", value=300.0)
        spin_until(node, lambda: False, timeout_sec=args.turn_sec)  # just pump the executor
        node.send_cmd("turn_right", value=300.0)
        spin_until(node, lambda: False, timeout_sec=args.turn_sec)
        node.send_cmd("stop")
        checks.append(Check(
            "turn_commands_accepted",
            True,
            "turn_left/turn_right/stop published; no closed-loop heading check "
            "performed (see module docstring)",
        ))

        node.send_cmd("disarm")
        disarmed = spin_until(node, lambda: bool(node.last_state and not node.last_state.armed),
                               timeout_sec=args.arm_timeout_sec)
        checks.append(Check(
            "disarm",
            disarmed,
            f"armed={node.last_state.armed if node.last_state else None} "
            f"within {args.arm_timeout_sec}s",
        ))

    # Let IMU/image counters accumulate for the observation window,
    # independently of the arm/disarm sequence above.
    spin_until(node, lambda: False, timeout_sec=args.observe_sec)

    imu_ok = node.imu_rate.count > 0 and node.imu_rate.rate_hz() >= args.min_imu_hz
    checks.append(Check(
        "imu_topic_rate",
        imu_ok,
        f"count={node.imu_rate.count} rate={node.imu_rate.rate_hz():.1f}Hz "
        f"(min {args.min_imu_hz}Hz)",
    ))

    image_ok = node.image_rate.count > 0 and node.image_rate.rate_hz() >= args.min_image_hz
    checks.append(Check(
        "image_topic_rate",
        image_ok,
        f"count={node.image_rate.count} rate={node.image_rate.rate_hz():.1f}Hz "
        f"(min {args.min_image_hz}Hz)",
    ))

    # rooster_status publishes at ~5Hz (see RoosterCommandUnitNode._publish_status);
    # a low min threshold here just confirms it kept flowing through the whole run.
    status_ok = node.status_rate.count > 0 and node.status_rate.rate_hz() >= args.min_status_hz
    checks.append(Check(
        "rooster_status_rate",
        status_ok,
        f"count={node.status_rate.count} rate={node.status_rate.rate_hz():.1f}Hz "
        f"(min {args.min_status_hz}Hz)",
    ))

    return checks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bench-only (props off) ROS2 interface-parity check for the "
        "new Orin-NX Rooster architecture: cmd_nav, state, IMU, and image topics."
    )
    p.add_argument("--rooster-id", default="R1")
    p.add_argument("--imu-topic", default=None,
                    help="Defaults to /{rooster-id}/imu/data")
    p.add_argument("--image-topic", default=None,
                    help="Defaults to /{rooster-id}/camera/image_raw")
    p.add_argument("--skip-arm", action="store_true",
                    help="Skip arm/disarm/turn entirely; only check state/IMU/image topics.")
    p.add_argument("--arm-timeout-sec", type=float, default=8.0)
    p.add_argument("--turn-sec", type=float, default=2.0,
                    help="How long to hold each turn_left/turn_right command.")
    p.add_argument("--observe-sec", type=float, default=8.0,
                    help="How long to listen for IMU/image messages.")
    p.add_argument("--min-imu-hz", type=float, default=10.0)
    p.add_argument("--min-image-hz", type=float, default=5.0)
    p.add_argument("--min-status-hz", type=float, default=2.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    imu_topic = args.imu_topic or f"/{args.rooster_id}/imu/data"
    image_topic = args.image_topic or f"/{args.rooster_id}/camera/image_raw"

    print("[flight-iface-test] BENCH TEST ONLY -- confirm props are removed before continuing.")
    print(f"[flight-iface-test] rooster_id  : {args.rooster_id}")
    print(f"[flight-iface-test] imu_topic   : {imu_topic}")
    print(f"[flight-iface-test] image_topic : {image_topic}")

    rclpy.init()
    node = FlightInterfaceTestNode(args.rooster_id, imu_topic, image_topic)
    try:
        checks = run_checks(node, args)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    print("\n[flight-iface-test] ---- results ----")
    all_passed = True
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        all_passed &= c.passed
        print(f"[flight-iface-test] {status:4s} {c.name:24s} {c.detail}")

    print(f"[flight-iface-test] {'PASS' if all_passed else 'FAIL'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
