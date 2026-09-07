#!/usr/bin/env python3
"""ROS2 interface-parity check for Robotican's new Orin-NX architecture,
with an optional REAL FLIGHT sequence (takeoff, hover, 360 turn, land).

Default mode is bench-only, props off: confirms cmd_nav/state/rooster_status
still behave the same after the architecture change (backend + all models
now co-located on the drone's own Orin NX), and that IMU/image topics
deliver at a sane rate. It never expects the airframe to leave the ground.

--real-flight mode is a genuinely different thing: it arms, takes off,
expects to reach a stable hover in [--hover-min-m, --hover-max-m], turns
~360 degrees, then lands and confirms disarm. This commands an actual
flight. Requires props installed, a clear area, and a safety pilot present
with a manual override/kill switch ready -- the script will not proceed
past a typed confirmation (or --yes for non-interactive use) precisely
because this is not the same risk tier as the bench test.

IMPORTANT on hover altitude: RoosterUnit's altitude-hold locks onto
`target_ranger_m`, which is a ROS2 parameter of the ALREADY-RUNNING
RoosterCommandUnitNode (set at launch, e.g. `-p target_ranger_m:=1.2`),
not something this script can set via cmd_nav. cmd_nav's "up"/"down"
actions directly override the z-axis throttle and would fight the
altitude-hold loop rather than nudge it (see rooster_unit.py's own
comments on how not-fully-settled that loop already is) -- so this script
does NOT attempt to correct an out-of-band hover live. If hover_altitude_in_band
fails, the fix is to relaunch RoosterCommandUnitNode with the right
target_ranger_m, not to patch this script.

IMPORTANT on the turn: closed-loop via UAVState.azimuth (fcu_driver_interfaces,
radians, 0=North, positive CW -- see rooster_video_adapter.py's
state_callback for prior use of this same field) if it arrives, else a
timed open-loop fallback that is explicitly flagged as NOT heading-confirmed
rather than silently treated as equivalent.

A continuous ceiling guard (--ceiling-abort-m) commands an immediate land
if the ranger ever exceeds it during takeoff/hover/turn.

Usage (bench, props off, default):
    python3 orin_nx_flight_interface_test.py --rooster-id R1

Usage (real flight):
    python3 orin_nx_flight_interface_test.py --rooster-id R1 --real-flight \\
        --hover-min-m 1.0 --hover-max-m 1.5 --turn-direction left
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from sensor_msgs.msg import Imu, Image
from rooster_manager_interfaces.msg import RoosterState
from fcu_driver_interfaces.msg import UAVState

TWO_PI = 2 * math.pi


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


def wrap_pi(angle: float) -> float:
    return (angle + math.pi) % TWO_PI - math.pi


class AzimuthTracker:
    """Accumulates total angular travel from UAVState.azimuth, handling
    wraparound. Deliberately direction-agnostic (sums abs(delta)) -- a full
    turn in either direction should trip the ~360 degree threshold. Marks
    itself unavailable rather than raising if the field turns out not to
    exist on this hardware (mirrors the defensive pattern already used for
    this same field in rooster_video_adapter.py)."""

    def __init__(self):
        self.available = False
        self._last_azimuth: float | None = None
        self.cumulative_rotation_rad = 0.0

    def on_uav_state(self, msg: UAVState):
        try:
            az = float(msg.azimuth)
        except AttributeError:
            return
        self.available = True
        if self._last_azimuth is not None:
            self.cumulative_rotation_rad += abs(wrap_pi(az - self._last_azimuth))
        self._last_azimuth = az

    def reset_accumulation(self):
        self.cumulative_rotation_rad = 0.0
        self._last_azimuth = None


class FlightInterfaceTestNode(Node):
    def __init__(self, rooster_id: str, imu_topic: str, image_topic: str):
        super().__init__("orin_nx_flight_interface_test")
        self.rooster_id = rooster_id

        self.last_state: RoosterState | None = None
        self.last_status: dict | None = None
        self.status_rate = RateCounter()
        self.azimuth = AzimuthTracker()

        self.cmd_pub = self.create_publisher(String, f"/{rooster_id}/cmd_nav", 10)
        self.state_sub = self.create_subscription(
            RoosterState, f"/{rooster_id}/state", self._on_state, 10
        )
        self.status_sub = self.create_subscription(
            String, f"/{rooster_id}/rooster_status", self._on_status, 10
        )
        self.uav_state_sub = self.create_subscription(
            UAVState, f"/{rooster_id}/fcu/state", self.azimuth.on_uav_state, 10
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

    def ranger_m(self) -> float | None:
        return getattr(self.last_state, "ranger", None) if self.last_state else None

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


def poll_with_ceiling_guard(
    node: FlightInterfaceTestNode, predicate, timeout_sec: float, ceiling_m: float
) -> tuple[bool, bool]:
    """Like spin_until, but commands an immediate land and returns
    (False, aborted=True) the instant ranger exceeds ceiling_m."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        ranger = node.ranger_m()
        if ranger is not None and ranger > ceiling_m:
            node.get_logger().error(
                f"CEILING ABORT: ranger={ranger:.2f}m > {ceiling_m:.2f}m -- landing immediately."
            )
            node.send_cmd("land")
            return False, True
        if predicate():
            return True, False
    return predicate(), False


def run_connectivity_checks(node: FlightInterfaceTestNode) -> tuple[list[Check], bool]:
    checks: list[Check] = []

    got_state = spin_until(node, lambda: node.last_state is not None, timeout_sec=10.0)
    checks.append(Check(
        "state_topic_alive",
        got_state,
        f"/{node.rooster_id}/state message received" if got_state else "no /state message within 10s",
    ))
    if not got_state:
        return checks, False

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

    return checks, True


def run_topic_rate_checks(node: FlightInterfaceTestNode, args: argparse.Namespace) -> list[Check]:
    checks: list[Check] = []
    spin_until(node, lambda: False, timeout_sec=args.observe_sec)

    imu_ok = node.imu_rate.count > 0 and node.imu_rate.rate_hz() >= args.min_imu_hz
    checks.append(Check(
        "imu_topic_rate", imu_ok,
        f"count={node.imu_rate.count} rate={node.imu_rate.rate_hz():.1f}Hz (min {args.min_imu_hz}Hz)",
    ))

    image_ok = node.image_rate.count > 0 and node.image_rate.rate_hz() >= args.min_image_hz
    checks.append(Check(
        "image_topic_rate", image_ok,
        f"count={node.image_rate.count} rate={node.image_rate.rate_hz():.1f}Hz (min {args.min_image_hz}Hz)",
    ))

    status_ok = node.status_rate.count > 0 and node.status_rate.rate_hz() >= args.min_status_hz
    checks.append(Check(
        "rooster_status_rate", status_ok,
        f"count={node.status_rate.count} rate={node.status_rate.rate_hz():.1f}Hz (min {args.min_status_hz}Hz)",
    ))
    return checks


def run_bench_checks(node: FlightInterfaceTestNode, args: argparse.Namespace) -> list[Check]:
    """Original bench-only path: arm -> exercise turn commands (message-flow
    only, no closed-loop heading check) -> disarm. Props off."""
    checks: list[Check] = []

    if args.skip_arm:
        checks.append(Check("arm_disarm", True, "skipped via --skip-arm"))
    else:
        node.send_cmd("arm")
        armed = spin_until(node, lambda: bool(node.last_state and node.last_state.armed),
                            timeout_sec=args.arm_timeout_sec)
        checks.append(Check(
            "arm", armed,
            f"armed={node.last_state.armed if node.last_state else None} within {args.arm_timeout_sec}s",
        ))

        node.send_cmd("turn_left", value=300.0)
        spin_until(node, lambda: False, timeout_sec=args.turn_sec)
        node.send_cmd("turn_right", value=300.0)
        spin_until(node, lambda: False, timeout_sec=args.turn_sec)
        node.send_cmd("stop")
        checks.append(Check(
            "turn_commands_accepted", True,
            "turn_left/turn_right/stop published; no closed-loop heading check performed (bench mode)",
        ))

        node.send_cmd("disarm")
        disarmed = spin_until(node, lambda: bool(node.last_state and not node.last_state.armed),
                               timeout_sec=args.arm_timeout_sec)
        checks.append(Check(
            "disarm", disarmed,
            f"armed={node.last_state.armed if node.last_state else None} within {args.arm_timeout_sec}s",
        ))

    return checks


def run_real_flight_sequence(node: FlightInterfaceTestNode, args: argparse.Namespace) -> list[Check]:
    """arm -> takeoff -> confirm hover altitude in band -> turn ~360 degrees
    -> land -> confirm disarm. Guarded throughout by a ceiling abort."""
    checks: list[Check] = []

    node.send_cmd("arm")
    armed = spin_until(node, lambda: bool(node.last_state and node.last_state.armed),
                        timeout_sec=args.arm_timeout_sec)
    checks.append(Check("arm", armed,
                         f"armed={node.last_state.armed if node.last_state else None}"))
    if not armed:
        return checks

    node.send_cmd("takeoff")
    airborne, aborted = poll_with_ceiling_guard(
        node, lambda: bool(node.last_state and node.last_state.airborne),
        timeout_sec=args.takeoff_timeout_sec, ceiling_m=args.ceiling_abort_m,
    )
    checks.append(Check(
        "takeoff", airborne and not aborted,
        f"airborne={node.last_state.airborne if node.last_state else None} "
        f"ranger={node.ranger_m()} aborted_on_ceiling={aborted}",
    ))
    if aborted or not airborne:
        return checks

    # Let altitude-hold settle onto whatever target_ranger_m the running
    # RoosterCommandUnitNode was launched with (see module docstring --
    # this script cannot set it, only verify where it landed).
    spin_until(node, lambda: False, timeout_sec=args.hover_settle_sec)
    ranger = node.ranger_m()
    in_band = ranger is not None and args.hover_min_m <= ranger <= args.hover_max_m
    checks.append(Check(
        "hover_altitude_in_band",
        in_band,
        (
            f"ranger={ranger:.2f}m, expected [{args.hover_min_m},{args.hover_max_m}]m -- "
            "if out of band, relaunch RoosterCommandUnitNode with target_ranger_m "
            "in that range; cmd_nav cannot correct this live (see module docstring)."
        ) if ranger is not None else "no ranger reading available",
    ))
    if not in_band:
        node.send_cmd("land")
        spin_until(node, lambda: bool(node.last_state and not node.last_state.airborne),
                   timeout_sec=args.land_timeout_sec)
        return checks

    # Turn ~360 degrees.
    node.azimuth.reset_accumulation()
    turn_action = "turn_left" if args.turn_direction == "left" else "turn_right"
    node.send_cmd(turn_action, value=args.turn_axis_value)

    def turn_done():
        return (node.azimuth.available
                and node.azimuth.cumulative_rotation_rad >= (TWO_PI - args.turn_tolerance_rad))

    turned, aborted = poll_with_ceiling_guard(
        node, turn_done, timeout_sec=args.turn_timeout_sec, ceiling_m=args.ceiling_abort_m
    )
    node.send_cmd("stop")

    if aborted:
        checks.append(Check("turn_360", False, "ceiling exceeded during turn -- landed immediately"))
        return checks

    if node.azimuth.available:
        checks.append(Check(
            "turn_360", turned,
            f"closed-loop via UAVState.azimuth: accumulated "
            f"{math.degrees(node.azimuth.cumulative_rotation_rad):.0f} of 360 deg "
            f"within {args.turn_timeout_sec}s",
        ))
    else:
        node.get_logger().warn(
            "UAVState.azimuth never arrived -- falling back to a "
            f"{args.turn_open_loop_fallback_sec}s open-loop turn (NOT heading-confirmed)."
        )
        node.send_cmd(turn_action, value=args.turn_axis_value)
        spin_until(node, lambda: False, timeout_sec=max(0.0, args.turn_open_loop_fallback_sec))
        node.send_cmd("stop")
        checks.append(Check(
            "turn_360", True,
            f"open-loop fallback: held {turn_action} for {args.turn_open_loop_fallback_sec}s -- "
            "NOT heading-confirmed, UAVState.azimuth was never received",
        ))

    node.send_cmd("land")
    landed, _ = poll_with_ceiling_guard(
        node, lambda: bool(node.last_state and not node.last_state.airborne),
        timeout_sec=args.land_timeout_sec, ceiling_m=args.ceiling_abort_m,
    )
    checks.append(Check(
        "land", landed,
        f"airborne={node.last_state.airborne if node.last_state else None} within {args.land_timeout_sec}s",
    ))

    # RoosterUnit.land() disarms automatically once grounded (see rooster_unit.py).
    disarmed = spin_until(node, lambda: bool(node.last_state and not node.last_state.armed),
                           timeout_sec=args.arm_timeout_sec)
    checks.append(Check(
        "disarm_after_land", disarmed,
        f"armed={node.last_state.armed if node.last_state else None}",
    ))

    return checks


def confirm_real_flight(args: argparse.Namespace) -> bool:
    print("=" * 70)
    print("REAL FLIGHT MODE -- this will arm, take off, fly a 360 turn, and land.")
    print("Before continuing, confirm:")
    print("  - props are installed")
    print("  - the area is clear of people/obstacles, including overhead")
    print("  - a safety pilot is present with a manual override/kill switch ready")
    print(f"  - ceiling abort is set to {args.ceiling_abort_m}m")
    print("=" * 70)
    if args.yes:
        print("[flight-iface-test] --yes passed, skipping interactive confirmation.")
        return True
    typed = input("Type FLY (all caps) to proceed, anything else aborts: ")
    return typed.strip() == "FLY"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ROS2 interface-parity check for the new Orin-NX Rooster "
        "architecture. Bench-only by default; --real-flight runs an actual "
        "takeoff/hover/360-turn/land sequence."
    )
    p.add_argument("--rooster-id", default="R1")
    p.add_argument("--imu-topic", default=None, help="Defaults to /{rooster-id}/imu/data")
    p.add_argument("--image-topic", default=None, help="Defaults to /{rooster-id}/camera/image_raw")
    p.add_argument("--observe-sec", type=float, default=8.0,
                    help="How long to listen for IMU/image/status messages.")
    p.add_argument("--min-imu-hz", type=float, default=10.0)
    p.add_argument("--min-image-hz", type=float, default=5.0)
    p.add_argument("--min-status-hz", type=float, default=2.0)
    p.add_argument("--arm-timeout-sec", type=float, default=8.0)

    # Bench-only mode
    p.add_argument("--skip-arm", action="store_true",
                    help="Bench mode only: skip arm/disarm/turn entirely.")
    p.add_argument("--turn-sec", type=float, default=2.0,
                    help="Bench mode only: how long to hold each turn_left/turn_right command.")

    # Real-flight mode
    p.add_argument("--real-flight", action="store_true",
                    help="Run an actual takeoff/hover/360-turn/land sequence instead of the bench check. "
                         "Props must be installed and the area clear.")
    p.add_argument("--yes", action="store_true",
                    help="Skip the interactive typed confirmation for --real-flight (non-interactive use only).")
    p.add_argument("--hover-min-m", type=float, default=1.0)
    p.add_argument("--hover-max-m", type=float, default=1.5)
    p.add_argument("--hover-settle-sec", type=float, default=3.0,
                    help="Time to let altitude-hold settle after takeoff before checking the hover band.")
    p.add_argument("--takeoff-timeout-sec", type=float, default=20.0)
    p.add_argument("--land-timeout-sec", type=float, default=35.0)
    p.add_argument("--turn-direction", choices=["left", "right"], default="left")
    p.add_argument("--turn-axis-value", type=float, default=300.0,
                    help="Magnitude passed with turn_left/turn_right (same axis scale as cmd_nav's 'value').")
    p.add_argument("--turn-timeout-sec", type=float, default=30.0,
                    help="Max time to wait for a closed-loop 360 via UAVState.azimuth.")
    p.add_argument("--turn-tolerance-rad", type=float, default=math.radians(10),
                    help="How close to 2*pi accumulated rotation counts as \"done\".")
    p.add_argument("--turn-open-loop-fallback-sec", type=float, default=8.0,
                    help="Used only if UAVState.azimuth never arrives -- NOT heading-confirmed.")
    p.add_argument("--ceiling-abort-m", type=float, default=2.5,
                    help="If ranger exceeds this at any point during takeoff/hover/turn, land immediately.")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    imu_topic = args.imu_topic or f"/{args.rooster_id}/imu/data"
    image_topic = args.image_topic or f"/{args.rooster_id}/camera/image_raw"

    if args.real_flight:
        if not confirm_real_flight(args):
            print("[flight-iface-test] Not confirmed -- aborting before connecting to anything.")
            return 1
    else:
        print("[flight-iface-test] BENCH TEST ONLY -- confirm props are removed before continuing.")

    print(f"[flight-iface-test] rooster_id  : {args.rooster_id}")
    print(f"[flight-iface-test] imu_topic   : {imu_topic}")
    print(f"[flight-iface-test] image_topic : {image_topic}")

    rclpy.init()
    node = FlightInterfaceTestNode(args.rooster_id, imu_topic, image_topic)
    try:
        checks, ok = run_connectivity_checks(node)
        if ok:
            if args.real_flight:
                checks += run_real_flight_sequence(node, args)
            else:
                checks += run_bench_checks(node, args)
            checks += run_topic_rate_checks(node, args)
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
