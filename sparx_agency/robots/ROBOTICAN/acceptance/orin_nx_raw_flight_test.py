#!/usr/bin/env python3
"""REAL FLIGHT test for Robotican's new Orin-NX architecture: arm, climb,
hover in POSITION mode, turn ~360 degrees, land, disarm -- with concurrent
camera frame capture. Built directly on the raw Rooster interface
(ManualControl/KeepAlive/RoosterState/force_arm), the same messages and
services Robotican's own force_arm_and_fly_example.py uses, modeled after
it directly. Deliberately does NOT go through our own
RoosterCommandUnitNode/RoosterUnit/cmd_nav layer -- that stack already has
its own test (orin_nx_flight_interface_test.py --real-flight); this script
exists to test the airframe/FCU/Orin-NX capability itself, with the fewest
possible dependencies on our own software being deployed and correctly
configured.

THIS COMMANDS AN ACTUAL FLIGHT. Props installed, area clear, safety pilot
present with a manual override/kill switch ready. Will not arm without a
typed confirmation (or --yes for non-interactive use).

Hover/climb/land constants (hover_z, climb_duration_sec, climb_settle_sec,
land_step, land_step_interval_sec, land_min_throttle_fraction,
ground_ranger_m) default to RoosterUnit's own already-tuned values for
POSITION mode on this airframe (see robots/ROBOTICAN/helpers/rooster_unit.py)
-- NOT the attached example's values, which are for MANUAL mode (no
altitude hold) and were tuned for a different, cruder kind of flight.
climb_z defaults to 700, deliberately NOT RoosterUnit's 1000: rooster_unit.py
documents 600 as too weak (re-triggers PX4's landing-detector mid-climb,
disarms) and 1000 as confirmed working; 700 sits between those two data
points and is itself untested. If it doesn't climb cleanly (same symptom
as 600 -- arms, attempts to climb, gets flagged still-landed), the known
fix is --climb-z 1000, not lower. This script's hover is still open-loop
(fixed throttle, no PD correction) -- it does not reimplement RoosterUnit's
altitude-hold loop, so it will drift more than the cmd_nav path does. A
continuous ceiling-abort guard (--ceiling-abort-m) lands immediately if
ranger ever exceeds it, during climb, hover, or the turn.

Turn is closed-loop via RoosterState.azimuth (radians, 0=North, positive
CW -- confirmed present directly on RoosterState, not just UAVState) when
it changes, else a timed open-loop fallback, explicitly flagged as such.
While turning, z is held at hover_z throughout -- NEVER defaulted to 0,
which the rooster_command_unit.py module docstring documents as having
caused an immediate fall in a prior live test.

Frame capture runs in a background thread for the whole flight (arm to
disarm), writing frame_%05d.jpg into --frames-dir from --camera-device via
V4L2/OpenCV. Independent of telemetry -- no MAVLink/timestamp correlation.

Usage:
    python3 orin_nx_raw_flight_test.py --rooster-id R1 \\
        --camera-device /dev/HD_CAMERA --frames-dir /home/iai/frames
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from fcu_driver_interfaces.msg import ManualControl
from rooster_handler_interfaces.msg import KeepAlive
from rooster_manager_interfaces.msg import RoosterState

import math

MAX_AXIS = 1000.0
TWO_PI = 2 * math.pi


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


def clamp(value: float, limit: float = MAX_AXIS) -> float:
    return max(-limit, min(limit, value))


def wrap_pi(angle: float) -> float:
    return (angle + math.pi) % TWO_PI - math.pi


class Axes:
    """Current manual-control axes, published at 40Hz by a timer. Any phase
    that wants to change ONE axis should read-modify-write through this,
    never construct a fresh all-zero message -- see module docstring on why
    z must never default to 0 mid-flight."""

    def __init__(self):
        self.x = self.y = self.z = self.r = 0.0

    def set(self, x=None, y=None, z=None, r=None):
        if x is not None:
            self.x = clamp(x)
        if y is not None:
            self.y = clamp(y)
        if z is not None:
            self.z = clamp(z)
        if r is not None:
            self.r = clamp(r)


class FrameCapture:
    """Background V4L2 capture, independent of the flight sequence and its
    telemetry -- no timestamp correlation attempted."""

    def __init__(self, device: str, frames_dir: Path, fps: float):
        self.device = device
        self.frames_dir = frames_dir
        self.period_sec = 1.0 / fps if fps > 0 else 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.frame_count = 0
        self.open_ok = False

    def start(self):
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        import cv2  # imported here so a missing cv2 only breaks capture, not the whole script
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        self.open_ok = cap.isOpened()
        if not self.open_ok:
            return
        idx = 0
        next_due = time.monotonic()
        while not self._stop.is_set():
            ok, frame = cap.read()
            if ok and frame is not None:
                now = time.monotonic()
                if now >= next_due:
                    idx += 1
                    path = self.frames_dir / f"frame_{idx:05d}.jpg"
                    cv2.imwrite(str(path), frame)
                    self.frame_count = idx
                    next_due = now + self.period_sec
        cap.release()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)


class RawFlightNode(Node):
    def __init__(self, rooster_id: str):
        super().__init__("orin_nx_raw_flight_test")
        self.id = rooster_id
        self.axes = Axes()
        self.last_state: RoosterState | None = None
        self.azimuth_start: float | None = None
        self.cumulative_rotation_rad = 0.0
        self._last_azimuth_for_accum: float | None = None

        self.manual_pub = self.create_publisher(ManualControl, f"/{self.id}/manual_control", 10)
        self.keep_alive_pub = self.create_publisher(KeepAlive, f"/{self.id}/keep_alive", 10)
        self.state_sub = self.create_subscription(
            RoosterState, f"/{self.id}/state", self._on_state, 10
        )
        self.force_arm_client = self.create_client(SetBool, f"/{self.id}/fcu/command/force_arm")

        self.create_timer(1.0 / 40.0, self._publish_manual)
        self.create_timer(1.0, self._publish_keep_alive)

    def _on_state(self, msg: RoosterState):
        self.last_state = msg

    def _publish_manual(self):
        msg = ManualControl()
        msg.x, msg.y, msg.z, msg.r = self.axes.x, self.axes.y, self.axes.z, self.axes.r
        msg.buttons = 0
        self.manual_pub.publish(msg)

    def _publish_keep_alive(self):
        msg = KeepAlive()
        msg.is_active = True
        msg.requested_flight_mode = KeepAlive.FLIGHT_MODE_POSITION
        msg.command_reboot = False
        self.keep_alive_pub.publish(msg)

    def ranger_m(self) -> float | None:
        return float(self.last_state.ranger) if self.last_state else None

    def armed(self) -> bool:
        return bool(self.last_state and self.last_state.armed)

    def airborne(self) -> bool:
        return bool(self.last_state and self.last_state.airborne)

    def azimuth_rad(self) -> float | None:
        return float(self.last_state.azimuth) if self.last_state else None

    def reset_rotation_accumulator(self):
        self.cumulative_rotation_rad = 0.0
        self._last_azimuth_for_accum = self.azimuth_rad()

    def accumulate_rotation(self):
        az = self.azimuth_rad()
        if az is None:
            return
        if self._last_azimuth_for_accum is not None:
            self.cumulative_rotation_rad += abs(wrap_pi(az - self._last_azimuth_for_accum))
        self._last_azimuth_for_accum = az

    def call_force_arm(self, arm: bool, timeout_sec: float) -> bool:
        if not self.force_arm_client.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().error("force_arm service not available")
            return False
        req = SetBool.Request()
        req.data = arm
        future = self.force_arm_client.call_async(req)
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        if not future.done():
            self.get_logger().error("force_arm call timed out")
            return False
        result = future.result()
        return bool(result and result.success)


def spin_until(node: Node, predicate, timeout_sec: float, on_tick=None) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if on_tick is not None:
            on_tick()
        if predicate():
            return True
    return predicate()


def step_down_land(node: RawFlightNode, args: argparse.Namespace) -> Check:
    """Steps z down from wherever it currently is, checking ranger/airborne
    after every step -- same stopping conditions as RoosterUnit._do_land.

    NEVER cuts z to 0 unless actually confirmed grounded (not airborne, or
    ranger <= ground_ranger_m). On timeout while still airborne, z is left
    at its last stepped value (>= land_min_throttle_fraction * hover_z),
    not zeroed -- cutting throttle at altitude is a free-fall, the exact
    failure mode RoosterUnit's own landing logic is built to avoid."""
    min_z = args.hover_z * args.land_min_throttle_fraction
    deadline = time.monotonic() + args.land_timeout_sec
    reason = "timeout"
    grounded = False
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        ranger = node.ranger_m()
        if not node.airborne() or (ranger is not None and ranger <= args.ground_ranger_m):
            reason = f"airborne={node.airborne()}, ranger={ranger}"
            grounded = True
            break
        node.axes.set(z=max(min_z, node.axes.z - args.land_step))
        spin_until(node, lambda: False, timeout_sec=args.land_step_interval_sec)

    if grounded:
        node.axes.set(z=0.0)
        detail = f"stopped because: {reason}, final airborne={node.airborne()}"
    else:
        # Timed out still airborne -- leave z at its last (floored) value,
        # do NOT cut throttle. This is a real failure: report it loudly.
        detail = (
            f"TIMEOUT WHILE STILL AIRBORNE at z={node.axes.z:.0f} "
            f"(ranger={node.ranger_m()}) -- throttle NOT cut, take manual control now."
        )

    landed = grounded and not node.airborne()
    return Check("land", landed, detail)


def run_flight(node: RawFlightNode, args: argparse.Namespace) -> list[Check]:
    checks: list[Check] = []

    got_state = spin_until(node, lambda: node.last_state is not None, timeout_sec=10.0)
    checks.append(Check("state_topic_alive", got_state,
                         "received" if got_state else "no /state within 10s"))
    if not got_state:
        return checks

    armed = node.call_force_arm(True, timeout_sec=args.arm_timeout_sec)
    armed = armed and spin_until(node, node.armed, timeout_sec=args.arm_timeout_sec)
    checks.append(Check("arm", armed, f"armed={node.armed()}"))
    if not armed:
        return checks

    ceiling_hit = threading.Event()

    def ceiling_guard():
        r = node.ranger_m()
        if r is not None and r > args.ceiling_abort_m:
            ceiling_hit.set()

    node.axes.set(x=0.0, y=0.0, z=args.climb_z, r=0.0)
    spin_until(node, lambda: ceiling_hit.is_set(), timeout_sec=args.climb_duration_sec,
               on_tick=ceiling_guard)

    node.axes.set(z=args.hover_z)
    spin_until(node, lambda: ceiling_hit.is_set(), timeout_sec=args.climb_settle_sec,
               on_tick=ceiling_guard)

    airborne = not ceiling_hit.is_set() and spin_until(
        node, lambda: node.airborne() or ceiling_hit.is_set(),
        timeout_sec=args.takeoff_timeout_sec, on_tick=ceiling_guard,
    )
    checks.append(Check(
        "takeoff", airborne and not ceiling_hit.is_set(),
        f"airborne={node.airborne()} ranger={node.ranger_m()} ceiling_hit={ceiling_hit.is_set()}",
    ))
    if ceiling_hit.is_set() or not airborne:
        checks.append(step_down_land(node, args))
        node.call_force_arm(False, timeout_sec=args.arm_timeout_sec)
        return checks

    spin_until(node, lambda: ceiling_hit.is_set(), timeout_sec=args.hover_hold_sec,
               on_tick=ceiling_guard)
    if ceiling_hit.is_set():
        checks.append(Check("hover", False, "ceiling exceeded during hover"))
        checks.append(step_down_land(node, args))
        node.call_force_arm(False, timeout_sec=args.arm_timeout_sec)
        return checks

    node.reset_rotation_accumulator()
    turn_sign = 1.0 if args.turn_direction == "right" else -1.0
    node.axes.set(r=turn_sign * args.turn_axis_value, z=args.hover_z)  # z preserved, see docstring

    def turn_tick():
        ceiling_guard()
        node.accumulate_rotation()

    turned = spin_until(
        node,
        lambda: ceiling_hit.is_set() or node.cumulative_rotation_rad >= (TWO_PI - args.turn_tolerance_rad),
        timeout_sec=args.turn_timeout_sec, on_tick=turn_tick,
    )
    azimuth_moved = node.cumulative_rotation_rad > math.radians(5)  # more than noise

    if ceiling_hit.is_set():
        node.axes.set(r=0.0, z=args.hover_z)
        checks.append(Check("turn_360", False, "ceiling exceeded during turn"))
        checks.append(step_down_land(node, args))
        node.call_force_arm(False, timeout_sec=args.arm_timeout_sec)
        return checks

    if azimuth_moved:
        node.axes.set(r=0.0, z=args.hover_z)
        checks.append(Check(
            "turn_360", turned,
            f"closed-loop via RoosterState.azimuth: accumulated "
            f"{math.degrees(node.cumulative_rotation_rad):.0f} of 360 deg "
            f"within {args.turn_timeout_sec}s",
        ))
    else:
        node.get_logger().warn(
            "RoosterState.azimuth never moved -- falling back to a "
            f"{args.turn_open_loop_fallback_sec}s open-loop turn (NOT heading-confirmed)."
        )
        spin_until(node, lambda: ceiling_hit.is_set(),
                   timeout_sec=max(0.0, args.turn_open_loop_fallback_sec), on_tick=ceiling_guard)
        node.axes.set(r=0.0, z=args.hover_z)
        checks.append(Check(
            "turn_360", True,
            f"open-loop fallback: held turn_axis_value={turn_sign * args.turn_axis_value} for "
            f"{args.turn_open_loop_fallback_sec}s -- NOT heading-confirmed, azimuth never moved",
        ))

    checks.append(step_down_land(node, args))

    disarmed = node.call_force_arm(False, timeout_sec=args.arm_timeout_sec)
    disarmed = disarmed and spin_until(node, lambda: not node.armed(), timeout_sec=args.arm_timeout_sec)
    checks.append(Check("disarm_after_land", disarmed, f"armed={node.armed()}"))

    return checks


def confirm_flight(args: argparse.Namespace) -> bool:
    print("=" * 70)
    print("REAL FLIGHT (raw interface, position mode) -- arm, climb, hover,")
    print("360 turn, land. Confirm:")
    print("  - props installed, area clear of people/obstacles, including overhead")
    print("  - safety pilot present with a manual override/kill switch ready")
    print(f"  - climb_z={args.climb_z} (untested -- 600 is documented too weak, 1000 confirmed "
          f"working, this is between them; if it doesn't climb cleanly, next try is --climb-z 1000)")
    print(f"  - ceiling abort at {args.ceiling_abort_m}m")
    print(f"  - camera device {args.camera_device} -> {args.frames_dir}")
    print(f"  - turn direction requested: {args.turn_direction} (sign convention for this "
          f"axis was flipped once before after a live test showed it backwards, and has not "
          f"been re-validated since -- watch the actual direction and be ready to abort/re-run "
          f"with the other --turn-direction if it's wrong)")
    print("=" * 70)
    if args.yes:
        print("[raw-flight-test] --yes passed, skipping interactive confirmation.")
        return True
    typed = input("Type FLY (all caps) to proceed, anything else aborts: ")
    return typed.strip() == "FLY"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Raw-interface real flight test: arm/climb/hover(position mode)/"
        "360-turn/land, with concurrent camera frame capture."
    )
    p.add_argument("--rooster-id", default="R1")
    p.add_argument("--yes", action="store_true", help="Skip the interactive typed confirmation.")
    p.add_argument("--arm-timeout-sec", type=float, default=8.0)
    p.add_argument("--ceiling-abort-m", type=float, default=2.5)

    # Climb/hover/land -- hover_z is RoosterUnit's own tuned POSITION-mode value.
    # climb_z=700 is BETWEEN the two known data points: 600 is documented as too
    # weak (re-triggers the landing-detector, disarms mid-climb -- see
    # rooster_unit.py), 1000 is documented as confirmed to climb. 700 itself is
    # untested -- if it doesn't climb cleanly (same re-trigger symptom), the
    # fallback is --climb-z 1000, not lower.
    p.add_argument("--climb-z", type=float, default=700.0,
                    help="Untested value between the documented-too-weak (600) and "
                         "documented-working (1000) points -- see module docstring.")
    p.add_argument("--hover-z", type=float, default=550.0)
    p.add_argument("--climb-duration-sec", type=float, default=1.0)
    p.add_argument("--climb-settle-sec", type=float, default=1.0)
    p.add_argument("--takeoff-timeout-sec", type=float, default=20.0)
    p.add_argument("--hover-hold-sec", type=float, default=3.0,
                    help="How long to hold hover before starting the turn.")
    p.add_argument("--land-step", type=float, default=75.0)
    p.add_argument("--land-step-interval-sec", type=float, default=1.0)
    p.add_argument("--land-min-throttle-fraction", type=float, default=0.3)
    p.add_argument("--ground-ranger-m", type=float, default=0.3)
    p.add_argument("--land-timeout-sec", type=float, default=30.0)

    # Turn
    p.add_argument("--turn-direction", choices=["left", "right"], default="left")
    p.add_argument("--turn-axis-value", type=float, default=300.0)
    p.add_argument("--turn-timeout-sec", type=float, default=30.0)
    p.add_argument("--turn-tolerance-rad", type=float, default=math.radians(10))
    p.add_argument("--turn-open-loop-fallback-sec", type=float, default=8.0)

    # Frame capture
    p.add_argument("--camera-device", default="/dev/HD_CAMERA")
    p.add_argument("--frames-dir", default="/home/iai/frames")
    p.add_argument("--capture-fps", type=float, default=10.0)
    p.add_argument("--no-capture", action="store_true", help="Skip frame capture entirely.")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not confirm_flight(args):
        print("[raw-flight-test] Not confirmed -- aborting before connecting to anything.")
        return 1

    capture = None
    if not args.no_capture:
        capture = FrameCapture(args.camera_device, Path(args.frames_dir), args.capture_fps)
        capture.start()

    rclpy.init()
    node = RawFlightNode(args.rooster_id)
    try:
        checks = run_flight(node, args)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        if capture is not None:
            capture.stop()

    print("\n[raw-flight-test] ---- results ----")
    all_passed = True
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        all_passed &= c.passed
        print(f"[raw-flight-test] {status:4s} {c.name:24s} {c.detail}")

    if capture is not None:
        print(
            f"[raw-flight-test] frame capture: opened={capture.open_ok} "
            f"frames_saved={capture.frame_count} dir={args.frames_dir}"
        )

    print(f"[raw-flight-test] {'PASS' if all_passed else 'FAIL'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
