#!/usr/bin/env python3
"""Emergency disarm. Run this from a SEPARATE terminal while a flight
script (e.g. orin_nx_raw_flight_test.py) is active.

Its only job is calling force_arm(False) on
/{rooster_id}/fcu/command/force_arm, repeatedly, until RoosterState.armed
confirms it actually took effect. NO confirmation prompt -- an emergency
stop that asks "are you sure?" first has missed the point. Does not touch
flight mode and does not publish ManualControl/KeepAlive at all: disarm
cuts motor output at the FCU regardless of what any other process (the
flight script) keeps publishing on those topics.

This is software last-resort, not the primary safety mechanism -- the
safety pilot's own manual RC override is. If this script fails to confirm
disarm, it says so explicitly and says to use that override immediately;
it does not retry forever or hide the failure.

Usage (second terminal, while a flight script is running):
    python3 orin_nx_emergency_disarm.py --rooster-id R1
"""

from __future__ import annotations

import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from rooster_manager_interfaces.msg import RoosterState


class EmergencyDisarmNode(Node):
    def __init__(self, rooster_id: str):
        super().__init__("orin_nx_emergency_disarm")
        self.id = rooster_id
        self.last_state: RoosterState | None = None
        self.state_sub = self.create_subscription(
            RoosterState, f"/{rooster_id}/state", self._on_state, 10
        )
        self.force_arm_client = self.create_client(SetBool, f"/{rooster_id}/fcu/command/force_arm")

    def _on_state(self, msg: RoosterState):
        self.last_state = msg

    def armed(self) -> bool:
        return bool(self.last_state and self.last_state.armed)

    def call_disarm(self, service_timeout_sec: float) -> bool:
        if not self.force_arm_client.wait_for_service(timeout_sec=service_timeout_sec):
            return False
        req = SetBool.Request()
        req.data = False
        future = self.force_arm_client.call_async(req)
        deadline = time.monotonic() + service_timeout_sec
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        if not future.done():
            return False
        result = future.result()
        return bool(result and result.success)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Emergency disarm -- call force_arm(False) repeatedly until confirmed."
    )
    p.add_argument("--rooster-id", default="R1")
    p.add_argument("--retries", type=int, default=10)
    p.add_argument("--retry-interval-sec", type=float, default=0.5)
    p.add_argument("--service-timeout-sec", type=float, default=3.0)
    p.add_argument("--confirm-timeout-sec", type=float, default=5.0,
                    help="How long to wait for RoosterState.armed to flip False after each attempt.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[emergency-disarm] EMERGENCY DISARM -- /{args.rooster_id}/fcu/command/force_arm")

    rclpy.init()
    node = EmergencyDisarmNode(args.rooster_id)
    confirmed = False
    try:
        # Best-effort initial state read -- not required to proceed.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and node.last_state is None:
            rclpy.spin_once(node, timeout_sec=0.1)

        for attempt in range(1, args.retries + 1):
            print(f"[emergency-disarm] attempt {attempt}/{args.retries}: calling force_arm(False)...")
            ok = node.call_disarm(args.service_timeout_sec)
            print(f"[emergency-disarm]   service call {'ok' if ok else 'FAILED'}")

            deadline = time.monotonic() + args.confirm_timeout_sec
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
                if not node.armed():
                    confirmed = True
                    break
            if confirmed:
                print(f"[emergency-disarm] CONFIRMED DISARMED (armed={node.armed()})")
                break
            print(f"[emergency-disarm]   not confirmed yet (armed={node.armed()}), retrying...")
            time.sleep(args.retry_interval_sec)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if not confirmed:
        print("[emergency-disarm] FAILED TO CONFIRM DISARM after all retries.")
        print("[emergency-disarm] USE THE SAFETY PILOT'S MANUAL OVERRIDE NOW.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
