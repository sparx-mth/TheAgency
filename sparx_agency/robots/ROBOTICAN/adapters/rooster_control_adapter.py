#!/usr/bin/env python3
import argparse
import time
from typing import List, Tuple

import rclpy
from rclpy.node import Node

from fcu_driver_interfaces.msg import ManualControl, UAVState
from rooster_handler_interfaces.msg import KeepAlive
from std_srvs.srv import Trigger, SetBool

from sparx_agency.robots.ROBOTICAN.helpers.manual_core import ManualCommandModel  # uses same axis logic as GUI


class PathRunnerNode(Node):
    """
    Minimal path runner:

    - Optionally arms the drone via /<ROOSTER_ID>/fcu/command/arm.
    - Publishes /<ROOSTER_ID>/manual_control at ~40 Hz during segments.
    - Publishes /<ROOSTER_ID>/keep_alive at ~1 Hz while running.
    - Reads path segments from a text file or from a hard-coded list.
    """

    def __init__(
        self,
        rooster_id: str,
        flight_mode: int,
        turtle: bool = False,
        arm_before_path: bool = True,
    ):
        super().__init__("path_runner")

        self.rooster_id = rooster_id
        self.flight_mode = int(flight_mode)
        self.arm_before_path = bool(arm_before_path)

        # Use same core model as GUI, but no logging.
        self.command_model = ManualCommandModel(step=10.0, turtle_scale=0.5)
        if turtle:
            self.command_model.toggle_turtle()

        manual_topic = f"/{self.rooster_id}/manual_control"
        keep_alive_topic = f"/{self.rooster_id}/keep_alive"
        arm_service_name = f"/{self.rooster_id}/fcu/command/force_arm"
        capture_start_service = f"/{self.rooster_id}/start_capture"
        capture_stop_service = f"/{self.rooster_id}/stop_capture"



        self.manual_pub = self.create_publisher(ManualControl, manual_topic, 10)
        self.keep_alive_pub = self.create_publisher(KeepAlive, keep_alive_topic, 10)


        # Service client for arming
        self.force_arm_client = self.create_client(SetBool, arm_service_name)
        self.start_capture_client = self.create_client(Trigger, capture_start_service)
        self.stop_capture_client = self.create_client(Trigger, capture_stop_service)

        self.last_uav_state = None
        self.uav_state_sub = self.create_subscription(
            UAVState,
            f"/{self.rooster_id}/fcu/state",
            self.uav_state_callback,
            10,
        )

        self.get_logger().info(
            f"PathRunnerNode for {self.rooster_id} (flight_mode={self.flight_mode})"
        )
        self.get_logger().info(
            f"Publishing to: {manual_topic}, {keep_alive_topic}"
        )

        self.get_logger().info(
            f"Arm service: {arm_service_name}, arm_before_path={self.arm_before_path}"
        )

    # ---------- low-level send functions ----------

    def _send_keep_alive(self):
        msg = KeepAlive()
        msg.is_active = True
        msg.requested_flight_mode = self.flight_mode
        msg.command_reboot = False
        self.keep_alive_pub.publish(msg)

    def _send_manual(self):
        axes = self.command_model.get_scaled_axes()
        msg = ManualControl()
        msg.x = axes.x
        msg.y = axes.y
        msg.z = axes.z
        msg.r = axes.r
        msg.buttons = 0
        self.manual_pub.publish(msg)

    # ---------- arming ----------

    def try_arm_drone(self, timeout: float = 5.0) -> bool:
        if self.force_arm_client is None:
            self.get_logger().error("No force_arm client created.")
            return False

        # 1) Warm up KeepAlive like the GUI does
        self.get_logger().info(f"{self.rooster_id}: warm-up keep_alive before force_arm")
        t0 = time.time()
        while time.time() - t0 < 2.0 and rclpy.ok():
            self._send_keep_alive()
            self.command_model.reset_axes()
            self._send_manual()
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(0.05)

        # 2) Wait for service
        self.get_logger().info(f"{self.rooster_id}: waiting for force_arm service...")
        if not self.force_arm_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("force_arm service not available")
            return False

        self.get_logger().info(f"{self.rooster_id}: calling force_arm=True")
        req = SetBool.Request()
        req.data = True
        future = self.force_arm_client.call_async(req)

        # 3) Wait for response
        end_time = time.time() + timeout
        while rclpy.ok() and not future.done() and time.time() < end_time:
            rclpy.spin_once(self, timeout_sec=0.05)

        if not future.done():
            self.get_logger().error("force_arm timed out")
            return False

        resp = future.result()
        self.get_logger().info(
            f"{self.rooster_id}: force_arm response: success={resp.success}, msg='{resp.message}'"
        )
        if not resp.success:
            return False

        # 4) Wait for UAVState.armed == True
        self.get_logger().info(f"{self.rooster_id}: waiting for UAVState.armed == True")
        t0 = time.time()
        while time.time() - t0 < timeout and rclpy.ok():
            if self.last_uav_state is not None and self.last_uav_state.armed:
                self.get_logger().info(f"{self.rooster_id}: UAV is armed (UAVState.armed=True)")
                return True
            rclpy.spin_once(self, timeout_sec=0.05)

        self.get_logger().warn(
            f"{self.rooster_id}: force_arm service succeeded but UAVState.armed is still False"
        )
        return False


    def uav_state_callback(self, msg: UAVState):
        self.last_uav_state = msg

    def call_service_capture(self, service_client, timeout=5.0) -> bool:
        self.get_logger().info(f"{self.rooster_id}: waiting for {service_client.srv_name} service...")
        if not service_client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(f"{service_client.srv_name} service not available")
            return False
        req = Trigger.Request()
        future = service_client.call_async(req)
        end_time = time.time() + timeout
        while rclpy.ok() and not future.done() and time.time() < end_time:
            rclpy.spin_once(self, timeout_sec=0.05)
        if not future.done():
            self.get_logger().error(f"{service_client.srv_name} timed out")
            return False
        resp = future.result()
        self.get_logger().info(
            f"{self.rooster_id}: {service_client.srv_name} response: success={resp.success}, msg='{resp.message}'"
        )
        return resp.success

    # ---------- high-level path execution ----------

    def run_path(
        self,
        segments: List[Tuple[str, float, float, float, float, float]],
        path_name: str = "path",
    ):
        """
        segments: list of (name, x, y, z, r, duration_sec)
        """
        if not segments:
            self.get_logger().warn("Empty path, nothing to do.")
            return

        # Try to start capture before arming (or after, your choice)
        ok = self.call_service_capture(service_client=self.start_capture_client)
        if not ok:
            self.get_logger().error("Aborting path: failed to start image capture.")
            return

        # Try to arm before starting
        if self.arm_before_path:
            ok = self.try_arm_drone()
            if not ok:
                self.call_service_capture(service_client=self.stop_capture_client)
                self.get_logger().error("Aborting path: failed to arm drone.")
                return

        self.get_logger().info(
            f"Starting path '{path_name}' with {len(segments)} segments."
        )

        # Start with zero axes
        self.command_model.reset_axes()
        self._send_manual()
        self._send_keep_alive()

        last_keep_alive = time.time()
        last_manual = time.time()


        for name, x, y, z, r, dur in segments:
            if not rclpy.ok():
                break

            self.get_logger().info(
                f"Segment '{name}': x={x}, y={y}, z={z}, r={r}, dur={dur}s"
            )

            # Set axes for this segment
            self.command_model.set_axes(x, y, z, r)

            start = time.time()
            end = start + dur

            while time.time() < end and rclpy.ok():
                now = time.time()

                # ~40 Hz manual control
                if now - last_manual >= 1.0 / 40.0:
                    self._send_manual()
                    last_manual = now

                # ~1 Hz keep-alive
                if now - last_keep_alive >= 1.0:
                    self._send_keep_alive()
                    last_keep_alive = now

                # Let ROS process timers, service responses, etc.
                rclpy.spin_once(self, timeout_sec=0.01)

            # small pause between segments (optional)
            self.command_model.reset_axes()
            for _ in range(5):
                self._send_manual()
                rclpy.spin_once(self, timeout_sec=0.01)
                time.sleep(0.02)

            if name == 'A':
                pass

        # Zero at the end
        self.get_logger().info("Path finished, zeroing axes.")
        self.command_model.reset_axes()
        self.call_service_capture(service_client=self.stop_capture_client)
        self.get_logger().info(f"{self.rooster_id}: calling force_arm=False")
        req = SetBool.Request()
        req.data = False
        future = self.force_arm_client.call_async(req)
        # 3) Wait for response
        end_time = time.time() + 5.0
        while rclpy.ok() and not future.done() and time.time() < end_time:
            rclpy.spin_once(self, timeout_sec=0.05)

        if not future.done():
            self.get_logger().error("force_arm timed out")
            return False

        resp = future.result()
        self.get_logger().info(
            f"{self.rooster_id}: force_arm response: success={resp.success}, msg='{resp.message}'"
        )
        if not resp.success:
            return False

        for _ in range(20):
            self._send_manual()
            if time.time() - last_keep_alive > 1.0:
                self._send_keep_alive()
                last_keep_alive = time.time()
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.02)


# ---------- helpers ----------

def parse_path_file(path_file: str):
    """
    Parse a text path file.

    Each non-empty, non-comment line can be:
        name x y z r duration
      or:
        x y z r duration   (name auto-generated)
    """
    segments = []
    auto_idx = 1

    with open(path_file, "r") as f:
        for idx, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split()
            if len(parts) == 5:
                # x y z r dur
                name = f"S{auto_idx}"
                auto_idx += 1
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2])
                    r = float(parts[3])
                    dur = float(parts[4])
                except ValueError:
                    raise ValueError(f"Line {idx}: cannot parse numbers: {raw!r}")
            elif len(parts) == 6:
                # name x y z r dur
                name = parts[0]
                try:
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    r = float(parts[4])
                    dur = float(parts[5])
                except ValueError:
                    raise ValueError(f"Line {idx}: cannot parse numbers: {raw!r}")
            else:
                raise ValueError(
                    f"Line {idx}: expected 5 or 6 values, got {len(parts)}: {raw!r}"
                )

            if dur <= 0:
                raise ValueError(f"Line {idx}: duration must be > 0")

            segments.append((name, x, y, z, r, dur))

    if not segments:
        raise ValueError("No valid segments in path file")

    return segments


def main():
    parser = argparse.ArgumentParser(
        description="Run a manual-control path on a Rooster via ROS 2."
    )
    parser.add_argument(
        "--rooster-id", "-r", default="R2", help="Rooster ID (R1/R2/R3...)"
    )
    parser.add_argument(
        "--flight-mode",
        "-m",
        type=int,
        default=1,
        help="Requested flight mode in KeepAlive (e.g. 1=GROUND_ROLL, 2=MANUAL)",
    )
    parser.add_argument(
        "--path", "-p", required=True, help="Path file (text) with segments"
    )
    parser.add_argument(
        "--turtle", action="store_true", help="Enable turtle (slow) mode scaling"
    )
    parser.add_argument(
        "--no-arm",
        action="store_true",
        help="Do not call /<ROOSTER_ID>/fcu/command/arm before running the path.",
    )
    args = parser.parse_args()

    segments = parse_path_file(args.path)

    rclpy.init()
    node = PathRunnerNode(
        rooster_id=args.rooster_id,
        flight_mode=args.flight_mode,
        turtle=args.turtle,
        arm_before_path=not args.no_arm,
    )
    try:
        node.run_path(segments, path_name=args.path)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()