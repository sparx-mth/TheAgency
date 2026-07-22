#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import threading
import time
from datetime import datetime
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
import websockets

from sparx_agency.robots.XTEND.automation import ControllerAutomation


def clamp_axis(v: float, max_abs: int) -> int:
    v = max(-1.0, min(1.0, float(v)))
    return int(v * max_abs)


class TwistReceiverNode(Node):
    def __init__(self, controller, cmd_vel_topic: str):
        super().__init__("xtend_twist_receiver")
        self.controller = controller

        self.sub = self.create_subscription(
            Twist,
            cmd_vel_topic,
            self.twist_cb,
            10,
        )

        self.get_logger().info(f"Listening for Twist on: {cmd_vel_topic}")

    def twist_cb(self, msg: Twist):
        self.controller.apply_twist(msg)


class OnlineXtendTwistController(ControllerAutomation):
    def __init__(
        self,
        host: str,
        port: int,
        frequency: float,
        robot_uid: str,
        cmd_vel_topic: str,
        axis_max: int,
        yaw_axis_max: int,
        twist_timeout_sec: float,
    ):
        super().__init__(host, port, frequency, robot_uid)

        # Avoid sharing the same dict object with base_command.
        self.send_command = copy.deepcopy(self.base_command)

        self.cmd_vel_topic = cmd_vel_topic
        self.axis_max = int(axis_max)
        self.yaw_axis_max = int(yaw_axis_max)
        self.twist_timeout_sec = float(twist_timeout_sec)

        self.last_twist_time: Optional[float] = None
        self.last_twist: Optional[Twist] = None

        self.ros_node: Optional[TwistReceiverNode] = None
        self.ros_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Continuous low-level command API
    # ------------------------------------------------------------------
    def set_axes(
        self,
        lateral: int = 0,
        vertical: int = 0,
        forward: int = 0,
        yaw: int = 0,
        marker_vertical: int = 0,
    ):
        self.send_command["axes"][0] = int(lateral)
        self.send_command["axes"][1] = int(vertical)
        self.send_command["axes"][2] = int(forward)
        self.send_command["axes"][3] = int(yaw)
        self.send_command["axes"][4] = int(marker_vertical)

    def stop_motion(self):
        self.set_axes(0, 0, 0, 0, 0)

    def apply_twist(self, msg: Twist):
        """
        Online Twist -> XTEND axes mapping.

        Tune signs here if the drone moves opposite to expected.
        """
        self.last_twist = msg
        self.last_twist_time = time.time()

        forward = clamp_axis(msg.linear.x, self.axis_max)
        lateral = 0 #clamp_axis(msg.linear.y, self.axis_max)
        vertical = 0 #clamp_axis(msg.linear.z, self.axis_max)
        yaw = clamp_axis(msg.angular.z, self.yaw_axis_max)

        self.set_axes(
            lateral=lateral,
            vertical=vertical,
            forward=forward,
            yaw=yaw,
        )

    # ------------------------------------------------------------------
    # Optional discrete actions still come from automation.py
    # arm_robot(), disarm_robot(), takeoff(), land()
    # ------------------------------------------------------------------

    async def stale_twist_watchdog(self):
        """
        If planner stops publishing, stop motion.
        This is important for online control.
        """
        while True:
            now = time.time()

            if self.last_twist_time is None:
                self.stop_motion()
            elif now - self.last_twist_time > self.twist_timeout_sec:
                self.stop_motion()

            await asyncio.sleep(0.05)

    def start_ros(self):
        self.ros_node = TwistReceiverNode(
            controller=self,
            cmd_vel_topic=self.cmd_vel_topic,
        )

        self.ros_thread = threading.Thread(
            target=rclpy.spin,
            args=(self.ros_node,),
            daemon=True,
        )
        self.ros_thread.start()

    def stop_ros(self):
        if self.ros_node is not None:
            self.ros_node.destroy_node()
            self.ros_node = None

    async def run_online(self):
        rclpy.init(args=None)
        self.start_ros()

        try:
            async with websockets.connect(self.uri) as websocket:
                print(f"✓ Connected to {self.uri}")
                print("[online] Sending latest Twist-derived command continuously.")
                print("[online] Press Ctrl+C to stop.")

                send_task = asyncio.create_task(self.send_message(websocket))
                receive_task = asyncio.create_task(self.receive_message(websocket))
                watchdog_task = asyncio.create_task(self.stale_twist_watchdog())

                try:
                    # Run forever until Ctrl+C.
                    await asyncio.Future()
                finally:
                    self.stop_motion()

                    for task in (send_task, receive_task, watchdog_task):
                        task.cancel()

                    await asyncio.gather(
                        send_task,
                        receive_task,
                        watchdog_task,
                        return_exceptions=True,
                    )

        finally:
            self.stop_ros()
            rclpy.shutdown()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="192.0.0.15")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--frequency", type=float, default=30.0)
    p.add_argument("--robot-uid", default="drndfb3eeb1")

    p.add_argument("--cmd-vel-topic", default="/cmd_vel")
    p.add_argument("--axis-max", type=int, default=500)
    p.add_argument("--yaw-axis-max", type=int, default=700)
    p.add_argument("--twist-timeout-sec", type=float, default=0.3)

    return p.parse_args()


def main():
    args = parse_args()

    controller = OnlineXtendTwistController(
        host=args.host,
        port=args.port,
        frequency=args.frequency,
        robot_uid=args.robot_uid,
        cmd_vel_topic=args.cmd_vel_topic,
        axis_max=args.axis_max,
        yaw_axis_max=args.yaw_axis_max,
        twist_timeout_sec=args.twist_timeout_sec,
    )

    try:
        asyncio.run(controller.run_online())
    except KeyboardInterrupt:
        print("\n[online] Stopped by user.")


if __name__ == "__main__":
    main()