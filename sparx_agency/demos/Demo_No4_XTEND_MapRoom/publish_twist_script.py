#!/usr/bin/env python3

import argparse
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


TWIST_SCRIPT = [
    # name, duration_sec, linear_x, linear_y, linear_z, angular_z
    ("forward_2m", 4.0, 0.5, 0.0, 0.0, 0.0),
    ("forward_2m", 4.0, 0.5, 0.0, 0.0, 0.0),
    ("stop_after_forward", 1.0, 0.0, 0.0, 0.0, 0.0),
    # ("turn_right_90deg", 3.2, 0.0, 0.0, 0.0, -0.5),
    # ("final_stop", 1.0, 0.0, 0.0, 0.0, 0.0),
]


class TwistScriptPublisher(Node):
    def __init__(self, topic: str, hz: float):
        super().__init__("twist_script_publisher")
        self.pub = self.create_publisher(Twist, topic, 10)
        self.period = 1.0 / hz

        self.get_logger().info(f"Publishing Twist script to {topic} at {hz} Hz")

    def publish_twist(self, linear_x, linear_y, linear_z, angular_z):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.linear.y = float(linear_y)
        msg.linear.z = float(linear_z)
        msg.angular.z = float(angular_z)
        self.pub.publish(msg)

    def run_script(self):
        for name, duration_sec, lx, ly, lz, az in TWIST_SCRIPT:
            self.get_logger().info(
                f"Step: {name}, duration={duration_sec}s, "
                f"linear.x={lx}, angular.z={az}"
            )

            start = time.time()
            while time.time() - start < duration_sec:
                self.publish_twist(lx, ly, lz, az)
                rclpy.spin_once(self, timeout_sec=0.0)
                time.sleep(self.period)

        self.get_logger().info("Script done. Publishing final zero Twist.")
        for _ in range(10):
            self.publish_twist(0.0, 0.0, 0.0, 0.0)
            time.sleep(self.period)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/cmd_vel")
    parser.add_argument("--hz", type=float, default=30.0)

    parser.add_argument("--forward-speed", type=float, default=0.5)
    parser.add_argument("--forward-distance", type=float, default=4.0)

    parser.add_argument("--yaw-rate", type=float, default=-0.5)
    parser.add_argument("--turn-deg", type=float, default=90.0)

    return parser.parse_args()


def main():
    args = parse_args()

    forward_duration = abs(args.forward_distance / args.forward_speed)
    turn_duration = abs((args.turn_deg * 3.141592653589793 / 180.0) / args.yaw_rate)

    global TWIST_SCRIPT
    TWIST_SCRIPT = [
        ("forward", forward_duration, args.forward_speed, 0.0, 0.0, 0.0),
        ("stop_after_forward", 1.0, 0.0, 0.0, 0.0, 0.0),
        ("turn", turn_duration, 0.0, 0.0, 0.0, args.yaw_rate),
        ("final_stop", 1.0, 0.0, 0.0, 0.0, 0.0),
    ]

    rclpy.init()
    node = TwistScriptPublisher(args.topic, args.hz)

    try:
        node.run_script()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()