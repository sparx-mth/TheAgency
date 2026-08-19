#!/usr/bin/env python3
"""Host-side flight: take off and fly a warehouse-exploration sweep via cmd_vel.

Runs on the host (small messages cross the host<->container DDS gap fine). Used
to fly the SJTU drone while the in-container recorder captures the real camera --
so the recorded footage is a genuine flight through the warehouse, not a hover.
"""
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty


def main():
    rclpy.init()
    node = rclpy.create_node("fly_explore")
    takeoff = node.create_publisher(Empty, "/simple_drone/takeoff", 1)
    cmd = node.create_publisher(Twist, "/simple_drone/cmd_vel", 1)
    time.sleep(1.0)
    for _ in range(3):
        takeoff.publish(Empty())
        time.sleep(1.0)
    time.sleep(4.0)  # let it climb to hover

    duration = float(__import__("os").environ.get("FLY_SECONDS", "55"))
    t0 = time.time()
    while rclpy.ok() and time.time() - t0 < duration:
        t = time.time() - t0
        tw = Twist()
        tw.linear.x = 0.45                       # forward through the aisle
        tw.angular.z = 0.5 * math.sin(t * 0.25)  # gentle yaw sweep to pan the camera
        if int(t) % 15 >= 12:                    # periodic turn into a new aisle
            tw.linear.x = 0.1
            tw.angular.z = 0.7
        cmd.publish(tw)
        time.sleep(0.1)

    for _ in range(5):
        cmd.publish(Twist())                     # stop, and keep saying so
        time.sleep(0.1)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

