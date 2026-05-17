#!/usr/bin/env python3

import argparse

import rclpy
from rclpy.executors import MultiThreadedExecutor

from sparx_agency.robots.ROBOTICAN.adapters.rooster_video_adapter import VideoStreamManager
from sparx_agency.robots.ROBOTICAN.adapters.rooster_twist_control_adapter import RoosterTwistControlNode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rooster-id", default="R1")
    parser.add_argument("--host-ip", default="192.168.131.24")
    parser.add_argument("--video-port", type=int, default=5001)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--flight-mode", type=int, default=1)
    parser.add_argument("--cmd-vel-topic", default=None)
    args = parser.parse_args()

    rclpy.init()

    rooster_id = args.rooster_id
    cmd_vel_topic = args.cmd_vel_topic or f"/{rooster_id}/cmd_vel"

    video_node = VideoStreamManager(
        drone_id=rooster_id,
        high_resolution=args.width,
        host_ip=args.host_ip,
        port=args.video_port,
    )

    control_node = RoosterTwistControlNode(
        rooster_id=rooster_id,
        flight_mode=args.flight_mode,
        cmd_vel_topic=cmd_vel_topic,
    )

    executor = MultiThreadedExecutor()
    executor.add_node(video_node)
    executor.add_node(control_node)

    try:
        control_node.get_logger().info(
            f"ROBOTICAN Twist demo running for {rooster_id}. "
            f"Send geometry_msgs/Twist to {cmd_vel_topic}"
        )
        executor.spin()

    except KeyboardInterrupt:
        pass

    finally:
        control_node.stop_motion()
        control_node.destroy_node()
        video_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()