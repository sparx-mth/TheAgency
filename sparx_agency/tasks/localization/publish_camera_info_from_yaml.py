#!/usr/bin/env python3

import argparse
import yaml
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy


class CameraInfoPublisher(Node):
    def __init__(self, yaml_path, topic, frame_id, rate_hz, qos_policy):
        super().__init__("camera_info_from_yaml_publisher")

        reliability = (
            QoSReliabilityPolicy.RELIABLE
            if qos_policy == "reliable"
            else QoSReliabilityPolicy.BEST_EFFORT
        )

        qos = QoSProfile(
            reliability=reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.pub = self.create_publisher(CameraInfo, topic, qos)
        self.msg = self.load_camera_info(yaml_path, frame_id)

        self.timer = self.create_timer(1.0 / rate_hz, self.publish_camera_info)

        self.get_logger().info(f"Publishing CameraInfo from YAML: {yaml_path}")
        self.get_logger().info(f"Topic: {topic}")
        self.get_logger().info(f"Frame ID: {frame_id}")

    def load_camera_info(self, yaml_path, frame_id):
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        msg = CameraInfo()
        msg.header.frame_id = frame_id

        # image size
        msg.width = int(data.get("image_width", data.get("width", 0)))
        msg.height = int(data.get("image_height", data.get("height", 0)))

        # K matrix
        if "camera_matrix" in data:
            msg.k = [float(x) for x in data["camera_matrix"]["data"]]
        elif "K" in data:
            msg.k = [float(x) for x in data["K"]]
        else:
            raise ValueError("Could not find camera_matrix/data or K in YAML")

        # D distortion
        if "distortion_coefficients" in data:
            msg.d = [float(x) for x in data["distortion_coefficients"]["data"]]
        elif "D" in data:
            msg.d = [float(x) for x in data["D"]]
        else:
            msg.d = []

        # distortion model
        msg.distortion_model = data.get("distortion_model", "plumb_bob")

        # R matrix
        if "rectification_matrix" in data:
            msg.r = [float(x) for x in data["rectification_matrix"]["data"]]
        elif "R" in data:
            msg.r = [float(x) for x in data["R"]]
        else:
            msg.r = [
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
            ]

        # P matrix
        if "projection_matrix" in data:
            msg.p = [float(x) for x in data["projection_matrix"]["data"]]
        elif "P" in data:
            msg.p = [float(x) for x in data["P"]]
        else:
            K = np.array(msg.k).reshape(3, 3)
            msg.p = [
                K[0, 0], K[0, 1], K[0, 2], 0.0,
                K[1, 0], K[1, 1], K[1, 2], 0.0,
                K[2, 0], K[2, 1], K[2, 2], 0.0,
            ]

        return msg

    def publish_camera_info(self):
        now = self.get_clock().now().to_msg()
        self.msg.header.stamp = now
        self.pub.publish(self.msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", required=True, help="Path to camera calibration YAML")
    parser.add_argument("--topic", default="/xtend/camera_info")
    parser.add_argument("--frame_id", default="camera")
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument("--qos", choices=["best_effort", "reliable"], default="best_effort")
    args = parser.parse_args()

    rclpy.init()
    node = CameraInfoPublisher(
        yaml_path=args.yaml,
        topic=args.topic,
        frame_id=args.frame_id,
        rate_hz=args.rate,
        qos_policy=args.qos,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()