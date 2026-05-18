#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


def load_camera_info_from_yaml(yaml_path: str, frame_id: str) -> CameraInfo:
    yaml_path = Path(yaml_path).expanduser()

    if not yaml_path.exists():
        raise FileNotFoundError(f"Camera YAML file does not exist: {yaml_path}")

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"Camera YAML is empty: {yaml_path}")

    msg = CameraInfo()
    msg.header.frame_id = frame_id

    # Image resolution
    msg.width = int(cfg.get("image_width", 0))
    msg.height = int(cfg.get("image_height", 0))

    if msg.width <= 0 or msg.height <= 0:
        raise ValueError(
            f"Invalid image_width/image_height in camera YAML: {yaml_path}"
        )

    # Distortion model
    msg.distortion_model = cfg.get("distortion_model", "plumb_bob")

    # Distortion coefficients D
    if "distortion_coefficients" in cfg and "data" in cfg["distortion_coefficients"]:
        msg.d = [float(x) for x in cfg["distortion_coefficients"]["data"]]
    else:
        msg.d = [
            float(cfg.get("k1", 0.0)),
            float(cfg.get("k2", 0.0)),
            float(cfg.get("p1", 0.0)),
            float(cfg.get("p2", 0.0)),
            float(cfg.get("k3", 0.0)),
        ]

    # Camera matrix K
    if "camera_matrix" not in cfg or "data" not in cfg["camera_matrix"]:
        raise ValueError(f"Missing camera_matrix.data in YAML: {yaml_path}")
    msg.k = [float(x) for x in cfg["camera_matrix"]["data"]]

    if len(msg.k) != 9:
        raise ValueError(f"camera_matrix.data must contain 9 values: {yaml_path}")

    # Rectification matrix R
    if "rectification_matrix" in cfg and "data" in cfg["rectification_matrix"]:
        msg.r = [float(x) for x in cfg["rectification_matrix"]["data"]]
    else:
        msg.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]

    if len(msg.r) != 9:
        raise ValueError(f"rectification_matrix.data must contain 9 values: {yaml_path}")

    # Projection matrix P
    if "projection_matrix" not in cfg or "data" not in cfg["projection_matrix"]:
        raise ValueError(f"Missing projection_matrix.data in YAML: {yaml_path}")
    msg.p = [float(x) for x in cfg["projection_matrix"]["data"]]

    if len(msg.p) != 12:
        raise ValueError(f"projection_matrix.data must contain 12 values: {yaml_path}")

    return msg


def find_rgb_files(rgb_dir: Path, extensions: list[str]) -> list[Path]:
    paths = []
    for ext in extensions:
        paths.extend(rgb_dir.glob(f"*.{ext}"))
        paths.extend(rgb_dir.glob(f"*.{ext.upper()}"))
    return sorted(set(paths))


class RgbFilePublisher(Node):
    def __init__(self, args):
        super().__init__("rgb_file_publisher")

        self.args = args
        self.bridge = CvBridge()

        self.rgb_dir = Path(args.rgb_dir).expanduser()

        if not self.rgb_dir.exists():
            raise RuntimeError(f"RGB directory does not exist: {self.rgb_dir}")

        self.rgb_paths = find_rgb_files(
            self.rgb_dir,
            extensions=args.rgb_extensions,
        )

        if not self.rgb_paths:
            raise RuntimeError(f"No RGB files found in: {self.rgb_dir}")

        if args.end_idx >= 0:
            self.rgb_paths = self.rgb_paths[args.start_idx : args.end_idx : args.step]
        else:
            self.rgb_paths = self.rgb_paths[args.start_idx :: args.step]

        if not self.rgb_paths:
            raise RuntimeError("No frames left after start/end/step filtering.")
        
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )


        self.rgb_pub = self.create_publisher(Image, args.rgb_topic, image_qos)

        # CameraInfo Publisher and Message Setup
        self.cam_info_pub = self.create_publisher(CameraInfo, args.camera_info_topic, image_qos)

        # Load CameraInfo from YAML instead of hard-coding calibration values
        self.cam_info_msg = load_camera_info_from_yaml(
            args.camera_config_yaml,
            args.frame_id,
        )

        self.get_logger().info(f"Loaded CameraInfo from YAML: {args.camera_config_yaml}")
        self.get_logger().info(
            f"CameraInfo: width={self.cam_info_msg.width}, "
            f"height={self.cam_info_msg.height}, "
            f"fx={self.cam_info_msg.p[0]:.2f}, "
            f"fy={self.cam_info_msg.p[5]:.2f}, "
            f"cx={self.cam_info_msg.p[2]:.2f}, "
            f"cy={self.cam_info_msg.p[6]:.2f}"
        )

        self.idx = 0
        self.finished_once = False


        period = 1.0 / max(args.publish_hz, 1e-6)
        self.timer = self.create_timer(period, self.timer_cb)

        self.get_logger().info(f"RGB dir: {self.rgb_dir}")
        self.get_logger().info(f"Publishing {len(self.rgb_paths)} RGB frames")
        self.get_logger().info(f"RGB topic: {args.rgb_topic}")
        self.get_logger().info(f"Publish Hz: {args.publish_hz}")
        self.get_logger().info(f"Loop: {args.loop}")



    def timer_cb(self):
        if self.idx >= len(self.rgb_paths):
            if self.args.loop:
                self.idx = 0
                self.finished_once = True
                self.get_logger().info("Looping back to first frame")
            else:
                self.get_logger().info("Finished publishing all frames")
                self.timer.cancel()
                rclpy.shutdown()
                return

        rgb_path = self.rgb_paths[self.idx]

        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            self.get_logger().warn(f"Could not read RGB: {rgb_path}")
            self.idx += 1
            return

        stamp = self.get_clock().now().to_msg()

        # Publish CameraInfo with the exact same timestamp
        self.cam_info_msg.header.stamp = stamp
        self.cam_info_pub.publish(self.cam_info_msg)

        rgb_msg = self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
        rgb_msg.header.stamp = stamp
        rgb_msg.header.frame_id = self.args.frame_id

        self.rgb_pub.publish(rgb_msg)

        if self.idx % max(int(self.args.publish_hz), 1) == 0:
            h, w = bgr.shape[:2]
            self.get_logger().info(
                f"published idx={self.idx} "
                f"rgb={w}x{h} "
                f"name={rgb_path.name}"
            )

        self.idx += 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Publish RGB images as synchronized ROS 2 Image topics."
    )

    parser.add_argument("--rgb-dir", required=True)

    parser.add_argument(
        "--rgb-extensions",
        nargs="+",
        default=["jpg", "jpeg", "png"],
    )

    parser.add_argument(
        "--rgb-topic",
        default="/xtend/rgb",
    )
    
    parser.add_argument(
        "--camera-info-topic",
        default="/xtend/camera_info",
    )

    parser.add_argument(
        "--camera-config-yaml",
        default="/home/nvidia/GIT/TheAgency/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_720_420.yaml",
        help="Path to camera calibration YAML file.",
    )
    
    parser.add_argument(
        "--frame-id",
        default="xtend_camera",
    )
    parser.add_argument(
        "--publish-hz",
        type=float,
        default=10.0,
    )

    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=-1)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--loop", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    rclpy.init()
    node = RgbFilePublisher(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()