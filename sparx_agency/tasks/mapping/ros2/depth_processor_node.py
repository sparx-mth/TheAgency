#!/usr/bin/env python3
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from sparx_agency.core.mapping.depth.depth_anything_v3 import DA3TensorRTModel
from sparx_agency.robots.common.helpers import load_camera_info_from_yaml, padded_camera_info, crop_resize_camera_info


class DepthProcessorNode(Node):
    """
    Generic live DA3 depth node.

    Inputs:
      image_topic        sensor_msgs/Image
      camera_info_topic  sensor_msgs/CameraInfo

    Outputs:
      depth_topic        sensor_msgs/Image, encoding 32FC1
      debug_topic        sensor_msgs/Image, encoding bgr8

    For DA3METRIC models:
      metric_depth_m = focal_px * net_output / metric_scale_divisor

    focal_px is taken from CameraInfo when available. If no CameraInfo was
    received yet, the fallback is the calibration loaded by DA3TensorRTModel.
    """

    def __init__(self):
        super().__init__("depth_processor_node")

        self.bridge = CvBridge()
        self.camera_info_msg: CameraInfo | None = None

        self.declare_parameter(
            "engine_path",
            str(Path.home() / "depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3METRIC-LARGE/DA3METRIC-LARGE_v1.engine"),
        )
        self.declare_parameter(
            "config_yaml",
            str(Path.home() / "GIT/TheAgency/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_720_420.yaml"),
        )


        self.declare_parameter("image_topic", "/xtend/rgb")
        self.declare_parameter("camera_info_topic", "/xtend/camera_info")
        self.declare_parameter("depth_topic", "/xtend/depth_m")
        # self.declare_parameter("debug_topic", "/xtend/depth_vis")

        self.declare_parameter("publish_debug", False)
        self.declare_parameter("clip_min_m", 0.0)
        self.declare_parameter("clip_max_m", 20.0)
        self.declare_parameter("apply_metric_focal_scaling", True)
        self.declare_parameter("metric_scale_divisor", 300.0)

        self.engine_path = self.get_parameter("engine_path").value
        self.config_yaml = self.get_parameter("config_yaml").value
        self.image_topic = self.get_parameter("image_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        # self.debug_topic = self.get_parameter("debug_topic").value

        self.publish_debug = bool(self.get_parameter("publish_debug").value)
        self.clip_min_m = float(self.get_parameter("clip_min_m").value)
        self.clip_max_m = float(self.get_parameter("clip_max_m").value)
        self.apply_metric_focal_scaling = bool(self.get_parameter("apply_metric_focal_scaling").value)
        self.metric_scale_divisor = float(self.get_parameter("metric_scale_divisor").value)

        rgb_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        depth_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.depth_model = DA3TensorRTModel(self.engine_path, self.config_yaml)

        self.pub_depth = self.create_publisher(Image, self.depth_topic, depth_qos)
        # self.pub_debug = self.create_publisher(Image, self.debug_topic, depth_qos)

        self.sub_image = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            rgb_qos,
        )

        base_info = load_camera_info_from_yaml(
            yaml_path=self.config_yaml,
            frame_id="xtend_camera",
        )

        # DA3 LARGEMETRIC MODEL: 728*420
        # self.camera_info_msg  = padded_camera_info(
        #     base=base_info,
        #     pad_left=4,
        #     pad_top=0,
        #     new_width=728,
        #     new_height=420,
        # )
        # DA3 SMALL MODEL 504*392
        self.camera_info_msg = crop_resize_camera_info(
            base=base_info,
            crop_left=90,
            crop_top=0,
            crop_width=540,
            crop_height=420,
            new_width=504,
            new_height=392,
        )

        self.get_logger().info("DepthProcessorNode started")
        self.get_logger().info(f"Image topic: {self.image_topic}")
        self.get_logger().info(f"CameraInfo topic: {self.camera_info_topic}")
        self.get_logger().info(f"Depth topic: {self.depth_topic}")
        # self.get_logger().info(f"Debug topic: {self.debug_topic}")


    def sanitize_depth(self, depth: np.ndarray) -> np.ndarray:
        depth = depth.astype(np.float32, copy=False)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        if self.clip_max_m > self.clip_min_m:
            depth = np.clip(depth, self.clip_min_m, self.clip_max_m)
        return depth

    def get_focal_px(self) -> float:
        if self.camera_info_msg is None:
            raise RuntimeError("CameraInfo was not loaded from config")

        # Prefer projection matrix P for rectified images.
        fx = float(self.camera_info_msg.p[0])
        fy = float(self.camera_info_msg.p[5])

        if fx <= 0.0 or fy <= 0.0:
            # Fallback to raw camera matrix K.
            fx = float(self.camera_info_msg.k[0])
            fy = float(self.camera_info_msg.k[4])

        focal_px = 0.5 * (fx + fy)

        if focal_px <= 0.0:
            raise ValueError(f"Invalid focal length from config: focal_px={focal_px}")

        return focal_px

    def convert_to_metric_depth(self, net_output: np.ndarray) -> np.ndarray:
        if not self.apply_metric_focal_scaling:
            return net_output.astype(np.float32, copy=False)

        focal_px = self.get_focal_px()
        if focal_px <= 0.0:
            raise ValueError(f"Invalid focal length: focal_px={focal_px}")

        metric_depth = (focal_px * net_output) / self.metric_scale_divisor
        return metric_depth.astype(np.float32, copy=False)

    def depth_to_debug(self, depth: np.ndarray) -> np.ndarray:
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    def publish_depth(self, depth: np.ndarray, header):
        msg = self.bridge.cv2_to_imgmsg(depth.astype(np.float32), encoding="32FC1")
        msg.header = header
        self.pub_depth.publish(msg)

    def image_callback(self, msg: Image):
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            net_output, _ = self.depth_model.infer_all(rgb)

            depth_m = self.convert_to_metric_depth(net_output)
            depth_m = self.sanitize_depth(depth_m)
            self.publish_depth(depth_m, msg.header)

            # if self.publish_debug:
            #     dbg = self.depth_to_debug(depth_m)
            #     dbg_msg = self.bridge.cv2_to_imgmsg(dbg, encoding="bgr8")
            #     dbg_msg.header = msg.header
            #     self.pub_debug.publish(dbg_msg)

        except Exception as e:
            self.get_logger().error(f"Processing failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = DepthProcessorNode()

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
