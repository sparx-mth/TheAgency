#!/usr/bin/env python3
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters

from sparx_agency.core.mapping.depth.depth_anything_v3 import DA3TensorRTModel


class DepthProcessorNode(Node):
    """
    Generic DA3 depth node.

    For DA3METRIC models, convert network output to metric depth in meters using:
        metric_depth_m = focal_px * net_output / 300.0
    where focal_px is typically 0.5 * (fx + fy) from CameraInfo.

    For models that already output meters, disable this scaling with:
        apply_metric_focal_scaling := false
    """

    def __init__(self):
        super().__init__('depth_processor_node')

        self.bridge = CvBridge()

        # Parameters
        self.declare_parameter(
            'engine_path',
            str(Path.home() / 'depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3METRIC-LARGE/DA3METRIC-LARGE_v1.engine')
        )
        self.declare_parameter(
            'config_yaml',
            str(Path.home() / 'GIT/TheAgency/sparx_agency/tasks/mapping/config/simple_drone_front_cam.yaml')
        )
        self.declare_parameter('rgb_topic', '/simple_drone/front/image_raw')
        self.declare_parameter('pub_depth_topic', '/sparx/depth/da3_raw')
        self.declare_parameter('pub_debug_topic', '/sparx/depth/da3_debug')
        self.declare_parameter('publish_debug', True)
        self.declare_parameter('clip_min_m', 0.0)
        self.declare_parameter('clip_max_m', 20.0)

        # DA3 metric-scaling behavior
        self.declare_parameter('apply_metric_focal_scaling', True)
        self.declare_parameter('metric_scale_divisor', 300.0)

        # Load params
        self.engine_path = self.get_parameter('engine_path').value
        self.config_yaml = self.get_parameter('config_yaml').value
        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.pub_depth_topic = self.get_parameter('pub_depth_topic').value
        self.pub_debug_topic = self.get_parameter('pub_debug_topic').value
        self.publish_debug = bool(self.get_parameter('publish_debug').value)
        self.clip_min_m = float(self.get_parameter('clip_min_m').value)
        self.clip_max_m = float(self.get_parameter('clip_max_m').value)
        self.apply_metric_focal_scaling = bool(self.get_parameter('apply_metric_focal_scaling').value)
        self.metric_scale_divisor = float(self.get_parameter('metric_scale_divisor').value)

        #qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        # ===== Model =====
        self.depth_model = DA3TensorRTModel(self.engine_path, self.config_yaml)

        # ===== Publishers =====
        self.pub_depth = self.create_publisher(Image, self.pub_depth_topic, image_qos)
        self.pub_debug = self.create_publisher(Image, self.pub_debug_topic, image_qos)

        # ===== Subscriber =====
        self.sub_image = self.create_subscription(
            Image,
            self.rgb_topic,
            self.image_callback,
            image_qos
        )

        self.get_logger().info('DepthProcessorNode started')
        self.get_logger().info(f'RGB topic: {self.rgb_topic}')
        self.get_logger().info(f'Depth topic: {self.pub_depth_topic}')
        self.get_logger().info(
            f'apply_metric_focal_scaling={self.apply_metric_focal_scaling}, '
            f'metric_scale_divisor={self.metric_scale_divisor}'
        )

    # ===== Utils =====
    def sanitize_depth(self, depth: np.ndarray) -> np.ndarray:
        depth = depth.astype(np.float32, copy=False)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        if self.clip_max_m > self.clip_min_m:
            depth = np.clip(depth, self.clip_min_m, self.clip_max_m)
        return depth

    def convert_to_metric_depth(self, net_output: np.ndarray) -> np.ndarray:
        if not self.apply_metric_focal_scaling:
            return net_output

        focal_px = 0.5 * (self.depth_model.intrinsics.fx + self.depth_model.intrinsics.fy)

        if focal_px <= 0.0:
            raise ValueError(f'Invalid focal length from CameraInfo: focal_px={focal_px}')

        metric_depth = (focal_px * net_output) / self.metric_scale_divisor
        return metric_depth.astype(np.float32, copy=False)

    def depth_to_debug(self, depth: np.ndarray) -> np.ndarray:
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    def publish_depth(self, depth: np.ndarray, header):
        msg = self.bridge.cv2_to_imgmsg(depth.astype(np.float32), encoding='32FC1')
        msg.header = header
        self.pub_depth.publish(msg)

    # ===== Callback =====
    def image_callback(self, msg: Image):
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            net_output, _ = self.depth_model.infer_all(rgb)

            depth_m = self.convert_to_metric_depth(net_output)
            depth_m = self.sanitize_depth(depth_m)
            # publish raw depth
            self.publish_depth(depth_m, msg.header)

            # debug image
            if self.publish_debug:
                dbg = self.depth_to_debug(depth_m)
                dbg_msg = self.bridge.cv2_to_imgmsg(dbg, encoding='bgr8')
                dbg_msg.header = msg.header
                self.pub_debug.publish(dbg_msg)

        except Exception as e:
            self.get_logger().error(f'Processing failed: {e}')


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


if __name__ == '__main__':
    main()
