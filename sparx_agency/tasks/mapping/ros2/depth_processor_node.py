#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from sparx_agency.core.mapping.depth.depth_anything_v3 import DA3TensorRTModel


class DepthProcessorNode(Node):
    def __init__(self):
        super().__init__('depth_processor_node')

        self.bridge = CvBridge()

        # ===== Parameters =====
        self.declare_parameter('engine_path', '')
        self.declare_parameter('config_yaml', '')
        self.declare_parameter('rgb_topic', '/camera/image_raw')
        self.declare_parameter('pub_depth_topic', '/sparx/depth/da3_raw')
        self.declare_parameter('pub_debug_topic', '/sparx/depth/da3_debug')
        self.declare_parameter('publish_debug', True)
        self.declare_parameter('clip_min_m', 0.0)
        self.declare_parameter('clip_max_m', 20.0)

        # ===== Load params =====
        self.engine_path = self.get_parameter('engine_path').value
        self.config_yaml = self.get_parameter('config_yaml').value
        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.pub_depth_topic = self.get_parameter('pub_depth_topic').value
        self.pub_debug_topic = self.get_parameter('pub_debug_topic').value
        self.publish_debug = bool(self.get_parameter('publish_debug').value)
        self.clip_min_m = float(self.get_parameter('clip_min_m').value)
        self.clip_max_m = float(self.get_parameter('clip_max_m').value)

        # ===== QoS =====
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # ===== Model =====
        self.depth_model = DA3TensorRTModel(self.engine_path, self.config_yaml)

        # ===== Publishers =====
        self.pub_depth = self.create_publisher(Image, self.pub_depth_topic, qos)
        self.pub_debug = self.create_publisher(Image, self.pub_debug_topic, qos)

        # ===== Subscriber =====
        self.sub_image = self.create_subscription(
            Image,
            self.rgb_topic,
            self.image_callback,
            qos
        )

        self.get_logger().info('DepthProcessorNode started')
        self.get_logger().info(f'RGB topic: {self.rgb_topic}')
        self.get_logger().info(f'Depth topic: {self.pub_depth_topic}')

    # ===== Utils =====
    def sanitize_depth(self, depth):
        depth = depth.astype(np.float32, copy=False)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        if self.clip_max_m > self.clip_min_m:
            depth = np.clip(depth, self.clip_min_m, self.clip_max_m)
        return depth

    def depth_to_debug(self, depth):
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    def publish_depth(self, depth, header):
        msg = self.bridge.cv2_to_imgmsg(depth.astype(np.float32), encoding='32FC1')
        msg.header = header
        self.pub_depth.publish(msg)

    # ===== Callback =====
    def image_callback(self, msg: Image):
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            depth, _ = self.depth_model.infer_all(rgb)
            depth = self.sanitize_depth(depth)

            # publish raw depth
            self.publish_depth(depth, msg.header)

            # debug image
            if self.publish_debug:
                dbg = self.depth_to_debug(depth)
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
        rclpy.shutdown()


if __name__ == '__main__':
    main()