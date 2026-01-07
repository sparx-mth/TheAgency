#!/usr/bin/env python3
import os
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from message_filters import Subscriber, ApproximateTimeSynchronizer

# ---- Your DepthAnythingV2 import (adjust to your repo) ----
# Example:
# from depth_anything_v2.dpt import DepthAnythingV2
# or: from sparx_agency.core.mapping.depth.depth_anything_v2 import DepthAnythingV2
from depth_anything_v2.dpt import DepthAnythingV2  

import torch


def strip_slash(frame_id: str) -> str:
    return frame_id[1:] if frame_id and frame_id.startswith("/") else frame_id


class DepthAnythingNode(Node):
    def __init__(self):
        super().__init__("depth_anything_node")
        self.declare_parameter("use_sim_time", True)

        # Topics
        self.rgb_topic = self.declare_parameter("rgb_topic", "/simple_drone/front/image_raw").value
        self.info_topic = self.declare_parameter("camera_info_topic", "/simple_drone/front/camera_info").value
        self.depth_topic = self.declare_parameter("depth_topic", "/simple_drone/front/depth").value
        self.depth_info_topic = self.declare_parameter("depth_camera_info_topic", "/simple_drone/front/depth/camera_info").value

        # Model
        self.checkpoint = self.declare_parameter("checkpoint", "checkpoints/depth_anything_v2_vits.pth").value
        self.encoder = self.declare_parameter("encoder", "vits").value
        self.input_size = int(self.declare_parameter("input_size", 518).value)

        # IMPORTANT: DepthAnything outputs *relative* depth. We expose a scale you can tune.
        self.depth_scale = float(self.declare_parameter("depth_scale", 5.0).value)  # meters-ish scaling
        self.depth_clip_min = float(self.declare_parameter("depth_clip_min", 0.3).value)
        self.depth_clip_max = float(self.declare_parameter("depth_clip_max", 20.0).value)

        # Behavior
        self.publish_camera_info = bool(self.declare_parameter("publish_camera_info", True).value)

        self.bridge = CvBridge()

        # QoS (sensor data)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.rgb_sub = Subscriber(self, Image, self.rgb_topic, qos_profile=qos)
        self.info_sub = Subscriber(self, CameraInfo, self.info_topic, qos_profile=qos)

        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.info_sub],
            queue_size=10,
            slop=0.05,   # 50ms
            allow_headerless=False
        )
        self.sync.registerCallback(self.cb)

        self.depth_pub = self.create_publisher(Image, self.depth_topic, qos)
        self.info_pub = self.create_publisher(CameraInfo, self.depth_info_topic, qos)

        self.model = self._load_model()

        self.get_logger().info(f"Subscribing rgb={self.rgb_topic}, info={self.info_topic}")
        self.get_logger().info(f"Publishing depth={self.depth_topic}, depth_info={self.depth_info_topic}")

    def _load_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Loading DepthAnythingV2 on {device}")

        model_config = {"encoder": self.encoder, "features": 64, "out_channels": [48, 96, 192, 384]}
        model = DepthAnythingV2(**model_config)

        ckpt = self.checkpoint
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        sd = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(sd)
        model.to(device).eval()
        return model

    @torch.no_grad()
    def cb(self, rgb_msg: Image, info_msg: CameraInfo):
        # Convert rgb
        cv_bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        # DepthAnything usually expects RGB numpy
        cv_rgb = cv_bgr[:, :, ::-1].copy()

        # Run model (returns HxW float32; relative depth)
        depth_rel = self.model.infer_image(cv_rgb, input_size=self.input_size).astype(np.float32)

        # Convert relative -> "meters-ish" by normalization + scale
        d = depth_rel
        d = d - np.nanmin(d)
        denom = (np.nanmax(d) + 1e-6)
        d = d / denom
        depth_m = d * self.depth_scale

        # Clip for point cloud sanity
        depth_m = np.clip(depth_m, self.depth_clip_min, self.depth_clip_max).astype(np.float32)

        # Publish depth image (32FC1)
        depth_msg = self.bridge.cv2_to_imgmsg(depth_m, encoding="32FC1")
        depth_msg.header = rgb_msg.header
        depth_msg.header.frame_id = strip_slash(info_msg.header.frame_id or rgb_msg.header.frame_id)

        self.depth_pub.publish(depth_msg)

        # Publish matching camera_info (passthrough, stamped like depth)
        if self.publish_camera_info:
            out_info = CameraInfo()
            out_info = info_msg
            out_info.header.stamp = depth_msg.header.stamp
            out_info.header.frame_id = depth_msg.header.frame_id
            self.info_pub.publish(out_info)


def main():
    rclpy.init()
    node = DepthAnythingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
