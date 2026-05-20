#!/usr/bin/env python3
from datetime import datetime
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

        self.save_image = True
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
        self.declare_parameter("small_lut_path", str(Path.home() / "GIT/TheAgency/sparx_agency/tasks/mapping/config/lut_small_depth.npz"),)

        self.declare_parameter("image_topic", "/xtend/rgb")
        self.declare_parameter("camera_info_topic", "/xtend/camera_info")
        self.declare_parameter("camera_info_mode", "crop_resize")
        # options: base, padded, crop_resize
        self.declare_parameter("depth_topic", "/xtend/depth_m")
        # self.declare_parameter("debug_topic", "/xtend/depth_vis")

        self.declare_parameter("publish_debug", False)
        self.declare_parameter("clip_min_m", 0.0)
        self.declare_parameter("clip_max_m", 20.0)
        self.declare_parameter("model_type", "small_lut")  # large_metric or small_lut

        self.declare_parameter("small_lut_clip_min_m", 0.2)
        self.declare_parameter("small_lut_clip_max_m", 10.0)
        self.declare_parameter("apply_metric_focal_scaling", True)
        self.declare_parameter("metric_scale_divisor", 300.0)

        self.engine_path = self.get_parameter("engine_path").value
        self.config_yaml = self.get_parameter("config_yaml").value
        self.camera_info_mode = self.get_parameter("camera_info_mode").value
        self.image_topic = self.get_parameter("image_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        # self.debug_topic = self.get_parameter("debug_topic").value

        self.publish_debug = bool(self.get_parameter("publish_debug").value)
        self.clip_min_m = float(self.get_parameter("clip_min_m").value)
        self.clip_max_m = float(self.get_parameter("clip_max_m").value)
        self.model_type = self.get_parameter("model_type").value
        self.small_lut_path = self.get_parameter("small_lut_path").value
        self.small_lut_clip_min_m = float(self.get_parameter("small_lut_clip_min_m").value)
        self.small_lut_clip_max_m = float(self.get_parameter("small_lut_clip_max_m").value)
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

        self.small_lut_raw = None
        self.small_lut_meters = None

        if self.model_type == "small_lut":
            self.load_small_lut(self.small_lut_path)
        elif self.model_type != "large_metric":
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        self.depth_model = DA3TensorRTModel(self.engine_path, self.config_yaml)

        self.pub_depth = self.create_publisher(Image, self.depth_topic, rgb_qos)
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

        if self.camera_info_mode == "base":
            self.camera_info_msg = base_info

        elif self.camera_info_mode == "padded":
            self.camera_info_msg = padded_camera_info(
                base=base_info,
                pad_left=4,
                pad_top=0,
                new_width=728,
                new_height=420,
            )

        elif self.camera_info_mode == "crop_resize":
            self.camera_info_msg = crop_resize_camera_info(
                base=base_info,
                crop_left=90,
                crop_top=0,
                crop_width=540,
                crop_height=420,
                new_width=504,
                new_height=392,
            )

        else:
            raise ValueError(f"Unsupported camera_info_mode: {self.camera_info_mode}")

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

    def load_small_lut(self, lut_path: str):
        if not lut_path:
            raise ValueError("model_type=small_lut requires small_lut_path")

        data = np.load(lut_path)

        raw = data["raw"].astype(np.float32)
        meters = data["meters"].astype(np.float32)

        order = np.argsort(raw)
        self.small_lut_raw = raw[order]
        self.small_lut_meters = meters[order]

        self.get_logger().info("Loaded small depth LUT:")
        for r, z in zip(self.small_lut_raw, self.small_lut_meters):
            self.get_logger().info(f"  raw={r:.6f} -> meters={z:.3f}")

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

    def convert_small_lut_to_metric_depth(self, net_output: np.ndarray) -> np.ndarray:
        if self.small_lut_raw is None or self.small_lut_meters is None:
            raise RuntimeError("Small LUT was not loaded")

        raw = net_output.astype(np.float32, copy=False)

        x = self.small_lut_raw.astype(np.float32)
        y = self.small_lut_meters.astype(np.float32)

        # Normal interpolation inside LUT range.
        depth_m = np.interp(raw, x, y).astype(np.float32)

        # Linear extrapolation below LUT range using first segment.
        low_mask = raw < x[0]
        if np.any(low_mask):
            dx = max(float(x[1] - x[0]), 1e-6)
            slope_low = float(y[1] - y[0]) / dx
            depth_m[low_mask] = y[0] + slope_low * (raw[low_mask] - x[0])

        # Linear extrapolation above LUT range using last segment.
        high_mask = raw > x[-1]
        if np.any(high_mask):
            dx = max(float(x[-1] - x[-2]), 1e-6)
            slope_high = float(y[-1] - y[-2]) / dx
            depth_m[high_mask] = y[-1] + slope_high * (raw[high_mask] - x[-1])

        # Safety clamp. Use wider than calibration range if you want extrapolation.
        depth_m = np.clip(
            depth_m,
            self.small_lut_clip_min_m,
            self.small_lut_clip_max_m,
        )

        return depth_m.astype(np.float32)

    def convert_to_metric_depth(self, net_output: np.ndarray) -> np.ndarray:
        if self.model_type == "small_lut":
            return self.convert_small_lut_to_metric_depth(net_output)

        if self.model_type == "large_metric":
            if not self.apply_metric_focal_scaling:
                return net_output.astype(np.float32, copy=False)

            focal_px = self.get_focal_px()
            if focal_px <= 0.0:
                raise ValueError(f"Invalid focal length: focal_px={focal_px}")

            metric_depth = (focal_px * net_output) / self.metric_scale_divisor
            return metric_depth.astype(np.float32, copy=False)

        raise ValueError(f"Unsupported model_type: {self.model_type}")

    def depth_to_debug(self, depth: np.ndarray) -> np.ndarray:
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    def publish_depth(self, depth: np.ndarray, header):
        if self.save_image:
            time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            np.save(f"/home/user/Pictures/depth{time_str}.npy", depth)
            self.save_image = False
            
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
