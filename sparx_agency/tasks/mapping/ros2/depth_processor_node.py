#!/usr/bin/env python3
import io
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header, String
from cv_bridge import CvBridge

from sparx_agency.core.mapping.depth.depth_anything_v3 import DA3TensorRTModel
from sparx_agency.robots.common.helpers import load_camera_info_from_yaml, padded_camera_info, crop_resize_camera_info


class DepthProcessorNode(Node):
    """
    Generic live DA3 depth node.

    Inputs:
      image_topic        sensor_msgs/Image

    Outputs:
      depth_topic        sensor_msgs/Image, encoding 32FC1

    For DA3METRIC LARGE:
      metric_depth_m = focal_px * net_output / metric_scale_divisor
      (apply_metric_focal_scaling=True, metric_scale_divisor=300.0)
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
        self.declare_parameter("small_lut_path", str(Path.home() / "GIT/TheAgency/sparx_agency/tasks/mapping/config/lut_small_depth.npz"),)

        self.declare_parameter("image_topic", "/xtend/rgb")
        self.declare_parameter("frame_path_topic", "")
        self.declare_parameter("camera_info_topic", "/xtend/camera_info")
        self.declare_parameter("camera_info_mode", "crop_resize")
        # options: base, padded, crop_resize
        self.declare_parameter("depth_topic", "/xtend/depth_m")
        self.declare_parameter("depth_encoding", "32FC1")  # options: 32FC1, 16UC1
        self.declare_parameter("depth_path_topic", "/xtend/depth_frame_path")
        self.declare_parameter("depth_dir", "/tmp/xtend_depth")
        self.declare_parameter("max_depth_kept", 30)
        self.declare_parameter("publish_depth_ros", True)
        # self.declare_parameter("debug_topic", "/xtend/depth_vis")

        self.declare_parameter("publish_debug", False)
        self.declare_parameter("publish_cloud", False)
        self.declare_parameter("pointcloud_topic", "/rgbd/pointcloud")
        self.declare_parameter("clip_min_m", 0.0)
        self.declare_parameter("clip_max_m", 20.0)
        self.declare_parameter("model_type", "large_metric")  # large_metric or small_lut

        self.declare_parameter("small_lut_clip_min_m", 0.2)
        self.declare_parameter("small_lut_clip_max_m", 10.0)
        self.declare_parameter("apply_metric_focal_scaling", True)
        self.declare_parameter("metric_scale_divisor", 300.0)
        self.declare_parameter("metric_output_scale", 1.0)  # e.g. 0.88 for DA2-metric-indoor-small

        self.engine_path = self.get_parameter("engine_path").value
        self.config_yaml = self.get_parameter("config_yaml").value
        self.camera_info_mode = self.get_parameter("camera_info_mode").value
        self.image_topic = self.get_parameter("image_topic").value
        self.frame_path_topic = str(self.get_parameter("frame_path_topic").value).strip()
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.depth_encoding = str(self.get_parameter("depth_encoding").value)
        self.depth_path_topic = str(self.get_parameter("depth_path_topic").value).strip()
        self.depth_dir = Path(str(self.get_parameter("depth_dir").value)).expanduser().resolve()
        self.max_depth_kept = int(self.get_parameter("max_depth_kept").value)
        self.publish_depth_ros = bool(self.get_parameter("publish_depth_ros").value)
        # self.debug_topic = self.get_parameter("debug_topic").value

        self.publish_debug = bool(self.get_parameter("publish_debug").value)
        self.publish_cloud = bool(self.get_parameter("publish_cloud").value)
        self.pointcloud_topic = self.get_parameter("pointcloud_topic").value
        self.clip_min_m = float(self.get_parameter("clip_min_m").value)
        self.clip_max_m = float(self.get_parameter("clip_max_m").value)
        self.model_type = self.get_parameter("model_type").value
        self.small_lut_path = self.get_parameter("small_lut_path").value
        self.small_lut_clip_min_m = float(self.get_parameter("small_lut_clip_min_m").value)
        self.small_lut_clip_max_m = float(self.get_parameter("small_lut_clip_max_m").value)
        self.apply_metric_focal_scaling = bool(self.get_parameter("apply_metric_focal_scaling").value)
        self.metric_scale_divisor = float(self.get_parameter("metric_scale_divisor").value)
        self.metric_output_scale = float(self.get_parameter("metric_output_scale").value)

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )


        self.small_lut_raw = None
        self.small_lut_meters = None

        if self.model_type == "small_lut":
            self.load_small_lut(self.small_lut_path)
        elif self.model_type != "large_metric":
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        if self.depth_encoding not in ("32FC1", "16UC1"):
            raise ValueError(f"Unsupported depth_encoding: {self.depth_encoding}")

        self.depth_model = DA3TensorRTModel(
            self.engine_path,
            self.config_yaml,
            log_fn=self.get_logger().info,
        )

        self.pub_depth = (
            self.create_publisher(Image, self.depth_topic, image_qos)
            if self.publish_depth_ros else None
        )
        self.pub_cloud = (
            self.create_publisher(PointCloud2, self.pointcloud_topic, image_qos)
            if self.publish_cloud else None
        )
        
        self._depth_seq = 0
        self.pub_depth_path: "rclpy.publisher.Publisher | None" = None
        if self.depth_path_topic:
            self.depth_dir.mkdir(parents=True, exist_ok=True)
            for f in self.depth_dir.glob("depth_*.npy"):
                f.unlink(missing_ok=True)
            for f in self.depth_dir.glob("depth_*.tmp"):
                f.unlink(missing_ok=True)
            self.pub_depth_path = self.create_publisher(String, self.depth_path_topic, 10)
            self.get_logger().info(f"Depth dir publisher: {self.depth_dir} -> {self.depth_path_topic}")
        # self.pub_debug = self.create_publisher(Image, self.debug_topic, image_qos)

        if self.frame_path_topic:
            self.create_subscription(
                String,
                self.frame_path_topic,
                self.frame_path_callback,
                image_qos,
            )
        else:
            self.create_subscription(
                Image,
                self.image_topic,
                self.image_callback,
                image_qos,
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
        if self.frame_path_topic:
            self.get_logger().info(f"Frame path topic: {self.frame_path_topic} (overrides image_topic)")
        self.get_logger().info(f"CameraInfo topic: {self.camera_info_topic}")
        self.get_logger().info(f"Depth topic: {self.depth_topic}")
        self.get_logger().info(f"Depth encoding: {self.depth_encoding}")
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

        # DA3 spec: focal from intrinsic matrix K (not P which may differ for rectified images).
        fx = float(self.camera_info_msg.k[0])
        fy = float(self.camera_info_msg.k[4])

        if fx <= 0.0 or fy <= 0.0:
            # Fallback to projection matrix P.
            fx = float(self.camera_info_msg.p[0])
            fy = float(self.camera_info_msg.p[5])

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

    def publish_depth(self, depth_m: np.ndarray, header):
        if self.pub_depth is None:
            return
        if self.depth_encoding == "32FC1":
            msg = self.bridge.cv2_to_imgmsg(
                depth_m.astype(np.float32, copy=False),
                encoding="32FC1",
            )

        elif self.depth_encoding == "16UC1":
            # ROS convention: 16UC1 depth is usually millimeters.
            depth_mm = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
            depth_mm = np.clip(depth_mm * 1000.0, 0.0, 65535.0).astype(np.uint16)

            msg = self.bridge.cv2_to_imgmsg(
                depth_mm,
                encoding="16UC1",
            )

        else:
            raise ValueError(f"Unsupported depth_encoding: {self.depth_encoding}")

        msg.header = header
        self.pub_depth.publish(msg)

    def _backproject_metric_depth(self, depth_m: np.ndarray) -> np.ndarray:
        """Backproject (H,W) metric depth to (N,3) float32 points in camera optical frame.

        Intrinsics from camera_info_msg are scaled to match the depth output resolution.
        """
        h, w = depth_m.shape
        cam_w = self.camera_info_msg.width or w
        cam_h = self.camera_info_msg.height or h
        sx = w / cam_w
        sy = h / cam_h
        fx = float(self.camera_info_msg.p[0]) * sx
        fy = float(self.camera_info_msg.p[5]) * sy
        cx = float(self.camera_info_msg.p[2]) * sx
        cy = float(self.camera_info_msg.p[6]) * sy

        u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        z = depth_m.flatten()
        valid = (z > 0.01) & (z < self.clip_max_m)
        x = (u.flatten() - cx) * z / fx
        y = (v.flatten() - cy) * z / fy
        return np.stack([x[valid], y[valid], z[valid]], axis=1).astype(np.float32)

    def _make_pc2_msg(self, points: np.ndarray, header) -> PointCloud2:
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * len(points)
        msg.is_dense = True
        msg.data = points.tobytes()
        return msg

    def frame_path_callback(self, msg: String):
        """
        Callback for the file-path-based RGB input (from OnlineNavBridgeDirPublisher).
        Message format: "{abs_path} {sec} {nanosec}"
        Reads the JPEG from disk, runs depth inference, and publishes with the
        original RGB stamp so downstream nodes see matching timestamps.
        """
        try:
            parts = msg.data.rsplit(" ", 2)
            path = parts[0]
            sec = int(parts[1]) if len(parts) >= 3 else 0
            nanosec = int(parts[2]) if len(parts) >= 3 else 0
        except Exception as e:
            self.get_logger().error(f"[frame_path] bad message format: {e}")
            return

        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            self.get_logger().error(f"[frame_path] cv2.imread failed: {path}", throttle_duration_sec=2.0)
            return

        header = Header()
        header.frame_id = self.camera_info_msg.header.frame_id
        header.stamp.sec = sec
        header.stamp.nanosec = nanosec

        rgb_stem = Path(path).stem  # e.g. "frame_00000216"
        self._run_inference_and_publish(bgr, header, rgb_stem=rgb_stem)

    def _run_inference_and_publish(self, bgr: np.ndarray, header, rgb_stem: str = ""):
        t0 = time.perf_counter()
        try:
            t1 = time.perf_counter()

            net_output = self.depth_model.infer_all(bgr)
            t2 = time.perf_counter()

            depth_m = self.convert_to_metric_depth(net_output)
            t3 = time.perf_counter()

            depth_m = self.sanitize_depth(depth_m)
            t4 = time.perf_counter()

            if self.publish_depth_ros:
                self.publish_depth(depth_m, header)
            t5 = time.perf_counter()

            if self.pub_depth_path is not None:
                if rgb_stem:
                    stem = rgb_stem
                else:
                    self._depth_seq += 1
                    stem = f"depth_{self._depth_seq:08d}"
                final_path = self.depth_dir / f"{stem}.npy"
                tmp_path = final_path.with_suffix(".tmp")
                buf = io.BytesIO()
                np.save(buf, depth_m)
                tmp_path.write_bytes(buf.getvalue())
                tmp_path.rename(final_path)
                if self.max_depth_kept > 0:
                    existing = sorted(self.depth_dir.glob("*.npy"))
                    for old in existing[: max(0, len(existing) - self.max_depth_kept)]:
                        old.unlink(missing_ok=True)
                path_msg = String()
                path_msg.data = f"{final_path} {header.stamp.sec} {header.stamp.nanosec}"
                self.pub_depth_path.publish(path_msg)

            if self.pub_cloud is not None:
                pts = self._backproject_metric_depth(depth_m)
                self.pub_cloud.publish(self._make_pc2_msg(pts, header))

            self.get_logger().info(
                "depth_node timing ms: "
                f"infer={(t2 - t1) * 1000.0:.1f}, "
                f"metric={(t3 - t2) * 1000.0:.1f}, "
                f"sanitize={(t4 - t3) * 1000.0:.1f}, "
                f"publish={(t5 - t4) * 1000.0:.1f}, "
                f"total={(t5 - t0) * 1000.0:.1f}",
                throttle_duration_sec=1.0,
            )
        except Exception as e:
            self.get_logger().error(f"Processing failed: {e}", throttle_duration_sec=2.0)

    def image_callback(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge failed: {e}", throttle_duration_sec=2.0)
            return
        self._run_inference_and_publish(bgr, msg.header)


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
