#!/usr/bin/env python3
"""
DA3BatchInferenceNode

Accumulates RGB frames in a sliding window buffer, periodically runs
DepthAnything3 multi-view inference, and publishes:
  /da3/pointcloud   sensor_msgs/PointCloud2   world-frame fused cloud
  /da3/confidence   sensor_msgs/Image         confidence map (last frame)
  /da3/mesh         visualization_msgs/Marker  triangle mesh (requires open3d)
"""
import threading
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker


class DA3BatchInferenceNode(Node):
    """
    Parameters
    ----------
    model_name          HuggingFace model id (default: depth-anything/da3metric-large)
    batch_size          Max frames per inference call (default: 10)
    batch_interval_s    Seconds between inference runs (default: 5.0)
    process_res         Inference resolution (default: 504)
    conf_thresh_pct     Confidence percentile filter for pointcloud (default: 40.0)
    max_depth_m         Depth clip max in metres (default: 10.0)
    frame_id            ROS frame_id for published cloud/mesh (default: map)
    rgb_topic           Input RGB topic (default: /rgbd/rgb)
    """

    def __init__(self):
        super().__init__("da3_batch_inference_node")
        self._bridge = CvBridge()
        self._model = None
        self._lock = threading.Lock()
        self._inference_running = False

        self.declare_parameter("model_name", "depth-anything/da3metric-large")
        self.declare_parameter("batch_size", 10)
        self.declare_parameter("batch_interval_s", 5.0)
        self.declare_parameter("process_res", 504)
        self.declare_parameter("conf_thresh_pct", 40.0)
        self.declare_parameter("max_depth_m", 10.0)
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("rgb_topic", "/rgbd/rgb")

        self.model_name = self.get_parameter("model_name").value
        self.batch_size = int(self.get_parameter("batch_size").value)
        self.process_res = int(self.get_parameter("process_res").value)
        self.conf_thresh_pct = float(self.get_parameter("conf_thresh_pct").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.frame_id = self.get_parameter("frame_id").value
        rgb_topic = self.get_parameter("rgb_topic").value
        interval = float(self.get_parameter("batch_interval_s").value)

        self._buffer: deque = deque(maxlen=self.batch_size)

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.sub_rgb = self.create_subscription(Image, rgb_topic, self._rgb_cb, qos)
        self.pub_cloud = self.create_publisher(PointCloud2, "/da3/pointcloud", qos)
        self.pub_conf = self.create_publisher(Image, "/da3/confidence", qos)
        self.pub_mesh = self.create_publisher(Marker, "/da3/mesh", 10)

        self.timer = self.create_timer(interval, self._timer_cb)
        self.get_logger().info(
            f"DA3BatchInferenceNode: model={self.model_name} "
            f"batch={self.batch_size} interval={interval}s"
        )

    def _rgb_cb(self, msg: Image):
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge failed: {e}", throttle_duration_sec=2.0)
            return
        rgb = bgr[:, :, ::-1].copy()
        with self._lock:
            self._buffer.append((rgb, msg.header.stamp))

    def _timer_cb(self):
        with self._lock:
            if self._inference_running or len(self._buffer) < 2:
                return
            frames = [item[0] for item in self._buffer]
            stamp = self._buffer[-1][1]
            self._inference_running = True

        thread = threading.Thread(
            target=self._run_inference, args=(frames, stamp), daemon=True
        )
        thread.start()

    def _load_model(self):
        if self._model is not None:
            return
        import torch
        from depth_anything_3.api import DepthAnything3

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Loading {self.model_name} on {device}…")
        self._model = DepthAnything3.from_pretrained(self.model_name).to(device)
        self.get_logger().info("Model loaded.")

    def _run_inference(self, frames: list, stamp):
        try:
            self._load_model()
            prediction = self._model.inference(
                frames,
                process_res=self.process_res,
                conf_thresh_percentile=self.conf_thresh_pct,
            )
            self._publish_cloud(prediction, stamp)
            self._publish_confidence(prediction, stamp)
            self._publish_mesh(prediction, stamp)
        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")
        finally:
            with self._lock:
                self._inference_running = False

    def _backproject_frame(self, depth: np.ndarray, K: np.ndarray, conf: np.ndarray) -> np.ndarray:
        """Returns (N, 3) float32 points in camera frame, filtered by depth and confidence."""
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        z = depth.flatten()
        x = (u.flatten() - K[0, 2]) * z / K[0, 0]
        y = (v.flatten() - K[1, 2]) * z / K[1, 1]

        valid = (z > 0.01) & (z < self.max_depth_m)
        if conf is not None:
            threshold = float(np.percentile(conf.flatten(), self.conf_thresh_pct))
            valid &= conf.flatten() >= threshold

        return np.stack([x[valid], y[valid], z[valid]], axis=1).astype(np.float32)

    def _to_world(self, pts_cam: np.ndarray, w2c: np.ndarray) -> np.ndarray:
        """Transform (N,3) camera-frame points to world frame using w2c [3,4]."""
        w2c_4x4 = np.vstack([w2c, [0.0, 0.0, 0.0, 1.0]])
        c2w = np.linalg.inv(w2c_4x4)
        ones = np.ones((len(pts_cam), 1), dtype=np.float32)
        pts_h = np.hstack([pts_cam, ones])
        return (c2w @ pts_h.T).T[:, :3].astype(np.float32)

    def _prediction_to_points(self, prediction) -> np.ndarray:
        """Fuse all frames into a single world-frame point cloud (N, 3)."""
        all_pts = []
        n = len(prediction.depth)
        for i in range(n):
            conf = prediction.conf[i] if prediction.conf is not None else None
            pts_cam = self._backproject_frame(prediction.depth[i], prediction.intrinsics[i], conf)
            if len(pts_cam) == 0:
                continue
            pts_world = self._to_world(pts_cam, prediction.extrinsics[i])
            all_pts.append(pts_world)
        if not all_pts:
            return np.zeros((0, 3), dtype=np.float32)
        return np.concatenate(all_pts, axis=0)

    def _publish_cloud(self, prediction, stamp):
        points = self._prediction_to_points(prediction)
        if len(points) == 0:
            return
        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
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
        self.pub_cloud.publish(msg)
        self.get_logger().info(f"Published cloud: {len(points)} pts", throttle_duration_sec=2.0)

    def _publish_confidence(self, prediction, stamp):
        if prediction.conf is None or len(prediction.conf) == 0:
            return
        conf = prediction.conf[-1].astype(np.float32)
        msg = self._bridge.cv2_to_imgmsg(conf, encoding="32FC1")
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        self.pub_conf.publish(msg)

    def _publish_mesh(self, prediction, stamp):
        try:
            import open3d as o3d
        except ImportError:
            return

        points = self._prediction_to_points(prediction)
        if len(points) < 100:
            return

        # Downsample to keep Poisson fast
        max_pts = 50_000
        if len(points) > max_pts:
            idx = np.random.choice(len(points), max_pts, replace=False)
            points = points[idx]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30)
        )
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())

        verts = np.asarray(mesh.vertices)
        tris = np.asarray(mesh.triangles)
        if len(tris) == 0:
            return

        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.frame_id
        marker.ns = "da3_mesh"
        marker.id = 0
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        marker.scale.x = marker.scale.y = marker.scale.z = 1.0
        marker.color = ColorRGBA(r=0.7, g=0.7, b=0.7, a=0.85)
        for tri in tris:
            for vi in tri:
                p = Point()
                p.x, p.y, p.z = float(verts[vi, 0]), float(verts[vi, 1]), float(verts[vi, 2])
                marker.points.append(p)
        self.pub_mesh.publish(marker)
        self.get_logger().info(f"Published mesh: {len(tris)} triangles", throttle_duration_sec=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = DA3BatchInferenceNode()
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
