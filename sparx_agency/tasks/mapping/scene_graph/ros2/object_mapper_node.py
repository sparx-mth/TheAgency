"""object_mapper_node — detections JSON + depth + odom -> world-XY landmarks.

Joins ``/perception/detections`` (std_msgs/String JSON from the detection
server bridge) with the SJTU depth camera and odometry to place objects on
the world map. Each detection's bbox is rescaled from RGB to depth
intrinsics, given a robust depth, back-projected through the odom pose
(optical -> body FLU -> world ENU, camera 20 cm ahead of the body origin),
and folded into a per-class deduplicated landmark map. The confirmed set is
published latched on ``/perception/objects`` when it changes, at most 1 Hz.

Ported from the sjtu_project ``semantic_mapper/object_mapper_node.py`` onto
the new core (:mod:`sparx_agency.core.mapping.objects`); the old node's
silent xacro-fallback intrinsics are deliberately gone — detections are
dropped (and counted) until both camera_infos have arrived.

Run:
    .venv/bin/python -m sparx_agency.tasks.mapping.scene_graph.ros2.object_mapper_node \
        --ros-args -p use_sim_time:=true
"""
from __future__ import annotations

import json
from collections import deque

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import CameraInfo, Image
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from sparx_agency.core.common.math.se3 import quaternion_matrix
from sparx_agency.core.mapping.objects.geometry import (
    backproject_bbox_to_world, rescale_bbox_between_intrinsics,
    robust_bbox_depth)
from sparx_agency.core.mapping.objects.landmarks import ObjectLandmarkMap
from sparx_agency.robots.SJTU.adapters.topics import (
    FRONT_CAMERA_INFO, FRONT_CAMERA_OFFSET_FLU, FRONT_DEPTH_CAMERA_INFO,
    FRONT_DEPTH_IMAGE, ODOM)
from sparx_agency.tasks.mapping.scene_graph.ros2.payloads import objects_payload
from sparx_agency.tasks.mapping.scene_graph.ros2.qos import (latched_qos,
                                                             sensor_qos)

DEPTH_DEQUE_LEN = 30    # ~2 s of 15 Hz depth
ODOM_DEQUE_LEN = 120    # a few seconds of odometry
PUBLISH_MIN_PERIOD_S = 1.0  # contract: at most 1 Hz


def _stamp_to_sec(stamp) -> float:
    """builtin_interfaces/Time -> float seconds."""
    return stamp.sec + stamp.nanosec * 1e-9


def _depth_metres(msg: Image) -> np.ndarray:
    """A ``sensor_msgs/Image`` as an HxW float32 array of metres.

    Decoded with numpy (no cv_bridge on this host), keyed on the encoding;
    an unknown encoding raises rather than guessing a scale. ``step`` is the
    row stride in bytes and is honoured — a padded row read as if tight
    shears the image diagonally.
    """
    if msg.encoding == "32FC1":
        dtype, scale = np.float32, 1.0
    elif msg.encoding == "16UC1":
        dtype, scale = np.uint16, 1e-3
    else:
        raise ValueError("unsupported depth encoding %r" % (msg.encoding,))
    itemsize = np.dtype(dtype).itemsize
    row = msg.step // itemsize if msg.step else msg.width
    data = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, row)
    return data[:, :msg.width].astype(np.float32) * scale


def _nearest(entries, t: float):
    """The (stamp, ...) tuple in ``entries`` nearest ``t``, or None."""
    if not entries:
        return None
    return min(entries, key=lambda e: abs(e[0] - t))


class ObjectMapperNode(Node):
    """Detections -> deduplicated world-XY object landmarks."""

    def __init__(self):
        super().__init__("object_mapper")

        p = self.declare_parameter
        p("detections_topic", "/perception/detections")
        # Sim topics come from robots/SJTU/adapters/topics.py, never spelled
        # here: that module is pinned against the plugin by the SJTU tests, so
        # a namespace or leaf rename moves every consumer at once instead of
        # leaving this node subscribed to a name nothing publishes (which
        # fails as "no data", not as an error).
        p("depth_topic", FRONT_DEPTH_IMAGE)
        p("rgb_info_topic", FRONT_CAMERA_INFO)
        p("depth_info_topic", FRONT_DEPTH_CAMERA_INFO)
        p("odom_topic", ODOM)
        # Matches the flown stack and the detection server's own --conf gate.
        # Raising it here filters a second time and silently: YOLO-World scores
        # the open-vocabulary hospital prompts ("wheelchair", "hospital bed",
        # "medical trolley") in the 0.3-0.4 band, so a 0.5 gate publishes an
        # empty confirmed set for a whole flight and reads as "nothing seen yet".
        p("conf_min", 0.25)
        p("max_stamp_gap_s", 0.25)
        p("min_depth_m", 0.30)
        # This aircraft's depth sensor clips at <far>10.0</far> (600x600 front
        # depth, sjtu_drone.urdf.xacro on xtend_integration_nadav), so 8 m keeps
        # margin inside the clip. The old branch clipped at 5 m -- check the
        # xacro before assuming this number travels to another sim checkout.
        p("max_depth_m", 8.0)
        p("dedupe_radius_m", 0.70)
        p("min_observations", 2)
        # PREFIX match, not equality: "door" also skips "door frame" and
        # "doorway". Doors are wanted in the DETECTOR vocabulary — they help
        # the room segmentation — but never as object landmarks, or every
        # doorway is confirmed as one and pollutes the per-room object lists
        # the room labels are inferred from.
        p("skip_classes", ["door"])

        g = lambda n: self.get_parameter(n).value
        self._conf_min = float(g("conf_min"))
        self._max_gap = float(g("max_stamp_gap_s"))
        self._min_depth = float(g("min_depth_m"))
        self._max_depth = float(g("max_depth_m"))
        # Empty entries are dropped: as a prefix, "" would skip every class,
        # which under the old exact match was merely a no-op.
        self._skip = set(s for s in (str(c).strip()
                                     for c in (g("skip_classes") or [])) if s)
        self._landmarks = ObjectLandmarkMap(
            dedupe_radius_m=float(g("dedupe_radius_m")),
            min_observations=int(g("min_observations")))

        # State
        self._depths = deque(maxlen=DEPTH_DEQUE_LEN)  # (t, HxW float32 m)
        self._odoms = deque(maxlen=ODOM_DEQUE_LEN)    # (t, p (3,), R (3,3))
        self._rgb_k = None      # (fx, fy, cx, cy)
        self._depth_k = None
        self._intrinsics_logged = False
        self._last_signature = None
        self._last_pub_t = None
        self._dirty = False
        self._counts = dict(dets=0, projected=0, dropped_gap=0,
                            skipped_conf=0, skipped_class=0,
                            skipped_depth=0, no_intrinsics=0, bad_msgs=0)

        sensor = sensor_qos()
        det_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.VOLATILE,
                             history=HistoryPolicy.KEEP_LAST, depth=5)
        latched = latched_qos()

        self.create_subscription(String, str(g("detections_topic")),
                                 self._detections_cb, det_qos)
        self.create_subscription(Image, str(g("depth_topic")),
                                 self._depth_cb, sensor)
        self.create_subscription(CameraInfo, str(g("rgb_info_topic")),
                                 self._rgb_info_cb, sensor)
        self.create_subscription(CameraInfo, str(g("depth_info_topic")),
                                 self._depth_info_cb, sensor)
        self.create_subscription(Odometry, str(g("odom_topic")),
                                 self._odom_cb, sensor)

        self._pub_objects = self.create_publisher(
            String, "/perception/objects", latched)

        self.create_timer(PUBLISH_MIN_PERIOD_S, self._flush)
        self.create_timer(10.0, self._heartbeat)

        # Latched liveness: late joiners see an empty confirmed set rather
        # than nothing at all.
        self._publish()

        self.get_logger().info(
            "object_mapper up: detections=%s depth=%s odom=%s | "
            "conf_min=%.2f gap<=%.2fs depth=[%.2f, %.2f]m dedupe=%.2fm "
            "min_obs=%d skip=%s cam_offset=%s" % (
                g("detections_topic"), g("depth_topic"), g("odom_topic"),
                self._conf_min, self._max_gap, self._min_depth,
                self._max_depth, float(g("dedupe_radius_m")),
                int(g("min_observations")), sorted(self._skip),
                FRONT_CAMERA_OFFSET_FLU))

    # ── input callbacks ──────────────────────────────────────────────
    def _rgb_info_cb(self, msg: CameraInfo):
        if self._rgb_k is None:
            self._rgb_k = (float(msg.k[0]), float(msg.k[4]),
                           float(msg.k[2]), float(msg.k[5]))
            self._log_intrinsics_once()

    def _depth_info_cb(self, msg: CameraInfo):
        if self._depth_k is None:
            self._depth_k = (float(msg.k[0]), float(msg.k[4]),
                             float(msg.k[2]), float(msg.k[5]))
            self._log_intrinsics_once()

    def _log_intrinsics_once(self):
        if self._intrinsics_logged or self._rgb_k is None \
                or self._depth_k is None:
            return
        self._intrinsics_logged = True
        self.get_logger().info(
            "intrinsics resolved: rgb fx=%.1f fy=%.1f cx=%.1f cy=%.1f | "
            "depth fx=%.1f fy=%.1f cx=%.1f cy=%.1f"
            % (self._rgb_k + self._depth_k))

    def _depth_cb(self, msg: Image):
        try:
            depth = _depth_metres(msg)
        except ValueError as exc:
            self._counts["bad_msgs"] += 1
            self.get_logger().error("depth image rejected: %s" % (exc,),
                                    throttle_duration_sec=5.0)
            return
        self._depths.append((_stamp_to_sec(msg.header.stamp), depth))

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        rotation = quaternion_matrix((q.x, q.y, q.z, q.w))[:3, :3]
        self._odoms.append((_stamp_to_sec(msg.header.stamp),
                            np.array([p.x, p.y, p.z]), rotation))

    # ── the join ─────────────────────────────────────────────────────
    def _detections_cb(self, msg: String):
        self._counts["dets"] += 1
        try:
            payload = json.loads(msg.data)
            stamp = float(payload["stamp"])
            detections = payload["detections"]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self._counts["bad_msgs"] += 1
            self.get_logger().error(
                "bad /perception/detections payload: %s" % (exc,),
                throttle_duration_sec=5.0)
            return

        if self._rgb_k is None or self._depth_k is None:
            self._counts["no_intrinsics"] += 1
            return

        depth_entry = _nearest(self._depths, stamp)
        odom_entry = _nearest(self._odoms, stamp)
        if (depth_entry is None or odom_entry is None
                or abs(depth_entry[0] - stamp) > self._max_gap
                or abs(odom_entry[0] - stamp) > self._max_gap):
            self._counts["dropped_gap"] += 1
            return
        depth_img = depth_entry[1]
        _, translation, rotation = odom_entry

        for det in detections:
            try:
                cls = str(det["cls"])
                conf = float(det["conf"])
                bbox = tuple(float(v) for v in det["xyxy"])
            except (KeyError, TypeError, ValueError) as exc:
                self._counts["bad_msgs"] += 1
                self.get_logger().error(
                    "bad detection entry: %s" % (exc,),
                    throttle_duration_sec=5.0)
                continue
            if conf < self._conf_min:
                self._counts["skipped_conf"] += 1
                continue
            if any(cls.startswith(s) for s in self._skip):
                self._counts["skipped_class"] += 1
                continue

            depth_bbox = rescale_bbox_between_intrinsics(
                bbox, self._rgb_k, self._depth_k)
            depth_m = robust_bbox_depth(
                depth_img, depth_bbox,
                min_depth_m=self._min_depth, max_depth_m=self._max_depth)
            if depth_m is None:
                self._counts["skipped_depth"] += 1
                continue
            wx, wy, _wz = backproject_bbox_to_world(
                depth_bbox, depth_m, self._depth_k, rotation, translation,
                camera_offset_body=FRONT_CAMERA_OFFSET_FLU)
            self._landmarks.observe(cls, (wx, wy))
            self._counts["projected"] += 1

        signature = tuple((lm.id, lm.count, round(lm.xy[0], 3),
                           round(lm.xy[1], 3))
                          for lm in self._landmarks.confirmed())
        if signature != self._last_signature:
            self._dirty = True

    # ── publishing (on change, at most 1 Hz) ─────────────────────────
    def _flush(self):
        if self._dirty:
            self._publish()

    def _publish(self):
        confirmed = self._landmarks.confirmed()
        payload = objects_payload(
            self.get_clock().now().nanoseconds * 1e-9, confirmed)
        self._pub_objects.publish(String(data=json.dumps(payload)))
        self._last_signature = tuple(
            (lm.id, lm.count, round(lm.xy[0], 3), round(lm.xy[1], 3))
            for lm in confirmed)
        self._dirty = False

    def _heartbeat(self):
        missing = []
        if not self._depths:
            missing.append("depth")
        if not self._odoms:
            missing.append("odom")
        if self._rgb_k is None:
            missing.append("rgb_info")
        if self._depth_k is None:
            missing.append("depth_info")
        c = self._counts
        self.get_logger().info(
            "hb dets=%d projected=%d dropped_gap=%d "
            "skip(conf=%d class=%d depth=%d) no_intr=%d bad=%d | "
            "landmarks=%d confirmed=%d %s" % (
                c["dets"], c["projected"], c["dropped_gap"],
                c["skipped_conf"], c["skipped_class"], c["skipped_depth"],
                c["no_intrinsics"], c["bad_msgs"], len(self._landmarks),
                len(self._landmarks.confirmed()),
                ("MISSING[%s]" % ",".join(missing)) if missing else "OK"))
        self._counts = dict.fromkeys(self._counts, 0)


def main():
    rclpy.init()
    node = ObjectMapperNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # SIGINT and SIGTERM. stop_scene_graph.sh sends SIGTERM,
        # which rclpy turns into ExternalShutdownException out of
        # spin() -- uncaught it printed a traceback on every clean
        # teardown and exited non-zero, so a normal stop read as a
        # crash in the node log.
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
