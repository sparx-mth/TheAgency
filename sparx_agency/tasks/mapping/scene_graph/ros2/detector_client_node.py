"""detector_client_node — RGB frames -> detection HTTP server -> JSON topic.

Thin bridge across the split-runtime boundary: the open-vocabulary detector
(YOLO-World, torch, GPU) runs in a conda-side HTTP server
(``serve/detection_server.py``); this node runs in the ROS2 ``.venv`` (no
torch, no cv_bridge) and only moves bytes.

Subscribes
----------
``/simple_drone/front/image_raw``  (sensor_msgs/Image, best-effort)
    RGB frames from the sim; throttled to ``min_period_s`` before posting.

Publishes
---------
``/perception/detections``  (std_msgs/String JSON, reliable volatile)
    ``{"stamp": <SOURCE RGB header float sec>, "w", "h", "ms",
    "detections": [{"cls", "conf", "xyxy"}]}``
``/perception/detections/debug_image``  (sensor_msgs/Image bgr8, optional)
    The posted frame with green boxes + labels drawn.

Failure policy: a dead/unreachable server produces a throttled warning
(~10 s) and the node keeps trying — liveness retry, not a silent
fallback. A malformed server reply raises inside the callback and is
logged as an error. At startup the node polls ``GET /health`` until the
server answers and logs its model + vocabulary once.

Run:
    .venv/bin/python -m sparx_agency.tasks.mapping.scene_graph.ros2.\
detector_client_node --ros-args -p use_sim_time:=true
"""
from __future__ import annotations

import json

import numpy as np
import requests
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import Image
from std_msgs.msg import String

from sparx_agency.robots.SJTU.adapters.topics import FRONT_IMAGE
from sparx_agency.tasks.mapping.scene_graph.ros2.detection_payload import (
    build_detections_payload,
    draw_detections,
)
from sparx_agency.tasks.mapping.scene_graph.serve.contract import encode_frame

DETECTIONS_TOPIC = "/perception/detections"
DEBUG_IMAGE_TOPIC = "/perception/detections/debug_image"


def _image_msg_to_bgr(msg: Image) -> np.ndarray:
    """A colour ``sensor_msgs/Image`` as an HxWx3 uint8 BGR array.

    Decoded with ``np.frombuffer`` (no cv_bridge on this host, per the
    sjtu_internvla_n1 pattern). ``step`` is the row stride in BYTES and is
    not always ``width*3`` — a padded row read as if tight shears the image.
    Deliberate near-duplicate of ``n1_run_recorder_node._imgmsg_to_bgr``:
    that copy prefers cv_bridge and silently tolerates odd encodings; this
    one must be venv-pure and raise on anything but rgb8/bgr8 because a
    channel-swapped frame quietly degrades the detector.

    Raises:
        ValueError: Encoding is not ``rgb8``/``bgr8``.
    """
    enc = (msg.encoding or "").lower()
    if enc not in ("rgb8", "bgr8"):
        raise ValueError("unsupported colour encoding %r (want rgb8/bgr8)"
                         % (msg.encoding,))
    stride = msg.step if msg.step else msg.width * 3
    rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, stride)
    img = rows[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
    if enc == "rgb8":
        img = img[:, :, ::-1]
    return np.ascontiguousarray(img)


class DetectorClientNode(Node):
    """Posts throttled RGB frames to the detection server, relays the JSON."""

    def __init__(self) -> None:
        super().__init__("detector_client")

        # From robots/SJTU/adapters/topics.py, never spelled here: that module
        # is pinned against the plugin by the SJTU tests, so a rename moves
        # every consumer at once rather than leaving this node subscribed to a
        # name nothing publishes (which fails as "no frames", not as an error).
        self.declare_parameter("rgb_topic", FRONT_IMAGE)
        self.declare_parameter("server_url", "http://127.0.0.1:8092")
        self.declare_parameter("timeout_s", 5.0)
        self.declare_parameter("min_period_s", 1.0)
        self.declare_parameter("publish_debug", True)

        gp = self.get_parameter
        self._rgb_topic = str(gp("rgb_topic").value)
        self._server_url = str(gp("server_url").value).rstrip("/")
        self._timeout_s = float(gp("timeout_s").value)
        self._min_period_s = float(gp("min_period_s").value)
        self._publish_debug = bool(gp("publish_debug").value)

        self._sess = requests.Session()
        self._last_post_t = None  # rclpy.time.Time of the last accepted frame
        self._health_ok = False
        self._n = dict(frames_in=0, posted=0, published=0, conn_errors=0,
                       bad_replies=0)
        self._last_ms = 0.0

        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                durability=DurabilityPolicy.VOLATILE,
                                history=HistoryPolicy.KEEP_LAST, depth=1)
        det_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.VOLATILE,
                             history=HistoryPolicy.KEEP_LAST, depth=5)

        self._pub_det = self.create_publisher(String, DETECTIONS_TOPIC,
                                              det_qos)
        self._pub_dbg = (self.create_publisher(Image, DEBUG_IMAGE_TOPIC, 2)
                         if self._publish_debug else None)
        self.create_subscription(Image, self._rgb_topic, self._rgb_cb,
                                 sensor_qos)

        # Startup health poll: retries until the server answers, then stops.
        self._health_timer = self.create_timer(2.0, self._health_tick)
        self.create_timer(10.0, self._heartbeat)

        self.get_logger().info(
            "detector_client params  rgb_topic=%s  server_url=%s  "
            "timeout_s=%.1f  min_period_s=%.2f  publish_debug=%s"
            % (self._rgb_topic, self._server_url, self._timeout_s,
               self._min_period_s, self._publish_debug))

    # -- startup health check ------------------------------------------
    def _health_tick(self) -> None:
        try:
            r = self._sess.get(self._server_url + "/health", timeout=3.0)
            r.raise_for_status()
            info = r.json()
        except (requests.RequestException, ValueError) as exc:
            self.get_logger().warning(
                "detection server %s not answering /health yet: %s"
                % (self._server_url, exc), throttle_duration_sec=10.0)
            return
        classes = info.get("classes", [])
        self.get_logger().info(
            "detection server up  model=%s  device=%s  classes(%d)=%s"
            % (info.get("model"), info.get("device"), len(classes),
               ", ".join(str(c) for c in classes)))
        self._health_ok = True
        self._health_timer.cancel()

    # -- frame path ----------------------------------------------------
    def _rgb_cb(self, msg: Image) -> None:
        self._n["frames_in"] += 1
        now = self.get_clock().now()
        if (self._last_post_t is not None
                and (now - self._last_post_t).nanoseconds * 1e-9
                < self._min_period_s):
            return
        self._last_post_t = now

        try:
            bgr = _image_msg_to_bgr(msg)
        except ValueError as exc:
            self.get_logger().error("bad RGB frame: %s" % exc,
                                    throttle_duration_sec=10.0)
            return

        jpeg = encode_frame(bgr)
        try:
            self._n["posted"] += 1
            r = self._sess.post(self._server_url + "/detect", data=jpeg,
                                headers={"Content-Type": "image/jpeg"},
                                timeout=self._timeout_s)
            r.raise_for_status()
            reply = r.json()
        except (requests.RequestException, ValueError) as exc:
            self._n["conn_errors"] += 1
            self.get_logger().warning(
                "POST /detect to %s failed: %s" % (self._server_url, exc),
                throttle_duration_sec=10.0)
            return

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        try:
            payload = build_detections_payload(stamp, reply)
        except ValueError as exc:
            self._n["bad_replies"] += 1
            self.get_logger().error("malformed /detect reply: %s" % exc,
                                    throttle_duration_sec=10.0)
            return

        self._pub_det.publish(String(data=json.dumps(payload)))
        self._n["published"] += 1
        self._last_ms = float(payload["ms"])

        if self._pub_dbg is not None:
            self._publish_debug_image(bgr, payload["detections"], msg)

    def _publish_debug_image(self, bgr: np.ndarray, detections,
                             src: Image) -> None:
        """Publish the overlay frame.

        Named ``_publish_debug_image``, not ``_publish_debug``: the latter is
        already the *bool parameter* attribute set in ``__init__``, and an
        instance attribute shadows a same-named method, so the call site
        raised ``TypeError: 'bool' object is not callable`` on the first
        frame that reached it and killed the node.
        """
        overlay = draw_detections(bgr, detections)
        out = Image()
        out.header = src.header
        out.height, out.width = overlay.shape[:2]
        out.encoding = "bgr8"
        out.is_bigendian = 0
        out.step = overlay.shape[1] * 3
        out.data = overlay.tobytes()
        self._pub_dbg.publish(out)

    # -- heartbeat -----------------------------------------------------
    def _heartbeat(self) -> None:
        self.get_logger().info(
            "detector_client hb  server=%s  frames_in=%d posted=%d "
            "published=%d conn_errors=%d bad_replies=%d last_ms=%.0f"
            % ("up" if self._health_ok else "WAITING",
               self._n["frames_in"], self._n["posted"], self._n["published"],
               self._n["conn_errors"], self._n["bad_replies"], self._last_ms))
        self._n = dict(frames_in=0, posted=0, published=0, conn_errors=0,
                       bad_replies=0)


def main() -> None:
    rclpy.init()
    node = DetectorClientNode()
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
