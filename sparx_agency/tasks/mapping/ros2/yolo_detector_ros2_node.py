#!/usr/bin/env python3
"""yolo_detector_ros2_node.py -- ROS2 sidecar: RGB -> TensorRT YOLO-World -> detections JSON.

The GPU half of the "lock onto a named object and fly to it" mission, run as a
**host-side ROS2 node** rather than inside the FALCON ROS1 container.

Why a sidecar. The FALCON image is built ``FROM ros:noetic-perception`` -- a stock
Ubuntu 20.04 ROS image with no CUDA, no TensorRT and no ``pycuda``. ``--runtime
nvidia`` bind-mounts JetPack's shared libraries into the container but NOT the
Python bindings, so ``import pycuda`` fails there. The Orin *host*, meanwhile,
already has the working ``tensorrt`` + ``pycuda`` environment that built the engines
and that ``tasks/planning/object_approach_offline`` was validated in. This node is
the detector, running there.

Nothing else has to move. The detector and the tracker/servo were always decoupled
by a plain ``std_msgs/String`` JSON topic (see
:mod:`sparx_agency.core.common.detection_message`), so only that topic -- and the
target topic -- cross the ROS1<->ROS2 bridge. No image is ever bridged: on the real
drone RGB arrives as a *frame path*, and this node reads the JPEG straight off the
host's disk, upstream of the bridge.

  ROS2 (host, GPU)                         bridge          ROS1 (falcon container)
  /xtend/rgb_frame_path ─► this node ─► /object_approach/detections ─► object_approach
                                   ◄─── /object_approach/goal ◄─── operator / mission

The target object is set by ``target_topic`` (a ``std_msgs/String``) or the
``target_object`` parameter; publishing a new goal re-prompts the detector at
runtime with no engine rebuild (a fresh CLIP text-embed). Open-vocabulary means the
class list is just a prompt. Inference is torch-free; only ``set_prompts`` touches
torch, and only when the target changes.

NOTE: the TRT engines are NOT portable -- build them on-target (see the
``yolo_world_trt`` README) and point ``backbone_engine`` / ``head_engine`` at them.

Run (on the Orin host, in the env that has tensorrt + pycuda)::

    source /opt/ros/humble/setup.bash
    PYTHONPATH=/path/to/repo python3 yolo_detector_ros2_node.py --ros-args \\
        -p target_object:=monitor \\
        -p backbone_engine:=/path/backbone.engine \\
        -p head_engine:=/path/head.engine \\
        -p text_weights:=/path/yolov8s-worldv2.pt
"""
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import String

from sparx_agency.core.common.detection_message import encode_detections
from sparx_agency.core.common.frame_path_message import (
    parse_frame_path_message,
    resolve_frame_path,
)
from sparx_agency.tasks.mapping.yolo_world_trt.runtime import YoloTRTDetector


def _sensor_qos():
    """Match the drone's high-rate frame-path publisher.

    BEST_EFFORT is not a preference -- a reliability mismatch means **no data
    flows** (the same trap documented in the bridge's ``bridge.yaml``).
    """
    return QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                      reliability=ReliabilityPolicy.BEST_EFFORT,
                      durability=DurabilityPolicy.VOLATILE)


def _latched_qos(depth=10):
    """TRANSIENT_LOCAL so a late-joining subscriber still sees the last target."""
    return QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=depth,
                      reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL)


class YoloDetectorRos2Node(Node):
    def __init__(self):
        super().__init__("yolo_detector")

        self.declare_parameter("image_transport", "frame_path")  # frame_path | topic
        self.declare_parameter("rgb_topic", "/xtend/rgb_frame_path")
        self.declare_parameter("target_topic", "/object_approach/goal")
        self.declare_parameter("detections_topic", "/object_approach/detections")
        self.declare_parameter("frames_dir", "")     # replay override; '' = live paths
        self.declare_parameter("target_object", "refrigerator")
        self.declare_parameter("extra_prompts", [""])
        self.declare_parameter("backbone_engine", "")
        self.declare_parameter("head_engine", "")
        self.declare_parameter("text_weights", "")
        self.declare_parameter("text_device", "cpu")
        self.declare_parameter("conf_thresh", -1.0)  # <=0 => engine manifest
        self.declare_parameter("iou_thresh", -1.0)
        self.declare_parameter("max_det", 0)         # <=0 => engine manifest
        self.declare_parameter("detect_hz", 2.0)

        P = lambda n: self.get_parameter(n).value    # noqa: E731 -- terse param read

        self.image_transport = str(P("image_transport")).strip().lower()
        if self.image_transport not in ("frame_path", "topic"):
            raise ValueError("image_transport must be 'frame_path' or 'topic', got %r"
                             % self.image_transport)
        self.rgb_topic = str(P("rgb_topic"))
        self.detections_topic = str(P("detections_topic"))
        self.frames_dir = str(P("frames_dir"))
        self.detect_hz = float(P("detect_hz"))

        target = str(P("target_object")).strip().lower()
        extra = [str(s).strip().lower() for s in (P("extra_prompts") or []) if str(s).strip()]
        self._prompts = [target] + [p for p in extra if p != target]

        self.detector = YoloTRTDetector(
            backbone_engine=str(P("backbone_engine")),
            head_engine=str(P("head_engine")),
            text_weights=str(P("text_weights")) or None,
            text_device=str(P("text_device")),
            conf_thresh=_opt_float(P("conf_thresh")),
            iou_thresh=_opt_float(P("iou_thresh")),
            max_det=(int(P("max_det")) if int(P("max_det")) > 0 else None),
        )
        self.detector.set_prompts(self._prompts)

        self.rgb = None            # HxWx3 RGB
        self.rgb_stamp = 0.0

        # Publisher before subscribers (a late consumer still gets the next frame).
        self.pub = self.create_publisher(String, self.detections_topic, 5)
        if self.image_transport == "frame_path":
            self.create_subscription(String, self.rgb_topic, self._rgb_path_cb,
                                     _sensor_qos())
        else:
            self.create_subscription(Image, self.rgb_topic, self._rgb_cb, _sensor_qos())
        self.create_subscription(String, str(P("target_topic")), self._target_cb,
                                 _latched_qos())
        self.create_timer(1.0 / max(self.detect_hz, 0.1), self._tick)

        self._banner()

    # ─── Sensor callbacks ────────────────────────────────────────────
    def _rgb_path_cb(self, msg):
        try:
            parsed = parse_frame_path_message(msg.data)
            path = resolve_frame_path(parsed.path, self.frames_dir)
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise OSError("cv2.imread returned None for %s" % path)
        except (ValueError, OSError) as e:
            self._warn("dropping RGB frame-path (%s)" % e)
            return
        self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.rgb_stamp = parsed.stamp_seconds

    def _rgb_cb(self, msg):
        arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        if msg.encoding == "bgr8":
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        self.rgb = arr.copy()
        self.rgb_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _target_cb(self, msg):
        target = str(msg.data).strip().lower()
        if not target or target == self._prompts[0]:
            return
        self.get_logger().info("goal %r -> %r (re-prompting)" % (self._prompts[0], target))
        self._prompts = [target] + [p for p in self._prompts[1:] if p != target]
        self.detector.set_prompts(self._prompts)

    # ─── Detection loop ──────────────────────────────────────────────
    def _tick(self):
        rgb, stamp = self.rgb, self.rgb_stamp
        if rgb is None:
            return
        try:
            dets = self.detector.detect(rgb)
        except Exception as e:                     # noqa: BLE001 -- keep the timer alive
            self._warn("detect error (%s: %s)" % (type(e).__name__, e))
            return
        h, w = rgb.shape[:2]
        self.pub.publish(String(data=encode_detections(dets, stamp, w, h)))

    def _warn(self, text):
        self.get_logger().warn(text, throttle_duration_sec=5.0)

    def _banner(self):
        L = self.get_logger().info
        L("=" * 64)
        L("yolo_detector (ROS2 sidecar: TensorRT YOLO-World -> detections JSON)")
        L("  rgb  in  = %s  (%s)" % (self.rgb_topic, self.image_transport))
        L("  dets out = %s  @ %.1f Hz" % (self.detections_topic, self.detect_hz))
        L("  conf=%.2f iou=%.2f max_det=%d" % (self.detector.conf_thresh,
                                               self.detector.iou_thresh,
                                               self.detector.max_det))
        L("  prompts  = %s" % (self._prompts,))
        L("  bridge /object_approach/detections into ROS1 for object_approach_node")
        L("=" * 64)


def _opt_float(value):
    """A non-positive value means "use the engine manifest default" (params carry no None)."""
    v = float(value)
    return v if v > 0.0 else None


def main():
    rclpy.init()
    node = YoloDetectorRos2Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()


# ============================================================================
# PARAMETERS (ROS2 --ros-args -p name:=value; defaults in parentheses). Detection
# maths is ROS-free in tasks.mapping.yolo_world_trt.runtime.YoloTRTDetector and the
# wire format in core.common.detection_message; this node owns ROS2 I/O only.
#
#   IO: image_transport (frame_path | topic)
#       frame_path -> rgb_topic (/xtend/rgb_frame_path)  std_msgs/String, BEST_EFFORT
#       topic      -> rgb_topic (/xtend/rgb)             sensor_msgs/Image, BEST_EFFORT
#       target_topic (/object_approach/goal)  std_msgs/String, TRANSIENT_LOCAL
#       detections_topic (/object_approach/detections)  std_msgs/String JSON
#       frames_dir ('')  resolve frame paths by basename here (offline replay)
#   target: target_object (refrigerator)  extra_prompts ([])
#   engines (TRT, built on-target): backbone_engine ('')  head_engine ('')
#       text_weights ('' -> torch-free; must be set to re-prompt at runtime)
#       text_device (cpu)  conf_thresh (<=0 -> manifest)  iou_thresh (<=0 -> manifest)
#       max_det (<=0 -> manifest)  detect_hz (2.0)
#
# Bridge /object_approach/detections (ROS2 -> ROS1) and /object_approach/goal so the
# ROS1 object_approach_node sees them; see falcon/bridge/bridge.yaml.
# ============================================================================
