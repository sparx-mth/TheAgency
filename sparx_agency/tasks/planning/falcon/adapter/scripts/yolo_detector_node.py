#!/usr/bin/env python3
"""yolo_detector_node.py -- ROS1 adapter: RGB -> TensorRT YOLO-World -> detections JSON.

Runs the open-vocabulary YOLO-World detector on the live RGB stream and publishes
detections for :mod:`object_approach_node`, which does the fast tracking + visual
approach. This is the **TensorRT** build (:class:`YoloTRTDetector`, the
backbone-on-DLA + head-on-GPU split from ``tasks/mapping/yolo_world_trt``) -- the
same detector validated offline in ``tasks/planning/object_approach_offline`` -- not
the stock ultralytics ``.pt`` path. Inference itself is torch-free; only setting the
prompts (the CLIP text branch) touches torch, and only when the target changes.

Kept a SEPARATE node from the servo/tracker on purpose (they are decoupled by a
plain ``std_msgs/String`` JSON topic), so the heavy TRT/torch deps stay off the
lean, Python-3.8-safe ``object_approach_node``.

The target object is set by ``~target_topic`` (a ``std_msgs/String`` -- the mission
"goal", e.g. ``"refrigerator"``); publishing a new goal re-prompts the detector at
runtime with no engine rebuild (a fresh CLIP text-embed). Open-vocabulary means the
class list is just a prompt.

Detections JSON on ``~detections_topic``::

    {"stamp": <float sec>, "w": <int>, "h": <int>,
     "detections": [{"label": "refrigerator", "score": 0.83,
                     "bbox": [x1, y1, x2, y2]}, ...]}

The ``stamp`` is the source frame's stamp so the consumer can seed its tracker on
the matching buffered frame. RGB arrives as a frame-path ``String`` (real XTEND) or
a raw ``Image`` (sim/bag), exactly like navdp_click / combination.

NOTE: the TRT engines are NOT portable -- build them on-target (see the
``yolo_world_trt`` README) and point ``~backbone_engine`` / ``~head_engine`` at
them. ``~text_weights`` (the ``.pt`` YOLO-World checkpoint) drives the text branch
and requires torch/ultralytics at prompt-set time only. See the file footer for the
rosparam list.
"""
import cv2
import numpy as np

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from sparx_agency.core.common.detection_message import encode_detections
from sparx_agency.core.common.frame_path_message import parse_frame_path_message
from sparx_agency.tasks.mapping.yolo_world_trt.runtime import YoloTRTDetector


def _opt_float(name, default_none=True):
    """Read an optional float rosparam: a non-positive value means "use the engine
    manifest default" (rosparams cannot carry ``None``)."""
    v = float(rospy.get_param(name, -1.0))
    return v if v > 0.0 else (None if default_none else v)


class YoloDetectorNode(object):
    def __init__(self):
        rospy.init_node("yolo_detector")
        G = rospy.get_param

        self.image_transport = str(G("~image_transport", "frame_path")).strip().lower()
        if self.image_transport not in ("frame_path", "topic"):
            raise ValueError("~image_transport must be 'frame_path' or 'topic', "
                             "got %r" % self.image_transport)
        _fp = self.image_transport == "frame_path"
        self.rgb_topic = G("~rgb_topic",
                           "/xtend/rgb_frame_path" if _fp else "/xtend/rgb")
        self.target_topic = G("~target_topic", "/object_approach/goal")
        self.detections_topic = G("~detections_topic", "/object_approach/detections")

        target = str(G("~target_object", "refrigerator")).strip().lower()
        # Optional extra prompts give YOLO-World contrastive context; the target is
        # always included. The consumer's confirmation gate filters by label anyway.
        extra = [str(s).strip().lower() for s in G("~extra_prompts", []) if str(s).strip()]
        self._prompts = [target] + [p for p in extra if p != target]

        self.detect_hz = float(G("~detect_hz", 2.0))   # detector rate; tracker is at camera rate

        # TensorRT YOLO-World: backbone (DLA) + head (GPU) engine split, built
        # on-target. conf/iou/max_det default to the head engine's manifest when a
        # non-positive rosparam is given.
        self.detector = YoloTRTDetector(
            backbone_engine=str(G("~backbone_engine", "")),
            head_engine=str(G("~head_engine", "")),
            text_weights=str(G("~text_weights", "")) or None,
            text_device=str(G("~text_device", "cpu")),
            conf_thresh=_opt_float("~conf_thresh"),
            iou_thresh=_opt_float("~iou_thresh"),
            max_det=(lambda n: int(n) if int(n) > 0 else None)(G("~max_det", 0)),
        )
        self.detector.set_prompts(self._prompts)

        self.rgb = None          # HxWx3 RGB
        self.rgb_stamp = 0.0

        # Publisher before subscribers (a late consumer still gets the next frame).
        self.pub = rospy.Publisher(self.detections_topic, String, queue_size=5)
        if _fp:
            rospy.Subscriber(self.rgb_topic, String, self._rgb_path_cb, queue_size=2)
        else:
            rospy.Subscriber(self.rgb_topic, Image, self._rgb_cb, queue_size=2)
        rospy.Subscriber(self.target_topic, String, self._target_cb, queue_size=1)

        self._banner()

    # ─── Sensor callbacks ────────────────────────────────────────────
    def _rgb_path_cb(self, msg):
        try:
            parsed = parse_frame_path_message(msg.data)
            bgr = cv2.imread(parsed.path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise OSError("cv2.imread returned None for %s" % parsed.path)
        except (ValueError, OSError) as e:
            rospy.logwarn_throttle(5.0, "yolo_detector: dropping RGB frame-path (%s)", e)
            return
        self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.rgb_stamp = float(parsed.sec) + float(parsed.nsec) * 1e-9

    def _rgb_cb(self, msg):
        arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        if msg.encoding == "bgr8":
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        self.rgb = arr.copy()
        self.rgb_stamp = msg.header.stamp.to_sec()

    def _target_cb(self, msg):
        target = str(msg.data).strip().lower()
        if not target or target == self._prompts[0]:
            return
        rospy.loginfo("yolo_detector: goal %r -> %r (re-prompting)",
                      self._prompts[0], target)
        self._prompts = [target] + [p for p in self._prompts[1:] if p != target]
        self.detector.set_prompts(self._prompts)

    # ─── Detection loop ──────────────────────────────────────────────
    def start(self):
        rospy.Timer(rospy.Duration(1.0 / max(self.detect_hz, 0.1)), self._tick)
        rospy.spin()

    def _tick(self, _evt):
        rgb, stamp = self.rgb, self.rgb_stamp
        if rgb is None:
            return
        try:
            dets = self.detector.detect(rgb)
        except Exception as e:                    # noqa: BLE001 -- keep the timer alive
            rospy.logwarn_throttle(5.0, "yolo_detector: detect error (%s: %s)",
                                   type(e).__name__, e)
            return
        h, w = rgb.shape[:2]
        self.pub.publish(String(data=encode_detections(dets, stamp, w, h)))

    def _banner(self):
        L = rospy.loginfo
        L("=" * 64)
        L("yolo_detector (TensorRT YOLO-World -> detections JSON)")
        L("  rgb  in  = %s  (%s)", self.rgb_topic, self.image_transport)
        L("  goal in  = %s", self.target_topic)
        L("  dets out = %s  @ %.1f Hz", self.detections_topic, self.detect_hz)
        L("  conf=%.2f iou=%.2f max_det=%d",
          self.detector.conf_thresh, self.detector.iou_thresh, self.detector.max_det)
        L("  prompts  = %s", self._prompts)
        L("=" * 64)


def main():
    try:
        YoloDetectorNode().start()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()


# ============================================================================
# ROSPARAMS (all private ~; defaults in parentheses). Detection maths is ROS-free
# in tasks.mapping.yolo_world_trt.runtime.YoloTRTDetector; this node owns ROS I/O.
#
#   IO: ~image_transport (frame_path | topic)
#       frame_path -> ~rgb_topic (/xtend/rgb_frame_path)  std_msgs/String
#       topic      -> ~rgb_topic (/xtend/rgb)             sensor_msgs/Image
#       ~target_topic (/object_approach/goal)  std_msgs/String  -- the mission goal
#       ~detections_topic (/object_approach/detections)  std_msgs/String JSON
#   target: ~target_object (refrigerator)  ~extra_prompts ([])
#   engines (TRT, built on-target): ~backbone_engine ('')  ~head_engine ('')
#       ~text_weights ('' -> torch-free; must be set to re-prompt at runtime)
#       ~text_device (cpu)  ~conf_thresh (<=0 -> manifest)  ~iou_thresh (<=0 -> manifest)
#       ~max_det (<=0 -> manifest)  ~detect_hz (2.0)
#
# The engines are not portable: build on the target device (yolo_world_trt README).
# set_prompts needs torch/ultralytics via ~text_weights; precompute embeddings for a
# fully torch-free deployment (runtime.set_text_features).
# ============================================================================
