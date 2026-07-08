#!/usr/bin/env python3
"""yolo_detector_node.py -- ROS1 adapter: RGB -> open-vocab YOLO -> detections JSON.

Runs the "OpenYOLO" (YOLO-World) open-vocabulary detector on the live RGB stream
and publishes detections for :mod:`object_approach_node`, which does the fast
tracking + visual approach. Kept a SEPARATE node on purpose: the detector pulls in
torch/ultralytics, while the servo/tracking node stays a lean, Python-3.8-safe
consumer. The two are decoupled by a plain ``std_msgs/String`` JSON topic, so this
node can also be replaced by any other detector (or a TRT build, or a sidecar
container à la the reference stack) that publishes the same message.

The target object is set by ``~target_topic`` (a ``std_msgs/String`` -- the mission
"goal", e.g. ``"refrigerator"``); publishing a new goal re-prompts the detector at
runtime with no reload. Open-vocabulary means the class list is just a prompt.

Detections JSON on ``~detections_topic``::

    {"stamp": <float sec>, "w": <int>, "h": <int>,
     "detections": [{"label": "refrigerator", "score": 0.83,
                     "bbox": [x1, y1, x2, y2]}, ...]}

The ``stamp`` is the source frame's stamp so the consumer can seed its tracker on
the matching buffered frame. RGB arrives as a frame-path ``String`` (real XTEND) or
a raw ``Image`` (sim/bag), exactly like navdp_click / combination.

NOTE: needs ``ultralytics`` in the runtime. The algorithm is ROS-free in
``core.mapping.detection`` (``YoloWorldDetector``); this node owns only ROS I/O.
See the file footer for the rosparam list.
"""
import json

import cv2
import numpy as np

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from sparx_agency.core.common.frame_path_message import parse_frame_path_message
from sparx_agency.core.mapping.detection import YoloWorldConfig, YoloWorldDetector


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
        self.detector = YoloWorldDetector(YoloWorldConfig(
            model_path=str(G("~model_path", "yolov8s-world.pt")),
            device=str(G("~device", "cuda:0")),
            conf_thresh=float(G("~conf_thresh", 0.25)),
            iou_thresh=float(G("~iou_thresh", 0.5)),
            imgsz=int(G("~imgsz", 640)),
        ))
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
        payload = {
            "stamp": float(stamp),
            "w": int(w), "h": int(h),
            "detections": [
                {"label": d.label, "score": float(d.score),
                 "bbox": [int(v) for v in d.bbox_xyxy]}
                for d in dets
            ],
        }
        self.pub.publish(String(data=json.dumps(payload)))

    def _banner(self):
        L = rospy.loginfo
        L("=" * 64)
        L("yolo_detector (open-vocab YOLO-World -> detections JSON)")
        L("  rgb  in  = %s  (%s)", self.rgb_topic, self.image_transport)
        L("  goal in  = %s", self.target_topic)
        L("  dets out = %s  @ %.1f Hz", self.detections_topic, self.detect_hz)
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
# in core.mapping.detection.YoloWorldDetector; this node owns ROS I/O only.
#
#   IO: ~image_transport (frame_path | topic)
#       frame_path -> ~rgb_topic (/xtend/rgb_frame_path)  std_msgs/String
#       topic      -> ~rgb_topic (/xtend/rgb)             sensor_msgs/Image
#       ~target_topic (/object_approach/goal)  std_msgs/String  -- the mission goal
#       ~detections_topic (/object_approach/detections)  std_msgs/String JSON
#   target: ~target_object (refrigerator)  ~extra_prompts ([])
#   model: ~model_path (yolov8s-world.pt) ~device (cuda:0) ~conf_thresh (0.25)
#       ~iou_thresh (0.5) ~imgsz (640) ~detect_hz (2.0)
#
# Requires ultralytics in the runtime. If Noetic lacks it, run this node in a
# sidecar and just bridge ~detections_topic to object_approach_node.
# ============================================================================
