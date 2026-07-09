#!/usr/bin/env python3
"""target_lock_viewer_node.py -- ROS1 adapter: show the object-approach HUD window.

A tiny, dependency-light on-screen viewer for the live target-lock overlay that
:mod:`object_approach_node` renders and publishes (``~image_topic``, default
``/object_approach/overlay``, ``sensor_msgs/Image`` ``bgr8``). It just decodes the
Image and ``cv2.imshow``-s it -- the exact HUD of the offline
``run_live_target_lock`` tool, but driven by the *real* mission's detections,
tracked box and shaped command instead of a second detector.

Kept a SEPARATE node (not folded into object_approach) so the GUI dependency and
the display loop stay off the control node, and so it can be turned off (headless
runs) or pointed at a remote ``image_transport`` republish without touching the
mission. Needs a display: run on the Jetson's own screen, or ``ssh -X`` / VNC in.
Press 'q' in the window (or Ctrl+C) to close it; the mission keeps running.

No ``cv_bridge`` dependency -- the Image is unpacked with numpy directly.
"""
import cv2
import numpy as np

import rospy
from sensor_msgs.msg import Image

_WINDOW = "object_approach -- target lock (live)"


class TargetLockViewerNode(object):
    def __init__(self):
        rospy.init_node("target_lock_viewer")
        self.image_topic = rospy.get_param("~image_topic", "/object_approach/overlay")
        self.show_hz = float(rospy.get_param("~show_hz", 20.0))
        self._latest = None
        rospy.Subscriber(self.image_topic, Image, self._img_cb, queue_size=1)
        rospy.loginfo("target_lock_viewer: showing %s (press 'q' to close)",
                      self.image_topic)

    def _img_cb(self, msg):
        try:
            self._latest = self._to_bgr(msg)
        except ValueError as e:
            rospy.logwarn_throttle(5.0, "target_lock_viewer: bad Image (%s)", e)

    @staticmethod
    def _to_bgr(msg):
        """Decode an 8-bit 3-channel sensor_msgs/Image into a BGR numpy array."""
        enc = (msg.encoding or "").lower()
        if enc not in ("bgr8", "rgb8"):
            raise ValueError("unsupported encoding %r (want bgr8/rgb8)" % msg.encoding)
        arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        return arr if enc == "bgr8" else cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def spin(self):
        # imshow / waitKey must run on the main thread; the subscriber (a separate
        # rospy thread) only stashes the latest frame.
        cv2.namedWindow(_WINDOW, cv2.WINDOW_AUTOSIZE)
        rate = rospy.Rate(max(self.show_hz, 1.0))
        try:
            while not rospy.is_shutdown():
                if self._latest is not None:
                    cv2.imshow(_WINDOW, self._latest)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
                try:
                    rate.sleep()
                except rospy.ROSInterruptException:
                    break
        finally:
            cv2.destroyAllWindows()


def main():
    try:
        TargetLockViewerNode().spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()


# ============================================================================
# ROSPARAMS (all private ~; defaults in parentheses):
#   ~image_topic (/object_approach/overlay)  sensor_msgs/Image bgr8|rgb8
#   ~show_hz (20.0)  window refresh rate
# ============================================================================
