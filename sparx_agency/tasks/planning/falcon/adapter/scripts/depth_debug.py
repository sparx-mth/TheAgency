#!/usr/bin/env python3
"""
depth_debug.py — minimal depth image debug viewer (ROS 1 / rospy).

ONE OpenCV window showing the raw depth image. No NavDP, no projection,
no body-frame conversion, no filtering, no clipping. Just the depth
image as it arrives from the depth model.

  • MOVE the mouse  → show pixel coords + depth value at that pixel
  • LEFT-CLICK       → freeze a marker on the clicked pixel and print
                       its depth to the terminal
  • r                → clear the click marker
  • p                → print the locked pixel's depth with full precision
  • q / ESC          → quit

The image you SEE is colorized only because OpenCV can't display a
float32 depth map directly. The VALUES used for the readout are the
raw, unmodified depth values from the topic — exactly what the model
produced. No clipping, no median, no patch averaging.

Run:
    rosrun <your_pkg> depth_debug.py
or:
    python3 depth_debug.py

Override the topic if you need to:
    _depth_topic:=/your/depth/topic
"""

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image


# ── shared mouse state (mouse_cb writes, main loop reads) ──────────
hover_px = None        # (x, y) under the cursor
click_px = None        # (x, y) of last left-click


def mouse_cb(event, x, y, flags, param):
    global hover_px, click_px
    if event == cv2.EVENT_MOUSEMOVE:
        hover_px = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        click_px = (x, y)


class DepthDebug:
    def __init__(self):
        rospy.init_node("depth_debug")
        self.depth_topic = rospy.get_param("~depth_topic", "/xtend/depth_m")
        self.depth = None
        self.encoding = None
        rospy.Subscriber(self.depth_topic, Image,
                         self._depth_cb, queue_size=5)
        rospy.loginfo("depth_debug listening on %s", self.depth_topic)

    def _depth_cb(self, msg):
        # Keep the depth EXACTLY as it arrives. Only handle the two
        # common encodings so we end up with metres in a float array;
        # no clipping, no filtering, no smoothing.
        self.encoding = msg.encoding
        if msg.encoding == "32FC1":
            self.depth = np.frombuffer(
                msg.data, np.float32).reshape(msg.height, msg.width).copy()
        elif msg.encoding == "16UC1":
            # 16UC1 is typically millimetres → metres. If your sensor
            # uses a different scale, change the divisor here.
            self.depth = np.frombuffer(
                msg.data, np.uint16).reshape(
                    msg.height, msg.width).astype(np.float32) / 1000.0
        else:
            rospy.logwarn_throttle(
                5.0, "Unhandled depth encoding: %s", msg.encoding)


def colorize_for_display(depth):
    """Colorize ONLY so we can show it on screen. The values used for
    the readouts come from the original `depth` array, not from this."""
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return np.zeros((*depth.shape, 3), np.uint8), 0.0, 0.0
    dmin = float(depth[valid].min())
    dmax = float(depth[valid].max())
    if dmax - dmin < 1e-6:
        dmax = dmin + 1e-6
    norm = np.zeros_like(depth, np.float32)
    norm[valid] = (depth[valid] - dmin) / (dmax - dmin)
    vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[~valid] = 0
    return vis, dmin, dmax


def main():
    node = DepthDebug()

    rospy.loginfo("Waiting for first depth frame on %s ...", node.depth_topic)
    while not rospy.is_shutdown() and node.depth is None:
        rospy.sleep(0.05)
    if rospy.is_shutdown():
        return

    h, w = node.depth.shape
    rospy.loginfo("Depth: %dx%d  encoding=%s", w, h, node.encoding)

    cv2.namedWindow("depth_debug", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("depth_debug", mouse_cb)

    global click_px
    while not rospy.is_shutdown():
        depth = node.depth
        if depth is None:
            rospy.sleep(0.01); continue

        vis, dmin, dmax = colorize_for_display(depth)

        # Hover crosshair + readout for the pixel under the mouse.
        status = "encoding=%s  size=%dx%d  range=[%.3f, %.3f] m" % (
            node.encoding, depth.shape[1], depth.shape[0], dmin, dmax)

        if hover_px is not None:
            hx, hy = hover_px
            if 0 <= hx < depth.shape[1] and 0 <= hy < depth.shape[0]:
                z = float(depth[hy, hx])           # raw value, untouched
                cv2.drawMarker(vis, (hx, hy), (255, 255, 255),
                               cv2.MARKER_CROSS, 16, 1)
                status = "pixel=(%d, %d)   depth=%.4f m   [raw]" % (hx, hy, z)

        # Persistent click marker + readout (yellow).
        if click_px is not None:
            cx_, cy_ = click_px
            if 0 <= cx_ < depth.shape[1] and 0 <= cy_ < depth.shape[0]:
                cz = float(depth[cy_, cx_])        # raw value, untouched
                cv2.circle(vis, (cx_, cy_), 8, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(vis, (cx_, cy_), 2, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(vis,
                            "(%d,%d) = %.4f m" % (cx_, cy_, cz),
                            (cx_ + 12, cy_ - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 1, cv2.LINE_AA)

        # Status bar at the bottom.
        bar = np.zeros((28, vis.shape[1], 3), np.uint8)
        cv2.putText(bar, status, (8, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (220, 220, 220), 1, cv2.LINE_AA)
        cv2.imshow("depth_debug", np.vstack([vis, bar]))

        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('r'):
            click_px = None
        elif key == ord('p') and click_px is not None:
            cx_, cy_ = click_px
            if 0 <= cx_ < depth.shape[1] and 0 <= cy_ < depth.shape[0]:
                rospy.loginfo("click pixel=(%d, %d)  depth=%.6f m",
                              cx_, cy_, float(depth[cy_, cx_]))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass