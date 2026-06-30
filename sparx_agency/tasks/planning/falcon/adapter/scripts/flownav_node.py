#!/usr/bin/env python3
"""flownav_node.py -- ROS1 adapter: RGB + goal image -> FlowNav -> world path.

The FlowNav sibling of ``navdp_click_node.py``. NavDP is point-goal (click a
pixel); FlowNav is **image-goal** -- it steers toward a target image. This node
therefore needs NO depth, NO camera intrinsics, and NO click UI: it just feeds
the live RGB frame + a goal image to the FlowNav policy and publishes the
returned trajectory as a world-frame ``nav_msgs/Path`` -- the SAME output
contract as NavDP, so ``path_corrector_node`` -> ``trajectory_simplifier_node``
-> ``waypoint_follower_node`` are unchanged.

Selected by the launch ``vla:=flownav`` arg (vs ``vla:=navdp``). It publishes its
RAW path on ``~path_topic`` (``/path/waypoints_flownav``); the corrector recentres
it and republishes ``/path/waypoints``.

The heavy TensorRT inference runs in a HOST process
(``tasks/planning/flownav/server/flownav_trt_server.py``) reached over loopback
HTTP, because the FALCON Noetic container has no TensorRT -- exactly like NavDP.
This node owns ONLY ROS concerns: subscriptions, the goal image, world anchoring,
and publishing.

All the maths is ROS-free and reused from ``core``:
  * FlowNav HTTP request/response   (flownav.client.FlowNavImageGoalClient)
  * body trajectory -> world path   (navdp.geometry.anchor_trajectory_to_world)
  * frame-path message parsing      (common.frame_path_message)

Goal image (pick one):
  * ``~goal_image`` -- a file path loaded once at startup (the simple default), or
  * ``~goal_image_topic`` -- a ``sensor_msgs/Image`` updated live.

Unlike NavDP's click-to-go, this runs CONTINUOUSLY at ``~rate_hz`` (reactive local
planning toward the goal image); each tick republishes a fresh latched path.

Run:
    rosrun falcon_adapter flownav_node.py
See the file footer for the full rosparam list.
"""
import cv2
import numpy as np

import rospy
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import Image
from std_msgs.msg import String

from sparx_agency.core.common.frame_path_message import parse_frame_path_message
from sparx_agency.core.common.math import se3
from sparx_agency.core.planning.flownav.client import (
    FlowNavClientError, FlowNavImageGoalClient,
)
from sparx_agency.core.planning.navdp import anchor_trajectory_to_world


class FlowNavNode:
    def __init__(self):
        G = rospy.get_param

        self.image_transport = str(G("~image_transport", "frame_path")).strip().lower()
        _fp = self.image_transport == "frame_path"
        self.rgb_topic = G("~rgb_topic", "/xtend/rgb_frame_path" if _fp else "/xtend/rgb")
        self.pose_topic = G("~pose_topic", "/xtend/localization")
        self.pose_type = str(G("~pose_type", "pose_stamped")).strip().lower()

        self.path_topic = G("~path_topic", "/path/waypoints_flownav")
        self.full_path_topic = G("~full_path_topic", "/path/waypoints_flownav_full")
        self.frame_id = G("~frame_id", "world")
        self.execute_fraction = float(G("~execute_fraction", 1.0))
        self.rate_hz = float(G("~rate_hz", 4.0))

        self.goal_image_path = str(G("~goal_image", "")).strip()
        self.goal_image_topic = str(G("~goal_image_topic", "")).strip()

        port = int(G("~port", 8889))
        self.client = FlowNavImageGoalClient(
            "http://127.0.0.1:%d" % port,
            timeout_s=float(G("~timeout_s", 10.0)), logger=rospy.logwarn)

        self.rgb = None                 # HxWx3 uint8 RGB
        self.pose_xyyaw = None          # (x, y, yaw)
        self.goal_rgb = None            # HxWx3 uint8 RGB
        self.n_published = 0

        if self.goal_image_path:
            self._load_goal_file(self.goal_image_path)
        if not self.goal_image_path and not self.goal_image_topic:
            raise ValueError("flownav_node needs a goal: set ~goal_image (file) "
                             "or ~goal_image_topic")

        # ── subscribers ──────────────────────────────────────────────
        if _fp:
            rospy.Subscriber(self.rgb_topic, String, self._rgb_path_cb, queue_size=2)
        else:
            rospy.Subscriber(self.rgb_topic, Image, self._rgb_cb, queue_size=2)
        if self.pose_type == "pose_stamped":
            rospy.Subscriber(self.pose_topic, PoseStamped, self._pose_stamped_cb,
                             queue_size=10)
        elif self.pose_type == "pose":
            rospy.Subscriber(self.pose_topic, Pose, self._pose_cb, queue_size=10)
        else:
            raise ValueError("~pose_type must be 'pose' or 'pose_stamped', got %r"
                             % self.pose_type)
        if self.goal_image_topic:
            rospy.Subscriber(self.goal_image_topic, Image, self._goal_cb, queue_size=1)

        # ── publishers (latched, same contract as navdp_click_node) ──
        self.pub_path = rospy.Publisher(self.path_topic, Path, queue_size=1, latch=True)
        self.pub_full = rospy.Publisher(self.full_path_topic, Path, queue_size=1, latch=True)

        self.client.reset()             # best-effort: clear the server frame buffer
        rospy.Timer(rospy.Duration(1.0 / max(0.1, self.rate_hz)), self._tick)
        rospy.loginfo("flownav_node up: %.1f Hz, goal=%s, -> %s",
                      self.rate_hz, self.goal_image_path or self.goal_image_topic,
                      self.path_topic)

    # ─── goal loading ────────────────────────────────────────────────
    def _load_goal_file(self, path):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("could not read ~goal_image %r" % path)
        self.goal_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _goal_cb(self, msg):
        arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        if msg.encoding == "bgr8":
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        self.goal_rgb = arr.copy()

    # ─── observation subscribers ─────────────────────────────────────
    def _rgb_path_cb(self, msg):
        try:
            path = parse_frame_path_message(msg.data).path
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("could not read %r" % path)
        except Exception as e:                       # noqa: BLE001
            rospy.logwarn_throttle(5.0, "flownav: bad RGB frame-path: %s", e)
            return
        self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _rgb_cb(self, msg):
        arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        if msg.encoding == "bgr8":
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        self.rgb = arr.copy()

    def _pose_cb(self, msg):
        yaw = se3.yaw_from_quaternion((msg.orientation.x, msg.orientation.y,
                                       msg.orientation.z, msg.orientation.w))
        self.pose_xyyaw = (float(msg.position.x), float(msg.position.y), yaw)

    def _pose_stamped_cb(self, msg):
        self._pose_cb(msg.pose)

    # ─── inference + publish ─────────────────────────────────────────
    def _tick(self, _event):
        rgb, pose, goal = self.rgb, self.pose_xyyaw, self.goal_rgb
        if rgb is None or pose is None or goal is None:
            rospy.loginfo_throttle(5.0, "flownav: waiting for RGB + pose + goal")
            return

        result = self.client.step(rgb, goal)
        if result is None:
            rospy.logwarn_throttle(5.0, "flownav: no result -- holding last path")
            return
        try:
            traj = self.client.best_trajectory(result)
        except FlowNavClientError as e:
            rospy.logwarn_throttle(5.0, "flownav: %s -- holding last path", e)
            return

        ox, oy, oyaw = pose
        full_world = anchor_trajectory_to_world(traj, ox, oy, oyaw)
        n = len(full_world)
        k = n if self.execute_fraction >= 1.0 else min(n, max(2, int(round(n * self.execute_fraction))))
        stamp = rospy.Time.now()
        self.pub_path.publish(self._make_path(full_world[:k], stamp))   # flown prefix
        self.pub_full.publish(self._make_path(full_world, stamp))       # display full
        self.n_published += 1
        if self.n_published == 1 or self.n_published % 20 == 0:
            rospy.loginfo("flownav: published %d paths (%d waypoints)",
                          self.n_published, n)

    def _make_path(self, world_xy, stamp):
        """Build a latched world-frame ``nav_msgs/Path`` from ``(x, y)`` pairs."""
        m = Path()
        m.header.stamp = stamp
        m.header.frame_id = self.frame_id
        for wx, wy in world_xy:
            ps = PoseStamped()
            ps.header = m.header
            ps.pose.position.x = float(wx)
            ps.pose.position.y = float(wy)
            ps.pose.orientation.w = 1.0   # identity; follower derives heading
            m.poses.append(ps)
        return m


def main():
    rospy.init_node("flownav")
    FlowNavNode()
    rospy.spin()


if __name__ == "__main__":
    main()


# ── rosparam list ─────────────────────────────────────────────────────────────
#   ~image_transport   frame_path | topic                (default frame_path)
#   ~rgb_topic         /xtend/rgb_frame_path (or /xtend/rgb)
#   ~pose_topic        /xtend/localization
#   ~pose_type         pose_stamped | pose
#   ~goal_image        path to a target RGB image file    (primary goal source)
#   ~goal_image_topic  sensor_msgs/Image goal topic       (optional, live goal)
#   ~path_topic        /path/waypoints_flownav            (raw -> path_corrector)
#   ~full_path_topic   /path/waypoints_flownav_full       (display only)
#   ~frame_id          world
#   ~execute_fraction  1.0                                (fraction of traj to fly)
#   ~rate_hz           4.0                                (continuous inference rate)
#   ~port              8889                               (FlowNav server loopback port)
#   ~timeout_s         10.0                               (HTTP per-request timeout)
