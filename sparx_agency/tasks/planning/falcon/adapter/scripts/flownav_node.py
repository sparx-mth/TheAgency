#!/usr/bin/env python3
"""flownav_node.py -- ROS1 adapter: RGB + goal image -> FlowNav -> world path.

The FlowNav sibling of ``navdp_click_node.py``. NavDP is point-goal (click a
pixel); FlowNav is **image-goal** -- it steers toward a target image. It feeds the
live RGB frame + a goal image to the FlowNav policy and publishes the returned
trajectory as a world-frame ``nav_msgs/Path`` -- the SAME output contract as NavDP,
so ``path_corrector_node`` -> ``trajectory_simplifier_node`` ->
``waypoint_follower_node`` are unchanged.

Selected by the launch ``vla:=flownav`` arg (vs ``vla:=navdp``). It publishes its
RAW path on ``~path_topic`` (``/path/waypoints_flownav``); the corrector recentres
it and republishes ``/path/waypoints``.

The heavy TensorRT inference runs in a HOST process
(``tasks/planning/vlas/flownav/serve/flownav_trt_server.py``) reached over loopback
HTTP, because the FALCON Noetic container has no TensorRT -- exactly like NavDP.

Goal image (pick one; or none -> the server's ``--goal-image`` is used):
  * ``~goal_image`` -- a file path loaded once at startup, or
  * ``~goal_image_topic`` -- a ``sensor_msgs/Image`` updated live.

Inference needs NO depth and NO intrinsics. The OPTIONAL ``~display`` window
(intrinsics ``~fx ~fy ~cx ~cy`` + ``~render_cam_height``, used for the overlay
only) shows the live RGB with the predicted trajectory drawn on it, the
goal-distance, and tints the whole frame GREEN once ``~arrival_distance`` is met.

Runs CONTINUOUSLY at ``~rate_hz`` (reactive local planning toward the goal image);
each tick republishes a fresh latched path. See the file footer for the rosparams.
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
from sparx_agency.core.common.types import Intrinsics
from sparx_agency.core.planning.vlas.flownav.client import (
    FlowNavClientError, FlowNavImageGoalClient,
)
from sparx_agency.core.planning.vlas.navdp import (
    anchor_trajectory_to_world, project_trajectory_to_pixels,
)

WINDOW = "FlowNav"


def _truthy(value):
    """Parse a rosparam flag without the ``bool('false') is True`` trap."""
    return str(value).strip().lower() in ("1", "true", "yes", "on")


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
        self.arrival_distance = float(G("~arrival_distance", 0.0))  # >0: hold + GREEN when dist <= this
        self.rate_hz = float(G("~rate_hz", 4.0))

        self.goal_image_path = str(G("~goal_image", "")).strip()
        self.goal_image_topic = str(G("~goal_image_topic", "")).strip()

        # Display (overlay only): intrinsics + ground-plane height to project the
        # body-frame trajectory onto the image. Defaults are the raw XTEND K.
        self.display = _truthy(G("~display", "false"))
        self.render_cam_height = float(G("~render_cam_height", 0.5))
        self.intr = Intrinsics(
            width=int(G("~img_width", 504)), height=int(G("~img_height", 294)),
            fx=float(G("~fx", 322.6351083474948)), fy=float(G("~fy", 323.3893307141174)),
            cx=float(G("~cx", 242.06479658679714)), cy=float(G("~cy", 90.03019076680604)))

        port = int(G("~port", 8889))
        self.client = FlowNavImageGoalClient(
            "http://127.0.0.1:%d" % port,
            timeout_s=float(G("~timeout_s", 10.0)), logger=rospy.logwarn)

        self.rgb = None                 # HxWx3 uint8 RGB (latest frame)
        self.pose_xyyaw = None          # (x, y, yaw)
        self.goal_rgb = None            # HxWx3 uint8 RGB (None -> use the server's goal)
        self.last_traj_body = None      # (T, 2) body waypoints from the last inference
        self.last_dist = -1.0           # last goal-distance head value
        self.n_published = 0
        self.arrived = False

        if self.goal_image_path:
            self._load_goal_file(self.goal_image_path)
        # No local goal -> "server-goal" mode: send only the obs frame; the server
        # applies its own --goal-image (which can read host paths the container
        # cannot mount). Otherwise the node sends the goal it loaded / subscribed.
        self.server_goal = not (self.goal_image_path or self.goal_image_topic)
        if self.server_goal:
            rospy.loginfo("flownav_node: no ~goal_image set -> using the server's "
                          "goal (start the server with --goal-image)")

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
        rospy.loginfo("flownav_node up: %.1f Hz, goal=%s, display=%s, -> %s",
                      self.rate_hz, self.goal_image_path or self.goal_image_topic or "server",
                      self.display, self.path_topic)

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

    # ─── inference + publish (one step) ──────────────────────────────
    def _step_once(self):
        rgb, pose = self.rgb, self.pose_xyyaw
        goal = None if self.server_goal else self.goal_rgb
        if rgb is None or pose is None or (not self.server_goal and goal is None):
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

        self.last_traj_body = traj
        self.last_dist = float(result.get("distance", -1.0))
        rospy.loginfo_throttle(2.0, "flownav: goal-distance %.2f (lower = closer)",
                               self.last_dist)

        # Arrival: when the image-similarity distance head says we are close, stop
        # re-planning and hold the last (short, near-goal) path. ~arrival_distance
        # <= 0 disables this; tune it from the logged distances on a real run.
        if self.arrival_distance > 0.0 and 0.0 <= self.last_dist <= self.arrival_distance:
            if not self.arrived:
                rospy.loginfo("flownav: GOAL REACHED (distance %.2f <= %.2f) -- "
                              "holding the last path", self.last_dist, self.arrival_distance)
            self.arrived = True
            return
        self.arrived = False

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

    # ─── display ─────────────────────────────────────────────────────
    def _render(self):
        """Show [current view + predicted trajectory | target image]; GREEN on arrival."""
        if self.rgb is None:
            canvas = np.zeros((self.intr.height, self.intr.width, 3), np.uint8)
            cv2.putText(canvas, "FlowNav: waiting for RGB...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            cv2.imshow(WINDOW, canvas)
            return
        vis = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR).copy()

        if self.last_traj_body is not None:
            pts = project_trajectory_to_pixels(self.last_traj_body, self.intr,
                                               self.render_cam_height)
            prev = None
            for p in pts:
                if p is None:
                    prev = None
                    continue
                cv2.circle(vis, p, 4, (0, 255, 255), -1)              # yellow waypoint
                if prev is not None:
                    cv2.line(vis, prev, p, (0, 180, 255), 2)          # orange link
                prev = p

        col = (0, 255, 0) if self.arrived else (255, 255, 255)
        cv2.putText(vis, "goal-distance: %.2f%s" % (self.last_dist,
                    "   REACHED" if self.arrived else ""),
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        cv2.putText(vis, "current view", (10, vis.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

        # Target panel on the right (resized to the live-frame height), so the
        # operator sees "where I am" next to "where I'm going".
        frame = vis
        if self.goal_rgb is not None:
            h = vis.shape[0]
            gbgr = cv2.cvtColor(self.goal_rgb, cv2.COLOR_RGB2BGR)
            gw = max(1, int(round(gbgr.shape[1] * h / gbgr.shape[0])))
            goal_panel = cv2.resize(gbgr, (gw, h))
            cv2.putText(goal_panel, "TARGET", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            sep = np.full((h, 4, 3), 60, np.uint8)
            frame = np.hstack([vis, sep, goal_panel])

        if self.arrived:                                             # paint the frame green
            green = np.zeros_like(frame)
            green[:, :, 1] = 255
            frame = cv2.addWeighted(frame, 0.55, green, 0.45, 0)
            cv2.putText(frame, "GOAL REACHED", (frame.shape[1] // 2 - 150, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
        cv2.imshow(WINDOW, frame)

    # ─── main loop ───────────────────────────────────────────────────
    def run(self):
        """Infer at ``rate_hz``; render the window (if enabled) at ~30 Hz.

        Subscriber callbacks fire on rospy background threads, so the main thread
        owns the OpenCV window (cv2 GUI must run on the main thread).
        """
        loop_hz = 30.0 if self.display else max(0.5, self.rate_hz)
        rate = rospy.Rate(loop_hz)
        interval = 1.0 / max(0.1, self.rate_hz)
        last_infer = -1e9
        last_goal_fetch = -1e9
        while not rospy.is_shutdown():
            now = rospy.get_time()
            # display + server-goal: the node holds no local goal image, so fetch the
            # server's goal once (retry every 2 s) just to show it in the TARGET panel.
            if self.display and self.goal_rgb is None and now - last_goal_fetch > 2.0:
                self.goal_rgb = self.client.get_goal()
                last_goal_fetch = now
            if now - last_infer >= interval:
                self._step_once()
                last_infer = now
            if self.display:
                self._render()
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):   # q / ESC
                    rospy.signal_shutdown("flownav window closed")
                    break
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break
        if self.display:
            cv2.destroyAllWindows()


def main():
    rospy.init_node("flownav")
    FlowNavNode().run()


if __name__ == "__main__":
    main()


# ── rosparam list ─────────────────────────────────────────────────────────────
#   ~image_transport   frame_path | topic                (default frame_path)
#   ~rgb_topic         /xtend/rgb_frame_path (or /xtend/rgb)
#   ~pose_topic        /xtend/localization
#   ~pose_type         pose_stamped | pose
#   ~goal_image        path to a target RGB image file    (optional; else the server's --goal-image)
#   ~goal_image_topic  sensor_msgs/Image goal topic       (optional, live goal)
#   ~path_topic        /path/waypoints_flownav            (raw -> path_corrector)
#   ~full_path_topic   /path/waypoints_flownav_full       (display only)
#   ~frame_id          world
#   ~execute_fraction  1.0                                (fraction of traj to fly)
#   ~arrival_distance  0.0                                (>0: hold + GREEN frame when goal-distance <= this; 0 = off)
#   ~rate_hz           4.0                                (continuous inference rate)
#   ~display           false                              (open an OpenCV window: RGB + trajectory + goal-distance)
#   ~fx ~fy ~cx ~cy    raw XTEND K                        (overlay projection only)
#   ~img_width ~img_height  504 / 294                     (overlay canvas size)
#   ~render_cam_height 0.5                                (ground-plane height for the overlay)
#   ~port              8889                               (FlowNav server loopback port)
#   ~timeout_s         10.0                               (HTTP per-request timeout)
