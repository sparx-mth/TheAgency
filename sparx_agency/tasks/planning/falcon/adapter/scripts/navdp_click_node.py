#!/usr/bin/env python3
"""navdp_click_node.py -- ROS1 adapter: click an RGB pixel -> NavDP -> world path.

A drop-in REPLACEMENT for ``astar_planner_node.py``. Instead of A* searching the
BEV grid to a clicked map goal, this node lets the operator click a pixel in the
live camera image; it asks the NavDP point-goal policy for a trajectory and
publishes that trajectory as a world-frame ``nav_msgs/Path``.

Like A*, it publishes its RAW trajectory on its own planner topic
(``~path_topic`` = ``/path/waypoints_navdp``), not directly on ``/path/waypoints``.
This lets the same planner-agnostic ``path_corrector_node`` recentre the NavDP
path off walls against the BEV (point its ``~input_path_topic`` at
``/path/waypoints_navdp``) and republish the corrected, flown path on
``/path/waypoints``. To fly NavDP UNcorrected, point the corrector's input
elsewhere (or set its ``enabled:=false``, which passes the input through), or
point ``~path_topic`` here straight at ``/path/waypoints``. Everything downstream
(``waypoint_follower_node`` flying ``/path/waypoints``, ``bev_click_goal_node``
drawing it) is unchanged.

All the maths is ROS-free and unit-tested in ``core.planning.navdp``:
  * pixel + depth -> body-frame point-goal      (geometry.pixel_to_pointgoal)
  * NavDP HTTP request/response                  (client.NavDPPointgoalClient)
  * body trajectory -> world path                (geometry.anchor_trajectory_to_world)
  * body trajectory -> image overlay             (geometry.project_trajectory_to_pixels)
This node owns ONLY ROS / UI concerns: subscriptions, the OpenCV window, the
click handling, intrinsics resolution and publishing the path.

One OpenCV window "NavDP click", three panels side-by-side:
    [ live RGB (left-click) | colorized depth | snapshot + best trajectory ]
plus a status bar.

  LEFT-CLICK on the RGB panel  -> set the goal pixel (yellow dot + readout)
  ENTER / SPACE                -> send (RGB, depth, goal) to NavDP, anchor the
                                  returned trajectory at the current pose, publish
                                  the world path, and freeze it on the 3rd panel
  r                            -> clear the click + snapshot (and hand the node
                                  back to the world goal, if one is set)
  q / ESC                      -> quit

TWO GOAL SOURCES, and the difference between them is why there are two:

  CLICK (a pixel).   A goal in the IMAGE. It is only meaningful for the frame it
      was clicked on -- fly a metre and the same pixel is a different place -- so
      it cannot be re-used, and no new inference runs until the next ENTER. The
      follower keeps flying the last published path until then: "click once,
      follow until I click again".
  WORLD GOAL (metres).  ``~goal_x/~goal_y``, or the latched ``~goal_topic``
      (``geometry_msgs/Point``) the BEV click and the mission director publish on.
      A world point does NOT move as the drone does, so the node can re-aim it
      itself: every ``~auto_period_s`` it re-projects the goal into the body frame
      against the CURRENT pose and re-infers. NavDP only ever accepts a body-frame
      (forward, left) point-goal, so that re-projection is the entire navigation
      loop -- as the drone advances, the same world point becomes a nearer,
      differently-bearing point-goal. Nobody has to press anything.

SWITCHING BETWEEN THEM is symmetric, and neither needs a restart or a param:

  camera click  -> click mode. Takes over a hands-off run instantly; the standing
                   world goal is remembered, not lost.
  BEV click     -> hands-off mode. Publishing a world goal drops any pending camera
                   click, so the drone starts flying the new point immediately.
  r             -> drop the camera click; the world goal (if any) resumes.

Both end in the same place: one body-frame point-goal handed to NavDP, whose
trajectory is anchored at the pose snapshotted with the frame it saw.

RGB and depth arrive as frame-path messages (``std_msgs/String`` of the form
"<path> <sec> <nsec>"): this node loads the ``.jpg`` and ``.npy`` from disk rather
than receiving raw images over ROS, cutting serialization/network cost. The
parsed paths point at the SAME frames FALCON's mapping uses; only the transport
differs.

IMPORTANT -- intrinsics must match the RGB/depth stream NavDP receives. NavDP
consumes the SAME ``/xtend/rgb_frame_path`` + ``/xtend/depth_frame_path`` frames
FALCON's voxel mapping does, so it uses the SAME intrinsics (the launch wires ``~fx ~fy ~cx ~cy``
to the shared ``cam_*`` args). On the real XTEND that stream is the RAW,
unrectified depth at 504x294, so the correct focals are the ``camera_matrix`` (K)
values -- the ``~fx ~fy ~cx ~cy`` defaults below -- NOT the rectified
``projection_matrix`` (P), which over-scales every metric distance (the same
K-vs-P trap documented in ``real_drone.launch``). In sim, ``sim_adapter``
resamples the stream to its P-target intrinsics, so the launch's ``cam_*`` (and
hence these) become those instead -- the rule is simply "match the live stream".
Point this at a different camera by passing the matching ``~fx ~fy ~cx ~cy
~img_width ~img_height`` (or a ``~camera_info_topic``). Because the click pixel
indexes the depth array directly, the loaded RGB and depth frames MUST share
one resolution -- the node fails loud at startup if they do not.

Run:
    rosrun falcon_adapter navdp_click_node.py
See the file footer for the full rosparam list.
"""
import time

import cv2
import numpy as np

import rospy
from geometry_msgs.msg import Point, Pose, PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from sparx_agency.core.common.frame_path_message import parse_frame_path_message
from sparx_agency.core.common.math import se3
from sparx_agency.core.common.types import Intrinsics
from sparx_agency.core.planning.navdp import (
    NavDPError,
    NavDPPointgoalClient,
    anchor_trajectory_to_world,
    pixel_to_pointgoal,
    point_to_pointgoal,
    project_trajectory_to_pixels,
    world_to_body_2d,
)

WINDOW = "NavDP click"


def colorize_depth(depth, dmax=10.0):
    """HxW metric depth -> BGR TURBO image for the depth panel."""
    d = np.clip(np.nan_to_num(depth, nan=0.0), 0.0, dmax)
    return cv2.applyColorMap((d / dmax * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)


class NavDPClickNode:
    def __init__(self):
        rospy.init_node("navdp_click")
        G = rospy.get_param

        self.drone_ns = G("~drone_ns", "")
        # Image transport. "frame_path" (default): RGB/depth arrive as tiny
        # std_msgs/String "<path> <sec> <nsec>" messages this node loads from disk
        # (the .jpg/.npy the drone wrote -- see _rgb_path_cb/_depth_path_cb).
        # "topic": RGB/depth arrive as raw sensor_msgs/Image straight off the wire
        # (Gazebo sim or an old bag replay, where nothing writes frame files).
        # Only the subscriber type + decode path differ; everything downstream
        # (click -> NavDP -> world path) is identical.
        self.image_transport = str(G("~image_transport", "frame_path")).strip().lower()
        if self.image_transport not in ("frame_path", "topic"):
            raise ValueError("~image_transport must be 'frame_path' or 'topic', "
                             "got %r" % self.image_transport)
        _fp = self.image_transport == "frame_path"
        self.rgb_topic = G("~rgb_topic",
                           "/xtend/rgb_frame_path" if _fp else "/xtend/rgb")
        self.depth_topic = G("~depth_topic",
                             "/xtend/depth_frame_path" if _fp else "/xtend/depth_m")
        self.pose_topic = G("~pose_topic", "/xtend/localization")
        # Message type on ~pose_topic: "pose_stamped" (geometry_msgs/PoseStamped)
        # or "pose" (geometry_msgs/Pose). The default localization
        # /xtend/localization is a PoseStamped -- present in bag playback, on the
        # real drone, and from sim_adapter -- so the default is pose_stamped. Point
        # this at the bare /gt_pose (pose_type:=pose) when running inside the nav
        # stack. Mirrors pose_adapter's ~in_type and real_drone's real_pose_type.
        self.pose_type = G("~pose_type", "pose_stamped")
        self.camera_info_topic = G("~camera_info_topic", "")  # "" -> use params

        # Output: world-frame raw NavDP path on its own planner topic, mirroring
        # A* on /path/waypoints_astar. The path_corrector recentres it against the
        # BEV and republishes /path/waypoints (the flown topic). Point this straight
        # at /path/waypoints to fly NavDP uncorrected.
        self.path_topic = G("~path_topic", "/path/waypoints_navdp")
        # Execute only the first fraction of the trajectory: NavDP is accurate near
        # the camera and drifts further out, so we fly only the near part and re-
        # infer (next ENTER) before going further. The EXECUTED prefix goes on
        # path_topic (-> corrector -> follower, which holds at its last waypoint, so
        # the drone stops there until the next inference); the FULL trajectory is
        # published on full_path_topic for display only. 1.0 = execute the whole
        # trajectory (legacy behaviour).
        self.execute_fraction = float(G("~execute_fraction", 1.0))
        self.full_path_topic = G("~full_path_topic", "/path/waypoints_navdp_full")
        self.frame_id = G("~frame_id", "world")

        # ── The second goal source: a WORLD point, re-aimed every inference ──
        # A click is a point in the IMAGE, so it is only meaningful for the frame
        # it was clicked on: fly a metre and the same pixel is a different place.
        # A world goal is the opposite -- it stays put while the drone moves, so
        # it is the one that can be flown without an operator. NavDP itself only
        # ever accepts a BODY-frame (forward, left) point-goal, so the world goal
        # is re-projected into the body frame at EVERY inference, against the pose
        # snapshotted with that frame. That re-projection IS the navigation: as the
        # drone advances, the same world point becomes a nearer, differently-bearing
        # body goal, and NavDP re-plans toward it.
        #
        # Two ways in, both world/map metres, the click always winning:
        #   ~goal_x/~goal_y  a fixed goal from the config file ('' = unset)
        #   ~goal_topic      geometry_msgs/Point, latched -- the BEV click and the
        #                    mission director both publish here; a new one retargets
        #                    mid-flight and re-arms an arrived goal.
        self.goal_topic = G("~goal_topic", "/waypoint_nav/goal")
        gx0, gy0 = G("~goal_x", ""), G("~goal_y", "")
        self.world_goal = None
        if str(gx0) != "" and str(gy0) != "":
            self.world_goal = (float(gx0), float(gy0))
        # Auto-inference: with a world goal there is nobody to press ENTER, so the
        # node re-infers on its own. Every tick re-projects the goal and publishes a
        # fresh route; without it the drone would fly one prefix and hold forever.
        self.auto_infer = bool(G("~auto_infer", True))
        self.auto_period_s = float(G("~auto_period_s", 2.0))
        # Stop re-inferring once the goal is this close: NavDP would otherwise keep
        # being asked to reach a point under the drone and jitter around it.
        self.goal_arrive_radius_m = float(G("~goal_arrive_radius_m", 0.5))
        self._next_auto_t = 0.0
        self._goal_arrived = False

        # Window layout. "full" = RGB + colorized depth + snapshot/overlay panels
        # (the legacy 3-up view). "rgb_only" = just the live RGB panel to click on,
        # for performance and a clean view (no depth colorize, no overlay render).
        self.display_mode = G("~display_mode", "full")
        if self.display_mode not in ("full", "rgb_only"):
            raise ValueError("~display_mode must be 'full' or 'rgb_only', got %r"
                             % self.display_mode)

        # Camera intrinsics matching the depth frames NavDP indexes.
        # Defaults are the real-XTEND RAW K-matrix at 504x294 (the same values
        # real_drone.launch hands FALCON). The launch overrides them with the
        # shared cam_* args -- raw K on the real drone, sim_adapter's P-target in
        # sim -- so navdp always tracks the live stream. Pass matching values for
        # any other camera.
        self.intr = Intrinsics(
            width=int(G("~img_width", 504)),
            height=int(G("~img_height", 294)),
            fx=float(G("~fx", 322.6351083474948)),
            fy=float(G("~fy", 323.3893307141174)),
            cx=float(G("~cx", 242.06479658679714)),
            cy=float(G("~cy", 90.03019076680604)))

        # Client-side overlay camera height (snapshot panel only). The trajectory
        # projects onto the true floor when this equals the drone's altitude, but
        # at the ~1 m flight height the dense near waypoints project BELOW the image
        # and vanish. So the default is a fixed 0.5 m (NavDP's ~ground-robot
        # training height): it keeps the near part of the route in-frame (the
        # overlay's purpose) at the cost of exact floor alignment. Set <= 0 to track
        # the live altitude instead, or lower (e.g. 0.3) to pull in even more near
        # waypoints. Overlay-only -- the flown/BEV path is unaffected by this.
        self.render_cam_height = float(G("~render_cam_height", 0.5))

        # Kept so the depth panel can be colorized to exactly what NavDP receives
        # (the client clips depth to this before encoding).
        self.depth_max_m = float(G("~depth_max_m", 5.0))
        self.client = NavDPPointgoalClient(
            "http://127.0.0.1:%d" % int(G("~port", 8888)),
            timeout_s=float(G("~timeout_s", 30.0)),
            depth_max_m=self.depth_max_m,
            logger=rospy.logwarn)

        # ── Shared sensor state (callbacks write, main loop reads) ──
        self.rgb = None
        self.depth = None
        self.altitude = float(G("~default_altitude", 0.8))  # until pose arrives
        self.pose_xyyaw = None                              # (x, y, yaw) world
        self.click_px = None                                # (u, v) on RGB panel
        self.hover_px = None                                # (u, v) on depth panel
        self._rgb_w = 0                                     # RGB panel width
        self._got_cam_info = False
        self._reset_done = False                            # latch intrinsics after

        if self.image_transport == "frame_path":
            rospy.Subscriber(self.rgb_topic, String, self._rgb_path_cb, queue_size=2)
            rospy.Subscriber(self.depth_topic, String, self._depth_path_cb, queue_size=2)
        else:  # "topic": raw Images decoded in place
            rospy.Subscriber(self.rgb_topic, Image, self._rgb_cb, queue_size=2)
            rospy.Subscriber(self.depth_topic, Image, self._depth_cb, queue_size=2)
        if self.pose_type == "pose_stamped":
            rospy.Subscriber(self.pose_topic, PoseStamped,
                             self._pose_stamped_cb, queue_size=10)
        elif self.pose_type == "pose":
            rospy.Subscriber(self.pose_topic, Pose, self._pose_cb, queue_size=10)
        else:
            raise ValueError("~pose_type must be 'pose' or 'pose_stamped', got %r"
                             % self.pose_type)
        if self.camera_info_topic:
            rospy.Subscriber(self.camera_info_topic, CameraInfo,
                             self._cam_info_cb, queue_size=1)
        if self.goal_topic:
            rospy.Subscriber(self.goal_topic, Point, self._goal_cb, queue_size=1)

        self.pub_path = rospy.Publisher(self.path_topic, Path,
                                        queue_size=1, latch=True)
        # Full (un-truncated) trajectory for display only -- the BEV viewer draws it
        # so the operator sees the whole NavDP route while only the near part flies.
        self.pub_full = rospy.Publisher(self.full_path_topic, Path,
                                        queue_size=1, latch=True)
        self.n_published = 0
        self._banner()

    # ─── Subscribers (frame-path String: load the file from disk) ────────────
    def _rgb_path_cb(self, msg):
        # "<path> <sec> <nsec>" -> load the .jpg. cv2 reads BGR; NavDP wants RGB.
        # A malformed message or unreadable file keeps the last frame (counted by
        # the throttled warn) rather than clearing the live view.
        try:
            path = parse_frame_path_message(msg.data).path
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise OSError("cv2.imread returned None for %s" % path)
        except (ValueError, OSError) as e:
            rospy.logwarn_throttle(5.0, "navdp_click: dropping RGB frame-path (%s)", e)
            return
        self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _depth_path_cb(self, msg):
        # "<path> <sec> <nsec>" -> load the .npy (HxW float32 meters). Keep the
        # last good frame on a bad message/file instead of clearing the view.
        try:
            path = parse_frame_path_message(msg.data).path
            arr = np.squeeze(np.load(path))
            if arr.ndim != 2:
                raise ValueError("depth %s has shape %r; expected HxW"
                                 % (path, arr.shape))
        except (ValueError, OSError) as e:
            rospy.logwarn_throttle(5.0, "navdp_click: dropping depth frame-path (%s)", e)
            return
        self.depth = np.ascontiguousarray(arr, dtype=np.float32)

    # ─── Subscribers (raw Image: decode in place; sim / bag replay) ──────────
    def _rgb_cb(self, msg):
        arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        if msg.encoding == "bgr8":
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        self.rgb = arr.copy()

    def _depth_cb(self, msg):
        if msg.encoding == "32FC1":
            self.depth = np.frombuffer(msg.data, np.float32).reshape(
                msg.height, msg.width).copy()
        elif msg.encoding == "16UC1":
            self.depth = (np.frombuffer(msg.data, np.uint16).reshape(
                msg.height, msg.width).astype(np.float32) / 1000.0)
        else:
            # Warn loudly so the operator sees WHY "Waiting for RGB + depth"
            # never clears, instead of silently dropping the frame.
            rospy.logwarn_throttle(5.0, "navdp_click: unsupported depth encoding "
                                   "%r (need 32FC1 or 16UC1); ignoring frame",
                                   msg.encoding)

    def _pose_cb(self, msg):
        yaw = se3.yaw_from_quaternion((msg.orientation.x, msg.orientation.y,
                                       msg.orientation.z, msg.orientation.w))
        self.pose_xyyaw = (float(msg.position.x), float(msg.position.y), yaw)
        self.altitude = float(msg.position.z)

    def _pose_stamped_cb(self, msg):
        self._pose_cb(msg.pose)

    def _cam_info_cb(self, msg):
        # ROS1 sensor_msgs/CameraInfo exposes UPPERCASE fields (K, P, R, D); ROS2
        # lowercases them. This node is rospy, so use K/P. Prefer the RAW 3x3
        # camera matrix K (fx=K[0] fy=K[4] cx=K[2] cy=K[5]): the loaded depth is
        # raw, unrectified, so K -- not the rectified projection matrix P --
        # back-projects it correctly (P over-scales metric distances; see the
        # K-vs-P note in real_drone.launch). Fall back to P (fx=P[0] fy=P[5]
        # cx=P[2] cy=P[6]) only if K is absent. The published info MUST describe
        # the actual 504x294 /xtend stream -- valid on the real drone, never in
        # sim (sim_adapter resamples the stream; nothing publishes a matching
        # camera_info there).
        if self._reset_done:
            return                     # intrinsics latched at reset; ignore late
        if any(msg.K):
            fx, fy, cx, cy = msg.K[0], msg.K[4], msg.K[2], msg.K[5]
        elif any(msg.P):
            fx, fy, cx, cy = msg.P[0], msg.P[5], msg.P[2], msg.P[6]
        else:
            return
        self.intr = Intrinsics(width=int(msg.width), height=int(msg.height),
                               fx=float(fx), fy=float(fy),
                               cx=float(cx), cy=float(cy))
        self._got_cam_info = True

    def _goal_cb(self, msg):
        """Latched world-frame goal (BEV click / mission director) -- retarget.

        A BEV click is also the MODE SWITCH: it drops any pending camera click, so
        the node leaves click-and-ENTER mode and starts flying the world goal
        hands-off straight away. Without that, a BEV click would silently do
        nothing until the operator pressed 'r' -- the click still won, and the new
        goal only took effect at some later, unrelated keypress.
        """
        goal = (float(msg.x), float(msg.y))
        if goal != self.world_goal:
            rospy.loginfo("NavDP world goal := (%.2f, %.2f)", goal[0], goal[1])
        self.world_goal = goal
        self._goal_arrived = False   # a new goal always re-arms inference
        self._next_auto_t = 0.0      # ...and acts on the next tick, not in 2 s
        if self.click_px is not None:
            rospy.loginfo("NavDP: world goal supersedes the camera click "
                          "-- switching to hands-off mode")
            self.click_px = None

    # ─── Inference + publish ─────────────────────────────────────
    def infer_and_publish(self, rgb, depth, pose_xyyaw, px, py):
        """Run one NavDP step for click ``(px, py)`` and publish the world path.

        Publishes the EXECUTED near prefix on ``path_topic`` (-> corrector ->
        follower) and the FULL trajectory on ``full_path_topic`` (display only).
        Returns the full body-frame trajectory ``(T, >=2)`` for the overlay, or None.
        """
        gx, gy, d, bz = pixel_to_pointgoal(px, py, depth, self.intr)
        side = "left" if gy > 0 else "right"
        rospy.loginfo("NavDP goal: %.2fm fwd  %.2fm %s  (click depth=%.2fm, "
                      "dz=%+.2fm, alt=%.2fm)", gx, abs(gy), side, d, bz,
                      self.altitude)
        return self._step(rgb, depth, pose_xyyaw, gx, gy, px, py)

    def infer_world_goal(self, rgb, depth, pose_xyyaw):
        """Run one NavDP step toward ``self.world_goal``, re-aimed from HERE.

        The world goal is converted to the body frame against ``pose_xyyaw`` --
        the pose snapshotted with this very RGB-D frame -- and range-limited to
        NavDP's input box by :func:`point_to_pointgoal`, which scales forward and
        lateral together so a goal beyond the box keeps its BEARING. So a distant
        goal is flown as "head that way", one leg per inference, rather than being
        clipped into a goal that points somewhere else.

        Returns the body trajectory, or None (no goal, arrived, or NavDP failed).
        """
        if self.world_goal is None or self._goal_arrived:
            return None
        gwx, gwy = self.world_goal
        ox, oy, _ = pose_xyyaw
        if np.hypot(gwx - ox, gwy - oy) <= self.goal_arrive_radius_m:
            rospy.loginfo("NavDP world goal (%.2f, %.2f) reached -- holding "
                          "(publish a new goal to resume)", gwx, gwy)
            self._goal_arrived = True
            return None
        fwd, left = world_to_body_2d(gwx, gwy, *pose_xyyaw)
        gx, gy = point_to_pointgoal(fwd, left)
        rospy.loginfo("NavDP world goal (%.2f, %.2f) -> body %.2fm fwd %.2fm "
                      "left -> pointgoal (%.2f, %.2f)", gwx, gwy, fwd, left, gx, gy)
        # -1: the server's overlay convention for "no click behind this goal".
        return self._step(rgb, depth, pose_xyyaw, gx, gy, -1, -1)

    def _step(self, rgb, depth, pose_xyyaw, gx, gy, px, py):
        """One NavDP inference for a body-frame point-goal, then publish."""
        result = self.client.pointgoal_step(rgb, depth, gx, gy,
                                             click_px=px, click_py=py,
                                             altitude=self.altitude)
        if result is None:
            rospy.logwarn("NavDP returned no result -- holding last path")
            return None
        try:
            traj = self.client.best_trajectory(result)
        except NavDPError as e:
            rospy.logwarn("NavDP: %s -- holding last path", e)
            return None

        ox, oy, oyaw = pose_xyyaw
        full_world = anchor_trajectory_to_world(traj, ox, oy, oyaw)
        n = len(full_world)
        # Execute only the near prefix (NavDP drifts further out); display the rest.
        # max(2, ...) keeps a flyable path; the follower holds at its last waypoint,
        # so the drone stops at the prefix end until the next ENTER re-infers.
        k = n if self.execute_fraction >= 1.0 else min(n, max(2, int(round(n * self.execute_fraction))))
        stamp = rospy.Time.now()
        self.pub_path.publish(self._make_path(full_world[:k], stamp))   # flown prefix
        self.pub_full.publish(self._make_path(full_world, stamp))       # display full
        self.n_published += 1
        end = traj[-1]
        rospy.loginfo("NavDP PUBLISHED: execute %d/%d waypoints (full route on %s)  "
                      "body_end=(%.2f, %.2f)  anchored@(%.2f, %.2f, %.0fdeg)",
                      k, n, self.full_path_topic, float(end[0]), float(end[1]),
                      ox, oy, np.degrees(oyaw))
        return traj

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

    # ─── Rendering ───────────────────────────────────────────────
    def _draw_live(self, rgb, depth):
        live = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self._rgb_w = live.shape[1]
        cv2.putText(live, "RGB  (left-click)", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        if self.click_px is not None:
            gx, gy, d, bz = pixel_to_pointgoal(
                self.click_px[0], self.click_px[1], depth, self.intr)
            cv2.circle(live, self.click_px, 12, (0, 255, 255), 2)
            cv2.circle(live, self.click_px, 4, (0, 255, 255), -1)
            side = "left" if gy > 0 else "right"
            cv2.putText(live, "%.1fm fwd  %.1fm %s  d=%.2fm  dz=%+.2fm"
                        % (gx, abs(gy), side, d, bz),
                        (self.click_px[0] + 14, self.click_px[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        return live

    def _draw_depth(self, depth):
        # Colorize to the same ceiling NavDP sees (the client clips to depth_max_m
        # before sending), so the panel shows the operator what the policy gets.
        vis = colorize_depth(depth, self.depth_max_m)
        cv2.putText(vis, "Depth  (hover)", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        status = ""
        if self.hover_px is not None:
            hx, hy = self.hover_px
            if 0 <= hx < depth.shape[1] and 0 <= hy < depth.shape[0]:
                cv2.drawMarker(vis, (hx, hy), (255, 255, 255),
                               cv2.MARKER_CROSS, 14, 1)
                status = "depth(%d,%d)=%.2fm" % (hx, hy, depth[hy, hx])
        return vis, status

    def _draw_snapshot(self, snap_rgb, traj, px, py):
        """Snapshot RGB + NavDP best trajectory drawn on the ground plane."""
        out = cv2.cvtColor(snap_rgb, cv2.COLOR_RGB2BGR)
        cam_h = (self.render_cam_height if self.render_cam_height > 0
                 else max(self.altitude, 0.1))
        pts = project_trajectory_to_pixels(traj, self.intr, cam_h)
        origin = (int(self.intr.cx), out.shape[0] - 1)
        prev = origin
        for p in pts:
            if p is not None:
                cv2.line(out, prev, p, (255, 255, 255), 4, cv2.LINE_AA)
                cv2.line(out, prev, p, (0, 255, 0), 2, cv2.LINE_AA)
                prev = p
        for p in pts:
            if p is not None:
                cv2.circle(out, p, 3, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(out, (px, py), 12, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(out, (px, py), 4, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(out, "NavDP best traj  render_h=%.2fm" % cam_h, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return out

    # ─── Mouse callback ──────────────────────────────────────────
    def _on_mouse(self, event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and self._rgb_w and x < self._rgb_w:
            self.click_px = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self._rgb_w:
            self.hover_px = ((x - self._rgb_w, y)
                             if self._rgb_w <= x < 2 * self._rgb_w else None)

    # ─── Main loop (OpenCV must run on the main thread) ──────────
    def spin(self):
        # When intrinsics come from a camera_info topic, wait briefly for the
        # first message so NavDP is reset with the stream's true intrinsics
        # (resolution/principal-point must match the RGB+depth NavDP receives).
        if self.camera_info_topic:
            t0 = time.time()
            while (not rospy.is_shutdown() and not self._got_cam_info
                   and time.time() - t0 < 2.0):
                time.sleep(0.05)
            if not self._got_cam_info:
                rospy.logwarn("No %s yet -- resetting NavDP with param intrinsics",
                              self.camera_info_topic)
        if not self.client.reset(self.intr):
            rospy.logfatal("Could not reach NavDP at %s", self.client.url)
            return
        # Latch intrinsics at reset: the server now holds exactly self.intr, so
        # ignore any later CameraInfo (a fixed camera's intrinsics are static)
        # rather than letting the node and server drift out of sync.
        self._reset_done = True
        rospy.loginfo("Waiting for RGB + depth ...")
        while not rospy.is_shutdown() and (self.rgb is None or self.depth is None):
            time.sleep(0.1)
        if rospy.is_shutdown():
            return
        if not self._validate_stream_resolution(self.rgb, self.depth):
            return
        rospy.loginfo("Ready. Click the RGB panel, then press ENTER.")

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, self._on_mouse)
        snap_vis = None

        while not rospy.is_shutdown():
            rgb, depth = self.rgb, self.depth
            if rgb is None or depth is None:
                time.sleep(0.01)
                continue

            live = self._draw_live(rgb, depth)
            default_status = ("ENTER=send  r=clear  q=quit  |  alt=%.2fm  published=%d"
                              % (self.altitude, self.n_published))
            if self.display_mode == "rgb_only":
                # Just the clickable RGB panel -- no depth colorize, no overlay
                # render (lighter; the route is still seen on the BEV viewer).
                top, status = live, default_status
            else:
                depth_vis, depth_status = self._draw_depth(depth)
                third = (snap_vis if snap_vis is not None
                         else self._placeholder(live))
                top = np.hstack([live, depth_vis, third])
                status = depth_status or default_status
            bar = np.zeros((28, top.shape[1], 3), np.uint8)
            cv2.putText(bar, status, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (220, 220, 220), 1)
            cv2.imshow(WINDOW, np.vstack([top, bar]))

            # Auto re-inference for the world goal. Runs HERE, on the main loop,
            # rather than on a rospy.Timer: inference and the ENTER path would
            # otherwise publish from two threads onto one latched topic. A pending
            # click always wins -- the operator's pixel goal is never overridden
            # by the standing world goal until 'r' clears it.
            if (self.auto_infer and self.click_px is None
                    and self.world_goal is not None and not self._goal_arrived
                    and self.pose_xyyaw is not None
                    and time.time() >= self._next_auto_t):
                self._next_auto_t = time.time() + self.auto_period_s
                self.infer_world_goal(rgb.copy(), depth.copy(), self.pose_xyyaw)

            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('r'):
                self.click_px = None
                snap_vis = None
            elif key in (13, 32):                       # ENTER / SPACE
                # NB: never `x or snap_vis` -- _on_enter returns an ndarray on
                # success, and bool(ndarray) raises. Keep the last snapshot on
                # failure (None).
                new_vis = self._on_enter()
                if new_vis is not None:
                    snap_vis = new_vis
        cv2.destroyAllWindows()

    def _on_enter(self):
        if self.click_px is None:
            rospy.loginfo("Click the RGB panel first.")
            return None
        if self.pose_xyyaw is None:
            rospy.logwarn("No pose yet -- cannot anchor the path.")
            return None
        # Snapshot RGB, depth and pose together so the body->world anchoring uses
        # the pose that matches the frame NavDP saw (immune to later drift).
        snap_rgb, snap_depth = self.rgb.copy(), self.depth.copy()
        snap_pose = self.pose_xyyaw
        px, py = self.click_px
        traj = self.infer_and_publish(snap_rgb, snap_depth, snap_pose, px, py)
        if traj is None:
            return None
        # rgb_only: skip the (costly) ground-plane overlay render -- there is no
        # snapshot panel to show it; the route is seen on the BEV viewer instead.
        if self.display_mode == "rgb_only":
            return None
        return self._draw_snapshot(snap_rgb, traj, px, py)

    def _validate_stream_resolution(self, rgb, depth):
        """Fail loud if the RGB/depth stream geometry is inconsistent.

        The click pixel is taken on the RGB panel and used to index the depth
        array and back-project with ``self.intr`` (and to draw the overlay), so
        all three must describe ONE image. A height mismatch would also crash the
        side-by-side ``np.hstack`` in the render loop. Returns ``True`` when it is
        safe to run.
        """
        rgb_hw, depth_hw = rgb.shape[:2], depth.shape[:2]
        if rgb_hw != depth_hw:
            rospy.logfatal(
                "RGB %dx%d and depth %dx%d differ; navdp_click indexes depth at "
                "the RGB click pixel and needs them aligned. Fix the stream.",
                rgb_hw[1], rgb_hw[0], depth_hw[1], depth_hw[0])
            return False
        if rgb_hw != (self.intr.height, self.intr.width):
            rospy.logwarn(
                "Stream is %dx%d but intrinsics are %dx%d (fx=%.1f cx=%.1f "
                "cy=%.1f); goals and overlay will be geometrically wrong -- pass "
                "intrinsics matching the live stream.", rgb_hw[1], rgb_hw[0],
                self.intr.width, self.intr.height, self.intr.fx, self.intr.cx,
                self.intr.cy)
        return True

    @staticmethod
    def _placeholder(live):
        third = np.zeros_like(live)
        cv2.putText(third, "Click + ENTER for trajectory",
                    (16, third.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (180, 180, 180), 1)
        return third

    def _banner(self):
        L = rospy.loginfo
        L("=" * 64)
        L("navdp_click (core NavDP point-goal -> world path)")
        L("  rgb   in  = %s", self.rgb_topic)
        L("  depth in  = %s", self.depth_topic)
        L("  pose  in  = %s  (%s)", self.pose_topic, self.pose_type)
        L("  navdp     = %s", self.client.url)
        L("  path  out = %s  (executed prefix -> path_corrector)", self.path_topic)
        L("  full  out = %s  (full route, display only)", self.full_path_topic)
        if self.execute_fraction >= 1.0:
            L("  execute   = whole trajectory")
        else:
            L("  execute   = first %.0f%% of the trajectory (hold at the prefix end "
              "until the next ENTER)", 100.0 * self.execute_fraction)
        L("  display   = %s", self.display_mode)
        if self.world_goal is not None:
            L("  world goal= (%.2f, %.2f)  (re-aimed every inference)",
              *self.world_goal)
        L("  goal  in  = %s  (world Point; a camera click overrides it)",
          self.goal_topic or "(none)")
        if self.auto_infer:
            L("  auto      = re-infer every %.1fs toward the world goal, until "
              "within %.2fm of it", self.auto_period_s, self.goal_arrive_radius_m)
        else:
            L("  auto      = OFF -- every inference needs an ENTER")
        L("  intrinsics: fx=%.1f fy=%.1f cx=%.1f cy=%.1f  (%dx%d)",
          self.intr.fx, self.intr.fy, self.intr.cx, self.intr.cy,
          self.intr.width, self.intr.height)
        if self.camera_info_topic:
            L("  camera_info override = %s", self.camera_info_topic)
        L("  render cam-height = %.2fm%s", self.render_cam_height,
          "" if self.render_cam_height > 0 else "  (tracks live altitude)")
        L("=" * 64)


def main():
    try:
        NavDPClickNode().spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()


# ============================================================================
# ROSPARAMS (all private ~; defaults in parentheses). The geometry + HTTP
# contract live in core.planning.navdp; this node owns ROS I/O, the OpenCV
# window, click handling and publishing the world path.
#
#   IO: ~image_transport (frame_path | topic) selects RGB/depth transport:
#       frame_path -> ~rgb_topic (/xtend/rgb_frame_path) ~depth_topic
#         (/xtend/depth_frame_path), both std_msgs/String "<path> <sec> <nsec>";
#         the .jpg/.npy are loaded from disk
#       topic -> ~rgb_topic (/xtend/rgb) ~depth_topic (/xtend/depth_m), raw
#         sensor_msgs/Image (sim/bag replay)
#       ~drone_ns ('') ~pose_topic (/xtend/localization)
#       ~pose_type (pose_stamped ; 'pose' for a bare geometry_msgs/Pose, e.g.
#         the nav stack's /gt_pose or Gazebo's /simple_drone/gt_pose)
#       ~camera_info_topic ('' = use the fx/fy/cx/cy params; K preferred over P)
#       ~path_topic (/path/waypoints_navdp; the EXECUTED prefix -> path_corrector_node,
#         which recentres it and republishes /path/waypoints. Point straight at
#         /path/waypoints to fly NavDP uncorrected.) ~frame_id (world)
#       ~full_path_topic (/path/waypoints_navdp_full; the FULL trajectory, display
#         only -- the BEV viewer draws it so you see the whole route while flying
#         only the near prefix)
#   world goal (the no-operator source; see the module docstring):
#       ~goal_topic (/waypoint_nav/goal; geometry_msgs/Point, world metres, latched
#         -- the BEV click and the mission director both publish here. A new goal
#         retargets mid-flight and re-arms an arrived one.)
#       ~goal_x ~goal_y ('' = unset; a fixed goal from the config file)
#       ~auto_infer (true; re-infer toward the world goal with no ENTER. A pending
#         camera click always wins -- press r to hand the node back.)
#       ~auto_period_s (2.0; seconds between auto re-inferences. Pair with
#         ~execute_fraction so a leg is roughly flown before the next one lands.)
#       ~goal_arrive_radius_m (0.5; stop re-inferring within this of the goal)
#   execution: ~execute_fraction (1.0; fly only the first fraction of the route,
#       then hold at the prefix end until the next ENTER re-infers -- NavDP is
#       accurate near the camera and drifts further out. 0.5 = first half.)
#   camera (MUST match the live depth frames; the launch wires these to
#       the shared cam_* args): ~fx ~fy ~cx ~cy ~img_width (504) ~img_height (294)
#       [real-XTEND raw-K defaults; sim uses sim_adapter's P-target via cam_*]
#   NavDP server: ~port (8888) ~timeout_s (30.0) ~depth_max_m (5.0)
#   window: ~display_mode (full = RGB + depth + snapshot/overlay; rgb_only = just
#       the clickable RGB panel, lighter -- the route is still seen on the BEV viewer)
#   overlay/misc: ~render_cam_height (0.5; fixed ground-plane height for the snapshot
#       overlay so the near waypoints stay in-frame at flight altitude. <=0 tracks
#       the live altitude; lower pulls in more near waypoints. Overlay-only.)
#       ~default_altitude (0.8; used until the first pose arrives)
# ============================================================================
