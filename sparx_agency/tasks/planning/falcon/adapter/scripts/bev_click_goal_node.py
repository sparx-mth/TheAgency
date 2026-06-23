#!/usr/bin/env python3
"""
bev_click_goal_node.py -- interactive 2D BEV viewer with click-to-navigate.

A matplotlib window (NOT RViz) showing the published BEV grid plus the drone
pose, the raw A* path, the APF-safe (recentred) path, the predicted drone
trajectory, and the last clicked goal:
    gray = unknown (-1)   white = free (0)   black = occupied (100)
    red path   = raw A*  (shortest -- hugs walls / cuts corners)
    green path = APF-safe (recentred toward free space -- the path flown)
    orange dashed = predicted (stop-and-turn)
The title shows the predicted-trajectory quality score (0..1) when available.

LEFT-CLICK anywhere publishes a geometry_msgs/Point to ~goal_topic; the planner
picks it up, replans, and the new path is drawn within one BEV update. This node
is purely a viewer + click adapter -- it does not move the drone, plan, or alter
the map.

Run as a sidecar in the FALCON container:
    rosrun falcon_adapter bev_click_goal_node.py
Requires matplotlib (apt-get install -y python3-matplotlib if missing).

  in   ~bev_topic  (OccupancyGrid)  /falcon/bev_2d
  in   ~path_topic (Path)           /path/waypoints      (APF-safe; drawn green)
  in   ~raw_path_topic (Path)       /path/waypoints_raw  (raw A*; drawn red)
  in   ~predicted_path_topic (Path) /path/predicted
  in   ~predicted_score_topic (Float32) /path/predicted_score
  in   ~drone_ns + /gt_pose (Pose)
  in   ~pose_stamped_topic (PoseStamped, optional)  e.g. /xtend/april_tag_pose
  out  ~goal_topic (Point, latched) /waypoint_nav/goal

The drone marker normally reads a bare Pose on ~drone_ns + /gt_pose (what the
rest of the nav stack publishes via pose_adapter). For standalone debugging --
e.g. a rosbag played straight through the ROS1<->ROS2 bridge with no nav stack
up -- set ~pose_stamped_topic to the raw localization (a PoseStamped such as
/xtend/april_tag_pose) and the dot is drawn directly from it, even before any
BEV map is published.
"""
import threading

import numpy as np
import rospy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from geometry_msgs.msg import Pose, PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Float32

from sparx_agency.core.common.math import se3


class BevClickGoalNode:
    def __init__(self):
        # disable_signals=True so matplotlib's main loop owns Ctrl+C
        rospy.init_node("bev_click_goal", disable_signals=True)
        G = rospy.get_param

        self.drone_ns = G("~drone_ns", "")
        self.bev_topic = G("~bev_topic", "/falcon/bev_2d")
        self.path_topic = G("~path_topic", "/path/waypoints")
        self.raw_path_topic = G("~raw_path_topic", "/path/waypoints_raw")
        self.predicted_path_topic = G("~predicted_path_topic", "/path/predicted")
        self.predicted_score_topic = G("~predicted_score_topic",
                                       "/path/predicted_score")
        self.goal_topic = G("~goal_topic", "/waypoint_nav/goal")
        self.pose_stamped_topic = G("~pose_stamped_topic", "")
        self.refresh_hz = float(G("~refresh_hz", 5.0))
        self.arrow_len = float(G("~arrow_len_m", 0.5))
        # View window used until the first BEV map arrives, so the drone is
        # visible even with no map yet (bridge+bag only). Matches the bev bbox.
        self.fb_extent = (float(G("~fallback_xmin", -6.0)),
                          float(G("~fallback_xmax", 6.0)),
                          float(G("~fallback_ymin", -6.0)),
                          float(G("~fallback_ymax", 6.0)))

        # Latest data + lock for cross-thread (ROS callbacks vs render) access
        self._bev = None
        self._path_xy = []              # APF-safe path (green) -- the path flown
        self._raw_xy = []               # raw A* path (red) -- shortest, viz only
        self._pred_xy = []              # predicted drone trajectory (rollout)
        self._pred_score = None         # 0..1 "how good" score
        self._drone_p = None            # (x, y, yaw)
        self._goal_xy = None
        self._lock = threading.Lock()

        self.goal_pub = rospy.Publisher(self.goal_topic, Point, queue_size=1, latch=True)
        rospy.Subscriber(self.bev_topic, OccupancyGrid, self._bev_cb, queue_size=1)
        rospy.Subscriber(self.path_topic, Path, self._path_cb, queue_size=1)
        rospy.Subscriber(self.raw_path_topic, Path, self._raw_path_cb, queue_size=1)
        rospy.Subscriber(self.predicted_path_topic, Path, self._pred_cb, queue_size=1)
        rospy.Subscriber(self.predicted_score_topic, Float32, self._pred_score_cb,
                         queue_size=1)
        rospy.Subscriber(self.drone_ns + "/gt_pose", Pose, self._pose_cb, queue_size=10)
        if self.pose_stamped_topic:
            rospy.Subscriber(self.pose_stamped_topic, PoseStamped,
                             self._pose_stamped_cb, queue_size=10)

        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect(
            "close_event",
            lambda _e: rospy.signal_shutdown("bev_click_goal window closed"))
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x (m)")
        self.ax.set_ylabel("y (m)")
        self.ax.grid(True, alpha=0.25)

        # Persistent artists (created lazily, updated in place)
        self._im = self._raw_line = self._path_line = self._pred_line = None
        self._drone_dot = self._drone_arrow = self._goal_marker = None
        self._limits_set = False        # axes window fixed on first render

        rospy.loginfo("=" * 64)
        rospy.loginfo("bev_click_goal: ready")
        rospy.loginfo("  bev  in  = %s", self.bev_topic)
        rospy.loginfo("  path in  = %s   (APF-safe, green)", self.path_topic)
        rospy.loginfo("  raw  in  = %s   (raw A*, red)", self.raw_path_topic)
        rospy.loginfo("  pose in  = %s/gt_pose", self.drone_ns)
        if self.pose_stamped_topic:
            rospy.loginfo("  pose in  = %s   (PoseStamped, direct)",
                          self.pose_stamped_topic)
        rospy.loginfo("  goal out = %s   (left-click to publish)", self.goal_topic)
        rospy.loginfo("=" * 64)

    # -- subscribers ----------------------------------------------------------
    def _bev_cb(self, msg):
        with self._lock:
            self._bev = msg

    def _path_cb(self, msg):
        pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        with self._lock:
            self._path_xy = pts

    def _raw_path_cb(self, msg):
        pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        with self._lock:
            self._raw_xy = pts

    def _pred_cb(self, msg):
        pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        with self._lock:
            self._pred_xy = pts

    def _pred_score_cb(self, msg):
        with self._lock:
            self._pred_score = float(msg.data)

    def _pose_cb(self, msg):
        o = msg.orientation
        yaw = se3.yaw_from_quaternion((o.x, o.y, o.z, o.w))
        with self._lock:
            self._drone_p = (msg.position.x, msg.position.y, yaw)

    def _pose_stamped_cb(self, msg):
        self._pose_cb(msg.pose)

    # -- click handler --------------------------------------------------------
    def _on_click(self, event):
        if event.inaxes != self.ax or event.button != 1:        # left only
            return
        if event.xdata is None or event.ydata is None:
            return
        gx, gy = float(event.xdata), float(event.ydata)
        rospy.loginfo("bev_click_goal: click -> goal (%.2f, %.2f)", gx, gy)
        with self._lock:
            self._goal_xy = (gx, gy)
            self._path_xy = []          # drop the stale safe path; replan redraws it
            self._raw_xy = []           # and the stale raw A* path
        m = Point()
        m.x, m.y, m.z = gx, gy, 0.0
        self.goal_pub.publish(m)

    # -- render (main thread, via FuncAnimation) ------------------------------
    def _render(self, _frame):
        with self._lock:
            bev, path = self._bev, list(self._path_xy)
            raw = list(self._raw_xy)
            drone, goal = self._drone_p, self._goal_xy
            pred, score = list(self._pred_xy), self._pred_score

        # Map background (optional): the drone is still drawn without it, so a
        # bridge+bag run with no bev_publisher up still shows where the drone is.
        if bev is not None:
            info = bev.info
            W, H, res = info.width, info.height, info.resolution
            ox, oy = info.origin.position.x, info.origin.position.y
            data = np.array(bev.data, dtype=np.int8).reshape(H, W)

            # Tri-color: unknown=gray, free=white, occupied=near-black
            rgb = np.full((H, W, 3), 180, dtype=np.uint8)
            rgb[data == 0] = (255, 255, 255)
            rgb[data == 100] = (30, 30, 30)

            extent = (ox, ox + W * res, oy, oy + H * res)
            if self._im is None:
                self._im = self.ax.imshow(rgb, origin="lower", extent=extent,
                                          interpolation="nearest")
            else:
                self._im.set_data(rgb)
                self._im.set_extent(extent)
            if not self._limits_set:
                self.ax.set_xlim(extent[0], extent[1])
                self.ax.set_ylim(extent[2], extent[3])
                self._limits_set = True
        elif not self._limits_set:
            # No map yet -- give the axes a sensible window so the dot is visible.
            self.ax.set_xlim(self.fb_extent[0], self.fb_extent[1])
            self.ax.set_ylim(self.fb_extent[2], self.fb_extent[3])
            self._limits_set = True

        if goal is not None:
            title = "current goal: (%.2f, %.2f)" % goal
        elif bev is None:
            title = "no BEV map yet -- showing drone pose only"
        else:
            title = "Left-click anywhere to set navigation goal"
        if score is not None:
            title += "   |  predicted quality: %.2f" % score
        self.ax.set_title(title)

        # Path overlays: raw A* (red, underneath) vs APF-safe path (green, on
        # top). The green path is what the drone actually flies; the red shows
        # how closely plain shortest-path A* hugged the walls before recentring.
        if self._raw_line is not None:
            self._raw_line.remove()
            self._raw_line = None
        if len(raw) >= 2:
            rxs, rys = [p[0] for p in raw], [p[1] for p in raw]
            self._raw_line, = self.ax.plot(rxs, rys, "-o", color="red",
                                           linewidth=1.6, markersize=3,
                                           alpha=0.75, zorder=2)

        if self._path_line is not None:
            self._path_line.remove()
            self._path_line = None
        if len(path) >= 2:
            xs, ys = [p[0] for p in path], [p[1] for p in path]
            self._path_line, = self.ax.plot(xs, ys, "-o", color="limegreen",
                                            linewidth=2.4, markersize=4,
                                            alpha=0.95, zorder=3)

        # Predicted trajectory overlay: the path the drone will ACTUALLY fly given
        # its stop-and-turn dynamics (orange dashed), vs the planned green path.
        if self._pred_line is not None:
            self._pred_line.remove()
            self._pred_line = None
        if len(pred) >= 2:
            pxs, pys = [p[0] for p in pred], [p[1] for p in pred]
            self._pred_line, = self.ax.plot(pxs, pys, "--", color="darkorange",
                                            linewidth=2.0, alpha=0.9, zorder=4)

        # Drone marker + heading arrow
        for artist in ("_drone_dot", "_drone_arrow"):
            a = getattr(self, artist)
            if a is not None:
                a.remove()
                setattr(self, artist, None)
        if drone is not None:
            x, y, yaw = drone
            self._drone_dot, = self.ax.plot([x], [y], "o", color="red", markersize=8,
                                            markeredgecolor="black", zorder=5)
            self._drone_arrow = self.ax.annotate(
                "", xy=(x + self.arrow_len * np.cos(yaw),
                        y + self.arrow_len * np.sin(yaw)),
                xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color="red", lw=2), zorder=5)

        # Goal marker
        if self._goal_marker is not None:
            self._goal_marker.remove()
            self._goal_marker = None
        if goal is not None:
            self._goal_marker, = self.ax.plot([goal[0]], [goal[1]], "*", color="lime",
                                              markersize=20, markeredgecolor="black",
                                              zorder=4)
        return []

    def spin(self):
        self.anim = FuncAnimation(self.fig, self._render,
                                  interval=int(1000.0 / self.refresh_hz),
                                  blit=False, cache_frame_data=False)
        try:
            plt.show()
        except KeyboardInterrupt:
            pass


def main():
    try:
        BevClickGoalNode().spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
