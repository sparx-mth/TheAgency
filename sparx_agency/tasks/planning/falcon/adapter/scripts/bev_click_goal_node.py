#!/usr/bin/env python3
"""
bev_click_goal_node.py -- interactive 2D BEV viewer with click-to-navigate.

A matplotlib window (NOT RViz) showing the published BEV grid plus the drone
pose, the raw A* path, the APF-safe (recentred) path, the predicted drone
trajectory, and the last clicked goal:
    gray = unknown (-1)   white = free (0)   black = occupied (100)
    red path     = raw A*  (shortest -- hugs walls / cuts corners)
    teal dashed  = A* global route (whole plan; stays put as a NavDP leg flies it)
    magenta path = APF-safe (recentred off walls -- before cleanup)
    green path   = cleaned/flown (simplified APF-safe path -- the path flown)
    blue dashed = full planner route (e.g. NavDP) when only its near prefix flies
    yellow arrows = repulsive force F_rep=-grad U_rep at each waypoint (obstacle push)
    gold field    = F_rep across the free space (how hard each wall section pushes)
    orange dashed = predicted (stop-and-turn)
    cyan          = smooth spline tracked by pure-pursuit (/path/smooth)
    blueviolet X  = pure-pursuit lookahead aim point (/path/lookahead)
The title shows the predicted-trajectory quality score (0..1) when available.

A SECOND window -- "drone thinking" -- shows the drone's reasoning: the running
narration every nav node publishes to /nav/thinking explaining why it is doing
what it is doing ("Stopping to turn", "Reached waypoint 3, heading for waypoint
4", "Replanning: obstacle on route", "Lost the chair from frame, searching").
Newest line last, warnings in orange and hard stops in red; a line thought
repeatedly collapses to "... (x3)" rather than flushing the reasoning that
explains it off the top. It is a separate window so the map keeps its whole
canvas and the log can be moved, resized or closed on its own -- closing it
leaves the map (and the drone) running. ~thinking_lines sets how many lines it
holds.

That window is OFF by default (~thinking_window:=true opens it). The reasoning
is persisted to a file by thought_logger_node regardless, so the window is a
convenience for a desk-side run rather than the record -- and the drone often
flies headless on the Jetson, where a stack that tries to open a GUI has
nowhere to put it.

LEFT-CLICK anywhere publishes a geometry_msgs/Point to ~goal_topic; the planner
picks it up, replans, and the new path is drawn within one BEV update. This node
is purely a viewer + click adapter -- it does not move the drone, plan, or alter
the map.

Overlays can be toggled live with the number keys to declutter the view (the
drone dot and goal star always stay on):
    1 full route   2 raw A*    3 safe    4 flown path   5 F_rep field
    6 F_rep arrows 7 predicted 8 smooth  9 lookahead    a A* global route
    0 all on/off
Initial visibility can also be preset per layer via the ~show_<layer> params
(e.g. ~show_pred:=false to start with the predicted trajectory hidden).

Run as a sidecar in the FALCON container:
    rosrun falcon_adapter bev_click_goal_node.py
Requires matplotlib (apt-get install -y python3-matplotlib if missing).

  in   ~bev_topic  (OccupancyGrid)  /falcon/bev_2d
  in   ~path_topic (Path)           /path/waypoints      (cleaned/flown; drawn green)
  in   ~raw_path_topic (Path)       /path/waypoints_raw  (raw A*; drawn red)
  in   ~astar_path_topic (Path)     /path/waypoints_astar (A* global route; teal dashed; '' = off)
  in   ~safe_path_topic (Path)      /path/waypoints_safe (APF-safe pre-cleanup; magenta; '' = off)
  in   ~full_path_topic (Path)      /path/waypoints_navdp_full  (full route, blue dashed; '' = off)
  in   ~forces_topic (MarkerArray)  /path/forces         (F_rep; drawn yellow)
  in   ~predicted_path_topic (Path) /path/predicted
  in   ~predicted_score_topic (Float32) /path/predicted_score
  in   ~smooth_path_topic (Path)    /path/smooth    (pure-pursuit spline; cyan; '' = off)
  in   ~lookahead_topic (PointStamped) /path/lookahead (pure-pursuit aim; blueviolet X; '' = off)
  in   ~drone_ns + /gt_pose (Pose)
  in   ~pose_stamped_topic (PoseStamped, optional)  e.g. /xtend/localization
  in   ~thinking_topic (String) /nav/thinking  (drone thinking log; '' = no window)
  ~thinking_window (bool, FALSE) open the thinking window at all; off suits a
       headless run and the thought_logger file keeps the record either way
  out  ~goal_topic (Point, latched) /waypoint_nav/goal

The drone marker normally reads a bare Pose on ~drone_ns + /gt_pose (what the
rest of the nav stack publishes via pose_adapter). For standalone debugging --
e.g. a rosbag played straight through the ROS1<->ROS2 bridge with no nav stack
up -- set ~pose_stamped_topic to the raw localization (a PoseStamped such as
/xtend/localization) and the dot is drawn directly from it, even before any
BEV map is published.
"""
import threading

import numpy as np
import rospy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

from geometry_msgs.msg import Point, PointStamped, Pose, PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Float32, String
from visualization_msgs.msg import Marker, MarkerArray

from sparx_agency.core.common.math import se3
from sparx_agency.core.common.thought_log import ThoughtLog
from sparx_agency.core.common.thought_message import parse_thought_message


def _get_bool_param(name, default):
    """Read a boolean rosparam, raising on a value that is not clearly boolean.

    roslaunch coerces ``value="true"`` / ``value="false"`` to a real Python bool
    but leaves an UNRECOGNISED string (the typo ``"fales"``) as a raw string --
    and ``bool("fales")`` is ``True``. A plain cast therefore looks like
    sanitisation while silently flipping a default-OFF flag ON, which for a
    window flag means a headless Jetson trying to open a GUI. Raise instead.

    (Also defined in astar_planner_node / trajectory_simplifier_node; worth
    hoisting into a shared adapter helper next time one of them is touched.)
    """
    value = rospy.get_param(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise ValueError(
        "%s must be a boolean (true/false), got %r -- refusing to guess"
        % (name, value))

# Thinking-panel line colours by level: normal reasoning stays quiet, trouble the
# drone is handling reads orange, a decision it could not resolve reads red.
_THOUGHT_COLORS = {"info": "0.15", "warn": "darkorange", "error": "red"}


class BevClickGoalNode:
    def __init__(self):
        # disable_signals=True so matplotlib's main loop owns Ctrl+C
        rospy.init_node("bev_click_goal", disable_signals=True)
        G = rospy.get_param

        self.drone_ns = G("~drone_ns", "")
        self.bev_topic = G("~bev_topic", "/falcon/bev_2d")
        self.path_topic = G("~path_topic", "/path/waypoints")
        self.raw_path_topic = G("~raw_path_topic", "/path/waypoints_raw")
        # A* GLOBAL route, subscribed straight from the planner so it draws
        # independently of the arbitrated combo/corrector pipeline. In combination
        # mode the raw/safe/flown overlays follow /path/waypoints_combo, which
        # time-shares between the echoed A* route (cruise) and the live NavDP leg
        # (follow) -- so the global A* route is otherwise invisible while a NavDP
        # leg flies. Drawn teal-dashed; "" disables it.
        self.astar_path_topic = G("~astar_path_topic", "/path/waypoints_astar")
        # The corrector's safe path BEFORE the trajectory_simplifier cleans it.
        # Drawn magenta so safety-corrected vs final-flown is visible while tuning
        # the simplifier. "" disables it (no publisher -> nothing drawn anyway).
        self.safe_path_topic = G("~safe_path_topic", "/path/waypoints_safe")
        # FULL planner route (e.g. NavDP's whole trajectory when only its near
        # prefix is executed): drawn dim/dashed for reference. Defaults to NavDP's
        # full-route topic so `rosrun bev_click_goal_node.py` shows the whole route
        # out of the box; harmless for A* (no publisher -> no blue line). "" = off.
        self.full_path_topic = G("~full_path_topic", "/path/waypoints_navdp_full")
        self.forces_topic = G("~forces_topic", "/path/forces")
        self.predicted_path_topic = G("~predicted_path_topic", "/path/predicted")
        self.predicted_score_topic = G("~predicted_score_topic",
                                       "/path/predicted_score")
        # Pure-pursuit overlays: the splined trajectory it tracks (cyan) and the
        # moving lookahead point it aims at (blueviolet). "" disables either; no
        # publisher (other controllers) -> nothing drawn anyway.
        self.smooth_path_topic = G("~smooth_path_topic", "/path/smooth")
        self.lookahead_topic = G("~lookahead_topic", "/path/lookahead")
        self.goal_topic = G("~goal_topic", "/waypoint_nav/goal")
        self.pose_stamped_topic = G("~pose_stamped_topic", "")
        # ── System-status HUD sources (top-left text box) ────────────
        # who is planning now (A*/NavDP, published by the hybrid arbiter),
        self.nav_status_topic = G("~nav_status_topic", "/nav/status")
        # the nav_mode this run was launched with (context when no arbiter publishes),
        self.nav_mode = str(G("~nav_mode", ""))
        # the drone command, to show forward-flight vs rotation-in-place live,
        self.cmd_vel_topic = G("~cmd_vel_topic", self.drone_ns + "/cmd_vel")
        # and A* replan events (first route / obstacle reroute / boxed STOP / shorter).
        self.astar_event_topic = G("~astar_event_topic", "/path/astar_event")
        # ── Drone thinking log (its own window) ──────────────────────
        # Every nav node narrates its decisions here; see thinking.py.
        # OFF by default: the reasoning is recorded to a file by thought_logger
        # (see thought_journal.py), so the window is a convenience for a
        # desk-side run, not the record -- and on a headless Jetson a stack that
        # tries to open one has nowhere to put it. Set ~thinking_window:=true
        # for the live view.
        self.thinking_topic = G("~thinking_topic", "/nav/thinking")
        self.thinking_window = _get_bool_param("~thinking_window", False)
        self.thinking_lines = int(G("~thinking_lines", 8))
        # Only the window needs the stream; with it off, do not subscribe at all.
        self.show_thinking = bool(self.thinking_topic) and self.thinking_window
        if self.show_thinking and self.thinking_lines < 1:
            raise ValueError(
                "~thinking_lines must be >= 1, got %d. To turn the thinking "
                "window off, set ~thinking_window:=false instead."
                % self.thinking_lines)
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
        self._path_xy = []              # cleaned, flown path (green)
        self._raw_xy = []               # raw planner path (red) -- shortest, viz only
        self._astar_xy = []             # A* GLOBAL route (teal) -- independent overlay
        self._safe_xy = []              # corrector's safe path (magenta) -- pre-cleanup
        self._full_xy = []              # full planner route (blue dashed) -- ref only
        self._forces = []               # (x,y,dx,dy) per-waypoint force arrows (yellow)
        self._field_arrows = []         # (x,y,dx,dy) coarse wall force-field (gold)
        self._pred_xy = []              # predicted drone trajectory (rollout)
        self._pred_score = None         # 0..1 "how good" score
        self._smooth_xy = []            # pure-pursuit splined trajectory (cyan)
        self._lookahead_xy = None       # pure-pursuit lookahead aim point (blueviolet)
        self._drone_p = None            # (x, y, yaw)
        self._goal_xy = None
        # System status (HUD): who plans, what the drone is doing, last A* event.
        self._nav_status = None         # "A*" / "NavDP (reason)" (from ~nav_status_topic)
        self._motion = None             # "FORWARD" / "ROTATING" / "HOLD" (from cmd_vel)
        self._astar_event = None        # last A* replan event text
        self._astar_event_t = None      # rospy.Time it arrived (for an age readout)
        # Drone thinking: the rolling narration drawn in the panel under the map.
        self._thoughts = ThoughtLog(capacity=max(1, self.thinking_lines))
        self._thought_t0 = None         # stamp of the first thought (for "+12.3s")
        self._thought_drops = 0         # malformed messages seen (heartbeat only)
        self._lock = threading.Lock()

        # Live per-layer visibility. Each maps a number key to an overlay; a
        # layer that is off is simply not recreated in _render, so its artist
        # disappears on the next frame. Defaults come from ~show_<layer> so a
        # launch file can start with a cluttered overlay (e.g. pred) hidden.
        self._keymap = {"1": "full", "2": "raw", "3": "safe", "4": "path",
                        "5": "field", "6": "forces", "7": "pred",
                        "8": "smooth", "9": "lookahead", "a": "astar"}
        # Declutter by default: show only the three overlays that matter for watching
        # the A*/NavDP hybrid -- [a] A* global route, [1] NavDP full leg, [4] flown
        # path -- and hide the rest. Any layer is still toggled live with its key, or
        # forced on/off from a launch file via ~show_<layer>.
        _default_on = {"astar", "full", "path"}
        self._show = {layer: bool(G("~show_" + layer, layer in _default_on))
                      for layer in self._keymap.values()}

        self.goal_pub = rospy.Publisher(self.goal_topic, Point, queue_size=1, latch=True)
        rospy.Subscriber(self.bev_topic, OccupancyGrid, self._bev_cb, queue_size=1)
        rospy.Subscriber(self.path_topic, Path, self._path_cb, queue_size=1)
        rospy.Subscriber(self.raw_path_topic, Path, self._raw_path_cb, queue_size=1)
        if self.astar_path_topic:
            rospy.Subscriber(self.astar_path_topic, Path, self._astar_path_cb,
                             queue_size=1)
        if self.safe_path_topic:
            rospy.Subscriber(self.safe_path_topic, Path, self._safe_path_cb, queue_size=1)
        if self.full_path_topic:
            rospy.Subscriber(self.full_path_topic, Path, self._full_path_cb, queue_size=1)
        rospy.Subscriber(self.forces_topic, MarkerArray, self._forces_cb, queue_size=1)
        rospy.Subscriber(self.predicted_path_topic, Path, self._pred_cb, queue_size=1)
        rospy.Subscriber(self.predicted_score_topic, Float32, self._pred_score_cb,
                         queue_size=1)
        if self.smooth_path_topic:
            rospy.Subscriber(self.smooth_path_topic, Path, self._smooth_cb, queue_size=1)
        if self.lookahead_topic:
            rospy.Subscriber(self.lookahead_topic, PointStamped, self._lookahead_cb,
                             queue_size=1)
        rospy.Subscriber(self.drone_ns + "/gt_pose", Pose, self._pose_cb, queue_size=10)
        if self.pose_stamped_topic:
            rospy.Subscriber(self.pose_stamped_topic, PoseStamped,
                             self._pose_stamped_cb, queue_size=10)
        if self.nav_status_topic:
            rospy.Subscriber(self.nav_status_topic, String, self._nav_status_cb,
                             queue_size=1)
        if self.cmd_vel_topic:
            rospy.Subscriber(self.cmd_vel_topic, Twist, self._cmd_vel_cb, queue_size=1)
        if self.astar_event_topic:
            rospy.Subscriber(self.astar_event_topic, String, self._astar_event_cb,
                             queue_size=5)
        if self.show_thinking:
            rospy.Subscriber(self.thinking_topic, String, self._thinking_cb,
                             queue_size=20)

        # The map owns its window outright. The thinking log gets a SEPARATE window
        # rather than a panel in this figure: sharing one figure costs the map real
        # estate on any fixed-height screen (the map shrinks by whatever the log
        # takes), and the operator can move, resize or close the log independently
        # of the map they are actually flying from.
        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        self.ax_log = self.fig_log = None      # stay None when the log is off
        if self.show_thinking:
            self._init_thinking_window()
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect(
            "close_event",
            lambda _e: rospy.signal_shutdown("bev_click_goal window closed"))
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x (m)")
        self.ax.set_ylabel("y (m)")
        self.ax.grid(True, alpha=0.25)
        self._add_legend()

        # System-status HUD (top-left): who is planning, what the drone is doing,
        # and the last A* replan event. Created once; _render updates its text.
        self._hud = self.ax.text(
            0.015, 0.985, "", transform=self.ax.transAxes, va="top", ha="left",
            fontsize=9, family="monospace", zorder=21,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85,
                      edgecolor="0.5"))

        # Persistent artists (created lazily, updated in place)
        self._im = self._raw_line = self._path_line = self._pred_line = None
        self._astar_line = None          # A* global route (teal dashed)
        self._safe_line = None           # corrector safe path (magenta)
        self._full_line = None           # full planner route (blue dashed)
        self._smooth_line = None         # pure-pursuit splined trajectory (cyan)
        self._lookahead_marker = None    # pure-pursuit lookahead point (blueviolet)
        self._forces_artist = None      # quiver of per-waypoint force arrows
        self._field_artist = None       # quiver of the coarse wall force-field
        self._drone_dot = self._drone_arrow = self._goal_marker = None
        self._limits_set = False        # axes window fixed on first render

        rospy.loginfo("=" * 64)
        rospy.loginfo("bev_click_goal: ready")
        rospy.loginfo("  bev  in  = %s", self.bev_topic)
        rospy.loginfo("  path in  = %s   (APF-safe, green)", self.path_topic)
        rospy.loginfo("  raw  in  = %s   (raw A*, red)", self.raw_path_topic)
        if self.astar_path_topic:
            rospy.loginfo("  astar in = %s   (A* global route, teal)",
                          self.astar_path_topic)
        if self.safe_path_topic:
            rospy.loginfo("  safe in  = %s   (corrector safe, magenta)",
                          self.safe_path_topic)
        if self.full_path_topic:
            rospy.loginfo("  full in  = %s   (full route, blue dashed)",
                          self.full_path_topic)
        rospy.loginfo("  force in = %s   (F_rep arrows, yellow)", self.forces_topic)
        rospy.loginfo("  pose in  = %s/gt_pose", self.drone_ns)
        if self.pose_stamped_topic:
            rospy.loginfo("  pose in  = %s   (PoseStamped, direct)",
                          self.pose_stamped_topic)
        rospy.loginfo("  goal out = %s   (left-click to publish)", self.goal_topic)
        rospy.loginfo("  HUD in   = %s (planner) + %s (motion) + %s (A* events)",
                      self.nav_status_topic, self.cmd_vel_topic, self.astar_event_topic)
        rospy.loginfo("  think in = %s   (%s)",
                      self.thinking_topic if self.show_thinking else "(disabled)",
                      "%d-line log in its own window" % self.thinking_lines
                      if self.show_thinking
                      else "no window; thought_logger still records to file")
        rospy.loginfo("  overlays = default ON: [a] A* route, [1] NavDP leg, [4] flown "
                      "path; others OFF")
        rospy.loginfo("  toggles  = keys 1-9/a per overlay, 0 = all on/off")
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

    def _astar_path_cb(self, msg):
        pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        with self._lock:
            self._astar_xy = pts

    def _safe_path_cb(self, msg):
        pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        with self._lock:
            self._safe_xy = pts

    def _full_path_cb(self, msg):
        pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        with self._lock:
            self._full_xy = pts

    def _forces_cb(self, msg):
        """Split ARROW markers into per-waypoint (ns 'apf_forces') and the coarse
        wall force-field (ns 'apf_field'); each as (x, y, dx, dy) start->tip."""
        wp, field = [], []
        for m in msg.markers:
            if m.action == Marker.DELETEALL or len(m.points) < 2:
                continue
            s, e = m.points[0], m.points[1]
            arrow = (s.x, s.y, e.x - s.x, e.y - s.y)
            (field if m.ns == "apf_field" else wp).append(arrow)
        with self._lock:
            self._forces = wp
            self._field_arrows = field

    def _pred_cb(self, msg):
        pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        with self._lock:
            self._pred_xy = pts

    def _pred_score_cb(self, msg):
        with self._lock:
            self._pred_score = float(msg.data)

    def _smooth_cb(self, msg):
        pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        with self._lock:
            self._smooth_xy = pts

    def _lookahead_cb(self, msg):
        with self._lock:
            self._lookahead_xy = (msg.point.x, msg.point.y)

    def _pose_cb(self, msg):
        o = msg.orientation
        yaw = se3.yaw_from_quaternion((o.x, o.y, o.z, o.w))
        with self._lock:
            self._drone_p = (msg.position.x, msg.position.y, yaw)

    def _pose_stamped_cb(self, msg):
        self._pose_cb(msg.pose)

    # -- system status (HUD) --------------------------------------------------
    def _nav_status_cb(self, msg):
        with self._lock:
            self._nav_status = msg.data

    def _astar_event_cb(self, msg):
        with self._lock:
            self._astar_event = msg.data
            self._astar_event_t = rospy.Time.now()

    def _thinking_cb(self, msg):
        """Record one narrated thought for the log panel.

        A malformed payload is dropped rather than raised: one node publishing
        junk must not take down the operator's view of every other node's
        reasoning. It is not silent either -- each drop is warned (throttled) with
        the reason, and counted for introspection.
        """
        try:
            thought = parse_thought_message(msg.data, default_stamp=self._now())
        except ValueError as e:
            with self._lock:
                self._thought_drops += 1
            rospy.logwarn_throttle(10.0, "bev_click_goal: dropping malformed "
                                         "thought on %s: %s", self.thinking_topic, e)
            return
        with self._lock:
            if self._thought_t0 is None:
                self._thought_t0 = thought.stamp
            self._thoughts.add(thought)

    @staticmethod
    def _now():
        """Wall/sim seconds, or 0.0 before the clock is up (bag not yet playing)."""
        return rospy.Time.now().to_sec()

    def _cmd_vel_cb(self, msg):
        """Classify the live command into forward-flight vs rotation-in-place so the
        HUD shows what the drone is doing right now (the one-axis follower does one
        or the other; a holonomic controller may do both)."""
        vx, vy, wz = msg.linear.x, msg.linear.y, msg.angular.z
        speed = (vx * vx + vy * vy) ** 0.5
        turning = abs(wz) >= 0.03            # rad/s
        moving = speed >= 0.03               # m/s
        if not moving and not turning:
            motion = "HOLD (stopped)"
        elif turning and not moving:
            motion = "ROTATING (turn in place)"
        elif moving and not turning:
            motion = "FORWARD FLIGHT"
        else:
            motion = "FORWARD + TURN"
        with self._lock:
            self._motion = motion

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
            self._path_xy = []          # drop the stale flown path; replan redraws it
            self._raw_xy = []           # and the stale raw planner path
            self._astar_xy = []         # and the stale A* global route
            self._safe_xy = []          # and the stale corrector safe path
            self._full_xy = []          # and the stale full planner route
            self._forces = []           # and the stale force arrows
            self._field_arrows = []     # and the stale wall force-field
            self._smooth_xy = []        # and the stale splined trajectory
            self._lookahead_xy = None   # and the stale lookahead point
        m = Point()
        m.x, m.y, m.z = gx, gy, 0.0
        self.goal_pub.publish(m)

    # -- keyboard toggles ------------------------------------------------------
    def _on_key(self, event):
        """Toggle overlay visibility live: number keys 1-9 flip one layer, 0
        flips all of them (off if any are on, else all back on).

        The numeric keypad reports its digits as ``kp_1`` .. ``kp_9`` (backend
        /NumLock dependent) rather than bare ``1`` .. ``9``, so strip that
        prefix first to accept both the top-row and right-hand number pads.
        """
        key = event.key or ""
        if key.lower().startswith("kp_"):
            key = key[3:]
        if key == "0":
            new = not any(self._show.values())
            for layer in self._show:
                self._show[layer] = new
            rospy.loginfo("bev_click_goal: all overlays %s",
                          "on" if new else "off")
            return
        layer = self._keymap.get(key)
        if layer is None:
            return
        self._show[layer] = not self._show[layer]
        rospy.loginfo("bev_click_goal: [%s] %s -> %s", key, layer,
                      "on" if self._show[layer] else "off")

    # -- legend ----------------------------------------------------------------
    def _add_legend(self):
        """A static colour key for every route/marker the viewer draws.

        Proxy ``Line2D`` handles (the real artists are recreated each frame, so
        they can't anchor a legend). Colours/styles mirror ``_render`` exactly:
        the three planner-pipeline paths (raw planner -> safe corrector ->
        cleaned/flown), the display-only routes, the repulsive-force overlays,
        and the drone/goal markers. Drawn once; it persists because ``_render``
        only updates artists in place and never clears the axes.
        """
        handles = [
            Line2D([0], [0], color="deepskyblue", marker="o", markersize=3,
                   linestyle="--", label="[1] full planner route (NavDP, display-only)"),
            Line2D([0], [0], color="red", marker="o", markersize=4,
                   label="[2] raw planner path (A*/NavDP, /path/waypoints_raw)"),
            Line2D([0], [0], color="magenta", marker="o", markersize=4,
                   label="[3] safe: corrector, pre-cleanup (/path/waypoints_safe)"),
            Line2D([0], [0], color="limegreen", marker="o", markersize=5, lw=2.4,
                   label="[4] cleaned / FLOWN path (/path/waypoints)"),
            Line2D([0], [0], color="gold", marker=">", linestyle="None",
                   label="[5] F_rep field over free space"),
            Line2D([0], [0], color="yellow", marker=">", markeredgecolor="black",
                   linestyle="None", label="[6] F_rep at each waypoint (obstacle push)"),
            Line2D([0], [0], color="darkorange", linestyle="--", lw=2,
                   label="[7] predicted stop-and-turn trajectory"),
            Line2D([0], [0], color="cyan", lw=2,
                   label="[8] smooth spline (pure-pursuit, /path/smooth)"),
            Line2D([0], [0], color="blueviolet", marker="X", markeredgecolor="black",
                   markersize=10, linestyle="None",
                   label="[9] pure-pursuit lookahead (/path/lookahead)"),
            Line2D([0], [0], color="teal", marker="s", markersize=4, linestyle="--",
                   label="[a] A* global route (/path/waypoints_astar)"),
            Line2D([0], [0], color="red", marker="o", markeredgecolor="black",
                   markersize=8, linestyle="None", label="drone pose + heading"),
            Line2D([0], [0], color="lime", marker="*", markeredgecolor="black",
                   markersize=12, linestyle="None", label="navigation goal"),
        ]
        self.ax.legend(handles=handles, loc="upper right", fontsize=7,
                       framealpha=0.85, ncol=1,
                       title="routes & markers  (keys 1-9/a toggle, 0 = all)"
                       ).set_zorder(20)

    # -- thinking window -------------------------------------------------------
    def _init_thinking_window(self):
        """Build the drone-thinking log's own window, beside the map's.

        A second figure, so the map keeps its whole window and the operator can
        move/resize/close this one independently. One text artist per line,
        created once and updated in place, so a busy log costs no artist churn
        per frame; the row count is fixed, which keeps the window's layout stable
        rather than jumping as thoughts arrive.
        """
        self.fig_log = plt.figure("drone thinking  (%s)" % self.thinking_topic,
                                  figsize=(11, 0.34 * self.thinking_lines + 0.5))
        self.ax_log = self.fig_log.add_axes([0.0, 0.0, 1.0, 1.0])
        self.ax_log.set_xticks([])
        self.ax_log.set_yticks([])
        for side in self.ax_log.spines.values():
            side.set_visible(False)
        # Closing the log must NOT take the node (or the map) down with it -- an
        # operator who declutters their screen is not asking to stop flying. Drop
        # the reference so _render stops touching a dead canvas.
        self.fig_log.canvas.mpl_connect("close_event", self._on_log_close)
        self._log_rows = [
            self.ax_log.text(0.008, 1.0 - (i + 0.6) / self.thinking_lines, "",
                             transform=self.ax_log.transAxes, va="center",
                             ha="left", fontsize=9, family="monospace")
            for i in range(self.thinking_lines)
        ]
        self._log_rows[0].set_text("(waiting for the drone's first thought)")
        self._log_rows[0].set_color("0.55")

    def _on_log_close(self, _evt):
        """The operator closed the thinking window; keep flying without it."""
        self.ax_log = None
        self.fig_log = None
        rospy.loginfo("bev_click_goal: thinking window closed; map keeps running")

    def _render_thoughts(self):
        """Refresh the thinking window: oldest at the top, newest at the bottom.

        Bottom-anchored like a terminal tail, so the drone's latest thought is
        always on the same line rather than wandering as the log fills. The
        newest line is bold -- with a static screenshot of a flight, that is the
        one an operator needs to find first.

        Driven from the map's animation rather than a second FuncAnimation: one
        timer keeps the two windows showing the same instant, and the log's canvas
        is redrawn explicitly here because an animation only redraws its own
        figure.
        """
        # Format inside the lock: entries() hands back LIVE entries, and a thought
        # arriving on the subscriber thread mutates the newest one in place as it
        # collapses a repeat. Rendering straight off them would read a half-updated
        # entry. Copy out finished strings instead, and hold the lock only for that.
        with self._lock:
            entries = self._thoughts.entries(limit=self.thinking_lines)
            # `t0 or ...` would be wrong here: the first thought's stamp is
            # legitimately 0.0 under a fresh sim clock, and falsy 0.0 would
            # re-baseline every line against itself -- printing +0.0s all flight.
            base = self._thought_t0 if self._thought_t0 is not None else 0.0
            lines = [("%+7.1fs  %-18.18s  %s"
                      % (e.thought.stamp - base, e.thought.source,
                         e.display_text()),
                      _THOUGHT_COLORS.get(e.thought.level, "0.15"))
                     for e in entries]
        if not lines:
            return
        pad = self.thinking_lines - len(lines)        # bottom-anchor the tail
        for i, row in enumerate(self._log_rows):
            if i < pad:
                row.set_text("")
                continue
            text, color = lines[i - pad]
            row.set_text(text)
            row.set_color(color)
            row.set_fontweight("bold" if i == self.thinking_lines - 1 else "normal")
        self.fig_log.canvas.draw_idle()

    # -- render (main thread, via FuncAnimation) ------------------------------
    def _render(self, _frame):
        with self._lock:
            bev, path = self._bev, list(self._path_xy)
            raw = list(self._raw_xy)
            astar = list(self._astar_xy)
            safe = list(self._safe_xy)
            full = list(self._full_xy)
            forces = list(self._forces)
            field_arrows = list(self._field_arrows)
            drone, goal = self._drone_p, self._goal_xy
            pred, score = list(self._pred_xy), self._pred_score
            smooth = list(self._smooth_xy)
            lookahead = self._lookahead_xy
            nav_status, motion = self._nav_status, self._motion
            event, event_t = self._astar_event, self._astar_event_t

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

        # Full planner route (blue dashed, bottommost): the WHOLE route the planner
        # proposed when only its near prefix is executed (e.g. NavDP). Shown for
        # reference -- the drone flies the green path and stops at its end.
        if self._full_line is not None:
            self._full_line.remove()
            self._full_line = None
        if self._show["full"] and len(full) >= 2:
            fxs, fys = [p[0] for p in full], [p[1] for p in full]
            self._full_line, = self.ax.plot(fxs, fys, "--o", color="deepskyblue",
                                            linewidth=1.4, markersize=2,
                                            alpha=0.6, zorder=1)

        # A* GLOBAL route (teal dashed): the whole planned route to the goal,
        # straight from astar_planner. In combination mode it stays put while the
        # NavDP leg (raw/green + blue-dashed full) flies along it, so the global
        # plan and the current NavDP leg are both visible at once. In plain A*
        # mode it coincides with the red raw overlay (same route pre-correction).
        if self._astar_line is not None:
            self._astar_line.remove()
            self._astar_line = None
        if self._show["astar"] and len(astar) >= 2:
            axs, ays = [p[0] for p in astar], [p[1] for p in astar]
            self._astar_line, = self.ax.plot(axs, ays, "--s", color="teal",
                                             linewidth=1.4, markersize=3,
                                             alpha=0.7, zorder=1.5)

        # Path overlays: raw A* (red, underneath) vs APF-safe path (green, on
        # top). The green path is what the drone actually flies; the red shows
        # how closely plain shortest-path A* hugged the walls before recentring.
        if self._raw_line is not None:
            self._raw_line.remove()
            self._raw_line = None
        if self._show["raw"] and len(raw) >= 2:
            rxs, rys = [p[0] for p in raw], [p[1] for p in raw]
            self._raw_line, = self.ax.plot(rxs, rys, "-o", color="red",
                                           linewidth=1.6, markersize=3,
                                           alpha=0.75, zorder=2)

        # Corrector's safe path (magenta), BEFORE the trajectory_simplifier cleans
        # it: shows what the cleanup removed/smoothed vs the green flown path.
        if self._safe_line is not None:
            self._safe_line.remove()
            self._safe_line = None
        if self._show["safe"] and len(safe) >= 2:
            sxs, sys = [p[0] for p in safe], [p[1] for p in safe]
            self._safe_line, = self.ax.plot(sxs, sys, "-o", color="magenta",
                                            linewidth=1.8, markersize=3,
                                            alpha=0.7, zorder=2)

        if self._path_line is not None:
            self._path_line.remove()
            self._path_line = None
        if self._show["path"] and len(path) >= 2:
            xs, ys = [p[0] for p in path], [p[1] for p in path]
            self._path_line, = self.ax.plot(xs, ys, "-o", color="limegreen",
                                            linewidth=2.4, markersize=4,
                                            alpha=0.95, zorder=3)

        # Coarse wall force-field (dim gold, thin): F_rep sampled across the free
        # space, so every wall section's push is visible -- dense/strong by the
        # walls, fading to nothing in open space.
        if self._field_artist is not None:
            self._field_artist.remove()
            self._field_artist = None
        if self._show["field"] and field_arrows:
            qx = [a[0] for a in field_arrows]
            qy = [a[1] for a in field_arrows]
            qu = [a[2] for a in field_arrows]
            qv = [a[3] for a in field_arrows]
            self._field_artist = self.ax.quiver(
                qx, qy, qu, qv, color="gold", angles="xy", scale_units="xy",
                scale=1.0, width=0.003, alpha=0.55, zorder=5)

        # Per-waypoint repulsive-force arrows (F_rep = -grad U_rep): bright yellow,
        # length proportional to force magnitude, pointing away from the walls.
        # Drawn in data units so they match the RViz MarkerArray arrows exactly.
        if self._forces_artist is not None:
            self._forces_artist.remove()
            self._forces_artist = None
        if self._show["forces"] and forces:
            fx = [a[0] for a in forces]
            fy = [a[1] for a in forces]
            fu = [a[2] for a in forces]
            fv = [a[3] for a in forces]
            self._forces_artist = self.ax.quiver(
                fx, fy, fu, fv, color="yellow", angles="xy",
                scale_units="xy", scale=1.0, width=0.006,
                edgecolor="black", linewidth=0.4, zorder=6)

        # Predicted trajectory overlay: the path the drone will ACTUALLY fly given
        # its stop-and-turn dynamics (orange dashed), vs the planned green path.
        if self._pred_line is not None:
            self._pred_line.remove()
            self._pred_line = None
        if self._show["pred"] and len(pred) >= 2:
            pxs, pys = [p[0] for p in pred], [p[1] for p in pred]
            self._pred_line, = self.ax.plot(pxs, pys, "--", color="darkorange",
                                            linewidth=2.0, alpha=0.9, zorder=4)

        # Pure-pursuit smooth (splined) trajectory the tracker follows (cyan): the
        # continuous path Pure Pursuit chases via its moving lookahead, vs the
        # piecewise green planner path it was splined from.
        if self._smooth_line is not None:
            self._smooth_line.remove()
            self._smooth_line = None
        if self._show["smooth"] and len(smooth) >= 2:
            mxs, mys = [p[0] for p in smooth], [p[1] for p in smooth]
            self._smooth_line, = self.ax.plot(mxs, mys, "-", color="cyan",
                                              linewidth=1.8, alpha=0.85, zorder=4)

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

        # Pure-pursuit lookahead point (blueviolet X, topmost): the moving target
        # on the smooth path the tracker is steering toward this instant.
        if self._lookahead_marker is not None:
            self._lookahead_marker.remove()
            self._lookahead_marker = None
        if self._show["lookahead"] and lookahead is not None:
            self._lookahead_marker, = self.ax.plot(
                [lookahead[0]], [lookahead[1]], "X", color="blueviolet",
                markersize=13, markeredgecolor="black", zorder=7)

        # System-status HUD (top-left)
        mode = self.nav_mode.upper() if self.nav_mode else "?"
        lines = ["mode: %s    planner: %s" % (mode, nav_status or "(awaiting status)"),
                 "motion: %s" % (motion or "(no /cmd_vel yet)")]
        if event:
            age = ("  (%.0fs ago)" % (rospy.Time.now() - event_t).to_sec()
                   if event_t is not None else "")
            lines.append("A*: %s%s" % (event, age))
        self._hud.set_text("\n".join(lines))

        # Drone thinking log (panel under the map)
        if self.ax_log is not None:
            self._render_thoughts()
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
