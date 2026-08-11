#!/usr/bin/env python3
"""nav_debug_recorder_node.py -- record a FALCON run for offline visual replay.

The per-tick certainty CSV (``certainty_log.py``, written by the flight node)
already captures pose, both command sets, drift and localization quality. What it
CANNOT hold is the BEV map and the routes -- large, changing arrays -- and the
planner's replan reasons. This node fills exactly that gap, WITHOUT touching any
flight-control code: it subscribes to the map, the three route layers and the
replan/blockage events and writes them, timestamped, into a per-run folder next
to the CSV, so :mod:`sparx_agency.tasks.planning.nav_debug.run_folder_nav_debug`
can replay the whole run frame by frame.

Written per the thought-journal/certainty-log conventions: flushed per line,
lands in ``$FALCON_LOG_DIR`` (the host bind-mount) so it survives the ``--rm``
container, and never takes the flight down over a diagnostic (every write is
guarded). BEV grids are de-duplicated (saved only when the map actually changes,
and no faster than ``~bev_min_interval_s``) so a minutes-long flight stays small.

Run folder layout (read by the offline player):
  manifest.json          run metadata + the certainty CSV path (for auto-pairing)
  telemetry.jsonl        {t,x,y,z,yaw, vx,vy,vz,wz}  -- pose + OUR command, per tick
  bev/<ms>.npy(+.json)   int8 occupancy grid + geometry sidecar (on change)
  bev_conf/<ms>.npy      int8 0..100 confidence, co-registered with bev/<ms>
  routes/<ms>.json       {astar,safe,final,goal,lookahead}  -- world xy (on change)
  events.jsonl           {t,wall,kind,text,x?,y?}  -- replan reasons + blockages

Python 3.8 compatible (runs in the FALCON Noetic container). See the rosparam
block at the bottom of the file.
"""
import json
import os
import time

import numpy as np
import rospy
import tf.transformations as tft
from geometry_msgs.msg import Point, PointStamped, Pose, Twist
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import String

from thought_journal import LOG_DIR_ENV     # sibling: shared $FALCON_LOG_DIR resolution


def _default_out_dir(now=None):
    # run_falcon.sh sets FALCON_RUN_DIR to the shared, single-stamp run folder so
    # the thought journal, the certainty CSV and this recording all land together.
    # Honour it; fall back to our own timestamped folder when run standalone.
    run = os.environ.get("FALCON_RUN_DIR")
    if run:
        return run
    base = (os.environ.get(LOG_DIR_ENV)
            or os.path.join(os.path.expanduser("~"), ".ros", "falcon"))
    stamp = time.strftime("%Y%m%d_%H%M%S", now or time.localtime())
    return os.path.join(base, "nav_debug_%s" % stamp)


def _classify(text):
    """Coarse replan bucket (kept in sync with nav_debug.session.classify_event)."""
    t = (text or "").lower()
    if "boxed in" in t:
        return "boxed_in"
    if "blockage" in t or "unseen obstacle" in t:
        return "blockage"
    if "obstacle on route" in t or "collision" in t:
        return "obstacle"
    if "rotat" in t:
        return "rotation"
    if "periodic" in t:
        return "time"
    return "info"


class NavDebugRecorder(object):
    def __init__(self):
        rospy.init_node("nav_debug_recorder")
        G = rospy.get_param

        self.out_dir = str(G("~out_dir", "") or _default_out_dir())
        self.record_hz = float(G("~record_hz", 15.0))
        self.bev_min_interval_s = float(G("~bev_min_interval_s", 0.5))

        self._latest_conf = None                    # latest confidence grid, paired to bev
        self._latest_cmd = (0.0, 0.0, 0.0, 0.0)     # vx, vy, vz, wz
        self._routes = {"astar": None, "safe": None, "final": None,
                        "goal": None, "lookahead": None}
        self._last_bev_bytes = None
        self._last_bev_save = rospy.Time(0)
        self._last_routes_key = None
        self._last_tel = rospy.Time(0)
        self._n = {"tel": 0, "bev": 0, "routes": 0, "events": 0}

        for sub in ("bev", "bev_conf", "routes"):
            _mkdir(os.path.join(self.out_dir, sub))
        self._tel = _open_append(os.path.join(self.out_dir, "telemetry.jsonl"))
        self._events = _open_append(os.path.join(self.out_dir, "events.jsonl"))
        self._write_manifest(G)

        rospy.Subscriber(G("~pose_topic", "/simple_drone/gt_pose"), Pose,
                         self._pose_cb, queue_size=10)
        rospy.Subscriber(G("~cmd_topic", "/simple_drone/cmd_vel"), Twist,
                         self._cmd_cb, queue_size=10)
        rospy.Subscriber(G("~bev_topic", "/falcon/bev_2d"), OccupancyGrid,
                         self._bev_cb, queue_size=1)
        rospy.Subscriber(G("~bev_conf_topic", "/falcon/bev_2d_conf"), OccupancyGrid,
                         self._conf_cb, queue_size=1)
        rospy.Subscriber(G("~astar_topic", "/path/waypoints_astar"), Path,
                         lambda m: self._route_cb("astar", m), queue_size=1)
        rospy.Subscriber(G("~safe_topic", "/path/waypoints_safe"), Path,
                         lambda m: self._route_cb("safe", m), queue_size=1)
        rospy.Subscriber(G("~final_topic", "/path/waypoints"), Path,
                         lambda m: self._route_cb("final", m), queue_size=1)
        rospy.Subscriber(G("~lookahead_topic", "/path/lookahead"), PointStamped,
                         self._lookahead_cb, queue_size=1)
        rospy.Subscriber(G("~goal_topic", "/waypoint_nav/goal"), Point,
                         self._goal_cb, queue_size=1)
        rospy.Subscriber(G("~astar_event_topic", "/path/astar_event"), String,
                         self._event_cb, queue_size=5)
        rospy.Subscriber(G("~blockage_topic", "/falcon/blockage"), PointStamped,
                         self._blockage_cb, queue_size=5)

        rospy.Timer(rospy.Duration(10.0), self._heartbeat)
        rospy.on_shutdown(self._close)
        rospy.loginfo("nav_debug_recorder: writing run -> %s (@ %.1f Hz)",
                      self.out_dir, self.record_hz)

    # ── telemetry (pose + our command) ───────────────────────────────────────
    def _cmd_cb(self, msg):
        self._latest_cmd = (msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.z)

    def _pose_cb(self, msg):
        now = rospy.Time.now()
        if self.record_hz > 0 and (now - self._last_tel).to_sec() < 1.0 / self.record_hz:
            return
        self._last_tel = now
        q = msg.orientation
        yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        vx, vy, vz, wz = self._latest_cmd
        self._emit(self._tel, {
            "t": round(now.to_sec(), 3),
            "x": round(msg.position.x, 4), "y": round(msg.position.y, 4),
            "z": round(msg.position.z, 4), "yaw": round(yaw, 5),
            "vx": round(vx, 4), "vy": round(vy, 4),
            "vz": round(vz, 4), "wz": round(wz, 4)}, "tel")

    # ── BEV map (+ confidence), de-duplicated ────────────────────────────────
    def _conf_cb(self, msg):
        self._latest_conf = _grid_from(msg)

    def _bev_cb(self, msg):
        now = rospy.Time.now()
        grid = _grid_from(msg)
        raw = grid.tobytes()
        if raw == self._last_bev_bytes:
            return                                   # unchanged map -> skip
        if (now - self._last_bev_save).to_sec() < self.bev_min_interval_s:
            return                                   # rate-limit map churn
        self._last_bev_bytes = raw
        self._last_bev_save = now
        name = "%d" % int(round(now.to_sec() * 1000.0))
        info = msg.info
        try:
            np.save(os.path.join(self.out_dir, "bev", name + ".npy"), grid)
            _write_json(os.path.join(self.out_dir, "bev", name + ".json"), {
                "t": round(now.to_sec(), 3), "resolution": info.resolution,
                "origin_x": info.origin.position.x, "origin_y": info.origin.position.y,
                "width": info.width, "height": info.height,
                "frame_id": msg.header.frame_id or "world"})
            if self._latest_conf is not None and self._latest_conf.shape == grid.shape:
                np.save(os.path.join(self.out_dir, "bev_conf", name + ".npy"),
                        self._latest_conf)
            self._n["bev"] += 1
        except (OSError, IOError) as e:
            rospy.logwarn_throttle(10.0, "nav_debug_recorder: BEV save failed (%s)", e)

    # ── routes (raw A* / corrected / final) + goal + aim point ───────────────
    def _route_cb(self, key, msg):
        self._routes[key] = [[round(p.pose.position.x, 3), round(p.pose.position.y, 3)]
                             for p in msg.poses]
        self._maybe_write_routes()

    def _goal_cb(self, msg):
        self._routes["goal"] = [round(msg.x, 3), round(msg.y, 3)]
        self._maybe_write_routes()

    def _lookahead_cb(self, msg):
        self._routes["lookahead"] = [round(msg.point.x, 3), round(msg.point.y, 3)]
        self._maybe_write_routes()

    def _maybe_write_routes(self):
        key = json.dumps(self._routes, sort_keys=True)
        if key == self._last_routes_key:
            return
        self._last_routes_key = key
        now = rospy.Time.now()
        payload = dict(self._routes)
        payload["t"] = round(now.to_sec(), 3)
        name = "%d.json" % int(round(now.to_sec() * 1000.0))
        try:
            _write_json(os.path.join(self.out_dir, "routes", name), payload)
            self._n["routes"] += 1
        except (OSError, IOError) as e:
            rospy.logwarn_throttle(10.0, "nav_debug_recorder: routes save failed (%s)", e)

    # ── replan / blockage events (the "why") ─────────────────────────────────
    def _event_cb(self, msg):
        text = msg.data or ""
        self._emit(self._events, {
            "t": round(rospy.Time.now().to_sec(), 3),
            "wall": time.strftime("%H:%M:%S"), "kind": _classify(text),
            "text": text}, "events")

    def _blockage_cb(self, msg):
        self._emit(self._events, {
            "t": round(rospy.Time.now().to_sec(), 3),
            "wall": time.strftime("%H:%M:%S"), "kind": "blockage",
            "text": "BLOCKAGE: unseen obstacle at (%.2f, %.2f)"
                    % (msg.point.x, msg.point.y),
            "x": round(msg.point.x, 3), "y": round(msg.point.y, 3)}, "events")

    # ── plumbing ─────────────────────────────────────────────────────────────
    def _write_manifest(self, G):
        cert = str(G("~certainty_log_path", "")
                   or rospy.get_param("/certainty/log_path", "") or "")
        _write_json(os.path.join(self.out_dir, "manifest.json"), {
            "created_wall": time.strftime("%Y-%m-%d %H:%M:%S"),
            "created_ros": round(rospy.Time.now().to_sec(), 3),
            "frame_id": str(G("~frame_id", "world")),
            "map_name": str(rospy.get_param("/map_config/name", "") or ""),
            "certainty_csv": cert,
            "bev_topic": str(G("~bev_topic", "/falcon/bev_2d"))})

    def _emit(self, fh, obj, counter):
        if fh is None:
            return
        try:
            fh.write(json.dumps(obj) + "\n")
            fh.flush()
            self._n[counter] += 1
        except (OSError, IOError, ValueError) as e:
            rospy.logwarn_throttle(10.0, "nav_debug_recorder: log write failed (%s)", e)

    def _heartbeat(self, _evt):
        rospy.loginfo("nav_debug hb  tel=%d bev=%d routes=%d events=%d  -> %s",
                      self._n["tel"], self._n["bev"], self._n["routes"],
                      self._n["events"], self.out_dir)

    def _close(self):
        for fh in (self._tel, self._events):
            try:
                if fh is not None:
                    fh.close()
            except (OSError, IOError):
                pass
        rospy.loginfo("nav_debug_recorder: closed run %s (tel=%d bev=%d routes=%d)",
                      self.out_dir, self._n["tel"], self._n["bev"], self._n["routes"])


def _grid_from(msg):
    """OccupancyGrid -> (H, W) int8 array on the ROS convention."""
    return np.asarray(msg.data, np.int8).reshape(msg.info.height, msg.info.width)


def _mkdir(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def _open_append(path):
    _mkdir(os.path.dirname(path))
    return open(path, "a")


def _write_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh)


def main():
    try:
        NavDebugRecorder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()


# ============================================================================
# ROSPARAMS (all private ~; defaults in parentheses):
#   ~out_dir ('' -> $FALCON_RUN_DIR if set [run_falcon.sh's shared run folder],
#            else $FALCON_LOG_DIR/nav_debug_<stamp>, else ~/.ros/falcon/nav_debug_<stamp>)
#   ~record_hz (15.0)          telemetry (pose+cmd) sampling rate
#   ~bev_min_interval_s (0.5)  min seconds between BEV saves (on top of de-dup)
#   ~certainty_log_path ('')   CSV to pair in the manifest (else /certainty/log_path)
#   ~frame_id (world)
#   Topics (defaults are the nav_stack topics):
#     ~pose_topic (/simple_drone/gt_pose)  ~cmd_topic (/simple_drone/cmd_vel)
#     ~bev_topic (/falcon/bev_2d)  ~bev_conf_topic (/falcon/bev_2d_conf)
#     ~astar_topic (/path/waypoints_astar)  ~safe_topic (/path/waypoints_safe)
#     ~final_topic (/path/waypoints)  ~lookahead_topic (/path/lookahead)
#     ~goal_topic (/waypoint_nav/goal)  ~astar_event_topic (/path/astar_event)
#     ~blockage_topic (/falcon/blockage)
#
# Replay the run it writes on the dev PC:
#   python -m sparx_agency.tasks.planning.nav_debug.run_folder_nav_debug \
#       --run $FALCON_LOG_DIR/nav_debug_<stamp>
# ============================================================================
