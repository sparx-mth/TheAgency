#!/usr/bin/env python3
"""nav_debug_recorder_node.py -- record a flight for offline visual replay.

The per-tick certainty CSV (``certainty_log.py``, written by the flight node)
already captures pose, both command sets, drift and localization quality. What it
CANNOT hold is the BEV map and the routes -- large, changing arrays -- the
planner's reasons, and the setpoint the aircraft was actually chasing. This node
fills exactly that gap, WITHOUT touching any flight-control code: it subscribes,
never publishes, and writes into a per-run folder next to the CSV so
:mod:`sparx_agency.tasks.planning.nav_debug.run_folder_nav_debug` can replay the
whole run frame by frame.

**Two nav chains, one recorder.** On XTEND this records the A*/click-to-fly
chain: the three route layers, the pure-pursuit aim point and the planner's
replan strings. On Sphera none of those topics has a publisher -- FALCON flies
its own exploration FSM -- so the same node also records FALCON's lane: the
100 Hz ``PositionCommand`` reference, the follower's control trace, FALCON's
planned/executed paths, its FSM/replan/recovery narration and per-update map
quality. Every topic is a rosparam and ``''`` disables it, so either chain runs
with the other's lanes simply absent.

This is the ROS1 half of the recording: ``bridge.yaml`` carries neither the
actuator topics nor Sphera's ground truth into ROS1, so what the drone was told
and what it actually did are recorded by a second, ROS2-side recorder and joined
offline on the ``wall`` clock -- see ``nav_debug/schema.py``.

Per the thought-journal/certainty-log conventions: flushed per line, landing in
``$FALCON_LOG_DIR`` (the host bind-mount) so it survives the ``--rm`` container,
and never taking the flight down over a diagnostic. BEV grids and route
snapshots are de-duplicated and rate-limited so a long flight stays small.

Run folder layout (names from ``nav_debug.schema``; read by the offline player):
  manifest.json          run metadata + the certainty CSV path (for auto-pairing)
  telemetry.jsonl        pose + OUR command, per tick
  reference.jsonl        the setpoint being chased (/planning/pos_cmd), rate-limited
  control.jsonl          the follower's tracking verdict and the terms behind it
  events.jsonl           replan / FSM / frontier / recovery / blockage events
  mapping.jsonl          map quality: sensor-pose age + BEV occupancy census
  routes/<ms>.json       {astar,safe,final,executed,goal,lookahead} world xy
  bev/<ms>.npy(+.json)   int8 occupancy grid + geometry sidecar (on change)
  bev_conf/<ms>.npy      int8 0..100 confidence, co-registered with bev/<ms>

Python 3.8 compatible (runs in the FALCON Noetic container). See the rosparam
block at the bottom of the file.
"""
import json
import time

import rospy
import tf.transformations as tft
from geometry_msgs.msg import Point, PointStamped, Pose, PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Int32, String

import nav_debug_run_folder as folder
import nav_debug_sources as sources

try:
    from quadrotor_msgs.msg import PositionCommand
except ImportError:          # no FALCON messages here -> no reference lane
    PositionCommand = None


class NavDebugRecorder(object):
    """Subscribe-only recorder for one flight, on either nav chain."""

    def __init__(self):
        rospy.init_node("nav_debug_recorder")
        G = rospy.get_param

        self.record_hz = float(G("~record_hz", 15.0))
        self.bev_min_interval_s = float(G("~bev_min_interval_s", 0.5))
        self.routes_min_interval_s = float(G("~routes_min_interval_s", 0.5))
        # FALCON's executed path grows all flight and is republished whole; keep
        # a decimated copy so a snapshot costs O(1) rather than O(flight).
        self.executed_max_points = int(G("~executed_max_points", 600))
        self.event_repeat_s = float(G("~event_repeat_s", 2.0))
        self.moving_eps = float(G("~reference_moving_eps",
                                  sources.DEFAULT_MOVING_EPS))

        self.run = folder.RunFolder(str(G("~out_dir", "") or folder.default_out_dir()),
                                    warn=_warn)
        self._latest_conf = None                    # latest confidence grid, paired to bev
        self._latest_cmd = (0.0, 0.0, 0.0, 0.0)     # vx, vy, vz, wz
        self._routes = {"astar": None, "safe": None, "final": None,
                        "executed": None, "goal": None, "lookahead": None}
        self._last_bev_bytes = None
        self._last_routes_key = None
        self._last_write = {}       # lane -> ROS time of its last write (rate limits)
        self._last_event = {}       # event source -> (text, ROS time) last logged
        self._pose_age_s = None     # freshest /map_ros/pose age, for mapping rows
        self._last_census = {}      # last BEV occupancy census, carried forward

        self._write_manifest(G)
        self._subscribe_flight(G)
        self._subscribe_falcon(G)
        rospy.Timer(rospy.Duration(10.0), self._heartbeat)
        rospy.on_shutdown(self._close)
        rospy.loginfo("nav_debug_recorder: writing run -> %s (@ %.1f Hz)",
                      self.run.out_dir, self.record_hz)

    # ── subscriptions ────────────────────────────────────────────────────────
    def _subscribe_flight(self, G):
        """Pose, command, the BEV map and the A*/click-to-fly route layers."""
        _sub(G("~pose_topic", "/simple_drone/gt_pose"), Pose, self._pose_cb, 10)
        _sub(G("~cmd_topic", "/simple_drone/cmd_vel"), Twist, self._cmd_cb, 10)
        _sub(G("~bev_topic", "/falcon/bev_2d"), OccupancyGrid, self._bev_cb)
        _sub(G("~bev_conf_topic", "/falcon/bev_2d_conf"), OccupancyGrid, self._conf_cb)
        _sub(G("~astar_topic", "/path/waypoints_astar"), Path,
             lambda m: self._route_cb("astar", m))
        _sub(G("~safe_topic", "/path/waypoints_safe"), Path,
             lambda m: self._route_cb("safe", m))
        _sub(G("~final_topic", "/path/waypoints"), Path,
             lambda m: self._route_cb("final", m))
        _sub(G("~lookahead_topic", "/path/lookahead"), PointStamped, self._lookahead_cb)
        _sub(G("~goal_topic", "/waypoint_nav/goal"), Point, self._goal_cb)
        _sub(G("~astar_event_topic", "/path/astar_event"), String,
             lambda m: self._log_event("astar", None, m.data), 5)
        _sub(G("~blockage_topic", "/falcon/blockage"), PointStamped,
             self._blockage_cb, 5)

    def _subscribe_falcon(self, G):
        """FALCON's exploration lane: reference, control trace, paths, narration."""
        _sub(G("~reference_topic", "/planning/pos_cmd"), PositionCommand,
             self._reference_cb)
        _sub(G("~control_trace_topic", folder.CONTROL_TRACE_TOPIC), String,
             self._control_cb, 20)
        # FALCON plans a continuous curve, not waypoints: its own trajectory is
        # the 'final' layer and what it has flown is the 'executed' one.
        _sub(G("~planned_path_topic", "/falcon/planned_path"), Path,
             lambda m: self._route_cb("final", m))
        _sub(G("~executed_path_topic", "/falcon/executed_path"), Path,
             lambda m: self._route_cb("executed", m, self.executed_max_points))
        _sub(G("~replan_topic", "/planning/replan"), Int32, self._replan_cb, 5)
        _sub(G("~go_status_topic", "/mission/go_status"), String,
             lambda m: self._log_event("go_status", None, m.data), 5)
        _sub(G("~recovery_topic", "/recovery/status"), String,
             lambda m: self._log_event("recovery", "recovery", m.data), 5)
        _sub(G("~thinking_topic", "/nav/thinking"), String, self._thinking_cb, 20)
        _sub(G("~map_pose_topic", "/map_ros/pose"), PoseStamped, self._map_pose_cb)

    # ── telemetry (pose + our command) ───────────────────────────────────────
    def _cmd_cb(self, msg):
        self._latest_cmd = (msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.z)

    def _pose_cb(self, msg):
        now, wall = _now()
        if not self._due("telemetry", self._period(), now):
            return
        q = msg.orientation
        yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        vx, vy, vz, wz = self._latest_cmd
        self.run.emit(folder.TELEMETRY_FILE, now, wall,
                      x=msg.position.x, y=msg.position.y, z=msg.position.z,
                      yaw=yaw, vx=vx, vy=vy, vz=vz, wz=wz)

    # ── the setpoint being chased, and the tracker's verdict on it ───────────
    def _reference_cb(self, msg):
        """One ``/planning/pos_cmd`` setpoint, rate-limited off its 100 Hz source."""
        now, wall = _now()
        if not self._due("reference", self._period(), now):
            return
        self.run.emit(folder.REFERENCE_FILE, now, wall,
                      **sources.reference_row(msg, now, self.moving_eps))

    def _control_cb(self, msg):
        """The follower's control trace, already shaped by its publisher."""
        decoded = sources.control_row(msg.data, *_now())
        if decoded is None:
            return
        t, wall, fields = decoded
        # The fattest lane (reference + tracking + terms + gate per tick); hold
        # it to the recorder's own rate rather than the publisher's.
        if not self._due("control", self._period(), t):
            return
        self.run.emit(folder.CONTROL_FILE, t, wall, **fields)

    # ── BEV map (+ confidence), de-duplicated, and its occupancy census ──────
    def _conf_cb(self, msg):
        self._latest_conf = sources.grid_from(msg)

    def _bev_cb(self, msg):
        now, wall = _now()
        grid = sources.grid_from(msg)
        raw = grid.tobytes()
        if raw == self._last_bev_bytes:
            return                                   # unchanged map -> skip
        if not self._due("bev", self.bev_min_interval_s, now):
            return                                   # rate-limit map churn
        self._last_bev_bytes = raw
        info = msg.info
        conf = (self._latest_conf
                if self._latest_conf is not None
                and self._latest_conf.shape == grid.shape else None)
        geometry = folder.row(now, wall, resolution=info.resolution,
                              origin_x=info.origin.position.x,
                              origin_y=info.origin.position.y,
                              width=info.width, height=info.height,
                              frame_id=msg.header.frame_id or "world")
        if self.run.save_bev(_ms(now), grid, geometry, conf):
            self._last_census = sources.grid_counts(grid)
            self._emit_map_stats(now, wall, self._last_census)

    # ── map quality ─────────────────────────────────────────────────────────
    def _map_pose_cb(self, msg):
        """How stale the pose the mapper fuses against is, at the mapper's rate."""
        now, wall = _now()
        stamp = msg.header.stamp.to_sec()
        self._pose_age_s = max(0.0, now - stamp) if stamp > 0.0 else 0.0
        if self._due("mapping", self._period(), now):
            self._emit_map_stats(now, wall, self._last_census)

    def _emit_map_stats(self, now, wall, stats):
        """One mapping row. Pose-only rows carry the last census forward.

        The pose lane ticks far faster than the BEV lane, and the loader's as-of
        join takes the newest row -- so emitting the census only on a map change
        left ~90% of frames reading a fabricated ``occ 0 free 0 unk 0``.
        """
        if self._pose_age_s is not None:
            stats = dict(stats, pose_age_s=self._pose_age_s)
        self.run.emit(folder.MAPPING_FILE, now, wall, **stats)

    # ── routes (A* layers, or FALCON's planned/executed curves) ─────────────
    def _route_cb(self, key, msg, max_points=0):
        self._routes[key] = sources.path_xy(msg, max_points)
        self._maybe_write_routes()

    def _goal_cb(self, msg):
        self._routes["goal"] = [msg.x, msg.y]
        self._maybe_write_routes()

    def _lookahead_cb(self, msg):
        self._routes["lookahead"] = [msg.point.x, msg.point.y]
        self._maybe_write_routes()

    @staticmethod
    def _routes_key(routes):
        """A cheap change signature: layer lengths plus their endpoints.

        Serializing the whole dict here cost O(path length) on every marker, at
        the marker rate, inside a ROS callback -- for a path that grows all
        flight. Lengths and endpoints separate the snapshots we would keep.
        """
        parts = []
        for name in ("astar", "safe", "final", "executed"):
            layer = routes.get(name)
            parts.append(None if layer is None
                         else (len(layer), layer[0], layer[-1]) if layer else (0,))
        parts.append(routes.get("goal"))
        parts.append(routes.get("lookahead"))
        return repr(parts)

    def _maybe_write_routes(self):
        # Rate-limit BEFORE computing the key: the check is the cheap part, and
        # the executed layer changes on every marker so the key never suppresses.
        now, wall = _now()
        if not self._due("routes", self.routes_min_interval_s, now):
            return
        key = self._routes_key(self._routes)
        if key == self._last_routes_key:
            return
        self._last_routes_key = key
        self.run.save_routes("%s.json" % _ms(now),
                             folder.row(now, wall, **self._routes))

    # ── events: the "why", in both vocabularies ─────────────────────────────
    def _replan_cb(self, msg):
        """FALCON's own verdict on the trajectory it is flying."""
        kind, text = sources.replan_event(msg.data)
        self._log_event("replan", kind, text)

    def _thinking_cb(self, msg):
        self._log_event("thinking", None, sources.thinking_text(msg.data))

    def _blockage_cb(self, msg):
        self._log_event("blockage", "blockage",
                        "BLOCKAGE: unseen obstacle at (%.2f, %.2f)"
                        % (msg.point.x, msg.point.y), msg.point.x, msg.point.y)

    def _log_event(self, source, kind, text, x=None, y=None):
        """Append one event, dropping a line this source is merely repeating.

        FALCON republishes its FINISH verdict every FSM tick and narration
        repeats while a state persists, so without this one stuck run fills the
        log with a single sentence.
        """
        text = str(text or "").strip()
        if not text:
            return
        now, wall = _now()
        seen = self._last_event.get(source)
        if seen is not None and seen[0] == text and now - seen[1] < self.event_repeat_s:
            return
        self._last_event[source] = (text, now)
        fields = {"kind": kind or sources.classify(text), "text": text,
                  "source": source}
        if x is not None and y is not None:
            fields["x"], fields["y"] = x, y
        self.run.emit(folder.EVENTS_FILE, now, wall, **fields)

    # ── plumbing ─────────────────────────────────────────────────────────────
    def _period(self):
        """Minimum seconds between rate-limited rows (0 = record everything)."""
        return 1.0 / self.record_hz if self.record_hz > 0 else 0.0

    def _due(self, lane, interval, now):
        """True if ``lane`` may write again; stamps it when it may."""
        if interval > 0 and now - self._last_write.get(lane, 0.0) < interval:
            return False
        self._last_write[lane] = now
        return True

    def _write_manifest(self, G):
        cert = str(G("~certainty_log_path", "")
                   or rospy.get_param("/certainty/log_path", "") or "")
        self.run.write_manifest({
            "created_wall": time.strftime("%Y-%m-%d %H:%M:%S"),
            "created_ros": round(rospy.Time.now().to_sec(), 3),
            "frame_id": str(G("~frame_id", "world")),
            "map_name": str(rospy.get_param("/map_config/name", "") or ""),
            "certainty_csv": cert,
            "bev_topic": str(G("~bev_topic", "/falcon/bev_2d")),
            "reference_topic": str(G("~reference_topic", "/planning/pos_cmd")),
            "control_trace_topic": str(G("~control_trace_topic",
                                         folder.CONTROL_TRACE_TOPIC))})

    def _heartbeat(self, _evt):
        rospy.loginfo("nav_debug hb  %s  -> %s", self.run.summary(), self.run.out_dir)

    def _close(self):
        self.run.close()
        rospy.loginfo("nav_debug_recorder: closed run %s (%s)",
                      self.run.out_dir, self.run.summary())


def _sub(topic, msg_type, callback, queue_size=1):
    """Subscribe, unless the topic is '' or its message type is unavailable.

    An empty topic is the documented way to switch one lane off; a missing type
    means this container has no such messages, which is not an error either.
    """
    topic = str(topic or "").strip()
    if topic and msg_type is not None:
        rospy.Subscriber(topic, msg_type, callback, queue_size=queue_size)


def _now():
    """(ROS seconds, host wall clock) -- the two clocks every row carries."""
    return rospy.Time.now().to_sec(), time.time()


def _ms(t):
    """Run-folder snapshot name: the ROS time in whole milliseconds."""
    return "%d" % int(round(t * 1000.0))


def _warn(message):
    rospy.logwarn_throttle(10.0, "nav_debug_recorder: %s", message)


def main():
    try:
        NavDebugRecorder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()


# ============================================================================
# ROSPARAMS (all private ~; defaults in parentheses). Any topic set to '' turns
# that lane off, so an XTEND run and a Sphera run share one node.
#   ~out_dir ('' -> $FALCON_RUN_DIR if set [run_falcon.sh's shared run folder],
#            else $FALCON_LOG_DIR/nav_debug_<stamp>, else ~/.ros/falcon/nav_debug_<stamp>)
#   ~record_hz (15.0)             telemetry + reference + map-pose sampling rate
#   ~bev_min_interval_s (0.5)     min seconds between BEV saves (on top of de-dup)
#   ~routes_min_interval_s (0.5)  min seconds between route snapshots
#   ~event_repeat_s (2.0)         drop a repeated line from one source for this long
#   ~reference_moving_eps (0.05)  reference speed (m/s) counted as "moving"
#   ~certainty_log_path ('')      CSV to pair in the manifest (else /certainty/log_path)
#   ~frame_id (world)
#   Topics -- XTEND A*/click-to-fly chain:
#     ~pose_topic (/simple_drone/gt_pose)  ~cmd_topic (/simple_drone/cmd_vel)
#     ~bev_topic (/falcon/bev_2d)  ~bev_conf_topic (/falcon/bev_2d_conf)
#     ~astar_topic (/path/waypoints_astar)  ~safe_topic (/path/waypoints_safe)
#     ~final_topic (/path/waypoints)  ~lookahead_topic (/path/lookahead)
#     ~goal_topic (/waypoint_nav/goal)  ~astar_event_topic (/path/astar_event)
#     ~blockage_topic (/falcon/blockage)
#   Topics -- FALCON/Sphera exploration chain:
#     ~reference_topic (/planning/pos_cmd, quadrotor_msgs/PositionCommand)
#     ~control_trace_topic (nav_debug.schema.CONTROL_TRACE_TOPIC, String JSON)
#     ~planned_path_topic (/falcon/planned_path)  -> the 'final' route layer
#     ~executed_path_topic (/falcon/executed_path) -> the 'executed' route layer
#     ~replan_topic (/planning/replan, std_msgs/Int32)
#     ~go_status_topic (/mission/go_status)  ~recovery_topic (/recovery/status)
#     ~thinking_topic (/nav/thinking)  ~map_pose_topic (/map_ros/pose)
#
# Replay the run it writes on the dev PC:
#   python -m sparx_agency.tasks.planning.nav_debug.run_folder_nav_debug \
#       --run $FALCON_LOG_DIR/nav_debug_<stamp>
# ============================================================================
