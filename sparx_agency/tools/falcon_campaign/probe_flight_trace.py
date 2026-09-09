"""Record the whole "why" of a flight: the plan, the tracking, and the map.

LOOP_MISSION.md section 4 asks every flight to answer two questions after the
fact -- how closely did the aircraft fly FALCON's B-spline, and how fast did it
map -- and neither is answerable from what the campaign recorded before this.
``truth.jsonl`` is the ROS 2 side (ground truth and the axis counts) and knows
nothing about the plan; ``coverage.jsonl`` samples one scalar volume every
~19 s.

Everything needed is already on the ROS 1 side and was simply never captured:

* ``/nav_debug/control_trace`` -- the follower publishes its complete per-tick
  internals (the reference it was given, the error split into along-track lag
  and cross-track, the feed-forward/damping/correction breakdown, every gate and
  reflex flag). ``nav_debug_record`` already defaults to true, so this has been
  published on every flight and thrown away.
* ``/planning/bspline`` -- the trajectory ITSELF, control points and knots and
  yaw points. Tracking error measured against ``pos_cmd`` is error against a
  point; measured against these it is error against the curve.
* ``/voxel_mapping/map_stats`` -- the mapping-rate counters added by
  ``patches/sparx_map_stats.sh`` at 2 Hz. Absent on an unpatched image, in which
  case ``map_coverage`` alone is still recorded and the row simply carries less.

The B-spline message is ``trajectory/Bspline`` -- confirmed live with
``rostopic type``, not assumed; it is built into the FALCON workspace, so this
probe only imports cleanly inside that container.

One file, one JSON object per line, each tagged with ``kind`` so the streams can
be split again offline. Rows are written as they arrive rather than sampled: the
control trace is already one row per control tick, and the plan events are what
they are. Subscribe-only -- it never publishes and never actuates.

Runs INSIDE the ``falcon`` container for the length of a flight:

    docker exec -d falcon bash -lc 'source /opt/ros/noetic/setup.bash && \\
        python3 -u /tmp/probe_flight_trace.py > /tmp/flight_trace.jsonl'
"""
from __future__ import print_function

import json
import sys

import rospy
from quadrotor_msgs.msg import PositionCommand
from trajectory.msg import Bspline
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, Float32, String

CONTROL_TRACE = "/nav_debug/control_trace"
BSPLINE = "/planning/bspline"
POS_CMD = "/planning/pos_cmd"
REPLAN = "/planning/replan"
ODOM = "/odom_world"
MAP_STATS = "/voxel_mapping/map_stats"
MAP_COVERAGE = "/voxel_mapping/map_coverage"
FRONTIER = "/planning_vis/frontier_pcl"

#: Odometry is 100+ Hz and the control trace already carries the pose the
#: follower acted on; this is the independent check, so a slow beat is enough.
ODOM_PERIOD_S = 0.1

#: pos_cmd runs at 100 Hz. Kept at the follower's own tick rate instead: 50 ms
#: is 2.5 cm of travel at cruise, far finer than any lag worth measuring, and
#: the full rate would write ~9 MB a flight for no extra resolution.
REF_PERIOD_S = 0.05


def emit(kind, **fields):
    """Write one row. Never raises -- a broken diagnostic must not end a flight."""
    try:
        row = {"kind": kind, "t": rospy.Time.now().to_sec()}
        row.update(fields)
        sys.stdout.write(json.dumps(row) + "\n")
    except Exception:                       # noqa: BLE001
        pass


class FlightTrace(object):
    """Subscribes every stream that explains a flight and writes them as JSONL."""

    def __init__(self):
        self._last_odom = 0.0
        self._last_ref = 0.0
        rospy.Subscriber(CONTROL_TRACE, String, self._control, queue_size=50)
        rospy.Subscriber(BSPLINE, Bspline, self._bspline, queue_size=10)
        rospy.Subscriber(POS_CMD, PositionCommand, self._pos_cmd, queue_size=10)
        rospy.Subscriber(REPLAN, Bool, self._replan, queue_size=10)
        rospy.Subscriber(ODOM, Odometry, self._odom, queue_size=1)
        rospy.Subscriber(MAP_STATS, String, self._map_stats, queue_size=5)
        rospy.Subscriber(MAP_COVERAGE, Float32, self._coverage, queue_size=5)
        rospy.Subscriber(FRONTIER, PointCloud2, self._frontier, queue_size=1)

    def _control(self, msg):
        """The follower's per-tick internals, already JSON -- passed through."""
        try:
            emit("ctrl", trace=json.loads(msg.data))
        except Exception:                   # noqa: BLE001
            emit("ctrl_raw", data=msg.data[:2000])

    def _bspline(self, msg):
        """The trajectory itself -- control points, knots and the YAW curve.

        Error measured against ``pos_cmd`` is error against a moving point;
        against these it is error against the curve, which is what the operator
        asked for. The yaw points are the half the follower currently discards.
        """
        emit("bspline",
             traj_id=int(msg.traj_id),
             order=int(msg.order),
             start_time=float(msg.start_time.to_sec()),
             knots=[float(k) for k in msg.knots],
             pos_pts=[(p.x, p.y, p.z) for p in msg.pos_pts],
             yaw_pts=[float(y) for y in msg.yaw_pts],
             yaw_dt=float(msg.yaw_dt))

    def _pos_cmd(self, msg):
        """The setpoint, including the yaw and yaw_dot the follower drops."""
        now = rospy.Time.now().to_sec()
        if now - self._last_ref < REF_PERIOD_S:
            return
        self._last_ref = now
        emit("ref",
             x=msg.position.x, y=msg.position.y, z=msg.position.z,
             vx=msg.velocity.x, vy=msg.velocity.y, vz=msg.velocity.z,
             ax=msg.acceleration.x, ay=msg.acceleration.y, az=msg.acceleration.z,
             yaw=float(msg.yaw), yaw_dot=float(msg.yaw_dot),
             traj_id=int(getattr(msg, "trajectory_id", 0)))

    def _replan(self, msg):
        emit("replan", value=bool(msg.data))

    def _odom(self, msg):
        now = rospy.Time.now().to_sec()
        if now - self._last_odom < ODOM_PERIOD_S:
            return
        self._last_odom = now
        p, v = msg.pose.pose.position, msg.twist.twist.linear
        q = msg.pose.pose.orientation
        emit("odom", x=p.x, y=p.y, z=p.z, vx=v.x, vy=v.y, vz=v.z,
             qx=q.x, qy=q.y, qz=q.z, qw=q.w)

    def _map_stats(self, msg):
        """The 2 Hz mapping counters. Absent until the image carries the patch."""
        try:
            emit("map", **json.loads(msg.data))
        except Exception:                   # noqa: BLE001
            emit("map_raw", data=msg.data[:2000])

    def _coverage(self, msg):
        emit("coverage", volume_m3=float(msg.data))

    def _frontier(self, msg):
        emit("frontier", points=int(msg.width) * int(msg.height))


def main():
    rospy.init_node("probe_flight_trace", anonymous=True, disable_signals=True)
    FlightTrace()
    emit("start", note="probe_flight_trace up")
    rospy.spin()


if __name__ == "__main__":
    main()
