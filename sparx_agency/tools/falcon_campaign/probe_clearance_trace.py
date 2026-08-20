"""Log the aircraft's clearance to the nearest MAPPED obstacle, for a whole flight.

The aircraft gets stuck about a dozen times a flight, costing roughly a quarter
of it. Two very different causes are consistent with that, and they need
opposite fixes:

* it is grazing geometry the map already holds -- the planner is routing too
  close, and more inflation or viewpoint clearance is the answer; or
* it is hitting geometry the map never had -- DA3 returns nothing closer than
  0.62 m, so an obstacle inside that radius is invisible, and no amount of
  planner margin helps.

Clearance at the moment of a PINNED event separates them: comfortably outside
the inflation radius means the obstacle was unmapped.

Runs INSIDE the ``falcon`` container for the length of a flight:

    docker exec -d falcon bash -lc 'source /opt/ros/noetic/setup.bash && \\
        python3 -u /tmp/probe_clearance_trace.py > /tmp/clearance.jsonl'
"""
from __future__ import print_function

import json
import sys

import numpy as np
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand

OCC = "/voxel_mapping/occupancy_grid_occupied"
POSE = "/voxel_mapping/interpolated_pose"
REF = "/planning/pos_cmd"
BAND_M = 0.4          # only obstacles within this much of the aircraft's height
PERIOD_S = 2.0


class Trace(object):
    """Samples clearance against the newest occupancy cloud."""

    def __init__(self):
        self.cloud = None
        self.pose = None
        self.ref = None
        rospy.Subscriber(OCC, PointCloud2, self._cloud, queue_size=1)
        rospy.Subscriber(POSE, Odometry, self._pose, queue_size=1)
        rospy.Subscriber(REF, PositionCommand, self._ref, queue_size=1)

    def _cloud(self, msg):
        pts = np.array(list(point_cloud2.read_points(
            msg, ("x", "y", "z"), skip_nans=True)), dtype="float32")
        self.cloud = pts if pts.size else None

    def _pose(self, msg):
        p = msg.pose.pose.position
        self.pose = (p.x, p.y, p.z)

    def _ref(self, msg):
        self.ref = (msg.position.x, msg.position.y, msg.position.z)

    def nearest_to(self, point):
        """Nearest mapped obstacle to an arbitrary point, in its height band."""
        if self.cloud is None or point is None:
            return None
        x, y, z = point
        band = self.cloud[np.abs(self.cloud[:, 2] - z) <= BAND_M]
        if not band.size:
            return None
        return float(np.hypot(band[:, 0] - x, band[:, 1] - y).min())

    def sample(self):
        """Nearest mapped obstacle in the aircraft's height band, metres."""
        if self.cloud is None or self.pose is None:
            return None
        x, y, z = self.pose
        band = self.cloud[np.abs(self.cloud[:, 2] - z) <= BAND_M]
        if not band.size:
            return None
        d = np.hypot(band[:, 0] - x, band[:, 1] - y)
        return float(d.min())


def main():
    """Print one JSON line per sample until the node is killed."""
    rospy.init_node("clearance_trace", anonymous=True, disable_signals=True)
    trace = Trace()
    rate = rospy.Rate(1.0 / PERIOD_S)
    while not rospy.is_shutdown():
        nearest = trace.sample()
        if nearest is not None:
            # The reference's OWN clearance is the half that says whether the
            # plan is unsafe or merely unfollowed.
            ref_clear = trace.nearest_to(trace.ref)
            row = {"wall": round(rospy.Time.now().to_sec(), 3),
                   "x": round(trace.pose[0], 2), "y": round(trace.pose[1], 2),
                   "nearest_m": round(nearest, 3)}
            if ref_clear is not None:
                row["ref_nearest_m"] = round(ref_clear, 3)
                row["pos_err_m"] = round(float(np.hypot(
                    trace.ref[0] - trace.pose[0],
                    trace.ref[1] - trace.pose[1])), 3)
            print(json.dumps(row))
            sys.stdout.flush()
        rate.sleep()
    return 0


if __name__ == "__main__":
    sys.exit(main())
