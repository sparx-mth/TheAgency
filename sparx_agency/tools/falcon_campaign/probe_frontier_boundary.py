"""Count the free/unknown boundary in FALCON's map, independently of FALCON.

When the frontier finder reports zero frontiers while four fifths of the box is
unexplored, there are two very different explanations and they need different
fixes: either the mapped free space really is sealed off by occupied voxels --
a doorway closed by noisy monocular depth -- or the boundary is there and the
finder's own criteria are rejecting it. A frontier is by definition a free
voxel touching an unknown one, so count those directly and the question is
settled.

Runs INSIDE the ``falcon`` container (ROS1, Python 3):

    docker exec falcon bash -lc 'source /opt/ros/noetic/setup.bash && \\
        python3 -u /tmp/probe_frontier_boundary.py'
"""
from __future__ import print_function

import sys

import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2

RES = 0.1                      # voxel size, metres (map resolution_fine)
FREE = "/voxel_mapping/occupancy_grid_free"
UNKNOWN = "/voxel_mapping/occupancy_grid_unknown"
NEIGHBOURS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def _keys(topic, timeout_s=20.0):
    """Voxel-index set for one cloud, or None if the topic stays quiet."""
    try:
        cloud = rospy.wait_for_message(topic, PointCloud2, timeout=timeout_s)
    except rospy.ROSException:
        return None
    return {(int(round(x / RES)), int(round(y / RES)), int(round(z / RES)))
            for x, y, z in point_cloud2.read_points(cloud, ("x", "y", "z"),
                                                    skip_nans=True)}


def main():
    """Print how many free voxels touch unknown space, and where."""
    rospy.init_node("probe_frontier_boundary", anonymous=True,
                    disable_signals=True)
    free, unknown = _keys(FREE), _keys(UNKNOWN)
    if free is None or unknown is None:
        print("no map published -- is exploration running?")
        return 1
    boundary = [v for v in free
                if any((v[0] + d[0], v[1] + d[1], v[2] + d[2]) in unknown
                       for d in NEIGHBOURS)]
    print("free %d  unknown %d  free-touching-unknown %d"
          % (len(free), len(unknown), len(boundary)))
    if boundary:
        zs = sorted(v[2] * RES for v in boundary)
        print("boundary z: min %.2f  median %.2f  max %.2f"
              % (zs[0], zs[len(zs) // 2], zs[-1]))
        xs = sorted(v[0] * RES for v in boundary)
        ys = sorted(v[1] * RES for v in boundary)
        print("boundary spans x %.1f..%.1f  y %.1f..%.1f"
              % (xs[0], xs[-1], ys[0], ys[-1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
