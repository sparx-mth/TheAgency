"""Histogram FALCON's voxel map by height, to see how much of the box is reached.

The mission is volume coverage, and the box is 4.8 m tall while the aircraft
cruises at ~1.2 m with a monocular camera. Whether the map fills the whole box
or only a band around flight altitude decides whether the next lever is speed
(cover more floor) or altitude (cover more height) -- so measure it rather than
argue about it.

Runs INSIDE the ``falcon`` container (ROS1, Python 3):

    docker exec falcon bash -lc 'source /opt/ros/noetic/setup.bash && \\
        python3 /tmp/probe_map_heights.py'

Prints one row per 0.25 m band: occupied voxels, free voxels, and the share of
all mapped voxels that band holds.
"""
from __future__ import print_function

import sys

import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2

BIN_M = 0.25
TOPICS = ("/voxel_mapping/occupancy_grid_occupied",
          "/voxel_mapping/occupancy_grid_free")


def _grab(topic, timeout_s):
    """One message from ``topic``, or None if it stays quiet."""
    try:
        return rospy.wait_for_message(topic, PointCloud2, timeout=timeout_s)
    except rospy.ROSException:
        return None


def _histogram(cloud):
    """Voxel count per height bin, keyed by the bin's lower edge."""
    bins = {}
    if cloud is None:
        return bins
    for _, _, z in point_cloud2.read_points(cloud, ("x", "y", "z"),
                                            skip_nans=True):
        key = int(z // BIN_M) * BIN_M
        bins[key] = bins.get(key, 0) + 1
    return bins


def main(timeout_s=20.0):
    """Print the height histogram of the current map."""
    rospy.init_node("probe_map_heights", anonymous=True, disable_signals=True)
    occ, free = (_histogram(_grab(t, timeout_s)) for t in TOPICS)
    if not occ and not free:
        print("no map published on %s -- is exploration running?" % (TOPICS,))
        return 1
    total = sum(occ.values()) + sum(free.values())
    print("%8s %10s %10s %8s" % ("z_from", "occupied", "free", "share"))
    for key in sorted(set(occ) | set(free)):
        n = occ.get(key, 0) + free.get(key, 0)
        print("%8.2f %10d %10d %7.1f%%"
              % (key, occ.get(key, 0), free.get(key, 0), 100.0 * n / total))
    print("total mapped voxels: %d" % total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
