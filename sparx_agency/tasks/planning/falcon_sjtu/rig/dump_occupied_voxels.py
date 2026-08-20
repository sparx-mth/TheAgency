#!/usr/bin/env python
"""Dump FALCON's occupied-voxel map to a plain xyz file, one voxel per line.

Runs INSIDE the Noetic container (Python 3.8, rospy) and is copied in by
``campaign_run.sh`` at the end of a leg. Kept as its own file rather than a
heredoc inside the campaign script because a heredoc nested inside a
``docker exec bash -lc '...'`` string does not terminate where it appears to,
which has silently produced empty artifacts in this package before.

Why it exists. When the aircraft physically touches an obstacle while tracking
its plan to within 0.03 m, three explanations remain and only one of them is
about the map: the planner routed through a known obstacle, the controller left
the plan, or **the map never knew the obstacle was there**. The first two are
already excluded by ``tracking.csv`` field 13. A contact point that is still
empty in the final occupied-voxel map settles the third.

The warehouse strikes this was written for land on the RIM of a 1.79 m pile --
1.07 to 1.37 m from the model origin against half-extents of 0.89 x 1.03 m, at
1.75 to 1.79 m altitude -- and the aircraft never flies above 2.34 m, so it only
ever sees those tops edge-on. That is exactly the geometry that maps badly.

Exits 0 on any failure. This is a diagnostic collected after the verdict is
already decided, and it must never be able to change or delay one.
"""
import sys


def main():
    # type: () -> int
    """Write the occupied-voxel cloud to the path given as argv[1]."""
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/occupied_voxels.xyz"
    try:
        import rospy
        from sensor_msgs.msg import PointCloud2
        import sensor_msgs.point_cloud2 as pc2
    except ImportError as exc:
        sys.stderr.write("no rospy/sensor_msgs: %s\n" % exc)
        return 0

    try:
        rospy.init_node("sparx_map_dump", anonymous=True, disable_signals=True)
        cloud = rospy.wait_for_message(
            "/voxel_mapping/occupancy_grid_occupied", PointCloud2, timeout=15.0)
    except Exception as exc:                      # noqa: BLE001 - diagnostic only
        sys.stderr.write("no occupancy cloud: %s\n" % exc)
        return 0

    try:
        with open(out, "w") as handle:
            for point in pc2.read_points(cloud, field_names=("x", "y", "z"),
                                         skip_nans=True):
                handle.write("%.3f %.3f %.3f\n" % (point[0], point[1], point[2]))
    except Exception as exc:                      # noqa: BLE001 - diagnostic only
        sys.stderr.write("could not write %s: %s\n" % (out, exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
