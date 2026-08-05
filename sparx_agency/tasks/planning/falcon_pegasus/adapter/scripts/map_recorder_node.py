#!/usr/bin/env python3
"""Record FALCON building its map, as an MP4, with no display attached.

This is the deliverable half of a run. FALCON's own visualisation is an RViz
config, which needs an X server, a human to point the camera and a screen
recorder; this node subscribes to the same data and writes frames straight to
disk instead.

What is drawn, and why each part is there, is documented in
:mod:`~sparx_agency.tasks.planning.falcon_pegasus.viz.exploration_frame`. The
short version: the map grows out of the dark as the aircraft flies, and a red
line shows the gap between where FALCON is commanding and where the aircraft
actually is -- the gap that only exists because this simulator has physics.

The occupancy clouds are only computed by FALCON when something is subscribed to
them, so running this node is what turns them on. That is also why the run
configs shrink ``vbox`` to a 20 cm slab: every extra layer is another full pass
over the voxel grid, twice a second, inside the same single-threaded node that
services the depth callbacks.

Run: ``rosrun falcon_pegasus map_recorder_node.py`` (normally from
``launch/falcon_pegasus.launch``).
"""
import math
import os

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32

from cloud_reader import cloud_to_xy
from sparx_agency.tasks.planning.falcon_pegasus.viz import exploration_frame

TRAIL_MIN_STEP_M = 0.05


def _quaternion_yaw(orientation):
    """Heading from a ROS quaternion, radians CCW from +x."""
    return math.atan2(
        2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
        1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z))


def _bounds_from_params():
    """The visualisation box FALCON is drawing, straight off the parameter server.

    Returns:
        ``(min_x, min_y, max_x, max_y)``.

    Raises:
        rospy.ROSException: If the map config has not been loaded. Guessing a
            box would produce a video of the wrong building.
    """
    keys = ("vbox_min_x", "vbox_min_y", "vbox_max_x", "vbox_max_y")
    missing = [k for k in keys if not rospy.has_param("/map_config/map_size/" + k)]
    if missing:
        raise rospy.ROSException(
            "/map_config/map_size/{%s} not set -- the run's YAML must be loaded "
            "before this node starts" % ",".join(missing))
    return tuple(float(rospy.get_param("/map_config/map_size/" + k)) for k in keys)


class MapRecorder(object):
    """Subscribes to FALCON's map and writes a video of it being built."""

    def __init__(self):
        self.output = rospy.get_param("~output", "/falcon_logs/exploration.mp4")
        self.fps = float(rospy.get_param("~fps", 10.0))
        self.title = rospy.get_param("~title", "FALCON exploration")
        width = int(rospy.get_param("~width", 1280))
        height = int(rospy.get_param("~height", 720))
        sight_m = float(rospy.get_param("/voxel_mapping/tsdf/raycast_max", 5.0))
        fov_deg = math.degrees(float(rospy.get_param(
            "/uav_model/sensing_parameters/fov/horizontal", math.pi / 2.0)))

        self.canvas = exploration_frame.ExplorationCanvas(
            _bounds_from_params(), size=(width, height), fov_deg=fov_deg,
            sight_m=sight_m)

        directory = os.path.dirname(self.output)
        if directory:
            try:
                os.makedirs(directory)
            except OSError:
                pass
        self.writer = cv2.VideoWriter(
            self.output, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
        if not self.writer.isOpened():
            raise rospy.ROSException("could not open %s for writing" % self.output)

        self.occupied = np.empty((0, 2), np.float32)
        self.free = np.empty((0, 2), np.float32)
        self.trail = []
        self.position = None
        self.yaw = 0.0
        self.speed = 0.0
        self.reference = None
        self.trajectory_id = 0
        self.coverage_m3 = 0.0
        self.depth_frames = 0
        self.frames = 0
        self.started_at = None

        rospy.Subscriber("/voxel_mapping/occupancy_grid_occupied", PointCloud2,
                         self._on_occupied, queue_size=1)
        rospy.Subscriber("/voxel_mapping/occupancy_grid_free", PointCloud2,
                         self._on_free, queue_size=1)
        rospy.Subscriber("/voxel_mapping/map_coverage", Float32, self._on_coverage,
                         queue_size=1)
        rospy.Subscriber("/uav_simulator/odometry", Odometry, self._on_odometry,
                         queue_size=1)
        rospy.Subscriber("/uav_simulator/sensor_pose", TransformStamped,
                         self._on_sensor_pose, queue_size=1)
        rospy.Subscriber("/planning/pos_cmd", PositionCommand, self._on_command,
                         queue_size=1)
        rospy.loginfo("[recorder] writing %s at %.0f fps, %dx%d", self.output,
                      self.fps, width, height)

    def _on_occupied(self, msg):
        self.occupied = cloud_to_xy(msg)

    def _on_free(self, msg):
        self.free = cloud_to_xy(msg)

    def _on_coverage(self, msg):
        self.coverage_m3 = float(msg.data)

    def _on_sensor_pose(self, _msg):
        self.depth_frames += 1

    def _on_odometry(self, msg):
        position = msg.pose.pose.position
        self.position = (position.x, position.y, position.z)
        self.yaw = _quaternion_yaw(msg.pose.pose.orientation)
        linear = msg.twist.twist.linear
        self.speed = math.sqrt(linear.x ** 2 + linear.y ** 2 + linear.z ** 2)
        if self.started_at is None:
            self.started_at = rospy.Time.now()
        # Only extend the trail when the aircraft has actually moved, or a
        # 100 Hz odometry stream fills it with thousands of identical points
        # while it hovers and the polyline draw stops keeping up.
        if not self.trail or math.hypot(position.x - self.trail[-1][0],
                                        position.y - self.trail[-1][1]) > TRAIL_MIN_STEP_M:
            self.trail.append((position.x, position.y))

    def _on_command(self, msg):
        self.reference = (msg.position.x, msg.position.y, msg.position.z)
        self.trajectory_id = msg.trajectory_id

    def _hud(self):
        """The status lines, phrased as what a viewer wants to know."""
        elapsed = 0.0 if self.started_at is None else (
            rospy.Time.now() - self.started_at).to_sec()
        error = 0.0
        if self.position is not None and self.reference is not None:
            error = math.sqrt(sum((self.reference[i] - self.position[i]) ** 2
                                  for i in range(3)))
        known = len(self.occupied) + len(self.free)
        return [
            "t %5.1f s   mapped %6.0f m3   known cells %6d   depth frames %5d"
            % (elapsed, self.coverage_m3, known, self.depth_frames),
            "speed %4.2f m/s   trajectory #%d   plan-to-aircraft gap %4.2f m"
            % (self.speed, self.trajectory_id, error),
        ]

    def tick(self, _event):
        """Render and write one video frame."""
        if self.position is None:
            return
        frame = exploration_frame.render(
            self.canvas, self.occupied, self.free, self.trail, self.position,
            self.yaw, self.reference, self._hud(), self.title)
        self.writer.write(frame)
        self.frames += 1

    def close(self):
        """Finish the file. Without this the MP4 has no moov atom and will not play."""
        self.writer.release()
        rospy.loginfo("[recorder] wrote %d frames to %s", self.frames, self.output)


def main():
    rospy.init_node("map_recorder")
    recorder = MapRecorder()
    rospy.on_shutdown(recorder.close)
    rospy.Timer(rospy.Duration(1.0 / recorder.fps), recorder.tick)
    rospy.spin()


if __name__ == "__main__":
    main()
