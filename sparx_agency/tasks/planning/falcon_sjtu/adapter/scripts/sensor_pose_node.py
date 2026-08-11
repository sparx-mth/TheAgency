#!/usr/bin/env python
"""Turn the drone's odometry into the two inputs FALCON's mapper and FSM need.

The SJTU sim publishes one ``nav_msgs/Odometry`` for the body. FALCON needs two
things it does not contain:

* ``/map_ros/pose`` -- a ``PoseStamped`` for the **camera**, not the body. The
  depth camera sits 0.2 m forward of the body origin (``front_cam_joint`` in
  the URDF) and looks along body x, and the mapper projects every depth pixel
  through this pose. Publish the body pose here and every obstacle is placed
  20 cm too close and every wall leans with the aircraft's tilt.
* ``/odom_world`` -- the same odometry with the twist rotated into the WORLD
  frame. The sim follows REP-147 and expresses the twist in the child (body)
  frame, but FALCON reads ``twist.twist.linear`` straight into the initial
  velocity of its next B-spline without rotating it. Handed the body-frame
  twist, every replan starts with a velocity pointing wherever the nose
  happens to face -- the same defect the Pegasus deployment worked around on
  its sender side, handled here on the adapter side because the sim's
  publisher is not ours to change.

Both outputs carry the input message's own stamp, untouched: the mapper pairs
depth to pose by timestamp (tolerance 0.05 s here, interpolating between
samples), and restamping would break the pairing.

Runs inside the ROS1 Noetic FALCON container on Python 3.8 -- no scipy, no
f-strings with =, no walrus. Plain numpy only.
"""
from __future__ import annotations

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

CAMERA_OFFSET_BODY = np.array([0.2, 0.0, 0.0])
"""Camera position in the body frame, metres -- front_cam_joint's origin."""

Q_BODY_CAMERA = np.array([0.5, -0.5, 0.5, -0.5])
"""Body-FLU to camera-optical (RDF) rotation, ``(w, x, y, z)``.

The quaternion of the T_b_c matrix every map yaml carries::

    [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]

i.e. camera x = -body y (right), camera y = -body z (down), camera z = body x
(forward). Verified numerically against that matrix rather than derived by
hand, because a sign error here maps the world mirrored and nothing crashes.
"""


def _quat_multiply(a, b):
    # type: (np.ndarray, np.ndarray) -> np.ndarray
    """Hamilton product, ``(w, x, y, z)`` convention."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _rotate(quaternion_wxyz, vector):
    # type: (np.ndarray, np.ndarray) -> np.ndarray
    """Rotate a vector by a quaternion: q * v * q^-1, without building a matrix."""
    w = quaternion_wxyz[0]
    axis = quaternion_wxyz[1:]
    return (vector + 2.0 * np.cross(axis, np.cross(axis, vector) + w * vector))


class SensorPoseNode(object):
    """Republish body odometry as a camera pose and a world-twist odometry."""

    def __init__(self):
        # type: () -> None
        self._pose_pub = rospy.Publisher("/map_ros/pose", PoseStamped, queue_size=10)
        self._odom_pub = rospy.Publisher("/odom_world", Odometry, queue_size=10)
        rospy.Subscriber(rospy.get_param("~odom_topic", "/simple_drone/odom"),
                         Odometry, self._on_odom, queue_size=10)

    def _on_odom(self, msg):
        # type: (Odometry) -> None
        orientation = msg.pose.pose.orientation
        body_q = np.array([orientation.w, orientation.x, orientation.y, orientation.z])
        position = msg.pose.pose.position
        body_p = np.array([position.x, position.y, position.z])

        # ── the camera pose the mapper projects through ──────────────────
        camera_p = body_p + _rotate(body_q, CAMERA_OFFSET_BODY)
        camera_q = _quat_multiply(body_q, Q_BODY_CAMERA)
        pose = PoseStamped()
        pose.header = msg.header            # the SAME stamp: pairing is by time
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = camera_p
        (pose.pose.orientation.w, pose.pose.orientation.x,
         pose.pose.orientation.y, pose.pose.orientation.z) = camera_q
        self._pose_pub.publish(pose)

        # ── the odometry the FSM plans from ──────────────────────────────
        twist = msg.twist.twist.linear
        world_v = _rotate(body_q, np.array([twist.x, twist.y, twist.z]))
        fixed = Odometry()
        fixed.header = msg.header
        fixed.child_frame_id = msg.child_frame_id
        fixed.pose = msg.pose
        fixed.twist = msg.twist
        (fixed.twist.twist.linear.x, fixed.twist.twist.linear.y,
         fixed.twist.twist.linear.z) = world_v
        self._odom_pub.publish(fixed)


def main():
    # type: () -> None
    rospy.init_node("sensor_pose_node")
    SensorPoseNode()
    rospy.spin()


if __name__ == "__main__":
    main()
