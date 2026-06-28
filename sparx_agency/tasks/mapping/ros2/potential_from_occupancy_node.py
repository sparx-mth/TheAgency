#!/usr/bin/env python3
from __future__ import annotations

import math

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import tf2_ros

from sparx_agency.core.mapping.costmap.potential_field_layer import PotentialFieldLayer


class PotentialFromOccupancyNode(Node):
    """
    Subscribes to a nav_msgs/OccupancyGrid, computes a repulsive potential
    field via PotentialFieldLayer, and publishes the result.

    Inputs:
      /map                       (nav_msgs/OccupancyGrid)

    Outputs:
      /potential_field/u_rep     (sensor_msgs/Image, 32FC1)
      /local_nav_vector          (geometry_msgs/Vector3Stamped)
      /local_nav_heading         (std_msgs/Float32)
    """

    def __init__(self):
        super().__init__('potential_from_occupancy_node')

        self.declare_parameter('map_topic', '/falcon/bev_2d')
        self.declare_parameter('base_frame', 'world')
        self.declare_parameter('map_frame', 'world')
        self.declare_parameter('sigma_m', 0.6)
        self.declare_parameter('occ_thresh', 0.65)
        self.declare_parameter('repulse_radius_m', 1.0)
        self.declare_parameter('smooth', True)

        sigma_m          = float(self.get_parameter('sigma_m').value)
        occ_thresh       = float(self.get_parameter('occ_thresh').value)
        repulse_radius_m = float(self.get_parameter('repulse_radius_m').value)
        self._smooth     = bool(self.get_parameter('smooth').value)

        self._layer = PotentialFieldLayer(
            occ_thresh=occ_thresh,
            sigma_m=sigma_m,
            repulse_radius_m=repulse_radius_m,
        )
        self._bridge = CvBridge()
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        map_topic = self.get_parameter('map_topic').value
        qos_map = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(OccupancyGrid, map_topic, self._on_map, qos_map)

        qos_pub = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._pub_u_rep      = self.create_publisher(Image,          '/potential_field/u_rep', qos_pub)
        self._pub_nav_vector = self.create_publisher(Vector3Stamped, '/local_nav_vector',       10)
        self._pub_nav_heading = self.create_publisher(Float32,       '/local_nav_heading',      10)

        self.get_logger().info(f'PotentialFromOccupancyNode ready — listening on {map_topic}')

    def _on_map(self, msg: OccupancyGrid):
        h   = msg.info.height
        w   = msg.info.width
        res = float(msg.info.resolution)

        # int8 [-1, 0..100] → float32 [NaN, 0..1]
        raw = np.array(msg.data, dtype=np.int8).reshape(h, w)
        occ = np.where(raw < 0, np.nan, raw.astype(np.float32) / 100.0)

        # OccupancyGrid row 0 = bottom (y-up); flip so row 0 = top for numpy
        occ = np.flipud(occ)

        U_rep, _ = self._layer.compute_from_prob_grid(occ, res)

        if self._smooth:
            U_rep = cv2.GaussianBlur(U_rep, (5, 5), 1.0)

        # gradient convention: [...,0]=fwd (row axis), [...,1]=left (col axis)
        g_row, g_col = np.gradient(U_rep, res)
        gradient = np.stack([-g_row, g_col], axis=-1).astype(np.float32)

        # Publish U_rep as 32FC1
        u_rep_msg = self._bridge.cv2_to_imgmsg(U_rep, encoding='32FC1')
        u_rep_msg.header.stamp    = self.get_clock().now().to_msg()
        u_rep_msg.header.frame_id = msg.header.frame_id
        self._pub_u_rep.publish(u_rep_msg)

        # Sample gradient at robot cell (fall back to map centre)
        robot_row, robot_col = self._robot_cell(msg)
        if robot_row is not None:
            disp_row = max(0, min(h - 1, (h - 1) - robot_row))
            disp_col = max(0, min(w - 1, robot_col))
        else:
            disp_row, disp_col = h // 2, w // 2

        v    = gradient[disp_row, disp_col].copy()
        norm = float(np.linalg.norm(v))
        if norm > 1e-6:
            v /= norm

        vec_msg = Vector3Stamped()
        vec_msg.header.stamp    = self.get_clock().now().to_msg()
        vec_msg.header.frame_id = self.get_parameter('base_frame').value
        vec_msg.vector.x = float(v[0])
        vec_msg.vector.y = float(v[1])
        vec_msg.vector.z = 0.0
        self._pub_nav_vector.publish(vec_msg)

        heading_msg      = Float32()
        heading_msg.data = float(math.atan2(float(v[1]), float(v[0])))
        self._pub_nav_heading.publish(heading_msg)

    def _robot_cell(self, msg: OccupancyGrid):
        """Return (row, col) of the robot in the occupancy grid, or (None, None)."""
        try:
            tf = self._tf_buffer.lookup_transform(
                msg.header.frame_id,
                self.get_parameter('base_frame').value,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
            rx  = tf.transform.translation.x
            ry  = tf.transform.translation.y
            ox  = msg.info.origin.position.x
            oy  = msg.info.origin.position.y
            res = float(msg.info.resolution)
            col = int((rx - ox) / res)
            row = int((ry - oy) / res)
            return row, col
        except Exception:
            return None, None


def main(args=None):
    rclpy.init(args=args)
    node = PotentialFromOccupancyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
