#!/usr/bin/env python3

import math
from collections import deque

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

import matplotlib.pyplot as plt


def yaw_from_quaternion(qx, qy, qz, qw):
    """
    Convert quaternion to yaw angle in radians.
    Assumes ROS convention:
      X forward, Y left, Z up
    """
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz)
    )


class AprilTagPosePlotter(Node):
    def __init__(self):
        super().__init__("apriltag_pose_plotter")

        self.topic_name = "/xtend/april_tag_pose"

        self.sub = self.create_subscription(
            PoseStamped,
            self.topic_name,
            self.pose_callback,
            10
        )

        self.times = []
        self.xs = []
        self.ys = []
        self.zs = []
        self.yaws = []

        self.start_time = None

        # How many orientation arrows to draw on the X-Y trajectory
        self.arrow_every_n = 5
        self.arrow_len = 0.15

        # Plot update rate
        self.plot_timer = self.create_timer(0.2, self.update_plot)

        self._setup_plot()

        self.get_logger().info(f"Listening to {self.topic_name}")

    def _setup_plot(self):
        plt.ion()

        self.fig = plt.figure(figsize=(10, 8))
        self.fig.suptitle("AprilTag Estimated Pose - 3D Trajectory")

        # 3D trajectory plot
        self.ax_3d = self.fig.add_subplot(1, 1, 1, projection="3d")

        self.traj_line, = self.ax_3d.plot([], [], [], marker="o", markersize=3, linewidth=1)
        self.current_point, = self.ax_3d.plot([], [], [], marker="x", markersize=10)

        self.ax_3d.set_title("3D Trajectory in World Frame")
        self.ax_3d.set_xlabel("X [m]")
        self.ax_3d.set_ylabel("Y [m]")
        self.ax_3d.set_zlabel("Z [m]")

        # Fixed axis limits: 0 to 5 meters
        self.ax_3d.set_xlim(0, 5)
        self.ax_3d.set_ylim(0, 5)
        self.ax_3d.set_zlim(0, 5)

        # Optional: nicer initial view angle
        self.ax_3d.view_init(elev=25, azim=-60)

        self.info_text = self.ax_3d.text2D(
            0.02, 0.95,
            "",
            transform=self.ax_3d.transAxes
        )

        plt.tight_layout()
        plt.show(block=False)

    def pose_callback(self, msg: PoseStamped):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.start_time is None:
            self.start_time = stamp

        t = stamp - self.start_time

        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z

        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w

        yaw_rad = yaw_from_quaternion(qx, qy, qz, qw)
        yaw_deg = math.degrees(yaw_rad)

        self.times.append(t)
        self.xs.append(x)
        self.ys.append(y)
        self.zs.append(z)
        self.yaws.append(yaw_deg)

        self.get_logger().info(
            f"Pose: x={x:.3f}, y={y:.3f}, z={z:.3f}, yaw={yaw_deg:.2f} deg"
        )

    def update_plot(self):
        if len(self.times) == 0:
            return

        # Remove old yaw arrows
        for artist in list(self.ax_3d.collections):
            artist.remove()

        # Update 3D trajectory line
        self.traj_line.set_data(self.xs, self.ys)
        self.traj_line.set_3d_properties(self.zs)

        # Update current point
        self.current_point.set_data([self.xs[-1]], [self.ys[-1]])
        self.current_point.set_3d_properties([self.zs[-1]])

        # Draw yaw arrows along the trajectory
        # Yaw is shown as direction in X-Y plane, with dz=0
        for i in range(0, len(self.xs), self.arrow_every_n):
            x = self.xs[i]
            y = self.ys[i]
            z = self.zs[i]

            yaw_rad = math.radians(self.yaws[i])

            dx = self.arrow_len * math.cos(yaw_rad)
            dy = self.arrow_len * math.sin(yaw_rad)
            dz = 0.0

            self.ax_3d.quiver(
                x, y, z,
                dx, dy, dz,
                length=1.0,
                normalize=False
            )

        # Keep fixed limits between 0 and 5 meters
        self.ax_3d.set_xlim(0, 3)
        self.ax_3d.set_ylim(0, 3 )
        self.ax_3d.set_zlim(0, 3)

        self.ax_3d.set_xlabel("X [m]")
        self.ax_3d.set_ylabel("Y [m]")
        self.ax_3d.set_zlabel("Z [m]")

        self.info_text.set_text(
            f"x={self.xs[-1]:.2f}, y={self.ys[-1]:.2f}, z={self.zs[-1]:.2f}, yaw={self.yaws[-1]:.1f}°"
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def main(args=None):
    rclpy.init(args=args)

    node = AprilTagPosePlotter()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            plt.pause(0.01)

    except KeyboardInterrupt:
        node.get_logger().info("Stopping AprilTag pose plotter")

    finally:
        node.destroy_node()
        rclpy.shutdown()
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()