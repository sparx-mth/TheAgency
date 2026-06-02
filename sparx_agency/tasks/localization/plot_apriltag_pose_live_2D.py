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

        self.frames = [] 
        self.xs = []
        self.ys = []
        self.yaws = []

        self.prev_raw_yaw = None
        self.yaw_offset = 0.0
        

        self.frame_count = 0 

        # How many orientation arrows to draw on the X-Y trajectory
        self.arrow_every_n = 5
        self.arrow_len = 0.15

        # Plot update rate
        self.plot_timer = self.create_timer(0.2, self.update_plot)

        self._setup_plot()

        self.get_logger().info(f"Listening to {self.topic_name}")

    def _setup_plot(self):
        plt.ion()

        # Create a figure with 2 subplots side-by-side
        self.fig, (self.ax_xy, self.ax_yaw) = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.suptitle("AprilTag Estimated Pose - 2D Trajectory & Yaw")

        # --- Setup 2D Trajectory Plot (X-Y) ---
        self.traj_line, = self.ax_xy.plot([], [], marker="o", markersize=3, linewidth=1, label="Trajectory")
        self.current_point, = self.ax_xy.plot([], [], marker="x", markersize=10, color='red', label="Current Pose")
        self.quiver_plot = None  # To hold the arrows

        self.ax_xy.set_title("2D Trajectory in World Frame")
        self.ax_xy.set_xlabel("X [m]")
        self.ax_xy.set_ylabel("Y [m]")

        # Fixed axis limits: -4 to 4 meters
        self.ax_xy.set_xlim(-1, 4)
        self.ax_xy.set_ylim(-4, 1)
        self.ax_xy.grid(True)
        self.ax_xy.legend()

        self.info_text = self.ax_xy.text(
            0.02, 0.95,
            "",
            transform=self.ax_xy.transAxes,
            bbox=dict(facecolor='white', alpha=0.8)
        )

        # --- Setup Yaw Plot ---
        self.yaw_line, = self.ax_yaw.plot([], [], color='orange', linewidth=2)
        self.ax_yaw.set_title("Yaw Angle Over Frames")
        self.ax_yaw.set_xlabel("Frame Index") 
        self.ax_yaw.set_ylabel("Yaw [deg]")
        self.ax_yaw.grid(True)

        plt.tight_layout()
        plt.show(block=False)

    def pose_callback(self, msg: PoseStamped):
        self.frame_count += 1 

        x = msg.pose.position.x
        y = msg.pose.position.y

        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w

        yaw_rad = yaw_from_quaternion(qx, qy, qz, qw)
        raw_yaw_deg = math.degrees(yaw_rad)

        if self.prev_raw_yaw is not None:
            delta = raw_yaw_deg - self.prev_raw_yaw
            if delta > 180:
                self.yaw_offset -= 360
            elif delta < -180:
                self.yaw_offset += 360

        self.prev_raw_yaw = raw_yaw_deg
        
        yaw_deg = raw_yaw_deg + self.yaw_offset

        self.frames.append(self.frame_count)
        self.xs.append(x)
        self.ys.append(y)
        self.yaws.append(yaw_deg)

    def update_plot(self):
        if len(self.frames) == 0:
            return

        # --- Update X-Y Trajectory ---
        self.traj_line.set_data(self.xs, self.ys)
        self.current_point.set_data([self.xs[-1]], [self.ys[-1]])

        # Remove old yaw arrows if they exist
        if self.quiver_plot is not None:
            self.quiver_plot.remove()

        # Prepare data for new yaw arrows
        arrow_x = []
        arrow_y = []
        arrow_dx = []
        arrow_dy = []

        for i in range(0, len(self.xs), self.arrow_every_n):
            yaw_rad = math.radians(self.yaws[i])
            arrow_x.append(self.xs[i])
            arrow_y.append(self.ys[i])
            arrow_dx.append(self.arrow_len * math.cos(yaw_rad))
            arrow_dy.append(self.arrow_len * math.sin(yaw_rad))

        # Draw new arrows
        if arrow_x:
            self.quiver_plot = self.ax_xy.quiver(
                arrow_x, arrow_y, arrow_dx, arrow_dy,
                color='black', scale_units='xy', angles='xy', scale=1, width=0.005
            )

        self.info_text.set_text(
            f"Frame: {self.frames[-1]}\nx={self.xs[-1]:.2f}, y={self.ys[-1]:.2f}\nyaw={self.yaws[-1]:.1f}°"
        )

        # --- Update Yaw Plot ---
        self.yaw_line.set_data(self.frames, self.yaws)
        
        # Dynamically adjust Yaw plot limits (show trailing window of 100 frames)
        current_frame = self.frames[-1]
        self.ax_yaw.set_xlim(0, max(50, current_frame + 10))
        
        if self.yaws:
            min_yaw = min(self.yaws)
            max_yaw = max(self.yaws)
            self.ax_yaw.set_ylim(min_yaw - 20, max_yaw + 20)

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