import json
import time
import tkinter as tk
from tkinter import ttk

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class DroneControlUI(Node):
    def __init__(self):
        super().__init__("drone_gui_publisher")
        self.cmd_nav_pub = self.create_publisher(String, "/xtend/cmd_nav", 10)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.active_action = None
        self.active_action_start_t = None

        self.root = tk.Tk()
        self.root.title("XTEND ROS Command UI")
        self.root.geometry("340x720")

        self.forward_value_var = tk.StringVar(value="400")
        self.turn_value_var = tk.StringVar(value="1000")
        self.timer_text = tk.StringVar(value="Current action: none")

        self.active_twist_linear_x = 0.0
        self.active_twist_angular_z = 0.0
        self.twist_publish_period_sec = 0.1  # 10 Hz
        self.last_twist_publish_t = 0.0

        self.build_ui()

    def build_ui(self):
        ttk.Label(
            self.root,
            text="XTEND ROS Command UI",
            font=("Arial", 13, "bold"),
        ).pack(pady=10)

        ttk.Label(
            self.root,
            textvariable=self.timer_text,
            font=("Arial", 11, "bold"),
            foreground="blue",
        ).pack(pady=8)

        ttk.Label(
            self.root,
            text="System Controls",
            font=("Arial", 12, "bold"),
        ).pack(pady=8)

        self.create_button("ARM", "arm", color="orange")
        self.create_button("TAKEOFF", "takeoff", color="green")
        self.create_button("LAND", "land", color="red")
        self.create_button("DISARM", "disarm", color="darkred")

        ttk.Separator(self.root, orient="horizontal").pack(fill="x", pady=10)

        ttk.Label(
            self.root,
            text="Thrust Values",
            font=("Arial", 12, "bold"),
        ).pack(pady=5)

        thrust_frame = ttk.Frame(self.root)
        thrust_frame.pack(pady=4)

        ttk.Label(thrust_frame, text="Forward:").grid(row=0, column=0, padx=4, pady=3)
        ttk.Entry(thrust_frame, textvariable=self.forward_value_var, width=8).grid(row=0, column=1, padx=4, pady=3)

        ttk.Label(thrust_frame, text="Turn:").grid(row=1, column=0, padx=4, pady=3)
        ttk.Entry(thrust_frame, textvariable=self.turn_value_var, width=8).grid(row=1, column=1, padx=4, pady=3)

        ttk.Separator(self.root, orient="horizontal").pack(fill="x", pady=10)

        ttk.Label(
            self.root,
            text="Hold-Style Movement",
            font=("Arial", 12, "bold"),
        ).pack(pady=5)

        self.create_button("FORWARD", "forward")
        self.create_button("TURN LEFT", "turn_left")
        self.create_button("TURN RIGHT", "turn_right")
        self.create_button("STOP", "stop", color="yellow")

    def create_button(self, label, action, color=None):
        btn = tk.Button(
            self.root,
            text=label,
            command=lambda: self.handle_button(action),
            bg=color if color else "lightgrey",
            width=24,
        )
        btn.pack(pady=5)

    def get_int_value(self, var: tk.StringVar, default: int, name: str) -> int:
        try:
            value = int(var.get())
        except ValueError:
            self.get_logger().warn(f"Invalid {name}: {var.get()}, using {default}")
            value = default

        return max(0, min(1000, value))

    def handle_button(self, action: str):
        if action == "forward":
            self.set_active_twist(linear_x=0.3, angular_z=0.0, name="forward_twist")
            self.start_timer("forward_twist")

        elif action == "turn_left":
            self.set_active_twist(linear_x=0.0, angular_z=0.65, name="turn_left_twist")
            self.start_timer("turn_left_twist")

        elif action == "turn_right":
            self.set_active_twist(linear_x=0.0, angular_z=-0.65, name="turn_right_twist")
            self.start_timer("turn_right_twist")



        elif action == "stop":
            self.active_twist_linear_x = 0.0
            self.active_twist_angular_z = 0.0
            self.send_twist(0.0, 0.0)
            self.send_cmd("stop", 0)
            self.stop_timer("stop")

        elif action in ("land", "disarm"):
            self.send_cmd(action, 0)
            self.stop_timer(action)

        elif action in ("land", "disarm"):
            self.send_twist(0.0, 0.0)
            self.send_cmd(action, 0)
            self.stop_timer(action)
        else:
            # arm / takeoff
            self.send_cmd(action, 0)

    def send_cmd(self, action, value):
        msg = String()
        command = {
            "action": action,
            "value": int(value),
        }
        msg.data = json.dumps(command)
        self.cmd_nav_pub.publish(msg)
        self.get_logger().info(f"Published cmd_nav: {msg.data}")

    def send_twist(self, linear_x=0.0, angular_z=0.0):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(msg)

        self.get_logger().info(
            f"Published Twist: linear.x={msg.linear.x:.3f}, angular.z={msg.angular.z:.3f}"
        )

    def start_timer(self, action_name: str):
        self.active_action = action_name
        self.active_action_start_t = time.time()
        self.timer_text.set(f"Current action: {action_name} | 0.000s")

    def stop_timer(self, reason: str):
        if self.active_action is not None and self.active_action_start_t is not None:
            elapsed = time.time() - self.active_action_start_t
            self.get_logger().info(
                f"UI action ended: {self.active_action}, duration={elapsed:.3f}s, reason={reason}"
            )

        self.active_action = None
        self.active_action_start_t = None
        self.timer_text.set("Current action: none")

    def set_active_twist(self, linear_x: float, angular_z: float, name: str):
        self.active_twist_linear_x = float(linear_x)
        self.active_twist_angular_z = float(angular_z)
        self.send_twist(self.active_twist_linear_x, self.active_twist_angular_z)
        self.start_timer(name)

    def update_timer_label(self):
        if self.active_action is None or self.active_action_start_t is None:
            return

        elapsed = time.time() - self.active_action_start_t
        self.timer_text.set(f"Current action: {self.active_action} | {elapsed:.3f}s")

    def run(self):
        while rclpy.ok():
            now = time.time()

            if self.active_action is not None:
                if now - self.last_twist_publish_t >= self.twist_publish_period_sec:
                    self.send_twist(
                        self.active_twist_linear_x,
                        self.active_twist_angular_z,
                    )
                    self.last_twist_publish_t = now

            self.update_timer_label()
            self.root.update_idletasks()
            self.root.update()
            rclpy.spin_once(self, timeout_sec=0.01)


def main():
    rclpy.init()
    gui = DroneControlUI()

    try:
        gui.run()
    except tk.TclError:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()