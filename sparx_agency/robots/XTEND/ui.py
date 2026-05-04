import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import tkinter as tk
from tkinter import ttk


class DroneControlUI(Node):
    def __init__(self):
        super().__init__('drone_gui_publisher')
        self.publisher_ = self.create_publisher(String, '/drone/cmd_nav', 10)

        # Setup the UI Window
        self.root = tk.Tk()
        self.root.title("XTEND Drone Controller")
        self.root.geometry("300x450")

        style = ttk.Style()
        style.configure('TButton', font=('Arial', 10), padding=5)

        # UI Layout
        ttk.Label(self.root, text="System Controls", font=('Arial', 12, 'bold')).pack(pady=10)
        self.create_button("ARM", "arm", 0, "orange")
        self.create_button("TAKEOFF", "takeoff", 0, "green")
        self.create_button("LAND", "land", 0, "red")
        self.create_button("DISARM", "disarm", 0, "darkred")

        ttk.Separator(self.root, orient='horizontal').pack(fill='x', pady=10)
        ttk.Label(self.root, text="Movement Controls", font=('Arial', 12, 'bold')).pack(pady=5)

        self.create_button("FORWARD (2s)", "forward", 2000)
        self.create_button("TURN LEFT", "rotate_left", 1500)
        self.create_button("TURN RIGHT", "rotate_right", 1500)
        self.create_button("GO DOWN", "move_down", 500)

    def create_button(self, label, action, value, color=None):
        """Helper to create buttons that publish JSON commands."""
        btn = tk.Button(
            self.root,
            text=label,
            command=lambda: self.send_cmd(action, value),
            bg=color if color else "lightgrey",
            width=20
        )
        btn.pack(pady=5)

    def send_cmd(self, action, value):
        """Publishes the JSON command to ROS 2."""
        msg = String()
        command = {"action": action, "value": value}
        msg.data = json.dumps(command)
        self.publisher_.publish(msg)
        self.get_logger().info(f"Published: {msg.data}")

    def run(self):
        # Update Tkinter and ROS 2 together
        while rclpy.ok():
            self.root.update_idletasks()
            self.root.update()
            rclpy.spin_once(self, timeout_sec=0.01)


def main():
    rclpy.init()
    gui = DroneControlUI()
    try:
        gui.run()
    except tk.TclError:  # Handle window being closed
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()