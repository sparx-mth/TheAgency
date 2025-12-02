import rclpy
from rclpy.node import Node
import threading
import tkinter as tk
from fcu_driver_interfaces.msg import Battery  # Using the message type from your ICD


class BatteryMonitorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("R1 Battery")

        # Window Setup: Small, Top-Right Corner, Always on Top
        self.width = 220
        self.height = 120

        # Get screen width to position in top-right
        screen_width = self.root.winfo_screenwidth()
        x_pos = screen_width - self.width - 20  # 20px margin from right
        y_pos = 20  # 20px margin from top

        self.root.geometry(f"{self.width}x{self.height}+{x_pos}+{y_pos}")
        self.root.attributes('-topmost', True)  # Always on top
        # self.root.overrideredirect(True) # Uncomment this to remove title bar (frameless)

        # UI Elements
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.lbl_percent = tk.Label(self.main_frame, text="--%", font=("Helvetica", 24, "bold"))
        self.lbl_percent.pack(pady=(15, 5))

        self.lbl_voltage = tk.Label(self.main_frame, text="-- V", font=("Helvetica", 14))
        self.lbl_voltage.pack(pady=5)

        # State tracking for "Pop" effect
        self.current_color = None

    def update_display(self, voltage, percentage):
        """
        Updates the GUI with new values.
        percentage is expected to be 0.0 to 1.0
        """
        # Convert to int 0-100
        pct_value = int(percentage * 100)

        # Update Text
        self.lbl_percent.config(text=f"{pct_value}%")
        self.lbl_voltage.config(text=f"{voltage:.2f} V")

        # Determine Color
        new_color = "gray"
        text_color = "black"

        if percentage > 0.30:
            new_color = "#4CAF50"  # Green
            text_color = "white"
        elif 0.15 <= percentage <= 0.30:
            new_color = "#FF9800"  # Orange
            text_color = "black"
        else:  # Below 0.15
            new_color = "#F44336"  # Red
            text_color = "white"

        # Apply Color Change
        if self.current_color != new_color:
            self.current_color = new_color
            self.main_frame.config(bg=new_color)
            self.lbl_percent.config(bg=new_color, fg=text_color)
            self.lbl_voltage.config(bg=new_color, fg=text_color)

            # "Pop" effect: If color changes (e.g. Green -> Orange), force focus
            self.root.deiconify()
            self.root.lift()
            self.root.attributes('-topmost', True)


class BatteryListener(Node):
    def __init__(self, gui):
        super().__init__('battery_monitor_gui')
        self.gui = gui

        # Subscribe to the R2 battery topic
        self.subscription = self.create_subscription(
            Battery,
            '/R1/fcu/battery',
            self.listener_callback,
            10
        )
        self.get_logger().info('Battery Monitor Started. Waiting for data...')

    def listener_callback(self, msg):
        # Schedule the GUI update in the main Tkinter thread
        # Note: Tkinter is not thread-safe, so we use after() logic or just invoke directly
        # (mostly okay for simple config updates, but best practice is queuing).
        # For this simple script, direct call is usually fine, but let's be safe.
        try:
            self.gui.root.after(0, self.gui.update_display, msg.voltage, msg.percentage)
        except Exception as e:
            self.get_logger().error(f'Error updating GUI: {e}')


def run_ros_node(node):
    rclpy.spin(node)


def main():
    # 1. Init ROS
    rclpy.init()

    # 2. Setup GUI
    root = tk.Tk()
    app = BatteryMonitorGUI(root)

    # 3. Setup ROS Node
    ros_node = BatteryListener(app)

    # 4. Run ROS in a separate background thread
    ros_thread = threading.Thread(target=run_ros_node, args=(ros_node,), daemon=True)
    ros_thread.start()

    # 5. Start GUI Loop (Main Thread)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        if rclpy.ok():
            ros_node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()