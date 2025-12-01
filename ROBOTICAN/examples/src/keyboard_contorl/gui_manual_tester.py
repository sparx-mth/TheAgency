#!/usr/bin/env python3

import time
import threading
from typing import Optional

import rclpy
from rclpy.node import Node

from fcu_driver_interfaces.msg import ManualControl
from rooster_handler_interfaces.msg import KeepAlive
from rooster_manager_interfaces.msg import RoosterState

from manual_core import ManualCommandModel, CsvLogger

# GUI imports
import tkinter as tk
from tkinter import ttk


class GuiManualControlNode(Node):
    """
    ROS 2 node for GUI-based manual control.

    - Publishes /<ROOSTER_ID>/manual_control at 40 Hz.
    - Publishes /<ROOSTER_ID>/keep_alive at 1 Hz.
    - Logs commands + RoosterState to CSV.

    ROLL mode semantics by default (x = forward/backward).
    """

    def __init__(self):
        super().__init__("gui_manual_control")

        # Parameters
        self.declare_parameter("rooster_id", "R1")
        self.declare_parameter("log_path", "gui_manual_log.csv")
        self.declare_parameter("flight_mode", 1)  # your GROUND_ROLL / ROLL enum

        self.rooster_id = self.get_parameter("rooster_id").get_parameter_value().string_value
        self.flight_mode = self.get_parameter("flight_mode").get_parameter_value().integer_value
        log_path = self.get_parameter("log_path").get_parameter_value().string_value

        # Core models
        self.command_model = ManualCommandModel(step=10.0, turtle_scale=0.5)
        self.logger = CsvLogger(self, log_path)

        # State
        self.last_state: Optional[RoosterState] = None
        self.shutdown_flag = False

        # Pulse control (for duration-based commands)
        self.pulse_active = False
        self.pulse_end_time = 0.0

        # Topics
        manual_topic = f"/{self.rooster_id}/manual_control"
        keep_alive_topic = f"/{self.rooster_id}/keep_alive"
        state_topic = f"/{self.rooster_id}/state"

        self.manual_pub = self.create_publisher(ManualControl, manual_topic, 10)
        self.keep_alive_pub = self.create_publisher(KeepAlive, keep_alive_topic, 10)
        self.create_subscription(RoosterState, state_topic, self.state_callback, 10)

        # Timers
        self.manual_timer = self.create_timer(1.0 / 40.0, self.manual_timer_cb)
        self.keep_alive_timer = self.create_timer(1.0, self.keep_alive_timer_cb)

        self.get_logger().info(
            f"GuiManualControlNode for {self.rooster_id} "
            f"(flight_mode={self.flight_mode}, ROLL mode semantics)."
        )

    # ---------- ROS callbacks ----------

    def state_callback(self, msg: RoosterState):
        self.last_state = msg

    def manual_timer_cb(self):
        # Handle pulse auto-stop
        now_wall = time.time()
        if self.pulse_active and now_wall >= self.pulse_end_time:
            self.command_model.reset_axes()
            self.pulse_active = False

        axes = self.command_model.get_scaled_axes()

        msg = ManualControl()
        msg.x = axes.x
        msg.y = axes.y
        msg.z = axes.z
        msg.r = axes.r
        msg.buttons = 0

        self.manual_pub.publish(msg)

        now_ros = self.get_clock().now().nanoseconds / 1e9
        self.logger.log_command("gui", msg, self.last_state, now_ros)

    def keep_alive_timer_cb(self):
        msg = KeepAlive()
        msg.is_active = True
        msg.requested_flight_mode = int(self.flight_mode)
        msg.command_reboot = False
        self.keep_alive_pub.publish(msg)

    # ---------- Methods used by the GUI ----------

    def set_axes_continuous(self, x: float, y: float, z: float, r: float):
        """Set axes and keep them until changed/zeroed."""
        self.get_logger().info(
            f"Set continuous: x={x}, y={y}, z={z}, r={r}"
        )
        self.command_model.set_axes(x, y, z, r)
        self.pulse_active = False

    def start_pulse(self, x: float, y: float, z: float, r: float, duration: float):
        """Set axes, hold for duration seconds, then auto-zero."""
        self.get_logger().info(
            f"Start pulse: x={x}, y={y}, z={z}, r={r}, duration={duration}"
        )
        self.command_model.set_axes(x, y, z, r)
        self.pulse_active = True
        self.pulse_end_time = time.time() + duration

    def zero_axes(self):
        self.get_logger().info("Zero all axes from GUI.")
        self.command_model.reset_axes()
        self.pulse_active = False

    def destroy_node(self):
        self.logger.close()
        super().destroy_node()


class ManualGuiApp:
    """
    Tkinter-based tiny GUI for manual control testing.

    - Sliders for x/y/z/r in [-1000, 1000].
    - Text field for x y z r.
    - Duration text field.
    - Buttons for "Set continuous", "Send pulse", "Zero all", "Quit".
    - State label that shows latest RoosterState.
    """

    def __init__(self, node: GuiManualControlNode):
        self.node = node

        self.root = tk.Tk()
        self.root.title("Manual Axes Tester")

        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Sliders for x, y, z, r
        self.scale_x = self._create_axis_slider(main, "x (forward/back)", 0)
        self.scale_y = self._create_axis_slider(main, "y (roll left/right)", 1)
        self.scale_z = self._create_axis_slider(main, "z (up/down)", 2)
        self.scale_r = self._create_axis_slider(main, "r (yaw)", 3)

        # Text input for x y z r
        ttk.Label(main, text="x y z r:").grid(row=4, column=0, sticky="e", pady=(10, 0))
        self.entry_axes = ttk.Entry(main, width=30)
        self.entry_axes.grid(row=4, column=1, columnspan=2, sticky="w", pady=(10, 0))
        self.entry_axes.insert(0, "0 0 0 0")

        btn_apply_axes = ttk.Button(main, text="Apply to sliders", command=self.apply_axes_from_text)
        btn_apply_axes.grid(row=4, column=3, sticky="w", padx=(5, 0), pady=(10, 0))

        # Duration
        ttk.Label(main, text="Duration [sec] (for pulse):").grid(row=5, column=0, sticky="e", pady=(5, 0))
        self.entry_duration = ttk.Entry(main, width=10)
        self.entry_duration.grid(row=5, column=1, sticky="w", pady=(5, 0))
        self.entry_duration.insert(0, "1.0")

        # Buttons
        btn_set = ttk.Button(main, text="Set continuous", command=self.on_set_continuous)
        btn_set.grid(row=6, column=0, sticky="ew", pady=(10, 0))

        btn_pulse = ttk.Button(main, text="Send pulse", command=self.on_send_pulse)
        btn_pulse.grid(row=6, column=1, sticky="ew", pady=(10, 0))

        btn_zero = ttk.Button(main, text="Zero all", command=self.on_zero)
        btn_zero.grid(row=6, column=2, sticky="ew", pady=(10, 0))

        btn_quit = ttk.Button(main, text="Quit", command=self.on_quit)
        btn_quit.grid(row=6, column=3, sticky="ew", pady=(10, 0))

        # State label
        self.state_label = ttk.Label(main, text="State: (waiting for /state...)")
        self.state_label.grid(row=7, column=0, columnspan=4, sticky="w", pady=(10, 0))

        # Periodic GUI update for state
        self.root.after(500, self.update_state_label)

    def _create_axis_slider(self, parent, text: str, row: int) -> tk.Scale:
        ttk.Label(parent, text=text).grid(row=row, column=0, sticky="e")
        scale = tk.Scale(
            parent,
            from_=-1000,
            to=1000,
            orient=tk.HORIZONTAL,
            length=250,
            resolution=10,
        )
        scale.set(0)
        scale.grid(row=row, column=1, columnspan=3, sticky="we", pady=3)
        return scale

    def get_slider_values(self):
        x = float(self.scale_x.get())
        y = float(self.scale_y.get())
        z = float(self.scale_z.get())
        r = float(self.scale_r.get())
        return x, y, z, r

    # ---------- Button handlers ----------

    def apply_axes_from_text(self):
        text = self.entry_axes.get().strip()
        parts = text.split()
        if len(parts) != 4:
            self._show_temp_status("Expected 4 values: x y z r")
            return
        try:
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])
            r = float(parts[3])
        except ValueError:
            self._show_temp_status("Failed to parse x y z r")
            return

        # Update sliders (clamped by slider range automatically)
        self.scale_x.set(x)
        self.scale_y.set(y)
        self.scale_z.set(z)
        self.scale_r.set(r)
        self._show_temp_status("Updated sliders from text.")

    def on_set_continuous(self):
        x, y, z, r = self.get_slider_values()
        self.node.set_axes_continuous(x, y, z, r)
        self._show_temp_status(f"Continuous: x={x}, y={y}, z={z}, r={r}")

    def on_send_pulse(self):
        x, y, z, r = self.get_slider_values()
        try:
            duration = float(self.entry_duration.get().strip())
        except ValueError:
            self._show_temp_status("Invalid duration")
            return
        if duration <= 0:
            self._show_temp_status("Duration must be > 0")
            return

        self.node.start_pulse(x, y, z, r, duration)
        self._show_temp_status(
            f"Pulse: x={x}, y={y}, z={z}, r={r}, dur={duration}s"
        )

    def on_zero(self):
        self.node.zero_axes()
        self.scale_x.set(0)
        self.scale_y.set(0)
        self.scale_z.set(0)
        self.scale_r.set(0)
        self._show_temp_status("Zeroed all axes.")

    def on_quit(self):
        self.node.get_logger().info("GUI requested quit.")
        self.node.shutdown_flag = True
        self.root.destroy()

    # ---------- State label + status ----------

    def update_state_label(self):
        st = self.node.last_state
        if st is None:
            txt = "State: (no /state yet)"
        else:
            txt = (
                f"State: roll={st.roll:.2f}, pitch={st.pitch:.2f}, "
                f"azimuth={st.azimuth:.2f}, "
                f"mode={st.flight_mode}, "
                f"armed={st.armed}, airborne={st.airborne}"
            )
        self.state_label.config(text=txt)

        # schedule next update
        self.root.after(500, self.update_state_label)

    def _show_temp_status(self, msg: str):
        # Reuse state label for quick feedback (simple but effective)
        self.state_label.config(text=f"{msg}")


def ros_spin_thread(node: GuiManualControlNode):
    while rclpy.ok() and not node.shutdown_flag:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.get_logger().info("ROS spin thread exiting")


def main(args=None):
    rclpy.init(args=args)
    node = GuiManualControlNode()

    # Start ROS spinning in background
    t = threading.Thread(target=ros_spin_thread, args=(node,), daemon=True)
    t.start()

    app = ManualGuiApp(node)
    try:
        app.root.mainloop()
    finally:
        node.shutdown_flag = True
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
