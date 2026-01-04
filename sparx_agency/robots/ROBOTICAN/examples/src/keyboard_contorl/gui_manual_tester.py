#!/usr/bin/env python3

import time
import threading
from typing import Optional, List, Tuple

import rclpy
from rclpy.node import Node

from rooster_handler_interfaces.msg import KeepAlive
from rooster_manager_interfaces.msg import RoosterState
from fcu_driver_interfaces.msg import ManualControl, UAVState
from std_srvs.srv import SetBool

from manual_core import ManualCommandModel, RLTransitionLogger

# GUI imports
import tkinter as tk
from tkinter import ttk, filedialog


class GuiManualControlNode(Node):
    """
    ROS 2 node for GUI-based manual control.

    - Publishes /<ROOSTER_ID>/manual_control at 40 Hz.
    - Publishes /<ROOSTER_ID>/keep_alive at 1 Hz.
    - Calls /<ROOSTER_ID>/fcu/command/force_arm (SetBool).
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
        self.rl_logger = RLTransitionLogger(self, log_path)

        # State
        self.last_state: Optional[RoosterState] = None
        self.last_uav_state: Optional[UAVState] = None
        self.shutdown_flag = False

        # Pulse / path control
        self.pulse_active = False
        self.pulse_end_time = 0.0
        self.path_running = False
        self.path_stop_flag = False

        # RL logging for the current pulse/action
        self.current_action_info = None  # dict with S, A, times
        self.pulse_logged = False

        # ROS handles (created per-rooster)
        self.manual_pub = None
        self.keep_alive_pub = None
        self.state_sub = None
        self.force_arm_client = None

        # Init pubs/subs/clients for initial rooster_id
        self._switch_rooster_id(self.rooster_id)

        # Timers
        self.manual_timer = self.create_timer(1.0 / 40.0, self.manual_timer_cb)
        self.keep_alive_timer = self.create_timer(1.0, self.keep_alive_timer_cb)
        self.keepalive_enabled = True
        self.get_logger().info(
            f"GuiManualControlNode for {self.rooster_id} "
            f"(flight_mode={self.flight_mode}, ROLL mode semantics)."
        )

    # ---------- topic/service wiring per-rooster ----------

    def _switch_rooster_id(self, new_id: str):
        """Destroy old pubs/subs/clients and recreate for a new rooster_id."""
        # Destroy old
        if self.manual_pub is not None:
            self.destroy_publisher(self.manual_pub)
        if self.keep_alive_pub is not None:
            self.destroy_publisher(self.keep_alive_pub)
        if self.state_sub is not None:
            self.destroy_subscription(self.state_sub)
        if getattr(self, "uav_state_sub", None) is not None:
            self.destroy_subscription(self.uav_state_sub)
        if self.force_arm_client is not None:
            self.destroy_client(self.force_arm_client)

        self.rooster_id = new_id

        manual_topic = f"/{self.rooster_id}/manual_control"
        keep_alive_topic = f"/{self.rooster_id}/keep_alive"
        state_topic = f"/{self.rooster_id}/state"
        force_arm_service = f"/{self.rooster_id}/fcu/command/force_arm"

        self.manual_pub = self.create_publisher(ManualControl, manual_topic, 10)
        self.keep_alive_pub = self.create_publisher(KeepAlive, keep_alive_topic, 10)
        self.state_sub = self.create_subscription(RoosterState,
                                                  state_topic,
                                                  self.state_callback,
                                                  10)
        self.uav_state_sub = self.create_subscription(UAVState,
                                                      f"/{self.rooster_id}/fcu/state",
                                                      self.uav_state_callback,
                                                      10,)
        self.force_arm_client = self.create_client(SetBool, force_arm_service)

        self.get_logger().info(
            f"Switched IO to {self.rooster_id}: "
            f"{manual_topic}, {keep_alive_topic}, {state_topic}, {force_arm_service}"
        )

    def set_rooster_id(self, new_id: str):
        """Called from GUI when user selects R1/R2/R3."""
        if new_id == self.rooster_id:
            return
        self._switch_rooster_id(new_id)

    def set_flight_mode(self, mode: int):
        """Set requested flight mode (used in KeepAlive)."""
        self.flight_mode = int(mode)
        self.get_logger().info(f"Flight mode set to {self.flight_mode} from GUI")


    # ---------- ROS callbacks ----------

    def state_callback(self, msg: RoosterState):
        self.last_state = msg

    def uav_state_callback(self, msg: UAVState):
        self.last_uav_state = msg

    def manual_timer_cb(self):
        now_wall = time.time()
        now_ros = self.get_clock().now().nanoseconds / 1e9

        # Check if pulse ended and RL transition not yet logged
        if self.pulse_active and now_wall >= self.pulse_end_time:
            # Auto-zero axes
            self.command_model.reset_axes()
            self.pulse_active = False

            if (self.current_action_info is not None) and (not self.pulse_logged):
                info = self.current_action_info
                s_before = info["s_before"]
                x = info["x"]
                y = info["y"]
                z = info["z"]
                r = info["r"]
                duration = info["duration"]
                t_start_ros = info["t_start_ros"]
                s_after = self.last_uav_state  # state after action

                self.rl_logger.log_transition(
                    s_before=s_before,
                    action_x=x,
                    action_y=y,
                    action_z=z,
                    action_r=r,
                    duration=duration,
                    t_start=t_start_ros,
                    t_end=now_ros,
                    s_after=s_after,
                )
                self.pulse_logged = True
                self.current_action_info = None

        # Normal command publishing (40 Hz)
        axes = self.command_model.get_scaled_axes()

        msg = ManualControl()
        msg.x = axes.x
        msg.y = axes.y
        msg.z = axes.z
        msg.r = axes.r
        msg.buttons = 0

        self.manual_pub.publish(msg)

    def keep_alive_timer_cb(self):
        if not self.keepalive_enabled:
            return
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

    def set_keepalive_enabled(self, enabled: bool):
        self.keepalive_enabled = bool(enabled)
        state = "ENABLED" if enabled else "DISABLED"
        self.get_logger().info(f"KeepAlive is now {state}")


    def start_pulse(self, x: float, y: float, z: float, r: float, duration: float):
        """Set axes, hold for duration seconds, then auto-zero and log RL transition."""
        self.get_logger().info(
            f"Start pulse: x={x}, y={y}, z={z}, r={r}, duration={duration}"
        )
        self.command_model.set_axes(x, y, z, r)
        now_wall = time.time()
        now_ros = self.get_clock().now().nanoseconds / 1e9

        # Mark pulse active
        self.pulse_active = True
        self.pulse_end_time = now_wall + duration
        self.pulse_logged = False

        # Capture S and A for RL logging
        self.current_action_info = {
            "x": x,
            "y": y,
            "z": z,
            "r": r,
            "duration": duration,
            "t_start_ros": now_ros,
            "s_before": self.last_uav_state,  # may be None; logger will handle
        }

    def zero_axes(self):
        self.get_logger().info("Zero all axes from GUI.")
        self.command_model.reset_axes()
        self.pulse_active = False
        self.pulse_logged = True
        self.current_action_info = None

    # --- Force arm ---

    def call_force_arm(self, arm: bool):
        if not self.force_arm_client.service_is_ready():
            self.get_logger().warn("force_arm service not ready")
            return

        req = SetBool.Request()
        req.data = arm

        future = self.force_arm_client.call_async(req)

        def _done(fut):
            try:
                resp = fut.result()
            except Exception as e:
                self.get_logger().error(f"force_arm({arm}) failed: {e}")
                return

            if resp.success:
                self.get_logger().info(
                    f"force_arm({arm}) success: {resp.message}"
                )
            else:
                self.get_logger().warn(
                    f"force_arm({arm}) reported failure: {resp.message}"
                )

        future.add_done_callback(_done)

    # --- Generic path runner ---

    def run_path(self, segments: List[Tuple[str, float, float, float, float, float]], path_name: str = "path"):
        """
        Run a list of segments, each:
            (name, x, y, z, r, duration_sec)
        """
        if self.path_running:
            self.get_logger().warn("Path already running, ignoring new request.")
            return

        if not segments:
            self.get_logger().warn("Empty path, nothing to run.")
            return

        self.path_running = True
        self.path_stop_flag = False
        self.get_logger().info(f"Starting {path_name} with {len(segments)} segments.")

        try:
            for name, x, y, z, r, dur in segments:
                if self.path_stop_flag or not rclpy.ok():
                    break
                self.get_logger().info(
                    f"Path segment {name}: x={x}, y={y}, z={z}, r={r}, dur={dur}s"
                )
                self.start_pulse(x, y, z, r, dur)
                end = time.time() + dur
                while time.time() < end and rclpy.ok() and not self.path_stop_flag:
                    time.sleep(0.05)
        finally:
            self.zero_axes()
            self.path_running = False
            self.path_stop_flag = False
            self.get_logger().info(f"{path_name} finished or stopped.")

    def stop_path(self):
        if self.path_running:
            self.get_logger().info("Stop path requested.")
            self.path_stop_flag = True

    def destroy_node(self):
        try:
            self.rl_logger.close()
        except Exception:
            pass
        super().destroy_node()


class ManualGuiApp:
    """
    Tkinter-based tiny GUI for manual control testing.

    - Robot selector: R1 / R2 / R3.
    - Sliders for x/y/z/r in [-1000, 1000].
    - Text field for x y z r (single segment).
    - Duration field for single pulse.
    - Path editor for multiple segments:
        each line:  name x y z r duration
        or:         x y z r duration  (name auto-generated)
    - Buttons: Set continuous, Send pulse, Zero all, Arm, Disarm,
               Run path (editor), Stop path, Load path from file, Quit.
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

        # --- Robot selector R1/R2/R3 ---
        ttk.Label(main, text="Robot:").grid(row=0, column=0, sticky="e")
        self.robot_var = tk.StringVar(value=self.node.rooster_id)
        self.robot_combo = ttk.Combobox(
            main,
            textvariable=self.robot_var,
            values=["R1", "R2", "R3"],
            state="readonly",
            width=5,
        )
        self.robot_combo.grid(row=0, column=1, sticky="w")
        self.robot_combo.bind("<<ComboboxSelected>>", self.on_robot_change)

        # --- Flight mode selector ---
        ttk.Label(main, text="Flight mode:").grid(row=0, column=2, sticky="e")

        self.flight_mode_var = tk.StringVar()
        # 1 = GROUND_ROLL (taxi/rolling)
        # 2 = MANUAL (in-air manual)
        self.flight_mode_map = {
            "GROUND_ROLL (1)": 1,
            "MANUAL (2)": 2,
        }

        # Pick the label that matches the node's current flight_mode
        cur_label = None
        for label, value in self.flight_mode_map.items():
            if value == self.node.flight_mode:
                cur_label = label
                break
        if cur_label is None:
            cur_label = "GROUND_ROLL (1)"

        self.flight_mode_var.set(cur_label)
        self.flight_mode_combo = ttk.Combobox(
            main,
            textvariable=self.flight_mode_var,
            values=list(self.flight_mode_map.keys()),
            state="readonly",
            width=15,
        )
        self.flight_mode_combo.grid(row=0, column=3, sticky="w")
        self.flight_mode_combo.bind("<<ComboboxSelected>>", self.on_flight_mode_change)

        # --- KeepAlive enable/disable ---
        self.keepalive_var = tk.BooleanVar(value=True)
        keepalive_check = ttk.Checkbutton(
            main,
            text="Send KeepAlive",
            variable=self.keepalive_var,
            command=self.on_keepalive_toggle,
        )



        # --- Sliders for x, y, z, r ---
        self.scale_x = self._create_axis_slider(main, "x (forward/back)", 1)
        self.scale_y = self._create_axis_slider(main, "y (roll left/right)", 2)
        self.scale_z = self._create_axis_slider(main, "z (up/down)", 3)
        self.scale_r = self._create_axis_slider(main, "r (yaw)", 4)

        # --- Text input for single-segment x y z r ---
        ttk.Label(main, text="Single segment x y z r:").grid(row=5, column=0, sticky="e", pady=(10, 0))
        self.entry_axes = ttk.Entry(main, width=30)
        self.entry_axes.grid(row=5, column=1, columnspan=2, sticky="w", pady=(10, 0))
        self.entry_axes.insert(0, "0 0 0 0")

        btn_apply_axes = ttk.Button(main, text="Apply to sliders", command=self.apply_axes_from_text)
        btn_apply_axes.grid(row=5, column=3, sticky="w", padx=(5, 0), pady=(10, 0))

        # --- Duration for single pulse ---
        ttk.Label(main, text="Duration [sec] (for pulse):").grid(row=6, column=0, sticky="e", pady=(5, 0))
        self.entry_duration = ttk.Entry(main, width=10)
        self.entry_duration.grid(row=6, column=1, sticky="w", pady=(5, 0))
        self.entry_duration.insert(0, "1.0")

        # --- Main buttons row (single segment control) ---
        btn_set = ttk.Button(main, text="Set continuous", command=self.on_set_continuous)
        btn_set.grid(row=7, column=0, sticky="ew", pady=(10, 0))

        btn_pulse = ttk.Button(main, text="Send pulse", command=self.on_send_pulse)
        btn_pulse.grid(row=7, column=1, sticky="ew", pady=(10, 0))

        btn_zero = ttk.Button(main, text="Zero all", command=self.on_zero)
        btn_zero.grid(row=7, column=2, sticky="ew", pady=(10, 0))

        btn_quit = ttk.Button(main, text="Quit", command=self.on_quit)
        btn_quit.grid(row=7, column=3, sticky="ew", pady=(10, 0))

        # --- Arm / Disarm ---
        btn_arm = ttk.Button(main, text="Arm motors", command=self.on_arm)
        btn_arm.grid(row=8, column=0, sticky="ew", pady=(5, 0))

        btn_disarm = ttk.Button(main, text="Disarm", command=self.on_disarm)
        btn_disarm.grid(row=8, column=1, sticky="ew", pady=(5, 0))

        # --- Path editor area ---
        ttk.Label(main, text="Path editor (name x y z r duration, or x y z r duration):").grid(
            row=9, column=0, columnspan=4, sticky="w", pady=(10, 0)
        )

        self.path_text = tk.Text(main, width=60, height=8)
        self.path_text.grid(row=10, column=0, columnspan=4, sticky="nsew", pady=(3, 0))

        # Example default path A-B-C
        default_path = (
            "# Example:\n"
            "# name x y z r duration_sec\n"
            "A 200 0 100 0 3\n"
            "B 200 0 250 0 5\n"
            "C 0 0 0 0 4\n"
        )
        self.path_text.insert("1.0", default_path)

        # Path buttons
        btn_run_path = ttk.Button(main, text="Run path (editor)", command=self.on_run_path)
        btn_run_path.grid(row=11, column=0, sticky="ew", pady=(5, 0))

        btn_stop_path = ttk.Button(main, text="Stop path", command=self.on_stop_path)
        btn_stop_path.grid(row=11, column=1, sticky="ew", pady=(5, 0))

        btn_load_path = ttk.Button(main, text="Load path from file", command=self.on_load_path)
        btn_load_path.grid(row=11, column=2, sticky="ew", pady=(5, 0))

        # --- State label ---
        self.state_label = ttk.Label(main, text="State: (waiting for /state...)")
        self.state_label.grid(row=12, column=0, columnspan=4, sticky="w", pady=(10, 0))

        keepalive_check.grid(row=14, column=0, columnspan=2, sticky="w", pady=(5, 5))


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

    # ---------- Robot / Flight mode selector ----------

    def on_robot_change(self, event=None):
        rid = self.robot_var.get().strip()
        if not rid:
            return
        self.node.set_rooster_id(rid)
        self._show_temp_status(f"Switched to {rid}")

    def on_flight_mode_change(self, event=None):
        label = self.flight_mode_var.get()
        mode = self.flight_mode_map.get(label)
        if mode is None:
            return
        self.node.set_flight_mode(mode)
        self._show_temp_status(f"Requested flight mode {label}")

    def on_keepalive_toggle(self):
        enabled = self.keepalive_var.get()
        self.node.set_keepalive_enabled(enabled)
        msg = "KeepAlive ON" if enabled else "KeepAlive OFF"
        self._show_temp_status(msg)

    # ---------- Single segment handlers ----------

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

    def on_arm(self):
        # Throttle must be 0 before arming
        self.scale_z.set(0)
        self.node.zero_axes()
        self.node.call_force_arm(True)
        self._show_temp_status("Requested ARM (z=0). Check 'armed=True' in state.")

    def on_disarm(self):
        self.node.call_force_arm(False)
        self._show_temp_status("Requested DISARM.")

    # ---------- Path editing / running ----------

    def parse_path_editor(self) -> Optional[List[Tuple[str, float, float, float, float, float]]]:
        text = self.path_text.get("1.0", "end").strip()
        lines = text.splitlines()
        segments: List[Tuple[str, float, float, float, float, float]] = []
        auto_name_index = 1

        for idx, line in enumerate(lines):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split()
            # Expect either: x y z r dur  (5 tokens)
            # or: name x y z r dur        (6 tokens)
            if len(parts) == 5:
                name = f"S{auto_name_index}"
                auto_name_index += 1
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2])
                    r = float(parts[3])
                    dur = float(parts[4])
                except ValueError:
                    msg = f"Line {idx + 1}: cannot parse numbers."
                    self._show_temp_status(msg)
                    return None
            elif len(parts) == 6:
                name = parts[0]
                try:
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    r = float(parts[4])
                    dur = float(parts[5])
                except ValueError:
                    msg = f"Line {idx + 1}: cannot parse numbers."
                    self._show_temp_status(msg)
                    return None
            else:
                msg = f"Line {idx + 1}: expected 5 or 6 values."
                self._show_temp_status(msg)
                return None

            if dur <= 0:
                msg = f"Line {idx + 1}: duration must be > 0."
                self._show_temp_status(msg)
                return None

            segments.append((name, x, y, z, r, dur))

        if not segments:
            self._show_temp_status("No valid segments in path editor.")
            return None

        return segments

    def on_run_path(self):
        segments = self.parse_path_editor()
        if not segments:
            return
        self._show_temp_status("Running path from editor...")
        threading.Thread(
            target=self.node.run_path,
            args=(segments, "GUI path"),
            daemon=True,
        ).start()

    def on_stop_path(self):
        self.node.stop_path()
        self._show_temp_status("Stop path requested.")

    def on_load_path(self):
        filename = filedialog.askopenfilename(
            title="Select path file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not filename:
            return
        try:
            with open(filename, "r") as f:
                content = f.read()
        except Exception as e:
            self._show_temp_status(f"Failed to load file: {e}")
            return

        self.path_text.delete("1.0", "end")
        self.path_text.insert("1.0", content)
        self._show_temp_status(f"Loaded path from file: {filename}")

    # ---------- State label + status ----------

    def update_state_label(self):
        st = self.node.last_state
        if st is None:
            txt = "State: (no /state yet)"
        else:
            txt = (
                f"State [{self.node.rooster_id}]: "
                f"roll={st.roll:.2f}, pitch={st.pitch:.2f}, "
                f"azimuth={st.azimuth:.2f}, "
                f"mode={st.flight_mode}, "
                f"armed={st.armed}, airborne={st.airborne}"
            )
        self.state_label.config(text=txt)

        # schedule next update
        self.root.after(500, self.update_state_label)

    def _show_temp_status(self, msg: str):
        self.state_label.config(text=msg)


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
