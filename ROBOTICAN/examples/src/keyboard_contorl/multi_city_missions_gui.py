#!/usr/bin/env python3

import os
import time
import threading
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node

from fcu_driver_interfaces.msg import ManualControl, UAVState
from rooster_handler_interfaces.msg import KeepAlive
from rooster_manager_interfaces.msg import RoosterState
from std_srvs.srv import SetBool

import tkinter as tk
from tkinter import ttk


@dataclass
class MissionSegment:
    name: str
    x: float
    y: float
    z: float
    r: float
    duration: float


@dataclass
class DroneController:
    rooster_id: str
    manual_pub: Optional[any] = None
    keep_alive_pub: Optional[any] = None
    state_sub: Optional[any] = None
    uav_state_sub: Optional[any] = None
    force_arm_client: Optional[any] = None

    flight_mode: int = 2  # MANUAL
    keepalive_enabled: bool = True

    # Control state
    current_x: float = 0.0
    current_y: float = 0.0
    current_z: float = 0.0
    current_r: float = 0.0

    pulse_active: bool = False
    pulse_end_time: float = 0.0

    # Mission state
    mission_running: bool = False
    mission_done: bool = False
    mission_error: Optional[str] = None
    stop_requested: bool = False

    last_state: Optional[RoosterState] = None
    last_uav_state: Optional[UAVState] = None

    # UI hook (set later)
    status_label: Optional[ttk.Label] = field(default=None, repr=False)


class MultiMissionNode(Node):
    """
    One node controlling three drones (R1, R2, R3),
    each with its own manual_control, keep_alive, state, force_arm, and mission.
    """

    def __init__(self, path_files):
        super().__init__("multi_city_missions_gui")

        self.path_files = path_files

        # Create controllers for R1, R2, R3
        self.controllers = {
            rid: DroneController(rooster_id=rid)
            for rid in ["R1", "R2", "R3"]
        }

        self._create_ros_handles()

        # Timers: manual_control at 40 Hz, keep_alive at 1 Hz
        self.manual_timer = self.create_timer(1.0 / 40.0, self.manual_timer_cb)
        self.keep_alive_timer = self.create_timer(1.0, self.keep_alive_timer_cb)

        self.get_logger().info("MultiMissionNode initialized for R1, R2, R3.")

    def _create_ros_handles(self):
        for rid, ctrl in self.controllers.items():
            manual_topic = f"/{rid}/manual_control"
            keep_alive_topic = f"/{rid}/keep_alive"
            state_topic = f"/{rid}/state"
            uav_state_topic = f"/{rid}/fcu/state"
            force_arm_service = f"/{rid}/fcu/command/force_arm"

            ctrl.manual_pub = self.create_publisher(ManualControl, manual_topic, 10)
            ctrl.keep_alive_pub = self.create_publisher(KeepAlive, keep_alive_topic, 10)

            ctrl.state_sub = self.create_subscription(
                RoosterState, state_topic,
                lambda msg, r=rid: self._state_callback(r, msg),
                10,
            )
            ctrl.uav_state_sub = self.create_subscription(
                UAVState, uav_state_topic,
                lambda msg, r=rid: self._uav_state_callback(r, msg),
                10,
            )
            ctrl.force_arm_client = self.create_client(SetBool, force_arm_service)

            self.get_logger().info(
                f"{rid}: pubs/subs created: {manual_topic}, {keep_alive_topic}, "
                f"{state_topic}, {uav_state_topic}, {force_arm_service}"
            )

    # ---------- Callbacks ----------

    def _state_callback(self, rid: str, msg: RoosterState):
        self.controllers[rid].last_state = msg

    def _uav_state_callback(self, rid: str, msg: UAVState):
        self.controllers[rid].last_uav_state = msg

    # ---------- Timers ----------

    def manual_timer_cb(self):
        now = time.time()
        for rid, ctrl in self.controllers.items():
            # Handle pulse auto-stop
            if ctrl.pulse_active and now >= ctrl.pulse_end_time:
                ctrl.current_x = 0.0
                ctrl.current_y = 0.0
                ctrl.current_z = 0.0
                ctrl.current_r = 0.0
                ctrl.pulse_active = False

            # Publish manual_control
            msg = ManualControl()
            msg.x = ctrl.current_x
            msg.y = ctrl.current_y
            msg.z = ctrl.current_z
            msg.r = ctrl.current_r
            msg.buttons = 0
            ctrl.manual_pub.publish(msg)

    def keep_alive_timer_cb(self):
        for rid, ctrl in self.controllers.items():
            if not ctrl.keepalive_enabled:
                continue

            msg = KeepAlive()
            msg.is_active = True
            msg.requested_flight_mode = int(ctrl.flight_mode)  # MANUAL = 2
            msg.command_reboot = False
            ctrl.keep_alive_pub.publish(msg)

    # ---------- High-level actions ----------

    def set_flight_mode_manual(self, rid: str):
        ctrl = self.controllers[rid]
        ctrl.flight_mode = 2  # MANUAL
        self.get_logger().info(f"{rid}: set flight_mode=MANUAL (2)")

    def call_force_arm(self, rid: str, arm: bool):
        ctrl = self.controllers[rid]
        client = ctrl.force_arm_client

        if not client.service_is_ready():
            self.get_logger().warn(f"{rid}: force_arm service not ready")
            return

        req = SetBool.Request()
        req.data = arm

        future = client.call_async(req)

        def _done(fut):
            try:
                resp = fut.result()
            except Exception as e:
                self.get_logger().error(f"{rid}: force_arm({arm}) failed: {e}")
                return

            if resp.success:
                self.get_logger().info(
                    f"{rid}: force_arm({arm}) success: {resp.message}"
                )
            else:
                self.get_logger().warn(
                    f"{rid}: force_arm({arm}) reported failure: {resp.message}"
                )

        future.add_done_callback(_done)

    # Parsing path file: each line = name x y z r duration
    def load_path_for(self, rid: str) -> List[MissionSegment]:
        path_file = self.path_files.get(rid)
        if path_file is None:
            raise RuntimeError(f"No path file defined for {rid}")

        path_file = os.path.abspath(path_file)
        if not os.path.exists(path_file):
            raise FileNotFoundError(f"{rid}: path file not found: {path_file}")

        segments: List[MissionSegment] = []
        with open(path_file, "r") as f:
            for idx, line in enumerate(f, start=1):
                raw = line.strip()
                if not raw or raw.startswith("#"):
                    continue
                parts = raw.split()
                if len(parts) != 6:
                    raise ValueError(
                        f"{rid}: line {idx}: expected 6 tokens "
                        f"(name x y z r duration), got {len(parts)}"
                    )
                name = parts[0]
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                r = float(parts[4])
                duration = float(parts[5])
                segments.append(MissionSegment(name, x, y, z, r, duration))

        if not segments:
            raise RuntimeError(f"{rid}: path file {path_file} has no segments")

        self.get_logger().info(f"{rid}: loaded {len(segments)} segments from {path_file}")
        return segments

    # Mission runner (blocking, run in a thread)
    def run_mission(self, rid: str):
        ctrl = self.controllers[rid]
        if ctrl.mission_running:
            self.get_logger().warn(f"{rid}: mission already running")
            return

        try:
            segments = self.load_path_for(rid)
        except Exception as e:
            ctrl.mission_error = str(e)
            ctrl.mission_running = False
            ctrl.mission_done = False
            self._update_status_label(rid, f"Error: {e}")
            self.get_logger().error(f"{rid}: failed to load path: {e}")
            return

        ctrl.mission_running = True
        ctrl.mission_done = False
        ctrl.mission_error = None
        ctrl.stop_requested = False
        self._update_status_label(rid, "Running")

        self.set_flight_mode_manual(rid)

        # Arm
        self.call_force_arm(rid, True)
        time.sleep(2.0)  # small delay for safety in sim

        self.get_logger().info(f"{rid}: starting mission with {len(segments)} segments")

        try:
            for seg in segments:
                if ctrl.stop_requested or not rclpy.ok():
                    break

                self.get_logger().info(
                    f"{rid}: segment {seg.name}: x={seg.x}, y={seg.y}, "
                    f"z={seg.z}, r={seg.r}, dur={seg.duration}s"
                )

                now = time.time()
                ctrl.current_x = seg.x
                ctrl.current_y = seg.y
                ctrl.current_z = seg.z
                ctrl.current_r = seg.r
                ctrl.pulse_active = True
                ctrl.pulse_end_time = now + seg.duration

                end = now + seg.duration
                while time.time() < end and rclpy.ok() and not ctrl.stop_requested:
                    time.sleep(0.05)

            # End of mission: zero axes and disarm
            ctrl.current_x = 0.0
            ctrl.current_y = 0.0
            ctrl.current_z = 0.0
            ctrl.current_r = 0.0
            ctrl.pulse_active = False

            self.call_force_arm(rid, False)
            time.sleep(1.0)

            if ctrl.stop_requested:
                self._update_status_label(rid, "Stopped")
            else:
                ctrl.mission_done = True
                self._update_status_label(rid, "Done")
            self.get_logger().info(f"{rid}: mission finished")
        except Exception as e:
            ctrl.mission_error = str(e)
            self._update_status_label(rid, f"Error: {e}")
            self.get_logger().error(f"{rid}: mission error: {e}")
        finally:
            ctrl.mission_running = False
            ctrl.stop_requested = False

    def request_stop_mission(self, rid: str):
        ctrl = self.controllers[rid]
        if ctrl.mission_running:
            ctrl.stop_requested = True
            self.get_logger().info(f"{rid}: stop mission requested")
            self._update_status_label(rid, "Stopping...")

    # UI status helpers
    def _update_status_label(self, rid: str, text: str):
        ctrl = self.controllers[rid]
        if ctrl.status_label is not None:
            try:
                ctrl.status_label.after(
                    0, lambda lbl=ctrl.status_label, t=text: lbl.config(text=f"Status: {t}")
                )
            except Exception:
                pass


class MultiMissionGUI:
    """
    Tkinter UI with 3 panels (R1, R2, R3) to launch city paths.
    """

    def __init__(self, node: MultiMissionNode):
        self.node = node

        self.root = tk.Tk()
        self.root.title("Multi-Drone City Missions")

        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Three columns, one per drone
        col = 0
        for rid in ["R1", "R2", "R3"]:
            frame = ttk.LabelFrame(main, text=rid, padding=10)
            frame.grid(row=0, column=col, padx=5, pady=5, sticky="nsew")
            main.columnconfigure(col, weight=1)
            self._build_drone_panel(frame, rid)
            col += 1

        # Optional global buttons
        global_frame = ttk.Frame(main, padding=5)
        global_frame.grid(row=1, column=0, columnspan=3, sticky="ew")

        btn_all_start = ttk.Button(global_frame, text="Start ALL missions", command=self.on_start_all)
        btn_all_start.grid(row=0, column=0, padx=5)

        btn_all_stop = ttk.Button(global_frame, text="Stop ALL missions", command=self.on_stop_all)
        btn_all_stop.grid(row=0, column=1, padx=5)

        btn_quit = ttk.Button(global_frame, text="Quit", command=self.on_quit)
        btn_quit.grid(row=0, column=2, padx=5)

    def _build_drone_panel(self, frame: ttk.Frame, rid: str):
        ctrl = self.node.controllers[rid]

        # Path file label
        path_file = self.node.path_files.get(rid, "N/A")
        ttk.Label(frame, text=f"Path file:\n{path_file}").grid(row=0, column=0, columnspan=2, sticky="w")

        # Status label
        status_label = ttk.Label(frame, text="Status: Idle")
        status_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(5, 5))
        ctrl.status_label = status_label

        # Buttons: Start, Stop
        btn_start = ttk.Button(frame, text="Start mission", command=lambda r=rid: self.on_start(r))
        btn_start.grid(row=2, column=0, sticky="ew", pady=3)

        btn_stop = ttk.Button(frame, text="Stop mission", command=lambda r=rid: self.on_stop(r))
        btn_stop.grid(row=2, column=1, sticky="ew", pady=3)

        # Show some state info (read-only)
        info_label = ttk.Label(frame, text="Mode / armed info from /state will appear in logs.")
        info_label.grid(row=3, column=0, columnspan=2, sticky="w", pady=(5, 0))

    # Button handlers

    def on_start(self, rid: str):
        ctrl = self.node.controllers[rid]
        if ctrl.mission_running:
            self.node._update_status_label(rid, "Already running")
            return
        self.node._update_status_label(rid, "Starting...")
        t = threading.Thread(target=self.node.run_mission, args=(rid,), daemon=True)
        t.start()

    def on_stop(self, rid: str):
        self.node.request_stop_mission(rid)

    def on_start_all(self):
        for rid in ["R1", "R2", "R3"]:
            self.on_start(rid)

    def on_stop_all(self):
        for rid in ["R1", "R2", "R3"]:
            self.on_stop(rid)

    def on_quit(self):
        self.root.destroy()


def ros_spin_thread(node: MultiMissionNode):
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
    node.get_logger().info("ROS spin thread exiting")


def main():
    # Map each drone to its path file
    path_files = {
        "R1": "txt/city_path_1.txt",
        "R2": "txt/city_path_2.txt",
        "R3": "txt/city_path_3.txt",
    }

    rclpy.init()
    node = MultiMissionNode(path_files=path_files)

    # Start ROS spinning in background
    t = threading.Thread(target=ros_spin_thread, args=(node,), daemon=True)
    t.start()

    gui = MultiMissionGUI(node)
    try:
        gui.root.mainloop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
