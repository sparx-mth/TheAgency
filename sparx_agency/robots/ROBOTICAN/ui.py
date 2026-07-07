#!/usr/bin/env python3
"""ui.py

Tkinter flight-control panel for one ROBOTICAN Rooster drone. Mirrors
robots/XTEND/ui.py's layout and idle-safety-guard pattern.

This UI has no direct FCU access: every button only publishes a JSON
action to /<rooster_id>/cmd_nav, and status is read back from
/<rooster_id>/rooster_status. That is the same channel a planner node
will use, so this UI and a planner are just two peers talking to the
same RoosterCommandUnitNode (see adapters/rooster_command_unit.py).
"""

import json
import time
import tkinter as tk
from tkinter import ttk

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GstVideo

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

Gst.init(None)

BATTERY_GREEN = "#4CAF50"
BATTERY_ORANGE = "#FF9800"
BATTERY_RED = "#F44336"


class RoosterControlUI(Node):
    def __init__(self):
        super().__init__("rooster_gui_publisher")

        self.declare_parameter("rooster_id", "R1")
        self.declare_parameter("video_port", 5001)
        self.declare_parameter("video_width", 540)
        self.declare_parameter("video_height", 360)
        self.rooster_id = self.get_parameter("rooster_id").value
        self.video_port = int(self.get_parameter("video_port").value)
        self.video_width = int(self.get_parameter("video_width").value)
        self.video_height = int(self.get_parameter("video_height").value)

        self.cmd_pub = self.create_publisher(String, f"/{self.rooster_id}/cmd_nav", 10)
        self.status_sub = self.create_subscription(
            String, f"/{self.rooster_id}/rooster_status", self._on_status, 10)

        self.armed = False
        self.airborne = False
        self.video_on = False
        self.battery_pct = 0.0
        self.battery_voltage = 0.0

        # Embedded GStreamer pipeline (renders into self.video_frame via
        # ximagesink window-handle overlay) - only tracks whether *this* UI
        # instance is playing one; the drone's actual video on/off state
        # comes back via rooster_status.video_on.
        self.video_pipeline = None
        self.video_placeholder = None

        self.active_action = None
        self.active_action_start_t = None

        # Idle safety guard: auto-DISARM if armed but nothing happens for a
        # while (mirrors robots/XTEND/ui.py's arm-idle guard).
        self.arm_pending_activity = False
        self.arm_sent_t = 0.0
        self.arm_idle_disarm_delay_sec = 30.0

        self.root = tk.Tk()
        self.root.title(f"ROBOTICAN {self.rooster_id} ROS Command UI")
        window_width = max(400, self.video_width + 40)
        self.root.geometry(f"{window_width}x{1200 + max(0, self.video_height - 240)}")

        self.forward_value_var = tk.StringVar(value="400")  # forward/back/lateral
        self.vertical_value_var = tk.StringVar(value="400")  # up/down
        self.turn_value_var = tk.StringVar(value="150")      # yaw
        self.timer_text = tk.StringVar(value="Current action: none")
        self.status_text = tk.StringVar(value="armed: ?   airborne: ?")

        self.build_ui()

    def build_ui(self):
        ttk.Label(
            self.root, text=f"ROBOTICAN {self.rooster_id} Command UI",
            font=("Arial", 13, "bold"),
        ).pack(pady=8)

        ttk.Label(
            self.root, textvariable=self.status_text, font=("Arial", 11, "bold"),
            foreground="darkgreen",
        ).pack(pady=4)

        self.battery_canvas = tk.Canvas(
            self.root, width=300, height=24, highlightthickness=1, highlightbackground="black",
        )
        self.battery_canvas.pack(pady=4)
        self.battery_rect = self.battery_canvas.create_rectangle(0, 0, 0, 24, fill="grey", width=0)
        self.battery_label_id = self.battery_canvas.create_text(150, 12, text="battery: ?")

        # ── Camera ────────────────────────────────────────────
        ttk.Label(self.root, text="Camera", font=("Arial", 12, "bold")).pack(pady=(8, 2))
        self.video_frame = tk.Frame(
            self.root, width=self.video_width, height=self.video_height, bg="black")
        self.video_frame.pack(pady=4)
        self.video_frame.pack_propagate(False)
        self.video_placeholder = tk.Label(
            self.video_frame, text="VIDEO OFF", fg="white", bg="black", font=("Arial", 12, "bold"),
        )
        self.video_placeholder.pack(expand=True)

        ttk.Label(
            self.root, textvariable=self.timer_text, font=("Arial", 11, "bold"),
            foreground="blue",
        ).pack(pady=6)

        # ── System controls ──────────────────────────────────
        ttk.Label(self.root, text="System Controls", font=("Arial", 12, "bold")).pack(pady=6)
        self._btn("ARM", "arm", "orange")
        self._btn("TAKEOFF", "takeoff", "green")
        self._btn("LAND", "land", "red")
        self._btn("DISARM", "disarm", "darkred")

        self.video_btn = tk.Button(
            self.root, text="VIDEO: OFF", command=self.on_video_button, bg="lightgrey", width=24,
        )
        self.video_btn.pack(pady=4)

        ttk.Separator(self.root, orient="horizontal").pack(fill="x", pady=8)

        # ── Axis values ───────────────────────────────────────
        ttk.Label(self.root, text="Axis Values", font=("Arial", 12, "bold")).pack(pady=4)
        tf = ttk.Frame(self.root)
        tf.pack(pady=4)
        ttk.Label(tf, text="Fwd/Back/Lateral:").grid(row=0, column=0, padx=4, pady=3, sticky="e")
        ttk.Entry(tf, textvariable=self.forward_value_var, width=7).grid(row=0, column=1, padx=4)
        ttk.Label(tf, text="Vertical:").grid(row=1, column=0, padx=4, pady=3, sticky="e")
        ttk.Entry(tf, textvariable=self.vertical_value_var, width=7).grid(row=1, column=1, padx=4)
        ttk.Label(tf, text="Turn rate:").grid(row=2, column=0, padx=4, pady=3, sticky="e")
        ttk.Entry(tf, textvariable=self.turn_value_var, width=7).grid(row=2, column=1, padx=4)

        ttk.Separator(self.root, orient="horizontal").pack(fill="x", pady=8)

        # ── Movement d-pad ────────────────────────────────────
        ttk.Label(self.root, text="Movement", font=("Arial", 12, "bold")).pack(pady=4)

        pad = ttk.Frame(self.root)
        pad.pack(pady=4)

        self._grid_btn(pad, "UP", "up", row=0, col=1)
        self._grid_btn(pad, "◄ LEFT", "left", row=1, col=0)
        self._grid_btn(pad, "FORWARD", "forward", row=1, col=1, color="lightblue")
        self._grid_btn(pad, "RIGHT ►", "right", row=1, col=2)
        self._grid_btn(pad, "BACKWARD", "backward", row=2, col=1)
        self._grid_btn(pad, "DOWN", "down", row=3, col=1)
        self._grid_btn(pad, "↺ TURN L", "turn_left", row=4, col=0)
        self._grid_btn(pad, "STOP", "stop", row=4, col=1, color="yellow")
        self._grid_btn(pad, "TURN R ↻", "turn_right", row=4, col=2)

        for r in range(5):
            pad.rowconfigure(r, pad=4)
        for c in range(3):
            pad.columnconfigure(c, pad=4)

    # ── Widget helpers ────────────────────────────────────────

    def _btn(self, label: str, action: str, color: str = "lightgrey"):
        tk.Button(
            self.root, text=label, command=lambda a=action: self.handle_button(a),
            bg=color, width=24,
        ).pack(pady=4)

    def _grid_btn(self, parent, label: str, action: str, row: int, col: int, color: str = "lightgrey"):
        tk.Button(
            parent, text=label, command=lambda a=action: self.handle_button(a),
            bg=color, width=10, height=2,
        ).grid(row=row, column=col, padx=3, pady=3)

    def _axis_value(self, var: tk.StringVar, default: float) -> float:
        try:
            return float(max(0, min(1000, int(var.get()))))
        except ValueError:
            return default

    # ── Button dispatch ───────────────────────────────────────

    def handle_button(self, action: str):
        if action in ("forward", "backward", "left", "right"):
            self.clear_arm_idle_guard(action)
            self.send_cmd(action, self._axis_value(self.forward_value_var, 400))
            self.start_timer(action)

        elif action in ("up", "down"):
            self.clear_arm_idle_guard(action)
            self.send_cmd(action, self._axis_value(self.vertical_value_var, 400))
            self.start_timer(action)

        elif action in ("turn_left", "turn_right"):
            self.clear_arm_idle_guard(action)
            self.send_cmd(action, self._axis_value(self.turn_value_var, 150))
            self.start_timer(action)

        elif action == "stop":
            self.send_cmd("stop", 0)
            self.stop_timer("stop")

        elif action == "arm":
            self.send_cmd("arm", 0)
            self.start_arm_idle_guard()

        elif action == "takeoff":
            self.clear_arm_idle_guard("takeoff")
            self.send_cmd("takeoff", 0)
            self.start_timer("takeoff")

        elif action == "land":
            self.clear_arm_idle_guard("land")
            self.send_cmd("land", 0)
            self.start_timer("land")

        elif action == "disarm":
            self.send_cmd("disarm", 0)
            self.clear_arm_idle_guard("manual_disarm")
            self.stop_timer("disarm")

        else:
            self.send_cmd(action, 0)

    # ── Video ──────────────────────────────────────────────────

    def on_video_button(self):
        if self.video_on:
            self.send_cmd("video_off", 0)
            self._stop_video_viewer()
        else:
            self.send_cmd("video_on", 0)
            self._start_video_viewer()

    def _start_video_viewer(self):
        """Play the RTP/H264 stream the drone sends to this host directly
        into self.video_frame, via ximagesink's window-handle overlay -
        same RTP/H264 pipeline as the ROBOTICAN video_example.py demo, just
        rendered into this widget instead of a separate autovideosink window."""
        if self.video_pipeline is not None:
            return
        if self.video_placeholder is not None:
            self.video_placeholder.destroy()
            self.video_placeholder = None
        self.video_frame.update_idletasks()
        # ximagesink doesn't reliably scale via set_render_rectangle (it's a
        # software sink, not xvimagesink) - it just clips at native pixel
        # size, showing e.g. only the top-left quarter of a 640x360 stream
        # in a 320x180 window. Force the scale inside the pipeline instead,
        # to the embedding widget's actual size, so the sink displays an
        # already-correctly-sized image 1:1.
        width = self.video_frame.winfo_width()
        height = self.video_frame.winfo_height()
        pipeline_str = (
            f"udpsrc port={self.video_port} "
            "caps=application/x-rtp,media=(string)video,clock-rate=(int)90000,"
            "encoding-name=(string)H264,payload=(int)96 ! "
            "rtph264depay ! decodebin ! videoconvert ! "
            # add-borders=false: the real stream's aspect doesn't exactly
            # match the panel's, so the default letterbox padding was
            # showing as black bars top/bottom. Stretch to fill instead.
            f"videoscale add-borders=false ! video/x-raw,width={width},height={height} ! "
            "ximagesink name=sink sync=false"
        )
        try:
            self.video_pipeline = Gst.parse_launch(pipeline_str)
        except Exception as e:
            self.get_logger().error(f"Failed to build video pipeline: {e}")
            self.video_pipeline = None
            return
        sink = self.video_pipeline.get_by_name("sink")
        GstVideo.VideoOverlay.set_window_handle(sink, self.video_frame.winfo_id())
        self.video_pipeline.set_state(Gst.State.PLAYING)
        self.get_logger().info(f"Video embedded, playing on port {self.video_port}")

    def _stop_video_viewer(self):
        if self.video_pipeline is not None:
            self.video_pipeline.set_state(Gst.State.NULL)
            self.video_pipeline = None
        if self.video_placeholder is None:
            self.video_placeholder = tk.Label(
                self.video_frame, text="VIDEO OFF", fg="white", bg="black", font=("Arial", 12, "bold"),
            )
            self.video_placeholder.pack(expand=True)

    # ── Safety guard ──────────────────────────────────────────

    def start_arm_idle_guard(self):
        self.arm_pending_activity = True
        self.arm_sent_t = time.time()
        self.get_logger().warn(
            f"ARM idle guard armed: auto-DISARM in {self.arm_idle_disarm_delay_sec:.1f}s if idle"
        )

    def clear_arm_idle_guard(self, reason: str):
        if self.arm_pending_activity:
            self.get_logger().info(f"ARM idle guard cleared: {reason}")
        self.arm_pending_activity = False
        self.arm_sent_t = 0.0

    def check_safety_guards(self):
        now = time.time()
        if self.arm_pending_activity and now - self.arm_sent_t >= self.arm_idle_disarm_delay_sec:
            self.get_logger().warn("ARM idle guard triggered: sending DISARM.")
            self.send_cmd("disarm", 0)
            self.arm_pending_activity = False
            self.arm_sent_t = 0.0

    # ── ROS I/O ───────────────────────────────────────────────

    def send_cmd(self, action: str, value: float):
        msg = String()
        msg.data = json.dumps({"action": action, "value": float(value)})
        self.cmd_pub.publish(msg)
        self.get_logger().info(f"cmd_nav: {msg.data}")

    def _on_status(self, msg: String):
        try:
            status = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self.armed = status.get("armed", False)
        self.airborne = status.get("airborne", False)
        self.status_text.set(f"armed: {self.armed}   airborne: {self.airborne}")

        self.video_on = status.get("video_on", False)
        self.video_btn.config(
            text=f"VIDEO: {'ON' if self.video_on else 'OFF'}",
            bg="lightgreen" if self.video_on else "lightgrey",
        )

        self.battery_pct = float(status.get("battery_pct", 0.0))
        self.battery_voltage = float(status.get("battery_voltage", 0.0))
        self._update_battery_bar()

    def _update_battery_bar(self):
        pct = max(0.0, min(1.0, self.battery_pct))
        if pct > 0.30:
            color = BATTERY_GREEN
        elif pct >= 0.15:
            color = BATTERY_ORANGE
        else:
            color = BATTERY_RED
        self.battery_canvas.coords(self.battery_rect, 0, 0, 300 * pct, 24)
        self.battery_canvas.itemconfig(self.battery_rect, fill=color)
        self.battery_canvas.itemconfig(
            self.battery_label_id, text=f"{pct * 100:.0f}%   {self.battery_voltage:.2f}V")

    # ── Timer label ───────────────────────────────────────────

    def start_timer(self, action_name: str):
        self.active_action = action_name
        self.active_action_start_t = time.time()
        self.timer_text.set(f"Current action: {action_name} | 0.000s")

    def stop_timer(self, reason: str):
        if self.active_action and self.active_action_start_t:
            elapsed = time.time() - self.active_action_start_t
            self.get_logger().info(
                f"Action ended: {self.active_action}, {elapsed:.3f}s, reason={reason}"
            )
        self.active_action = None
        self.active_action_start_t = None
        self.timer_text.set("Current action: none")

    def update_timer_label(self):
        if self.active_action and self.active_action_start_t:
            elapsed = time.time() - self.active_action_start_t
            self.timer_text.set(f"Current action: {self.active_action} | {elapsed:.3f}s")

    # ── Main loop ─────────────────────────────────────────────

    def run(self):
        while rclpy.ok():
            self.check_safety_guards()
            self.update_timer_label()
            self.root.update_idletasks()
            self.root.update()
            rclpy.spin_once(self, timeout_sec=0.01)


def main():
    rclpy.init()
    gui = RoosterControlUI()
    try:
        gui.run()
    except tk.TclError:
        pass
    finally:
        gui._stop_video_viewer()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
