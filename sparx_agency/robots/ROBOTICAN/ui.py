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

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class RoosterControlUI(Node):
    def __init__(self):
        super().__init__("rooster_gui_publisher")

        self.declare_parameter("rooster_id", "R1")
        self.rooster_id = self.get_parameter("rooster_id").value

        self.cmd_pub = self.create_publisher(String, f"/{self.rooster_id}/cmd_nav", 10)
        self.status_sub = self.create_subscription(
            String, f"/{self.rooster_id}/rooster_status", self._on_status, 10)

        self.armed = False
        self.airborne = False

        self.active_action = None
        self.active_action_start_t = None

        # Idle safety guard: auto-DISARM if armed but nothing happens for a
        # while (mirrors robots/XTEND/ui.py's arm-idle guard).
        self.arm_pending_activity = False
        self.arm_sent_t = 0.0
        self.arm_idle_disarm_delay_sec = 30.0

        self.root = tk.Tk()
        self.root.title(f"ROBOTICAN {self.rooster_id} ROS Command UI")
        self.root.geometry("400x900")

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
        rclpy.shutdown()


if __name__ == "__main__":
    main()
