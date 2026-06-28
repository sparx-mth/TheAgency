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
        self.root.geometry("400x860")

        self.forward_value_var = tk.StringVar(value="400")   # forward / backward / lateral
        self.vertical_value_var = tk.StringVar(value="400")  # up / down
        self.turn_value_var = tk.StringVar(value="1000")      # yaw
        self.timer_text = tk.StringVar(value="Current action: none")

        # Active Twist state — all 4 axes
        self._active_lx = 0.0   # forward / backward
        self._active_ly = 0.0   # lateral left / right  (linear.y)
        self._active_lz = 0.0   # up / down             (linear.z)
        self._active_az = 0.0   # yaw                   (angular.z)

        self.twist_publish_period_sec = 0.1  # 10 Hz
        self.last_twist_publish_t = 0.0

        # Safety guards
        self.land_pending_disarm = False
        self.land_sent_t = 0.0
        self.land_disarm_delay_sec = 10.0

        self.arm_pending_activity = False
        self.arm_sent_t = 0.0
        self.arm_idle_disarm_delay_sec = 30.0

        self.build_ui()

    def _any_active(self) -> bool:
        return any([self._active_lx, self._active_ly, self._active_lz, self._active_az])

    def build_ui(self):
        ttk.Label(
            self.root, text="XTEND ROS Command UI", font=("Arial", 13, "bold"),
        ).pack(pady=8)

        ttk.Label(
            self.root, textvariable=self.timer_text, font=("Arial", 11, "bold"), foreground="blue",
        ).pack(pady=6)

        # ── System controls ──────────────────────────────────
        ttk.Label(self.root, text="System Controls", font=("Arial", 12, "bold")).pack(pady=6)
        self._btn("ARM",     "arm",     "orange")
        self._btn("TAKEOFF", "takeoff", "green")
        self._btn("LAND",    "land",    "red")
        self._btn("DISARM",  "disarm",  "darkred")

        ttk.Separator(self.root, orient="horizontal").pack(fill="x", pady=8)

        # ── Thrust values ─────────────────────────────────────
        ttk.Label(self.root, text="Thrust Values", font=("Arial", 12, "bold")).pack(pady=4)
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
        BW, BH = 10, 2  # button width / height (chars)

        # Row 0 — UP in centre
        self._grid_btn(pad, "UP",       "up",       row=0, col=1)
        # Row 1 — lateral + forward
        self._grid_btn(pad, "◄ LEFT",   "left",     row=1, col=0)
        self._grid_btn(pad, "FORWARD",  "forward",  row=1, col=1, color="lightblue")
        self._grid_btn(pad, "RIGHT ►",  "right",    row=1, col=2)
        # Row 2 — backward
        self._grid_btn(pad, "BACKWARD", "backward", row=2, col=1)
        # Row 3 — DOWN in centre
        self._grid_btn(pad, "DOWN",     "down",     row=3, col=1)
        # Row 4 — yaw + stop
        self._grid_btn(pad, "↺ TURN L", "turn_left",  row=4, col=0)
        self._grid_btn(pad, "STOP",     "stop",       row=4, col=1, color="yellow")
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

    # ── Value helpers ─────────────────────────────────────────

    def _fwd_thrust(self) -> float:
        try:
            v = int(self.forward_value_var.get())
        except ValueError:
            v = 400
        return float(max(0, min(1000, v)))

    def _vert_thrust(self) -> float:
        try:
            v = int(self.vertical_value_var.get())
        except ValueError:
            v = 400
        return float(max(0, min(1000, v)))

    def _turn_rate(self) -> float:
        try:
            v = int(self.turn_value_var.get())
        except ValueError:
            v = 1000
        return float(max(0, min(1000, v)))

    # ── Button dispatch ───────────────────────────────────────

    def handle_button(self, action: str):
        t = self._fwd_thrust()
        v = self._vert_thrust()
        r = self._turn_rate()

        if action == "forward":
            self.clear_arm_idle_guard("forward")
            self._set_twist(lx=t / 400.0 * 0.3, name="forward")

        elif action == "backward":
            self.clear_arm_idle_guard("backward")
            self._set_twist(lx=-(t / 400.0 * 0.3), name="backward")

        elif action == "left":
            self.clear_arm_idle_guard("left")
            self._set_twist(ly=t / 400.0 * 0.3, name="lateral_left")

        elif action == "right":
            self.clear_arm_idle_guard("right")
            self._set_twist(ly=-(t / 400.0 * 0.3), name="lateral_right")

        elif action == "up":
            self.clear_arm_idle_guard("up")
            self._set_twist(lz=v / 400.0 * 0.3, name="up")

        elif action == "down":
            self.clear_arm_idle_guard("down")
            self._set_twist(lz=-(v / 400.0 * 0.3), name="down")

        elif action == "turn_left":
            self.clear_arm_idle_guard("turn_left")
            self._set_twist(az=r / 1000.0 * 0.65, name="turn_left")

        elif action == "turn_right":
            self.clear_arm_idle_guard("turn_right")
            self._set_twist(az=-(r / 1000.0 * 0.65), name="turn_right")

        elif action == "stop":
            self._zero_twist()
            self.send_cmd("stop", 0)
            self.stop_timer("stop")

        elif action == "arm":
            self.send_cmd("arm", 0)
            self.start_arm_idle_guard()

        elif action == "takeoff":
            self.clear_arm_idle_guard("takeoff")
            self.send_cmd("takeoff", 0)

        elif action == "land":
            self._zero_twist()
            self.send_cmd("land", 0)
            self.start_land_disarm_guard()
            self.stop_timer("land")

        elif action == "disarm":
            self._zero_twist()
            self.send_cmd("disarm", 0)
            self.clear_all_safety_guards("manual_disarm")
            self.stop_timer("disarm")

        else:
            self.send_cmd(action, 0)

    # ── Twist state ───────────────────────────────────────────

    def _set_twist(self, lx=0.0, ly=0.0, lz=0.0, az=0.0, name: str = ""):
        self._active_lx = float(lx)
        self._active_ly = float(ly)
        self._active_lz = float(lz)
        self._active_az = float(az)
        self.send_twist(lx, ly, lz, az)
        self.start_timer(name)

    def _zero_twist(self):
        self._active_lx = 0.0
        self._active_ly = 0.0
        self._active_lz = 0.0
        self._active_az = 0.0
        self.send_twist(0.0, 0.0, 0.0, 0.0)

    # ── Safety guards ─────────────────────────────────────────

    def start_land_disarm_guard(self):
        self.land_pending_disarm = True
        self.land_sent_t = time.time()
        self.get_logger().warn(
            f"LAND safety guard armed: auto-DISARM in {self.land_disarm_delay_sec:.1f}s"
        )

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

    def clear_all_safety_guards(self, reason: str):
        if self.land_pending_disarm or self.arm_pending_activity:
            self.get_logger().info(f"Safety guards cleared: {reason}")
        self.land_pending_disarm = False
        self.land_sent_t = 0.0
        self.arm_pending_activity = False
        self.arm_sent_t = 0.0

    def check_safety_guards(self):
        now = time.time()
        if self.land_pending_disarm and now - self.land_sent_t >= self.land_disarm_delay_sec:
            self.get_logger().warn("LAND safety guard triggered: sending DISARM.")
            self.send_cmd("disarm", 0)
            self.land_pending_disarm = False
            self.land_sent_t = 0.0
        if self.arm_pending_activity and now - self.arm_sent_t >= self.arm_idle_disarm_delay_sec:
            self.get_logger().warn("ARM idle guard triggered: sending DISARM.")
            self.send_cmd("disarm", 0)
            self.arm_pending_activity = False
            self.arm_sent_t = 0.0

    # ── ROS publish ───────────────────────────────────────────

    def send_cmd(self, action: str, value: int):
        msg = String()
        msg.data = json.dumps({"action": action, "value": int(value)})
        self.cmd_nav_pub.publish(msg)
        self.get_logger().info(f"cmd_nav: {msg.data}")

    def send_twist(self, linear_x=0.0, linear_y=0.0, linear_z=0.0, angular_z=0.0):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.linear.y = float(linear_y)
        msg.linear.z = float(linear_z)
        msg.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(msg)
        self.get_logger().info(
            f"Twist: lx={msg.linear.x:.3f} ly={msg.linear.y:.3f} "
            f"lz={msg.linear.z:.3f} az={msg.angular.z:.3f}"
        )

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
            now = time.time()
            if self._any_active():
                if now - self.last_twist_publish_t >= self.twist_publish_period_sec:
                    self.send_twist(
                        self._active_lx,
                        self._active_ly,
                        self._active_lz,
                        self._active_az,
                    )
                    self.last_twist_publish_t = now
            self.check_safety_guards()
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
