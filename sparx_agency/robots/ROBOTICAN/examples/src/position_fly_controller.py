#!/usr/bin/env python3
"""
position_fly_controller.py
==========================
Fly one or more Rooster drones in POSITION mode (FLIGHT_MODE_POSITION = 3).

Supports up to N drones (default: R1, R2).  One drone is "active" at a time;
the keyboard always talks to the active drone.  Press 1/2/…/N to switch.
Press 'b' to toggle BROADCAST mode – in broadcast every keyboard command and
every path segment is sent to ALL drones simultaneously.

Keyboard bindings
-----------------
  1 / 2 / …  – select active drone (by index)
  b           – toggle broadcast mode (send to all drones)
  w / s       – forward / backward   (x, accumulates)
  j / l       – strafe left / right  (y, accumulates)
  i / k       – climb / descend      (z, accumulates)
  a / d       – yaw CCW / CW         (r, MOMENTARY – auto-resets after 0.5 s)
  SPACE       – zero all axes / stop path / exit hover-lock
  h           – HOVER-LOCK: zero axes and block keyboard until SPACE
  f           – arm + takeoff on active (or all, if broadcast)
  e           – DISARM active (or all, if broadcast)
  p           – load a path file and run it on active (or all, if broadcast)
  t           – toggle turtle mode
  q           – quit

Path file format (compatible with city_path_2.txt)
---------------------------------------------------
  # comment
  name  x  y  z  r  duration_sec   (6 tokens)
  x  y  z  r  duration_sec         (5 tokens, name auto-assigned)

ROS 2 parameters
-----------------
  rooster_ids        str    "R1,R2"   comma-separated list of drone IDs
  path_file          str    ""        global auto-run path (all drones, fallback)
  path_file_<ID>     str    ""        per-drone auto-run path, e.g. path_file_R1
                                      If set, this drone auto-runs its path after
                                      takeoff while other drones stay manual.
                                      Overrides the global path_file for that drone.
  step               float  50.0      keyboard axis increment
  climb_z            float  600.0     takeoff climb value
  hover_z            float  550.0     hover z after climb/path
  log_dir            str    "."       directory for CSV logs
"""

import sys
import csv
import time
import threading
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import rclpy
from rclpy.node import Node

from std_srvs.srv import SetBool
from fcu_driver_interfaces.msg import ManualControl
from rooster_handler_interfaces.msg import KeepAlive
from rooster_manager_interfaces.msg import RoosterState


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FLIGHT_MODE_POSITION = 3
PUBLISH_RATE_HZ = 40.0
KEEP_ALIVE_RATE_HZ = 1.0
MAX_AXIS = 1000.0
MIN_AXIS = -1000.0
TURTLE_SCALE = 0.4

# Yaw momentary control: each keypress sets r to this value for YAW_TIMEOUT seconds
YAW_RATE = 150.0      # axis units (out of 1000) – tune to taste
YAW_TIMEOUT = 0.5     # seconds before r auto-resets to 0 after last key press


# ---------------------------------------------------------------------------
# Per-drone axis model
# ---------------------------------------------------------------------------
class AxisModel:
    def __init__(self, step: float):
        self._x = self._y = self._z = self._r = 0.0
        self.step = step
        self.turtle = False

    def set(self, x, y, z, r):
        self._x = self._clamp(x)
        self._y = self._clamp(y)
        self._z = self._clamp(z)
        self._r = self._clamp(r)

    def reset(self):
        self._x = self._y = self._z = self._r = 0.0

    def increment(self, axis: str, direction: int):
        """Accumulating increment for x/y/z. NOT used for r (use set_yaw instead)."""
        delta = self.step * direction
        if axis == "x":   self._x = self._clamp(self._x + delta)
        elif axis == "y": self._y = self._clamp(self._y + delta)
        elif axis == "z": self._z = self._clamp(self._z + delta)

    def set_yaw(self, direction: int):
        """Set yaw to a fixed momentary rate (+/-). Call zero_yaw() to stop."""
        self._r = self._clamp(YAW_RATE * direction)

    def zero_yaw(self):
        """Reset yaw to 0 (called by auto-reset timer)."""
        self._r = 0.0

    def toggle_turtle(self) -> bool:
        self.turtle = not self.turtle
        return self.turtle

    def scaled(self) -> Tuple[float, float, float, float]:
        s = TURTLE_SCALE if self.turtle else 1.0
        return self._x * s, self._y * s, self._z * s, self._r * s

    @property
    def raw(self) -> Tuple[float, float, float, float]:
        return self._x, self._y, self._z, self._r

    @staticmethod
    def _clamp(v: float) -> float:
        from sparx_agency.robots.common.helpers import clamp
        return clamp(v, MIN_AXIS, MAX_AXIS)


# ---------------------------------------------------------------------------
# CSV logger (one per drone)
# ---------------------------------------------------------------------------
class CsvLogger:
    HEADER = ["wall_time", "drone", "x", "y", "z", "r",
              "armed", "flight_mode", "airborne", "roll", "pitch", "azimuth"]

    def __init__(self, path: str, drone_id: str):
        self._drone = drone_id
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADER)

    def log(self, x, y, z, r, state: Optional[RoosterState], wall_time: float):
        armed = fm = airborne = roll = pitch = azimuth = ""
        if state:
            armed, fm, airborne = state.armed, state.flight_mode, state.airborne
            roll, pitch, azimuth = round(state.roll, 3), round(state.pitch, 3), round(state.azimuth, 3)
        self._writer.writerow([round(wall_time, 4), self._drone, x, y, z, r,
                                armed, fm, airborne, roll, pitch, azimuth])

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Per-drone handle (publishers, state, axes, arm, path)
# ---------------------------------------------------------------------------
class DroneHandle:
    def __init__(self, drone_id: str, node: Node, step: float,
                 climb_z: float, hover_z: float, log_dir: str,
                 auto_path_file: Optional[str] = None):
        self.id = drone_id
        self.node = node
        self.climb_z = climb_z
        self.hover_z = hover_z

        # Per-drone auto-run path file (set from path_file_<ID> or global path_file)
        self.auto_path_file: Optional[str] = auto_path_file

        # Axes
        self.axes = AxisModel(step)

        # ROS I/O
        rid = drone_id
        self.manual_pub = node.create_publisher(ManualControl, f"/{rid}/manual_control", 10)
        self.keep_alive_pub = node.create_publisher(KeepAlive, f"/{rid}/keep_alive", 10)
        self.state_sub = node.create_subscription(
            RoosterState, f"/{rid}/state", self._state_cb, 10)
        self.force_arm_client = node.create_client(SetBool, f"/{rid}/fcu/command/force_arm")

        # State
        self.last_state: Optional[RoosterState] = None
        self.armed = False
        self.arm_pending = False
        self.takeoff_done = False
        self.path_running = False
        self.path_stop_flag = False

        # Yaw auto-reset timer (cancelled/restarted on each yaw keypress)
        self._yaw_timer: Optional[threading.Timer] = None

        # Logger
        log_path = str(Path(log_dir) / f"position_fly_{drone_id}.csv")
        self.logger = CsvLogger(log_path, drone_id)

        path_info = f"  auto-path: {auto_path_file}" if auto_path_file else "  (manual keyboard)"
        node.get_logger().info(f"DroneHandle created for {drone_id}{path_info}")

    def _state_cb(self, msg: RoosterState):
        self.last_state = msg
        self.armed = msg.armed

    # ---- publish helpers ----

    def publish_manual(self):
        x, y, z, r = self.axes.scaled()
        msg = ManualControl()
        msg.x, msg.y, msg.z, msg.r, msg.buttons = x, y, z, r, 0
        self.manual_pub.publish(msg)
        self.logger.log(x, y, z, r, self.last_state, time.time())

    def publish_keep_alive(self):
        msg = KeepAlive()
        msg.is_active = True
        msg.requested_flight_mode = FLIGHT_MODE_POSITION
        msg.command_reboot = False
        self.keep_alive_pub.publish(msg)

    # ---- yaw momentary control ----

    def apply_yaw(self, direction: int):
        """Set a momentary yaw rate; auto-resets to 0 after YAW_TIMEOUT seconds."""
        self.axes.set_yaw(direction)
        if self._yaw_timer is not None:
            self._yaw_timer.cancel()
        self._yaw_timer = threading.Timer(YAW_TIMEOUT, self._yaw_expire)
        self._yaw_timer.daemon = True
        self._yaw_timer.start()

    def _yaw_expire(self):
        self.axes.zero_yaw()
        self._yaw_timer = None

    # ---- arm + takeoff ----

    def request_arm_and_takeoff(self, on_done=None):
        """Kick off arm→climb sequence.  on_done() called when climb finishes."""
        if self.arm_pending:
            self.node.get_logger().warn(f"[{self.id}] Arm already in progress.")
            return
        if self.armed:
            self.node.get_logger().warn(f"[{self.id}] Already armed.")
            return
        self.axes.reset()
        self.arm_pending = True
        self.node.get_logger().info(f"[{self.id}] Zeroed axes, requesting ARM...")

        def _do_arm():
            if not self.force_arm_client.service_is_ready():
                self.node.get_logger().warn(f"[{self.id}] force_arm service not ready, waiting 2s...")
                time.sleep(2.0)
            req = SetBool.Request()
            req.data = True
            future = self.force_arm_client.call_async(req)

            def _done(fut):
                try:
                    resp = fut.result()
                except Exception as e:
                    self.node.get_logger().error(f"[{self.id}] force_arm error: {e}")
                    self.arm_pending = False
                    return
                if resp.success:
                    self.node.get_logger().info(f"[{self.id}] Armed! Climbing...")
                    self.arm_pending = False
                    self._climb(on_done)
                else:
                    self.node.get_logger().warn(f"[{self.id}] Arm refused: {resp.message}")
                    self.arm_pending = False

            future.add_done_callback(_done)

        threading.Thread(target=_do_arm, daemon=True).start()

    def _climb(self, on_done=None):
        self.axes.set(0.0, 0.0, self.climb_z, 0.0)

        def _wait():
            time.sleep(3.0)
            self.axes.set(0.0, 0.0, self.hover_z, 0.0)
            self.takeoff_done = True
            self.node.get_logger().info(
                f"[{self.id}] Climb done – hovering at z={self.hover_z}. Keyboard active.")
            if on_done:
                on_done(self)

        threading.Thread(target=_wait, daemon=True).start()

    # ---- path runner ----

    def run_path(self, segments: List[Tuple], path_name: str = "path"):
        if self.path_running:
            self.node.get_logger().warn(f"[{self.id}] Path already running – stop first (SPACE).")
            return
        self.path_running = True
        self.path_stop_flag = False
        self.node.get_logger().info(f"[{self.id}] Starting '{path_name}' ({len(segments)} segs).")
        threading.Thread(target=self._path_runner, args=(segments, path_name), daemon=True).start()

    def _path_runner(self, segments, path_name):
        try:
            for name, x, y, z, r, dur in segments:
                if self.path_stop_flag or not rclpy.ok():
                    break
                self.node.get_logger().info(
                    f"  [{self.id}|{path_name}] '{name}': x={x} y={y} z={z} r={r} dur={dur}s")
                self.axes.set(x, y, z, r)
                deadline = time.time() + dur
                while time.time() < deadline and not self.path_stop_flag and rclpy.ok():
                    time.sleep(0.05)
        finally:
            self.axes.set(0.0, 0.0, self.hover_z, 0.0)
            self.path_running = False
            self.path_stop_flag = False
            self.node.get_logger().info(f"[{self.id}] '{path_name}' done. Hovering.")

    def stop_path(self):
        if self.path_running:
            self.path_stop_flag = True
        self.axes.set(0.0, 0.0, self.hover_z, 0.0)

    # ---- disarm ----

    def request_disarm(self):
        """Call force_arm(False) to disarm the drone."""
        self.axes.reset()
        if not self.force_arm_client.service_is_ready():
            self.node.get_logger().warn(f"[{self.id}] force_arm service not ready for disarm.")
            return
        req = SetBool.Request()
        req.data = False
        future = self.force_arm_client.call_async(req)

        def _done(fut):
            try:
                resp = fut.result()
            except Exception as ex:
                self.node.get_logger().error(f"[{self.id}] disarm error: {ex}")
                return
            if resp.success:
                self.node.get_logger().info(f"[{self.id}] Disarmed.")
            else:
                self.node.get_logger().warn(f"[{self.id}] Disarm refused: {resp.message}")

        future.add_done_callback(_done)

    def close(self):
        self.path_stop_flag = True
        if self._yaw_timer:
            self._yaw_timer.cancel()
        self.logger.close()


# ---------------------------------------------------------------------------
# Path file parser
# ---------------------------------------------------------------------------
def parse_path_file(filepath: str) -> Optional[List[Tuple]]:
    segments = []
    auto_idx = 1
    try:
        lines = Path(filepath).read_text().splitlines()
    except OSError as e:
        print(f"[path] Cannot open '{filepath}': {e}")
        return None

    for lineno, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        try:
            if len(parts) == 5:
                name = f"S{auto_idx}"; auto_idx += 1
                x, y, z, r, dur = (float(p) for p in parts)
            elif len(parts) == 6:
                name = parts[0]
                x, y, z, r, dur = (float(p) for p in parts[1:])
            else:
                print(f"[path] Line {lineno}: expected 5 or 6 tokens. Skipping.")
                continue
        except ValueError as e:
            print(f"[path] Line {lineno}: parse error – {e}. Skipping.")
            continue
        if dur <= 0:
            print(f"[path] Line {lineno}: duration must be > 0. Skipping.")
            continue
        segments.append((name, x, y, z, r, dur))

    return segments or None


# ---------------------------------------------------------------------------
# Main ROS 2 node
# ---------------------------------------------------------------------------
class MultiDronePositionNode(Node):
    """
    Controls N drones in POSITION mode with a single keyboard.

    Active drone: receives keyboard commands.
    Broadcast mode: keyboard commands go to ALL drones.
    """

    def __init__(self):
        super().__init__("position_fly_controller")

        # ---- parameters ----
        self.declare_parameter("rooster_ids", "R1,R2")
        self.declare_parameter("path_file", "")   # global fallback path for all drones
        self.declare_parameter("step", 50.0)
        self.declare_parameter("climb_z", 600.0)
        self.declare_parameter("hover_z", 550.0)
        self.declare_parameter("log_dir", ".")

        ids_str        = self.get_parameter("rooster_ids").get_parameter_value().string_value
        global_path    = self.get_parameter("path_file").get_parameter_value().string_value.strip()
        step           = self.get_parameter("step").get_parameter_value().double_value
        climb_z        = self.get_parameter("climb_z").get_parameter_value().double_value
        hover_z        = self.get_parameter("hover_z").get_parameter_value().double_value
        log_dir        = self.get_parameter("log_dir").get_parameter_value().string_value

        # ---- per-drone path file parameters ----
        # Declare path_file_<ID> for every drone so they can each have their own path.
        # Priority: path_file_<ID>  >  global path_file  >  nothing (manual only)
        drone_ids = [d.strip() for d in ids_str.split(",") if d.strip()]
        per_drone_paths: Dict[str, Optional[str]] = {}
        for did in drone_ids:
            param_name = f"path_file_{did}"
            self.declare_parameter(param_name, "")
            val = self.get_parameter(param_name).get_parameter_value().string_value.strip()
            if val:
                per_drone_paths[did] = val          # drone-specific path wins
            elif global_path:
                per_drone_paths[did] = global_path  # fall back to global path
            else:
                per_drone_paths[did] = None         # pure manual for this drone

        # ---- drone handles ----
        self.drones: List[DroneHandle] = [
            DroneHandle(did, self, step, climb_z, hover_z, log_dir,
                        auto_path_file=per_drone_paths[did])
            for did in drone_ids
        ]
        self.active_idx = 0           # index into self.drones
        self.broadcast = False        # send to all drones at once
        self.hover_locked = False     # when True, keyboard axis keys are blocked

        # ---- global flags ----
        self.shutdown_flag = False

        # ---- timers ----
        self.create_timer(1.0 / PUBLISH_RATE_HZ, self._manual_timer_cb)
        self.create_timer(1.0 / KEEP_ALIVE_RATE_HZ, self._keep_alive_cb)

        self.get_logger().info(
            f"MultiDronePositionNode ready – drones: {drone_ids}, "
            f"flight_mode=POSITION({FLIGHT_MODE_POSITION})"
        )
        self._print_status()

    # ---- convenience ----

    @property
    def active(self) -> DroneHandle:
        return self.drones[self.active_idx]

    def _targets(self) -> List[DroneHandle]:
        """Return all drones if broadcast, else just the active one."""
        return self.drones if self.broadcast else [self.active]

    def _print_status(self):
        drone_str = "  ".join(
            f"[{'*' if i == self.active_idx else ' '}{d.id}]"
            for i, d in enumerate(self.drones)
        )
        bcast   = "  [BROADCAST]" if self.broadcast else ""
        hlocked = "  *** HOVER-LOCKED (SPACE to unlock) ***" if self.hover_locked else ""
        print(f"\n  Active: {drone_str}{bcast}{hlocked}", flush=True)
        print("  Keys: 1/2=select  b=broadcast  w/s=fwd/back  j/l=strafe  "
              "i/k=up/dn  a/d=yaw(momentary)  SPACE=zero/unlock  "
              "h=hover-lock  f=arm  e=disarm  p=path  t=turtle  q=quit\n",
              flush=True)

    # ---- ROS timers ----

    def _manual_timer_cb(self):
        for drone in self.drones:
            drone.publish_manual()

    def _keep_alive_cb(self):
        for drone in self.drones:
            drone.publish_keep_alive()

    # ---- keyboard handler ----

    def handle_key(self, ch: str) -> bool:
        """Returns True if quit requested."""
        quit_req = False

        # --- drone selection ---
        if ch.isdigit():
            idx = int(ch) - 1
            if 0 <= idx < len(self.drones):
                self.active_idx = idx
                self.broadcast = False
                self._print_status()
            else:
                print(f"  No drone #{ch} (only {len(self.drones)} drones)", flush=True)
            return False

        # --- broadcast toggle ---
        if ch == "b":
            self.broadcast = not self.broadcast
            self._print_status()
            return False

        # --- flight commands (apply to targets) ---
        targets = self._targets()

        # --- hover-lock: only SPACE, h, e, q, 1/2, b are accepted while locked ---
        axis_blocked = self.hover_locked and ch not in (" ", "h", "e", "q", "f")

        if ch == "h":
            if self.hover_locked:
                # unlock
                self.hover_locked = False
                self.get_logger().info("Hover-lock OFF – keyboard active.")
                self._print_status()
            else:
                # lock: zero everything and hold
                self.hover_locked = True
                for d in targets:
                    d.stop_path()
                    d.axes.reset()
                self.get_logger().info("HOVER-LOCK ON – axes zeroed, drone holds position.")
                self._print_status()
        elif ch == " ":
            # SPACE always works: clears lock + zeros/stops
            self.hover_locked = False
            for d in targets:
                d.stop_path() if d.path_running else d.axes.reset()
            self.get_logger().info("Axes zeroed / path stopped / hover-lock cleared.")
            self._print_status()
        elif axis_blocked:
            print("  [HOVER-LOCKED] Press SPACE or h to unlock.", flush=True)
        elif ch == "w":
            for d in targets: d.axes.increment("x", +1)
        elif ch == "s":
            for d in targets: d.axes.increment("x", -1)
        elif ch == "l":
            for d in targets: d.axes.increment("y", +1)
        elif ch == "j":
            for d in targets: d.axes.increment("y", -1)
        elif ch == "i":
            for d in targets: d.axes.increment("z", +1)
        elif ch == "k":
            for d in targets: d.axes.increment("z", -1)
        elif ch == "d":
            # Momentary yaw: sets rate, auto-resets after YAW_TIMEOUT seconds
            for d in targets: d.apply_yaw(+1)
        elif ch == "a":
            for d in targets: d.apply_yaw(-1)
        elif ch == "f":
            def _on_takeoff_done(drone: DroneHandle):
                # Each drone runs its own assigned path (or none if manual-only)
                path = drone.auto_path_file
                if path:
                    segs = parse_path_file(path)
                    if segs:
                        drone.run_path(segs, Path(path).name)
                    else:
                        drone.node.get_logger().warn(
                            f"[{drone.id}] Could not parse path file: {path}")
                else:
                    drone.node.get_logger().info(
                        f"[{drone.id}] No auto-path assigned – staying in manual mode.")
            for d in targets:
                d.request_arm_and_takeoff(on_done=_on_takeoff_done)
        elif ch == "e":
            for d in targets:
                d.request_disarm()
                self.get_logger().info(f"[{d.id}] Disarm requested.")
        elif ch == "t":
            for d in targets:
                turtle = d.axes.toggle_turtle()
                self.get_logger().info(f"[{d.id}] Turtle: {'ON' if turtle else 'OFF'}")
        elif ch == "p":
            self._prompt_and_run_path(targets)
        elif ch == "q":
            for d in self.drones:
                d.axes.reset()
            self.get_logger().info("Quit requested.")
            quit_req = True

        # print current axes for active drone
        if not ch.isdigit() and ch not in ("b", "h", " ") and not quit_req:
            x, y, z, r = self.active.axes.raw
            bcast_tag  = " [BCAST]" if self.broadcast else f" [{self.active.id}]"
            turtle_tag = " [turtle]" if self.active.axes.turtle else ""
            lock_tag   = " [HOVER-LOCKED]" if self.hover_locked else ""
            path_tags  = "  ".join(
                f"[{d.id}:path]" for d in self.drones if d.path_running
            )
            print(f"  {bcast_tag} x={x:.0f}  y={y:.0f}  z={z:.0f}  r={r:.0f}"
                  f"{turtle_tag}{lock_tag}  {path_tags}", flush=True)

        return quit_req

    def _prompt_and_run_path(self, targets: List[DroneHandle]):
        """Prompt for path file in the terminal and run it on targets."""
        print("\nEnter path file (absolute or relative to CWD):", end=" ", flush=True)
        try:
            filepath = input().strip()
        except EOFError:
            return
        if not filepath:
            return
        segments = parse_path_file(filepath)
        if segments is None:
            print(f"[path] No valid segments in '{filepath}'.")
            return
        for d in targets:
            d.run_path(segments, Path(filepath).name)

    # ---- cleanup ----

    def destroy_node(self):
        for d in self.drones:
            d.close()
        super().destroy_node()


# ---------------------------------------------------------------------------
# Keyboard input thread
# ---------------------------------------------------------------------------
def keyboard_thread(node: MultiDronePositionNode):
    import termios, tty, select
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        while rclpy.ok() and not node.shutdown_flag:
            rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
            if rlist:
                ch = sys.stdin.read(1)
                if ch and node.handle_key(ch):
                    node.shutdown_flag = True
                    break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        node.get_logger().info("Keyboard thread stopped.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = MultiDronePositionNode()

    kb = threading.Thread(target=keyboard_thread, args=(node,), daemon=True)
    kb.start()

    try:
        while rclpy.ok() and not node.shutdown_flag:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        for d in node.drones:
            d.axes.reset()
        node.destroy_node()
        rclpy.shutdown()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
