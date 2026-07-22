#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import time
import tkinter as tk
from tkinter import ttk
import csv
from datetime import datetime
from pathlib import Path
import websockets

from sparx_agency.robots.XTEND.automation import ControllerAutomation


class XtendDirectUIController(ControllerAutomation):
    def __init__(
            self,
            host: str,
            port: int,
            frequency: float,
            robot_uid: str,
            forward_value: int = 500,
            yaw_right_value: int = 1000,
            right_90_ms: int = 1500,
            log_dir: str = "xtend_ui_logs",
    ):
        super().__init__(host, port, frequency, robot_uid)

        self.forward_value = int(forward_value)
        self.yaw_right_value = int(yaw_right_value)
        self.right_90_ms = int(right_90_ms)

        self.cmd_queue: asyncio.Queue[str] = asyncio.Queue()

        self.active_action = None
        self.active_action_start_t = None
        self.action_log = []
        self.init_logs(log_dir)

        self.root = tk.Tk()
        self.root.title("XTEND Direct Drone Controller")
        self.root.geometry("340x710")

        self.active_action = None
        self.active_action_start_t = None
        self.timer_text = tk.StringVar(value="Current action: none")
        self.timer_text.set("Current action: none")

        self.forward_value_var = tk.StringVar(value=str(self.forward_value))
        self.turn_value_var = tk.StringVar(value=str(self.yaw_right_value))

        self._build_ui()

    # ------------------------------------------------------------
    # Direct continuous control helpers
    # ------------------------------------------------------------
    def set_axes(
        self,
        lateral: int = 0,
        vertical: int = 0,
        forward: int = 0,
        yaw: int = 0,
        marker_vertical: int = 0,
    ):
        self.send_command["axes"][0] = int(lateral)
        self.send_command["axes"][1] = int(vertical)
        self.send_command["axes"][2] = int(forward)
        self.send_command["axes"][3] = int(yaw)
        self.send_command["axes"][4] = int(marker_vertical)

    def init_logs(self, log_dir):
        self.log_dir =Path.home() / "Documents" / log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.telemetry_log_path = self.log_dir / f"xtend_telemetry_{self.run_stamp}.csv"
        self.action_log_path = self.log_dir / f"xtend_actions_{self.run_stamp}.csv"

        self.telemetry_fp = open(self.telemetry_log_path, "w", newline="", encoding="utf-8")
        self.telemetry_writer = csv.writer(self.telemetry_fp)
        self.telemetry_writer.writerow([
            "time_sec",
            "iso_time",
            "robot_uid",
            "x",
            "y",
            "z",
            "bearing_raw",
            "active_action",
        ])

        print(f"[log] telemetry: {self.telemetry_log_path}")
        print(f"[log] actions:   {self.action_log_path}")



    def save_action_log_csv(self):
        with open(self.action_log_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow([
                "index",
                "action",
                "start_t",
                "end_t",
                "duration_sec",
                "reason",
            ])

            for i, entry in enumerate(self.action_log):
                writer.writerow([
                    i,
                    entry["action"],
                    entry["start_t"],
                    entry["end_t"],
                    entry["duration_sec"],
                    entry["reason"],
                ])

        print(f"[log] saved actions: {self.action_log_path}")

    def hold_forward(self):
        value = self.get_int_from_ui(
            self.forward_value_var,
            self.forward_value,
            "forward thrust",
        )
        self.start_action_timer(f"forward_{value}")
        print(f"[ui] HOLD FORWARD value={value}")
        self.set_axes(forward=value, yaw=0)

    def hold_left_turn(self):
        value = self.get_int_from_ui(
            self.turn_value_var,
            self.yaw_right_value,
            "turn thrust",
        )
        self.start_action_timer(f"turn_left_{value}")
        print(f"[ui] HOLD LEFT TURN value={value}")
        self.set_axes(forward=0, yaw=-value)

    def hold_right_turn(self):
        value = self.get_int_from_ui(
            self.turn_value_var,
            self.yaw_right_value,
            "turn thrust",
        )
        self.start_action_timer(f"turn_right_{value}")
        print(f"[ui] HOLD RIGHT TURN value={value}")
        self.set_axes(forward=0, yaw=value)

    def stop_motion(self):
        self.end_action_timer(reason="stop")
        self.set_axes(0, 0, 0, 0, 0)

    async def land_safe(self):
        self.end_action_timer(reason="land")
        self.stop_motion()
        await self.land()

    async def disarm_safe(self):
        self.end_action_timer(reason="disarm")
        self.stop_motion()
        await self.disarm_robot()

    def start_action_timer(self, action_name: str):
        self.end_action_timer(reason=f"interrupted_by_{action_name}")

        self.active_action = action_name
        self.active_action_start_t = time.time()

        self.timer_text.set(f"Current action: {action_name} | 0.000s")
        print(f"[action] START {action_name}")

    def end_action_timer(self, reason: str = "stop"):
        if self.active_action is None or self.active_action_start_t is None:
            return

        now = time.time()
        duration = now - self.active_action_start_t

        entry = {
            "action": self.active_action,
            "start_t": self.active_action_start_t,
            "end_t": now,
            "duration_sec": duration,
            "reason": reason,
        }
        self.action_log.append(entry)

        print(
            f"[action] END {self.active_action} "
            f"duration={duration:.3f}s reason={reason}"
        )

        self.active_action = None
        self.active_action_start_t = None

    def print_action_summary(self):
        print("\n[action summary]")
        if not self.action_log:
            print("  no timed actions yet")
            return

        for i, entry in enumerate(self.action_log):
            print(
                f"  {i:02d}: {entry['action']} "
                f"{entry['duration_sec']:.3f}s "
                f"reason={entry['reason']}"
            )

    def update_action_timer_label(self):
        if self.active_action is None or self.active_action_start_t is None:
            return

        elapsed = time.time() - self.active_action_start_t
        self.timer_text.set(
            f"Current action: {self.active_action} | {elapsed:.3f}s"
        )

    def get_int_from_ui(self, var: tk.StringVar, default: int, name: str) -> int:
        try:
            value = int(var.get())
        except ValueError:
            print(f"[ui] invalid {name}: {var.get()}, using default={default}")
            return default

        value = max(0, min(1000, value))
        return value

    # ------------------------------------------------------------
    # Demo sequence
    # ------------------------------------------------------------
    async def run_forward_then_right_then_land(self):
        """
        Sequence:
          1. move forward for 2 sec at thrust 500
          2. without stopping forward, add right yaw for approx 90 deg
          3. stop motion
          4. land
        """
        print("[demo] forward 2 sec")
        self.set_axes(forward=self.forward_value, yaw=0)
        await asyncio.sleep(2.0)

        print(f"[demo] forward + right turn for {self.right_90_ms} ms")
        self.set_axes(forward=self.forward_value, yaw=self.yaw_right_value)
        await asyncio.sleep(self.right_90_ms * 0.001)

        print("[demo] stop motion before landing")
        self.stop_motion()
        await asyncio.sleep(0.5)

        print("[demo] land")
        await self.land_safe()

    async def run_full_demo(self):
        """
        Full sequence:
          arm -> takeoff -> forward 2 sec -> forward+right 90 -> land
        """
        print("[demo] FULL DEMO: arm")
        await self.arm_robot()
        await asyncio.sleep(3.0)

        print("[demo] FULL DEMO: takeoff")
        await self.takeoff()
        await asyncio.sleep(2.0)

        await self.run_forward_then_right_then_land()

    # ------------------------------------------------------------
    # UI
    # ------------------------------------------------------------
    def _button(self, text: str, command: str, color: str = "lightgrey"):
        btn = tk.Button(
            self.root,
            text=text,
            command=lambda: self.cmd_queue.put_nowait(command),
            bg=color,
            width=28,
            height=1,
        )
        btn.pack(pady=4)

    def _build_ui(self):
        ttk.Label(
            self.root,
            text="XTEND Direct Controller",
            font=("Arial", 13, "bold"),
        ).pack(pady=10)

        ttk.Label(
            self.root,
            text="State Controls",
            font=("Arial", 11, "bold"),
        ).pack(pady=5)

        self._button("ARM", "arm", "orange")
        self._button("TAKEOFF", "takeoff", "green")
        self._button("LAND", "land", "red")
        self._button("DISARM", "disarm", "darkred")
        self._button("PRINT ACTION SUMMARY", "summary", "lightblue")

        ttk.Separator(self.root, orient="horizontal").pack(fill="x", pady=10)



        ttk.Label(
            self.root,
            text="Thrust values",
            font=("Arial", 11, "bold"),
        ).pack(pady=5)

        thrust_frame = ttk.Frame(self.root)
        thrust_frame.pack(pady=4)

        ttk.Label(thrust_frame, text="Forward:").grid(row=0, column=0, padx=4, pady=3)
        ttk.Entry(thrust_frame, textvariable=self.forward_value_var, width=8).grid(row=0, column=1, padx=4, pady=3)

        ttk.Label(thrust_frame, text="Turn:").grid(row=1, column=0, padx=4, pady=3)
        ttk.Entry(thrust_frame, textvariable=self.turn_value_var, width=8).grid(row=1, column=1, padx=4, pady=3)

        ttk.Label(
            self.root,
            text="Continuous Motion",
            font=("Arial", 11, "bold"),
        ).pack(pady=5)

        self._button("HOLD FORWARD", "hold_forward")
        self._button("TURN LEFT", "hold_left")
        self._button("TURN RIGHT", "hold_right")
        self._button("STOP MOTION", "stop", "yellow")

        ttk.Label(
            self.root,
            textvariable=self.timer_text,
            font=("Arial", 11, "bold"),
            foreground="blue",
        ).pack(pady=8)
        #
        # ttk.Separator(self.root, orient="horizontal").pack(fill="x", pady=10)
        #
        # ttk.Label(
        #     self.root,
        #     text="Demo",
        #     font=("Arial", 11, "bold"),
        # ).pack(pady=5)
        #
        # self._button("FWD 2s → FWD+RIGHT90 → LAND", "demo", "lightblue")
        # self._button("FULL: ARM → TAKEOFF → DEMO", "full_demo", "lightgreen")
        #
        # ttk.Label(
        #     self.root,
        #     text=(
        #         f"forward={self.forward_value}, "
        #         f"yaw_right={self.yaw_right_value}, "
        #         f"right90={self.right_90_ms}ms"
        #     ),
        #     font=("Arial", 9),
        # ).pack(pady=10)

    async def ui_loop(self):
        try:
            while True:
                self.update_action_timer_label()
                self.root.update_idletasks()
                self.root.update()
                await asyncio.sleep(0.02)
        except tk.TclError:
            print("[ui] window closed")

    async def command_worker(self):
        while True:
            cmd = await self.cmd_queue.get()
            print(f"[ui] command={cmd}")

            try:
                if cmd == "arm":
                    await self.arm_robot()

                elif cmd == "takeoff":
                    await self.takeoff()

                elif cmd == "land":
                    await self.land_safe()

                elif cmd == "disarm":
                    await self.disarm_safe()

                elif cmd == "hold_forward":
                    self.hold_forward()

                elif cmd == "hold_left":
                    self.hold_left_turn()

                elif cmd == "hold_right":
                    self.hold_right_turn()

                elif cmd == "stop":
                    self.stop_motion()

                elif cmd == "summary":
                    self.print_action_summary()

                elif cmd == "demo":
                    await self.run_forward_then_right_then_land()

                elif cmd == "full_demo":
                    await self.run_full_demo()

                else:
                    print(f"[ui] unknown command: {cmd}")

            finally:
                self.cmd_queue.task_done()

    async def run(self):
        async with websockets.connect(self.uri) as websocket:
            print(f"✓ Connected to {self.uri}")

            send_task = asyncio.create_task(self.send_message(websocket))
            receive_task = asyncio.create_task(self.receive_message(websocket))
            ui_task = asyncio.create_task(self.ui_loop())
            worker_task = asyncio.create_task(self.command_worker())

            try:
                await ui_task
            finally:
                self.stop_motion()
                self.end_action_timer(reason="shutdown")
                self.print_action_summary()
                self.save_action_log_csv()

                if hasattr(self, "telemetry_fp") and not self.telemetry_fp.closed:
                    self.telemetry_fp.close()

                for task in (send_task, receive_task, worker_task):
                    task.cancel()

                await asyncio.gather(
                    send_task,
                    receive_task,
                    worker_task,
                    return_exceptions=True,
                )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="192.0.0.15")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--frequency", type=float, default=30.0)
    p.add_argument("--robot-uid", default="drndfb3eeb1")

    p.add_argument("--forward-value", type=int, default=400)
    p.add_argument("--yaw-right-value", type=int, default=1000)
    p.add_argument("--right-90-ms", type=int, default=1500)

    p.add_argument(
        "--log-dir",
        default="xtend_ui_logs",
        help="Directory for telemetry/action logs.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    controller = XtendDirectUIController(
        host=args.host,
        port=args.port,
        frequency=args.frequency,
        robot_uid=args.robot_uid,
        forward_value=args.forward_value,
        yaw_right_value=args.yaw_right_value,
        right_90_ms=args.right_90_ms,
        log_dir=args.log_dir,
    )

    try:
        asyncio.run(controller.run())
    except KeyboardInterrupt:
        print("\n[main] stopped by user")
        controller.stop_motion()


if __name__ == "__main__":
    main()