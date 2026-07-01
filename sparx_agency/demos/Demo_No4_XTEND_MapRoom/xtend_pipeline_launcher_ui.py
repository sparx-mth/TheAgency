#!/usr/bin/env python3
"""
XTEND pipeline launcher UI.

Starts common XTEND Jetson commands over SSH/tmux, starts PC commands locally,
and keeps the full command list in one place.

Safety:
- Use the UI for ARM / TAKEOFF / LAND / DISARM.
- Only online_nav_bridge_publisher.py should own the XTEND WebSocket.
- Movement path: planner/replayer -> /cmd_vel -> online bridge (built-in Twist converter) -> drone.
"""
from __future__ import annotations

import shlex
import subprocess
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import Literal

JETSON_SSH_DEFAULT = "user@192.0.0.89"
JETSON_REPO = "/home/user/agency_ws"

JETSON_ENV = """
cd /home/user/GIT/TheAgency
source /opt/ros/humble/setup.bash
source /home/user/GIT/TheAgency/venv/bin/activate
export ROS_DOMAIN_ID=5
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/opt/ros/humble/opt/rviz_ogre_vendor/lib:/opt/ros/humble/lib/aarch64-linux-gnu:/opt/ros/humble/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages:/home/user/GIT/TheAgency:/home/user/GIT/TheAgency/sparx_agency:${PYTHONPATH}
""".replace("/home/user/GIT/TheAgency", JETSON_REPO)

PC_ENV = """
cd /home/user1/GIT/TheAgency
source /opt/ros/jazzy/setup.bash
source /home/user1/GIT/TheAgency/venv/bin/activate
export ROS_DOMAIN_ID=5
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/ros/jazzy/lib
export PYTHONPATH=/usr/lib/python3.12/dist-packages:/opt/ros/jazzy/lib/python3.12/site-packages:/home/user1/GIT/TheAgency:${PYTHONPATH}
"""


@dataclass(frozen=True)
class LaunchItem:
    name: str
    machine: Literal["jetson", "pc", "manual"]
    tmux_name: str
    description: str
    command: str
    enabled_by_default: bool = True


LAUNCH_ITEMS: list[LaunchItem] = [
    LaunchItem(
        name="1. XTEND online bridge + RGB publisher",
        machine="jetson",
        tmux_name="xtend_bridge",
        description="Owns XTEND WebSocket, publishes /xtend/rgb as 504x294 resized frames, /xtend/bearing, /xtend/local_telemetry, subscribes to /xtend/cmd_nav and /cmd_vel (Twist).",
        command="""
        python3 /home/user/GIT/TheAgency/sparx_agency/robots/XTEND/online_nav_bridge_publisher.py \
          --camera-info-yaml /home/user/GIT/TheAgency/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_504_294_resize.yaml \
          --preprocess-mode resize \
          --output-width 504 \
          --output-height 294
        """,
    ),
    LaunchItem(
        name="2. DA3 Small depth processor",
        machine="jetson",
        tmux_name="xtend_depth",
        description="Subscribes to /xtend/rgb 504x294, runs DA3-SMALL, converts raw depth to meters using LUT, publishes /xtend/depth_m.",
        command="""
        python3 /home/user/GIT/TheAgency/sparx_agency/tasks/mapping/ros2/depth_processor_node.py \
          --ros-args \
          -p image_topic:=/xtend/rgb \
          -p depth_topic:=/xtend/depth_m \
          -p engine_path:=/home/user/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3-SMALL/DA3-SMALL.fp16-294x504.engine \
          -p config_yaml:=/home/user/GIT/TheAgency/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_720_420.yaml \
          -p model_type:=small_lut \
          -p camera_info_mode:=resize \
          -p apply_metric_focal_scaling:=false \
          -p small_lut_clip_min_m:=0.2 \
          -p small_lut_clip_max_m:=8.0
        """,
    ),
    LaunchItem(
        name="3. Optional Twist replayer",
        machine="jetson",
        tmux_name="xtend_twist_replayer",
        description="Optional: replays a JSONL Twist log onto /cmd_vel. Edit LOG_PATH before running.",
        enabled_by_default=False,
        command="""
LOG_PATH="/home/user/GIT/TheAgency/cmd_log.jsonl"
python3 /home/user/GIT/TheAgency/sparx_agency/tasks/planning/twist_replayer.py \
  --ros-args \
  -p log_path:="${LOG_PATH}" \
  -p topic:=/cmd_vel \
  -p speed:=1.0 \
  -p loop:=false
""",
    ),
    LaunchItem(
        name="5. Optical-flow depth velocity node",
        machine="jetson",
        tmux_name="xtend_flow_depth",
        description="Subscribes to RGB/depth and estimates velocity from optical flow + depth.",
        command="""
python3 -m sparx_agency.tasks.localization.ros2.depth_optical.flow_depth_velocity_node_separted \
  --ros-args \
  -p use_sim_time:=false \
  -p show_debug:=false \
  -p csv_filename:=/home/user/GIT/TheAgency/sparx_agency/tasks/localization/ros2/depth_optical/csv_eval/zone_velocities_log.csv \
  -p image_topic:=/xtend/rgb \
  -p depth_topic:=/xtend/depth_m \
  -p depth_scale:=0.8
""",
    ),
    LaunchItem(
        name="6. Velocity integrator",
        machine="jetson",
        tmux_name="xtend_velocity_integrator",
        description="Integrates velocity estimate into pose/odom.",
        command="""
python3 -m sparx_agency.tasks.localization.ros2.depth_optical.velocity_integrator \
  --ros-args \
  -p use_sim_time:=false \
  -p target_frame:=odom \
  -p init_from_gt:=false
""",
    ),
    LaunchItem(
        name="7. Static transform odom -> xtend_camera",
        machine="jetson",
        tmux_name="xtend_static_tf",
        description="Publishes static transform odom -> xtend_camera.",
        command="ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 odom xtend_camera",
    ),
    LaunchItem(
        name="8. PC manual UI",
        machine="pc",
        tmux_name="xtend_pc_ui",
        description="Manual ARM/TAKEOFF/LAND/DISARM/STOP UI. Movement can publish Twist.",
        command="python3 /home/user1/GIT/TheAgency/sparx_agency/robots/XTEND/ui.py",
    ),
    LaunchItem(
        name="9. Planner: hospital world",
        machine="manual",
        tmux_name="planner_hospital",
        description="Manual planner step on Jetson/container: starts hospital environment.",
        enabled_by_default=False,
        command="""
cd /home/user/GIT/sjtu_project/falcon_docker
./run_hospital.sh office
""",
    ),
    LaunchItem(
        name="10. Planner container: FALCON adapter",
        machine="manual",
        tmux_name="planner_falcon",
        description="Run inside the planner container.",
        enabled_by_default=False,
        command="roslaunch falcon_adapter real_drone.launch map_name:=office",
    ),
    LaunchItem(
        name="11. Planner ROS bridge docker",
        machine="manual",
        tmux_name="planner_ros_bridge",
        description="Manual ROS bridge step.",
        enabled_by_default=False,
        command="""
cd /home/user/GIT/sjtu_project/ros_bridge_docker
./run_bridge.sh
""",
    ),
]


def normalize_command(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def wrap_with_env(machine: str, command: str) -> str:
    env = JETSON_ENV if machine == "jetson" else PC_ENV
    if machine == "jetson":
        command = command.replace("/home/user/GIT/TheAgency", JETSON_REPO)
    return normalize_command(env) + "\n" + normalize_command(command)


class XtendPipelineLauncher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("XTEND Pipeline Launcher")
        self.geometry("1180x760")
        self.jetson_ssh_var = tk.StringVar(value=JETSON_SSH_DEFAULT)
        self.status_var = tk.StringVar(value="Ready.")
        self.selected_item: LaunchItem | None = None
        self.item_vars: dict[str, tk.BooleanVar] = {}
        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=8)
        ttk.Label(top, text="Jetson SSH:").pack(side="left")
        ttk.Entry(top, textvariable=self.jetson_ssh_var, width=28).pack(side="left", padx=6)
        ttk.Button(top, text="Start selected", command=self.start_selected).pack(side="left", padx=4)
        ttk.Button(top, text="Stop selected tmux", command=self.stop_selected).pack(side="left", padx=4)
        ttk.Button(top, text="Start checked Jetson core", command=self.start_checked_jetson).pack(side="left", padx=4)
        ttk.Button(top, text="Stop all known tmux", command=self.stop_all_known).pack(side="left", padx=4)

        main = ttk.PanedWindow(self, orient="horizontal")
        main.pack(fill="both", expand=True, padx=10, pady=6)
        left = ttk.Frame(main)
        right = ttk.Frame(main)
        main.add(left, weight=1)
        main.add(right, weight=2)

        ttk.Label(left, text="Nodes / Commands", font=("Arial", 11, "bold")).pack(anchor="w")
        self.listbox = tk.Listbox(left, height=28, exportselection=False)
        self.listbox.pack(fill="both", expand=True, pady=4)
        self.listbox.bind("<<ListboxSelect>>", self.on_select)

        for item in LAUNCH_ITEMS:
            self.listbox.insert("end", item.name)
            self.item_vars[item.tmux_name] = tk.BooleanVar(value=item.enabled_by_default)

        checks = ttk.LabelFrame(left, text="Checked for batch start")
        checks.pack(fill="x", pady=6)
        for item in LAUNCH_ITEMS:
            if item.machine == "jetson":
                ttk.Checkbutton(checks, text=item.name, variable=self.item_vars[item.tmux_name]).pack(anchor="w")

        self.desc_text = tk.Text(right, height=5, wrap="word")
        self.desc_text.pack(fill="x", pady=(0, 6))
        ttk.Label(right, text="Command", font=("Arial", 11, "bold")).pack(anchor="w")
        self.cmd_text = tk.Text(right, height=22, wrap="none")
        self.cmd_text.pack(fill="both", expand=True)

        btns = ttk.Frame(right)
        btns.pack(fill="x", pady=8)
        ttk.Button(btns, text="Copy command", command=self.copy_command).pack(side="left", padx=4)
        ttk.Button(btns, text="Copy env + command", command=self.copy_full_command).pack(side="left", padx=4)
        ttk.Button(btns, text="Run local terminal", command=self.run_local_terminal).pack(side="left", padx=4)
        ttk.Button(btns, text="Run Jetson tmux over SSH", command=self.run_jetson_tmux).pack(side="left", padx=4)
        ttk.Button(btns, text="Copy tmux attach command", command=self.copy_attach_command).pack(side="left", padx=4)

        ttk.Label(
            right,
            text="Safety: verify drone state manually. Only online_nav_bridge_publisher should own the WebSocket. Use UI for ARM/TAKEOFF/LAND/DISARM.",
            foreground="darkred",
        ).pack(anchor="w", pady=4)
        ttk.Label(self, textvariable=self.status_var, relief="sunken", anchor="w").pack(fill="x", padx=10, pady=4)
        self.listbox.selection_set(0)
        self.on_select()

    def current_index(self) -> int | None:
        sel = self.listbox.curselection()
        return int(sel[0]) if sel else None

    def on_select(self, _event=None):
        idx = self.current_index()
        if idx is None:
            return
        item = LAUNCH_ITEMS[idx]
        self.selected_item = item
        self.desc_text.delete("1.0", "end")
        self.desc_text.insert("end", f"{item.name}\nMachine: {item.machine}\nTmux: {item.tmux_name}\n\n{item.description}")
        cmd = item.command.replace("/home/user/GIT/TheAgency", JETSON_REPO) if item.machine == "jetson" else item.command
        self.cmd_text.delete("1.0", "end")
        self.cmd_text.insert("end", normalize_command(cmd))

    def get_command_text(self) -> str:
        return normalize_command(self.cmd_text.get("1.0", "end"))

    def copy_to_clipboard(self, text: str, label: str):
        self.clipboard_clear()
        self.clipboard_append(text)
        self.status_var.set(f"Copied {label}.")

    def copy_command(self):
        self.copy_to_clipboard(self.get_command_text(), "command")

    def copy_full_command(self):
        if self.selected_item is None:
            return
        item = self.selected_item
        full = self.get_command_text() if item.machine == "manual" else wrap_with_env(item.machine, self.get_command_text())
        self.copy_to_clipboard(full, "env + command")

    def run_local_terminal(self):
        if self.selected_item is None:
            return
        item = self.selected_item
        if item.machine != "pc" and not messagebox.askyesno("Run locally?", "This item is not marked as PC/local. Run it locally anyway?"):
            return
        script = wrap_with_env("pc", self.get_command_text())
        self._spawn_terminal(script, title=item.tmux_name)
        self.status_var.set(f"Started local terminal for {item.name}")

    def start_selected(self):
        if self.selected_item is None:
            return
        if self.selected_item.machine == "jetson":
            self.run_jetson_tmux()
        elif self.selected_item.machine == "pc":
            self.run_local_terminal()
        else:
            self.copy_command()
            messagebox.showinfo("Manual command", "This command is manual. It was copied to the clipboard.")

    def run_jetson_tmux(self):
        if self.selected_item is None:
            return
        self._start_jetson_tmux(self.selected_item)

    def start_checked_jetson(self):
        count = 0
        for item in LAUNCH_ITEMS:
            if item.machine == "jetson" and self.item_vars[item.tmux_name].get():
                self._start_jetson_tmux(item, quiet=True)
                count += 1
        self.status_var.set(f"Started {count} Jetson tmux sessions.")

    def _start_jetson_tmux(self, item: LaunchItem, quiet: bool = False):
        if item.machine != "jetson":
            if not quiet:
                messagebox.showwarning("Not Jetson", "This item is not a Jetson command.")
            return
        ssh_target = self.jetson_ssh_var.get().strip()
        script = wrap_with_env("jetson", item.command)
        tmux_cmd = f"bash -lc {shlex.quote(script)}"
        remote_cmd = (
            f"tmux kill-session -t {shlex.quote(item.tmux_name)} 2>/dev/null || true; "
            f"tmux new-session -d -s {shlex.quote(item.tmux_name)} {shlex.quote(tmux_cmd)}; "
            f"tmux ls"
        )
        result = subprocess.run(["ssh", ssh_target, remote_cmd], text=True, capture_output=True, check=False)
        if result.returncode != 0:
            messagebox.showerror("SSH/tmux failed", result.stderr.strip() or result.stdout.strip())
            self.status_var.set(f"Failed to start {item.tmux_name}")
            return
        if not quiet:
            self.status_var.set(f"Started Jetson tmux session: {item.tmux_name}")

    def stop_selected(self):
        if self.selected_item is None:
            return
        self._stop_tmux(self.selected_item.tmux_name)

    def stop_all_known(self):
        for item in LAUNCH_ITEMS:
            if item.machine == "jetson":
                self._stop_tmux(item.tmux_name, quiet=True)
        self.status_var.set("Requested stop for all known Jetson tmux sessions.")

    def _stop_tmux(self, tmux_name: str, quiet: bool = False):
        ssh_target = self.jetson_ssh_var.get().strip()
        remote_cmd = f"tmux kill-session -t {shlex.quote(tmux_name)} 2>/dev/null || true"
        subprocess.run(["ssh", ssh_target, remote_cmd], check=False)
        if not quiet:
            self.status_var.set(f"Stopped Jetson tmux session: {tmux_name}")

    def copy_attach_command(self):
        if self.selected_item is None:
            return
        ssh_target = self.jetson_ssh_var.get().strip()
        cmd = f"ssh -t {ssh_target} 'tmux attach -t {self.selected_item.tmux_name}'"
        self.copy_to_clipboard(cmd, "tmux attach command")

    def _spawn_terminal(self, script: str, title: str):
        candidates = [
            ["gnome-terminal", "--title", title, "--", "bash", "-lc", script],
            ["xterm", "-T", title, "-e", f"bash -lc {shlex.quote(script)}"],
            ["konsole", "--new-tab", "-p", f"tabtitle={title}", "-e", "bash", "-lc", script],
        ]
        last_error = None
        for cmd in candidates:
            try:
                subprocess.Popen(cmd)
                return
            except FileNotFoundError as exc:
                last_error = exc
        raise RuntimeError(f"No supported terminal found. Last error: {last_error}")


def main():
    app = XtendPipelineLauncher()
    app.mainloop()


if __name__ == "__main__":
    main()
