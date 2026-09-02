#!/usr/bin/env python3
"""
ROBOTICAN Rooster position-mode pipeline launcher UI.

Manages ROS2 nodes for position-mode (FLIGHT_MODE_POSITION=3) flight
with one or more Rooster drones, against either a live drone or the
SPHERA simulation.

Architecture
------------
- This launcher runs on the **host** (Python 3.12).
- All ROS2 nodes run **inside the Docker container** (ROS Foxy + CycloneDDS).
  Container: sphera-backend:rooster-with-sparx  (docker compose service: `it`)
  TheAgency/sparx_agency/robots/ROBOTICAN is mounted inside at:
    /home/rooster/sparx_agency/robots/ROBOTICAN
- network_mode: host  → container topics visible on host at ROS_DOMAIN_ID=6.

Quickstart
----------
1. cd ~/rqs_iai_ws/src && docker compose up -d it
2. python3 rooster_pipeline_launcher_ui_position.py
3. Start node 1 (position fly controller) → terminal opens inside container.
4. In that terminal: press f to arm+takeoff, w/s/j/l/i/k to fly, e to disarm.
"""
from __future__ import annotations

import shlex
import subprocess
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import Literal

CONTAINER_NAME_DEFAULT = "it"

# Environment sourced inside the container before every node command.
# ROS_DOMAIN_ID=6 matches the SPHERA simulator's domain.
CONTAINER_ENV = """\
export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.8/site-packages:/home/rooster
source /opt/ros/foxy/setup.bash
source /home/rooster/workspace/install/setup.bash
export ROS_DOMAIN_ID=6
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/rooster/workspace/src/cyclonedds.xml
export PYTHONUNBUFFERED=1"""

# Environment for nodes that run on the host PC (e.g. RViz with Jazzy).
PC_ENV = """\
cd /home/user1/GIT/TheAgency
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=6
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export PYTHONUNBUFFERED=1"""

_CONTROLLER = (
    "/home/rooster/sparx_agency/robots/ROBOTICAN/examples/src/position_fly_controller.py"
)


@dataclass(frozen=True)
class LaunchItem:
    name: str
    machine: Literal["container", "pc", "manual"]
    proc_key: str        # substring used by pkill to stop this node
    description: str
    command: str         # raw command shown/edited in UI (no env prefix)
    enabled_by_default: bool = True


LAUNCH_ITEMS: list[LaunchItem] = [
    LaunchItem(
        name="1. Position fly controller (keyboard)",
        machine="container",
        proc_key="position_fly_controller",
        description=(
            "Multi-drone keyboard controller in POSITION mode (flight_mode=3).\n\n"
            "Publishes /{id}/manual_control @ 40 Hz and /{id}/keep_alive @ 1 Hz.\n"
            "Arms via /{id}/fcu/command/force_arm (SetBool) service.\n\n"
            "Keys:\n"
            "  1/2/…   select active drone\n"
            "  b       toggle broadcast (all drones)\n"
            "  w/s     forward / backward  (x, accumulates)\n"
            "  j/l     strafe left / right (y, accumulates)\n"
            "  i/k     climb / descend     (z, accumulates)\n"
            "  a/d     yaw CCW / CW        (momentary, auto-resets 0.5 s)\n"
            "  SPACE   zero all axes / stop path / exit hover-lock\n"
            "  h       hover-lock (holds position, blocks axis keys)\n"
            "  f       arm + takeoff  (active or all if broadcast)\n"
            "  e       disarm\n"
            "  p       load & run a path file\n"
            "  t       toggle turtle (slow) mode\n"
            "  q       quit"
        ),
        command=(
            f"python3 {_CONTROLLER} \\\n"
            "  --ros-args \\\n"
            "  -p rooster_ids:=R1 \\\n"
            "  -p step:=50.0 \\\n"
            "  -p climb_z:=600.0 \\\n"
            "  -p hover_z:=550.0 \\\n"
            "  -p log_dir:=/tmp"
        ),
    ),
    LaunchItem(
        name="2. Drone state monitor",
        machine="container",
        proc_key="ros2 topic echo /R1/state",
        description=(
            "Streams RoosterState from /R1/state.\n"
            "Shows: armed, flight_mode, airborne, roll, pitch, azimuth.\n\n"
            "Edit the command to change drone ID (R1 → R2 etc.)."
        ),
        enabled_by_default=False,
        command="ros2 topic echo /R1/state",
    ),
    LaunchItem(
        name="3. KeepAlive rate check",
        machine="container",
        proc_key="ros2 topic hz /R1/keep_alive",
        description=(
            "Checks KeepAlive publish rate on /R1/keep_alive.\n"
            "Expected: ~1 Hz.  Silence = controller not running."
        ),
        enabled_by_default=False,
        command="ros2 topic hz /R1/keep_alive",
    ),
    LaunchItem(
        name="4. ManualControl rate check",
        machine="container",
        proc_key="ros2 topic hz /R1/manual_control",
        description=(
            "Checks ManualControl publish rate on /R1/manual_control.\n"
            "Expected: ~40 Hz.  Low rate = control loop stalled."
        ),
        enabled_by_default=False,
        command="ros2 topic hz /R1/manual_control",
    ),
    LaunchItem(
        name="5. Active topic list",
        machine="container",
        proc_key="ros2 topic list",
        description=(
            "Prints all active ROS2 topics visible inside the container.\n"
            "Useful to confirm the drone backend is publishing."
        ),
        enabled_by_default=False,
        command="ros2 topic list",
    ),
    LaunchItem(
        name="6. RViz (host / Jazzy)",
        machine="pc",
        proc_key="rviz2",
        description=(
            "Opens RViz2 on the host with Jazzy.\n"
            "Requires rmw_cyclonedds_cpp + ROS_DOMAIN_ID=6 to see drone topics."
        ),
        enabled_by_default=False,
        command="rviz2",
    ),
    LaunchItem(
        name="7. Manual: start container",
        machine="manual",
        proc_key="",
        description=(
            "Start the Docker container before launching any nodes.\n"
            "Run this command in a host terminal once per session."
        ),
        enabled_by_default=False,
        command="cd ~/rqs_iai_ws/src && docker compose up -d it",
    ),
    LaunchItem(
        name="8. Manual: attach to container shell",
        machine="manual",
        proc_key="",
        description=(
            "Open an interactive shell inside the running container.\n"
            "Useful for debugging topics, services, or running ad-hoc commands."
        ),
        enabled_by_default=False,
        command="docker exec -it it bash",
    ),
    LaunchItem(
        name="9. Rooster command unit (R1)",
        machine="container",
        proc_key="rooster_command_unit",
        description=(
            "Single command gateway for R1 — the one node that actually talks\n"
            "to the FCU (arm/disarm/takeoff/land/move). Listens on /R1/cmd_nav\n"
            "(String JSON) and publishes /R1/rooster_status.\n\n"
            "Run this instead of item 1 when using the Rooster manual UI (item 10)\n"
            "or, later, a planner — both just publish to /R1/cmd_nav.\n\n"
            "cmd_nav actions: arm, disarm, takeoff, land, forward, backward,\n"
            "left, right, up, down, turn_left, turn_right, stop."
        ),
        command=(
            "python3 /home/rooster/sparx_agency/robots/ROBOTICAN/adapters/rooster_command_unit.py \\\n"
            "  --ros-args \\\n"
            "  -p rooster_id:=R1 \\\n"
            "  -p climb_z:=600.0 \\\n"
            "  -p hover_z:=550.0"
        ),
    ),
    LaunchItem(
        name="10. Rooster manual UI (Tkinter, host)",
        machine="pc",
        proc_key="ROBOTICAN/ui.py",
        description=(
            "ARM / TAKEOFF / LAND / DISARM + movement d-pad for R1.\n"
            "Runs on the host (like the XTEND UI) — publishes /R1/cmd_nav.\n"
            "Requires item 9 (Rooster command unit) running in the container."
        ),
        command="python3 /home/user1/GIT/TheAgency/sparx_agency/robots/ROBOTICAN/ui.py --ros-args -p rooster_id:=R1",
    ),
]


def normalize_command(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def wrap_with_env(machine: str, command: str) -> str:
    env = CONTAINER_ENV if machine == "container" else PC_ENV
    return normalize_command(env) + "\n" + normalize_command(command)


class RoosterPositionLauncher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ROBOTICAN Position-Mode Launcher")
        self.geometry("1260x820")
        self.container_var = tk.StringVar(value=CONTAINER_NAME_DEFAULT)
        self.status_var = tk.StringVar(value="Ready.")
        self.selected_item: LaunchItem | None = None
        self.item_vars: dict[str, tk.BooleanVar] = {}
        self._build_ui()

    def _build_ui(self):
        # ── top bar ───────────────────────────────────────────────────────────
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=8)
        ttk.Label(top, text="Container:").pack(side="left")
        ttk.Entry(top, textvariable=self.container_var, width=18).pack(side="left", padx=6)
        ttk.Button(top, text="Start selected", command=self.start_selected).pack(side="left", padx=4)
        ttk.Button(top, text="Stop selected proc", command=self.stop_selected).pack(side="left", padx=4)
        ttk.Button(top, text="Start checked (container)", command=self.start_checked_container).pack(side="left", padx=4)
        ttk.Button(top, text="Stop all container procs", command=self.stop_all_container).pack(side="left", padx=4)
        ttk.Button(top, text="Check container", command=self.check_container).pack(side="left", padx=4)

        # ── arm / disarm quick-action row ─────────────────────────────────────
        arm_row = ttk.Frame(self)
        arm_row.pack(fill="x", padx=10, pady=(0, 6))
        ttk.Label(arm_row, text="Quick actions:").pack(side="left")
        for drone_id in ("R1", "R2"):
            ttk.Button(
                arm_row, text=f"ARM {drone_id}",
                command=lambda d=drone_id: self.force_arm(d, arm=True),
            ).pack(side="left", padx=4)
            ttk.Button(
                arm_row, text=f"DISARM {drone_id}",
                command=lambda d=drone_id: self.force_arm(d, arm=False),
            ).pack(side="left", padx=4)

        # ── main paned area ───────────────────────────────────────────────────
        main = ttk.PanedWindow(self, orient="horizontal")
        main.pack(fill="both", expand=True, padx=10, pady=6)
        left = ttk.Frame(main)
        right = ttk.Frame(main)
        main.add(left, weight=1)
        main.add(right, weight=2)

        ttk.Label(left, text="Nodes / Commands", font=("Arial", 11, "bold")).pack(anchor="w")
        self.listbox = tk.Listbox(left, height=30, exportselection=False)
        self.listbox.pack(fill="both", expand=True, pady=4)
        self.listbox.bind("<<ListboxSelect>>", self.on_select)

        for item in LAUNCH_ITEMS:
            self.listbox.insert("end", item.name)
            self.item_vars[item.proc_key] = tk.BooleanVar(value=item.enabled_by_default)

        checks = ttk.LabelFrame(left, text="Checked for batch start")
        checks.pack(fill="x", pady=6)
        for item in LAUNCH_ITEMS:
            if item.machine == "container":
                ttk.Checkbutton(
                    checks, text=item.name,
                    variable=self.item_vars[item.proc_key],
                ).pack(anchor="w")

        # ── right panel: description + command ───────────────────────────────
        self.desc_text = tk.Text(right, height=6, wrap="word")
        self.desc_text.pack(fill="x", pady=(0, 6))
        ttk.Label(right, text="Command", font=("Arial", 11, "bold")).pack(anchor="w")
        self.cmd_text = tk.Text(right, height=20, wrap="none")
        self.cmd_text.pack(fill="both", expand=True)

        btns = ttk.Frame(right)
        btns.pack(fill="x", pady=8)
        ttk.Button(btns, text="Copy command", command=self.copy_command).pack(side="left", padx=4)
        ttk.Button(btns, text="Copy env + command", command=self.copy_full_command).pack(side="left", padx=4)
        ttk.Button(btns, text="Run in container terminal", command=self.run_in_container_terminal).pack(side="left", padx=4)
        ttk.Button(btns, text="Run local terminal (PC)", command=self.run_local_terminal).pack(side="left", padx=4)

        ttk.Label(
            right,
            text="Safety: ARM sends a real arm command. Verify drone state before flying.",
            foreground="darkred",
        ).pack(anchor="w", pady=4)

        ttk.Label(self, textvariable=self.status_var, relief="sunken", anchor="w").pack(
            fill="x", padx=10, pady=4
        )
        self.listbox.selection_set(0)
        self.on_select()

    # ── selection ─────────────────────────────────────────────────────────────

    def on_select(self, _event=None):
        sel = self.listbox.curselection()
        if not sel:
            return
        item = LAUNCH_ITEMS[int(sel[0])]
        self.selected_item = item
        self.desc_text.delete("1.0", "end")
        self.desc_text.insert(
            "end",
            f"{item.name}\nMachine: {item.machine}\n\n{item.description}",
        )
        self.cmd_text.delete("1.0", "end")
        self.cmd_text.insert("end", normalize_command(item.command))

    def get_command_text(self) -> str:
        return normalize_command(self.cmd_text.get("1.0", "end"))

    # ── clipboard helpers ──────────────────────────────────────────────────────

    def _copy(self, text: str, label: str):
        self.clipboard_clear()
        self.clipboard_append(text)
        self.status_var.set(f"Copied {label}.")

    def copy_command(self):
        self._copy(self.get_command_text(), "command")

    def copy_full_command(self):
        if self.selected_item is None:
            return
        item = self.selected_item
        if item.machine == "manual":
            self._copy(self.get_command_text(), "command")
        else:
            self._copy(wrap_with_env(item.machine, self.get_command_text()), "env + command")

    # ── launch helpers ─────────────────────────────────────────────────────────

    def start_selected(self):
        if self.selected_item is None:
            return
        item = self.selected_item
        if item.machine == "container":
            self.run_in_container_terminal()
        elif item.machine == "pc":
            self.run_local_terminal()
        else:
            self.copy_command()
            messagebox.showinfo("Manual command", "This is a manual command — copied to clipboard.")

    def run_in_container_terminal(self):
        if self.selected_item is None:
            return
        item = self.selected_item
        if item.machine != "container":
            messagebox.showwarning("Not a container node", "This item does not run in the container.")
            return
        script = wrap_with_env("container", self.get_command_text())
        container = self.container_var.get().strip()
        self._spawn_container_terminal(script, container=container, title=item.name)
        self.status_var.set(f"Started container terminal: {item.name}")

    def run_local_terminal(self):
        if self.selected_item is None:
            return
        item = self.selected_item
        if item.machine not in ("pc", "container") and not messagebox.askyesno(
            "Run locally?", "This item is not marked as PC/local. Run locally anyway?"
        ):
            return
        script = wrap_with_env("pc", self.get_command_text())
        self._spawn_terminal(script, title=item.name)
        self.status_var.set(f"Started local terminal: {item.name}")

    def start_checked_container(self):
        count = 0
        for item in LAUNCH_ITEMS:
            if item.machine == "container" and self.item_vars[item.proc_key].get():
                script = wrap_with_env("container", item.command)
                container = self.container_var.get().strip()
                self._spawn_container_terminal(script, container=container, title=item.name)
                count += 1
        self.status_var.set(f"Started {count} container terminal(s).")

    # ── stop helpers ───────────────────────────────────────────────────────────

    def stop_selected(self):
        if self.selected_item is None:
            return
        item = self.selected_item
        if not item.proc_key:
            self.status_var.set("No proc_key for this item — nothing to stop.")
            return
        self._pkill_container(item.proc_key)
        self.status_var.set(f"Sent stop for: {item.proc_key}")

    def stop_all_container(self):
        for item in LAUNCH_ITEMS:
            if item.machine == "container" and item.proc_key:
                self._pkill_container(item.proc_key, quiet=True)
        self.status_var.set("Sent stop for all known container processes.")

    def _pkill_container(self, proc_key: str, quiet: bool = False):
        container = self.container_var.get().strip()
        kill_cmd = f"pkill -f {shlex.quote(proc_key)} 2>/dev/null || true"
        result = subprocess.run(
            ["docker", "exec", container, "bash", "-c", kill_cmd],
            check=False, capture_output=True, text=True,
        )
        if not quiet and result.returncode not in (0, 1):
            messagebox.showwarning(
                "pkill", result.stderr.strip() or f"pkill returned {result.returncode}"
            )

    # ── arm / disarm ───────────────────────────────────────────────────────────

    def force_arm(self, drone_id: str, arm: bool):
        action_str = "true" if arm else "false"
        label = "ARM" if arm else "DISARM"
        if arm and not messagebox.askyesno(
            f"Confirm {label}",
            f"Send force_arm({action_str}) to /{drone_id}/fcu/command/force_arm?\n\n"
            "Make sure the drone is in a safe position before arming.",
        ):
            return
        container = self.container_var.get().strip()
        srv_cmd = (
            "source /opt/ros/foxy/setup.bash && "
            "source /home/rooster/workspace/install/setup.bash && "
            "export ROS_DOMAIN_ID=6 && "
            "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && "
            "export CYCLONEDDS_URI=file:///home/rooster/workspace/src/cyclonedds.xml && "
            f"ros2 service call /{drone_id}/fcu/command/force_arm "
            f"std_srvs/srv/SetBool '{{data: {action_str}}}'"
        )
        try:
            result = subprocess.run(
                ["docker", "exec", container, "bash", "-c", srv_cmd],
                check=False, capture_output=True, text=True, timeout=10,
            )
        except subprocess.TimeoutExpired:
            messagebox.showerror(f"{label} timeout", "Service call timed out after 10 s.")
            self.status_var.set(f"{label} {drone_id}: timed out")
            return
        if result.returncode != 0:
            messagebox.showerror(
                f"{label} failed",
                result.stderr.strip() or result.stdout.strip() or f"Exit code {result.returncode}",
            )
            self.status_var.set(f"{label} {drone_id}: FAILED")
        else:
            self.status_var.set(f"{label} {drone_id}: sent.")

    # ── container health check ─────────────────────────────────────────────────

    def check_container(self):
        container = self.container_var.get().strip()
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name=^{container}$", "--format", "{{.Names}}\t{{.Status}}"],
            check=False, capture_output=True, text=True,
        )
        output = result.stdout.strip()
        if container in output:
            messagebox.showinfo("Container status", f"Running:\n{output}")
            self.status_var.set(f"Container '{container}' is running.")
        else:
            messagebox.showwarning(
                "Container not found",
                f"Container '{container}' does not appear to be running.\n\n"
                "Start it with:\n  cd ~/rqs_iai_ws/src && docker compose up -d it",
            )
            self.status_var.set(f"Container '{container}' not running.")

    # ── terminal spawners ──────────────────────────────────────────────────────

    def _spawn_container_terminal(self, script: str, container: str, title: str):
        docker_cmd = f"docker exec -it {shlex.quote(container)} bash -c {shlex.quote(script)}"
        self._spawn_terminal_with_cmd(docker_cmd, title=title)

    def _spawn_terminal(self, script: str, title: str):
        self._spawn_terminal_with_cmd(f"bash -lc {shlex.quote(script)}", title=title)

    def _spawn_terminal_with_cmd(self, cmd: str, title: str):
        candidates = [
            ["gnome-terminal", "--title", title, "--", "bash", "-c", cmd],
            ["xterm", "-T", title, "-e", f"bash -c {shlex.quote(cmd)}"],
            ["konsole", "--new-tab", "-p", f"tabtitle={title}", "-e", "bash", "-c", cmd],
        ]
        last_error = None
        for args in candidates:
            try:
                subprocess.Popen(args)
                return
            except FileNotFoundError as exc:
                last_error = exc
        messagebox.showerror(
            "No terminal found",
            f"Could not open a terminal window.\nLast error: {last_error}",
        )


def main():
    app = RoosterPositionLauncher()
    app.mainloop()


if __name__ == "__main__":
    main()
