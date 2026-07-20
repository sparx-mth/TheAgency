"""The launcher window: pick a command, set its parameters, start it.

Selecting a command resolves it into a :class:`~.item.LaunchPlan` -- its full
parameter set, discovered from the files that declare it, with the operator's
saved overrides applied -- and shows that on the Parameters tab. Editing any
parameter re-renders the Command tab, and Start runs exactly what is in it.

Plans are built on first selection and then cached, so the mission's three
hundred parameters are read once rather than on every click.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from sparx_agency.tasks.common.launch_params.editor import ParameterEditor
from sparx_agency.tasks.common.launch_params.store import (DEFAULT_PATH as
                                                           DEFAULT_STORE_PATH)
from sparx_agency.tasks.common.launch_params.store import ParamStore

from . import auto_pipeline, remote
from .environments import (JETSON_REPO, JETSON_SSH_DEFAULT, normalize_command,
                           wrap_with_env)
from .item import MACHINES, LaunchItem, LaunchPlan
from .items import LAUNCH_ITEMS

#: Demo-mode buttons, and whether each needs confirming before it is published.
DEMO_MODES = (("IDLE", "idle", False), ("FLY_STRAIGHT", "fly_straight", False),
              ("TURNING", "turning", False), ("VISUAL_SERVOING", "visual_servoing", False),
              ("FINISH", "finish", True))


class XtendPipelineLauncher(tk.Tk):
    """The launcher window."""

    def __init__(self, items: list[LaunchItem] = None, store: ParamStore = None) -> None:
        super().__init__()
        self.title("XTEND Pipeline Launcher — RGBD Mapping + Object Mission")
        self.geometry("1500x900")

        self.items = list(items if items is not None else LAUNCH_ITEMS)
        self.store = store if store is not None else ParamStore()
        self._plans: dict[str, LaunchPlan] = {}
        self._selected: LaunchItem | None = None

        self.ssh_target = tk.StringVar(value=JETSON_SSH_DEFAULT)
        self.status = tk.StringVar(value="Ready.")
        self.machine = tk.StringVar(value="jetson")
        self.checked: dict[str, tk.BooleanVar] = {
            item.tmux_name: tk.BooleanVar(value=item.enabled_by_default)
            for item in self.items}
        self._build()

    # ── layout ────────────────────────────────────────────────────

    def _build(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=8)
        ttk.Label(top, text="Jetson SSH:").pack(side="left")
        ttk.Entry(top, textvariable=self.ssh_target, width=24).pack(side="left", padx=6)
        for text, action in (("Start selected", self.start_selected),
                             ("Stop selected", self.stop_selected),
                             ("Start checked", self.start_checked),
                             ("AUTO perception bring-up", self.start_auto),
                             ("Stop all known", self.stop_all)):
            ttk.Button(top, text=text, command=action).pack(side="left", padx=3)

        modes = ttk.Frame(self)
        modes.pack(fill="x", padx=10, pady=(0, 6))
        ttk.Label(modes, text="Demo mode:").pack(side="left")
        for label, mode, confirm in DEMO_MODES:
            ttk.Button(modes, text=label,
                       command=lambda m=mode, c=confirm: self.publish_mode(m, c)
                       ).pack(side="left", padx=3)

        panes = ttk.PanedWindow(self, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=10, pady=6)
        left, right = ttk.Frame(panes), ttk.Frame(panes)
        panes.add(left, weight=1)
        panes.add(right, weight=3)

        ttk.Label(left, text="Commands", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        self.listbox = tk.Listbox(left, exportselection=False, activestyle="none")
        self.listbox.pack(fill="both", expand=True, pady=4)
        for item in self.items:
            self.listbox.insert("end", item.name)
        self.listbox.bind("<<ListboxSelect>>", lambda _e: self.on_select())

        batch = ttk.LabelFrame(left, text="Included in “Start checked”")
        batch.pack(fill="x", pady=6)
        for item in self.items:
            if item.machine == "jetson":
                ttk.Checkbutton(batch, text=item.name,
                                variable=self.checked[item.tmux_name]).pack(anchor="w")

        self.tabs = ttk.Notebook(right)
        self.tabs.pack(fill="both", expand=True)
        self._build_parameters_tab()
        self._build_command_tab()
        self._build_about_tab()

        ttk.Label(self, textvariable=self.status, relief="sunken",
                  anchor="w").pack(fill="x", padx=10, pady=4)
        self.listbox.selection_set(0)
        self.on_select()

    def _build_parameters_tab(self) -> None:
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="Parameters")

        bar = ttk.Frame(tab)
        bar.pack(fill="x", pady=(6, 0), padx=6)
        ttk.Label(bar, text="Run on:").pack(side="left")
        ttk.Combobox(bar, textvariable=self.machine, values=list(MACHINES),
                     state="readonly", width=9).pack(side="left", padx=(4, 12))
        ttk.Button(bar, text="Save these as my defaults",
                   command=self.save_defaults).pack(side="left", padx=3)
        ttk.Button(bar, text="Forget saved",
                   command=self.forget_defaults).pack(side="left", padx=3)

        self.editor = ParameterEditor(tab, on_change=self.refresh_command)
        self.editor.pack(fill="both", expand=True, padx=6, pady=6)

    def _build_command_tab(self) -> None:
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="Command")
        ttk.Label(tab, text="This is what Start runs. Editing here overrides the "
                            "parameters — until you change one, which rewrites it.",
                  foreground="#444444").pack(anchor="w", padx=6, pady=(6, 2))
        self.command_text = tk.Text(tab, wrap="none", height=20)
        self.command_text.pack(fill="both", expand=True, padx=6)

        row = ttk.Frame(tab)
        row.pack(fill="x", pady=8, padx=6)
        for text, action in (("Copy command", self.copy_command),
                             ("Copy env + command", self.copy_full_command),
                             ("Copy tmux attach", self.copy_attach)):
            ttk.Button(row, text=text, command=action).pack(side="left", padx=3)
        ttk.Label(row, text="AUTO arms the drone and takes off. Check the area first.",
                  foreground="#a61b1b").pack(side="right")

    def _build_about_tab(self) -> None:
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="About this command")
        self.about_text = tk.Text(tab, wrap="word", height=20)
        self.about_text.pack(fill="both", expand=True, padx=6, pady=6)

    # ── selection ─────────────────────────────────────────────────

    def plan_for(self, item: LaunchItem) -> LaunchPlan:
        """The item's resolved plan, built on first use and cached after."""
        if item.tmux_name not in self._plans:
            self._plans[item.tmux_name] = LaunchPlan.build(item, self.store)
        return self._plans[item.tmux_name]

    def on_select(self) -> None:
        """Show the highlighted command: its parameters, command and description."""
        selection = self.listbox.curselection()
        if not selection:
            return
        item = self.items[int(selection[0])]
        self._selected = item
        plan = self.plan_for(item)
        self.machine.set(plan.machine)

        self.editor.show(plan.params, tuple(plan.problems))
        self.refresh_command()

        saved = self.store.get(item.tmux_name)
        self.about_text.delete("1.0", "end")
        self.about_text.insert("end", "%s\n\nRuns on: %s     tmux session: %s%s\n\n%s\n" % (
            item.name, item.machine, item.tmux_name,
            "     saved overrides: %d" % len(saved) if saved else "",
            item.description))

    def refresh_command(self) -> None:
        """Re-render the Command tab from the current parameter values."""
        if self._selected is None:
            return
        self.command_text.delete("1.0", "end")
        self.command_text.insert("end", self.plan_for(self._selected).command_text())

    def current_command(self) -> str:
        """Whatever is in the Command box, which is what Start runs."""
        return normalize_command(self.command_text.get("1.0", "end"))

    # ── saved defaults ────────────────────────────────────────────

    def save_defaults(self) -> None:
        if self._selected is None:
            return
        plan = self.plan_for(self._selected)
        changed = len(plan.params.changed())
        self._guarded(
            lambda: plan.save(self.store),
            "Saved %d changed parameter(s) for %s to %s"
            % (changed, self._selected.tmux_name, self.store.path) if changed else
            "Nothing is changed from default, so nothing was saved for %s"
            % self._selected.tmux_name)

    def forget_defaults(self) -> None:
        if self._selected is None:
            return
        self._guarded(
            lambda: self.plan_for(self._selected).forget(self.store),
            "Forgot the saved parameters for %s. The values on screen are unchanged "
            "until you reset them." % self._selected.tmux_name)

    # ── starting and stopping ─────────────────────────────────────

    def start_selected(self) -> None:
        """Start the highlighted command on whichever machine is chosen."""
        if self._selected is None:
            return
        machine = self.machine.get()
        if machine == "manual":
            self.copy_full_command()
            messagebox.showinfo("Manual command",
                                "Set to run manually, so it was copied to the clipboard "
                                "instead of started.")
            return
        if machine == "pc":
            self._guarded(lambda: remote.spawn_local_terminal(
                wrap_with_env("pc", self.current_command()), self._selected.tmux_name),
                "Started a local terminal for %s" % self._selected.name)
            return
        self._start_on_jetson(self._selected, self.current_command())

    def _start_on_jetson(self, item: LaunchItem, command: str, quiet: bool = False) -> None:
        self._guarded(
            lambda: remote.start_tmux_over_ssh(
                self.ssh_target.get().strip(), item.tmux_name,
                wrap_with_env("jetson", command)),
            None if quiet else "Started Jetson tmux session: %s" % item.tmux_name)

    def start_checked(self) -> None:
        """Start every ticked Jetson command, each with its own parameters."""
        started = []
        for item in self.items:
            if item.machine == "jetson" and self.checked[item.tmux_name].get():
                self._start_on_jetson(item, self.plan_for(item).command_text(), quiet=True)
                started.append(item.tmux_name)
        self.status.set("Started %d session(s): %s" % (len(started), ", ".join(started)))

    def start_auto(self) -> None:
        """Run the scripted perception bring-up, with the current parameters in it."""
        if not messagebox.askyesno(
                "Start AUTO perception bring-up?",
                "This starts bridge, depth and the mode manager, then SENDS ARM AND "
                "TAKEOFF, waits 30 seconds, and starts localization, TF and octomap.\n\n"
                "It does NOT start the object mission — items 11-13 do that.\n\n"
                "Is the drone clear to arm and take off?"):
            return
        script = auto_pipeline.build(
            lambda name: self.plan_for(self._item(name)).command_text())
        self._guarded(
            lambda: remote.start_tmux_over_ssh(
                self.ssh_target.get().strip(), auto_pipeline.AUTO_SESSION,
                wrap_with_env("jetson", script), hold_open=True),
            "AUTO bring-up started in tmux: %s" % auto_pipeline.AUTO_SESSION)

    def stop_selected(self) -> None:
        if self._selected is None:
            return
        remote.stop_tmux_over_ssh(self.ssh_target.get().strip(), self._selected.tmux_name)
        self.status.set("Stopped Jetson tmux session: %s" % self._selected.tmux_name)

    def stop_all(self) -> None:
        """Stop every session this launcher knows how to start."""
        target = self.ssh_target.get().strip()
        names = [item.tmux_name for item in self.items if item.machine == "jetson"]
        for name in names + [auto_pipeline.AUTO_SESSION]:
            remote.stop_tmux_over_ssh(target, name)
        self.status.set("Requested stop for %d known session(s)." % (len(names) + 1))

    def publish_mode(self, mode: str, confirm: bool) -> None:
        if confirm and not messagebox.askyesno(
                "Confirm demo mode change",
                "Publish demo mode request: %s?\n\nFINISH triggers stop -> land -> "
                "disarm if the mode manager is running." % mode):
            return
        self._guarded(
            lambda: remote.publish_demo_mode(self.ssh_target.get().strip(),
                                             JETSON_REPO, mode),
            "Published demo mode request: %s" % mode)

    # ── clipboard ─────────────────────────────────────────────────

    def copy_command(self) -> None:
        self._copy(self.current_command(), "command")

    def copy_full_command(self) -> None:
        if self._selected is None:
            return
        machine = self.machine.get()
        text = (self.current_command() if machine == "manual"
                else wrap_with_env(machine, self.current_command()))
        self._copy(text, "environment + command")

    def copy_attach(self) -> None:
        if self._selected is None:
            return
        self._copy(remote.attach_command(self.ssh_target.get().strip(),
                                         self._selected.tmux_name),
                   "tmux attach command")

    def _copy(self, text: str, label: str) -> None:
        self.clipboard_clear()
        self.clipboard_append(text)
        self.status.set("Copied %s to the clipboard." % label)

    # ── helpers ───────────────────────────────────────────────────

    def _item(self, tmux_name: str) -> LaunchItem:
        """The catalog item with this session name.

        Raises:
            KeyError: If nothing declares it -- which means the AUTO script and
                the catalog have gone out of step, and starting a partial
                pipeline would be worse than saying so.
        """
        for item in self.items:
            if item.tmux_name == tmux_name:
                return item
        raise KeyError("no launch item declares the tmux session %r" % tmux_name)

    def _guarded(self, action, success: str | None) -> None:
        """Run ``action``, reporting either its success or its error."""
        try:
            action()
        except (remote.RemoteError, KeyError, ValueError, OSError) as error:
            messagebox.showerror("Command failed", str(error))
            self.status.set(str(error).splitlines()[0])
            return
        if success:
            self.status.set(success)


def main() -> None:
    """Open the launcher, surviving an unreadable store of saved parameters.

    A corrupt overrides file must not brick the tool the operator is standing in
    front of -- but it must not pass unnoticed either, or they will wonder for an
    hour why yesterday's settings are gone. So: say so, then carry on with none.
    """
    try:
        store = ParamStore()
    except ValueError as error:
        # Move the unreadable file aside rather than overwrite it: it holds the
        # settings, and whatever corrupted it is worth being able to look at.
        spoiled = DEFAULT_STORE_PATH.with_suffix(".broken")
        DEFAULT_STORE_PATH.replace(spoiled)
        store = ParamStore()
        messagebox.showwarning(
            "Saved parameters could not be read",
            "%s\n\nIt has been moved to %s and the launcher started with no saved "
            "overrides. Every command still has its own defaults; re-save the ones "
            "you want kept." % (error, spoiled))
    XtendPipelineLauncher(store=store).mainloop()
