"""A launchable command, and the live parameter set behind it.

:class:`LaunchItem` is the static declaration -- what to run, where, and where
its parameters are documented. :class:`LaunchPlan` is what the UI actually holds:
that declaration resolved against the repo, against the operator's saved
overrides, and re-rendered into the command Start will run.

The split matters because the catalog is data. Adding a command means adding a
:class:`LaunchItem`, and its parameter screen comes from the sources it names --
never from a second list that has to be kept in step with the first.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

from sparx_agency.tasks.common.launch_params import command as cmd
from sparx_agency.tasks.common.launch_params import discovery
from sparx_agency.tasks.common.launch_params.spec import ParamSet, ParamSpec
from sparx_agency.tasks.common.launch_params.store import ParamStore

Machine = Literal["jetson", "pc", "manual"]

#: Where a command can be started from. "manual" only copies it to the clipboard.
MACHINES: tuple[str, ...] = ("jetson", "pc", "manual")


@dataclass(frozen=True)
class LaunchItem:
    """One command the launcher can start.

    Attributes:
        name: Display name, numbered so the list reads in startup order.
        machine: Default machine. The UI can override it per run, because which
            box owns the GPU or the container is a site decision, not a
            property of the command.
        tmux_name: The tmux session it runs in, and the key its saved parameter
            overrides are filed under.
        description: What it does and what it publishes.
        command: The command itself. Either plain text whose parameters are
            recovered by :func:`~...launch_params.command.parse`, or -- when
            ``template`` is set -- text with ``{slot}`` / ``{env}`` / ``{params}``
            markers, for commands whose parameters sit inside quotes or are not
            in one contiguous run.
        template: Whether ``command`` is a template. Commands that wrap an inner
            shell (``docker exec ... bash -lc '...'``) must set this: splitting
            them on whitespace would tear the quoting apart.
        param_sources: Files declaring the command's full parameter set, coarse
            to fine -- see :class:`~...launch_params.discovery.Source`.
        params: Parameters declared here rather than discovered: the template's
            own slots, and any default this launcher pins that the underlying
            tool leaves unset. Merged last, so they win.
        enabled_by_default: Whether it is ticked for the batch start.
    """

    name: str
    machine: Machine
    tmux_name: str
    description: str
    command: str
    template: bool = False
    param_sources: tuple[discovery.Source, ...] = ()
    params: tuple[ParamSpec, ...] = ()
    enabled_by_default: bool = True


@dataclass
class LaunchPlan:
    """A :class:`LaunchItem` resolved into an editable, runnable state.

    Attributes:
        item: The declaration this was built from.
        params: Every parameter the command accepts, at its current value.
            Edited in place by the parameter editor.
        parsed: The command's fixed skeleton, for non-template items.
        problems: Sources that could not be read, and saved overrides that no
            longer match any parameter.
        machine: The machine chosen for this run; starts at ``item.machine``.
    """

    item: LaunchItem
    params: ParamSet = field(default_factory=ParamSet)
    parsed: cmd.ParsedCommand | None = None
    problems: list[str] = field(default_factory=list)
    machine: str = "jetson"

    @classmethod
    def build(cls, item: LaunchItem, store: ParamStore | None = None) -> "LaunchPlan":
        """Resolve an item: discover its parameters, then apply what was saved.

        The layering is deliberate. Discovery supplies every parameter that
        exists, with its documented default. The command's own parameters land
        on top, because what a command spells out IS what a plain Start runs --
        the localization node defaults ``alpha`` to 0.8, but the command asks
        for 0.2, so 0.2 is the default this screen must show and reset to.
        Saved overrides come last, as they are the operator's own decisions.

        Args:
            item: The command to resolve.
            store: Where the operator's overrides were saved, if any.

        Returns:
            The resolved plan.
        """
        found = discovery.discover(item.param_sources)
        plan = cls(item=item, params=found.params,
                   problems=list(found.problems), machine=item.machine)

        # COPY the declared parameters. They are module-level constants, shared
        # between items on purpose (every container command wants the same
        # `container` knob) -- but a ParamSet holds them by reference and the
        # editor writes into them, so without this, setting the container on the
        # RViz screen would silently change the BEV viewer's too, and each would
        # save an override the operator never made there.
        declared = [replace(param) for param in item.params]
        if item.template:
            plan.params.extend(declared)
        else:
            plan.parsed = cmd.parse(item.command)
            plan.params.extend(plan.parsed.params)
            plan.params.extend(declared)

        if store is not None:
            unknown = plan.params.apply(store.get(item.tmux_name))
            if unknown:
                plan.problems.append(
                    "saved values no longer match any parameter and were not "
                    "applied: " + ", ".join(sorted(unknown)))
        return plan

    def command_text(self) -> str:
        """The command as it stands, with the current parameter values in it."""
        if self.item.template:
            return cmd.render_template(self.item.command, self.params)
        return cmd.render(self.parsed or cmd.ParsedCommand(), self.params)

    def reset(self) -> None:
        """Put every parameter back to its built-in default."""
        self.params.reset()

    def save(self, store: ParamStore) -> None:
        """Persist only the parameters moved off their default."""
        store.put(self.item.tmux_name, self.params.as_dict())

    def forget(self, store: ParamStore) -> None:
        """Drop this command's saved overrides, leaving the current values alone."""
        store.put(self.item.tmux_name, {})
