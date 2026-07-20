"""One parameter, and an ordered set of them.

The vocabulary the whole package speaks. A :class:`ParamSpec` is one knob a
command exposes: what it is called, what it defaults to, what it is set to right
now, what it means, and -- crucially -- *how it is written on a command line*,
which differs per launcher world (``-p k:=v`` for ROS2, ``k:=v`` for roslaunch,
``--k v`` for argparse, ``K=v`` for a shell env override).

Keeping the syntax on the parameter, rather than on the command, is what lets a
single command mix them: ``NAV_MODE=hybrid ./run_object_mission.sh vel_x:=0.3``
is one env parameter and one roslaunch parameter rendered side by side.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

#: ``-p name:=value`` after ``--ros-args`` (a ROS2 node parameter).
ROS2 = "ros2"
#: ``-r from:=to`` after ``--ros-args`` (a ROS2 remapping). A separate syntax
#: because it is not a parameter of the node: it rewires a name, and writing it
#: with ``-p`` would set a parameter nobody reads instead of remapping anything.
ROS2_REMAP = "ros2_remap"
#: ``name:=value`` (a roslaunch ``<arg>``, or a mission-script launch override).
ROSLAUNCH = "roslaunch"
#: ``--name value`` (an argparse option).
CLI = "cli"
#: ``--name`` with no value (an argparse ``store_true`` flag).
FLAG = "flag"
#: ``NAME=value`` written before the command (a shell environment override).
ENV = "env"
#: A value a command template substitutes itself, via its own ``{name}`` slot --
#: a container name, a workspace path. Editable like any other parameter, but it
#: is never written out as an option, because it is part of the command's shape.
SLOT = "slot"

SYNTAXES = (ROS2, ROS2_REMAP, ROSLAUNCH, CLI, FLAG, ENV, SLOT)

#: Values a FLAG parameter may take. "on" renders the flag, "off" omits it.
FLAG_CHOICES = ("off", "on")


@dataclass
class ParamSpec:
    """A single knob of a launchable command.

    Attributes:
        name: The parameter's name, without any syntax decoration (``vel_x``,
            not ``vel_x:=``).
        default: The built-in default, as it would appear on a command line.
            This is what "Reset" restores, and the value against which
            :attr:`changed` is judged.
        value: What it is set to right now. Starts equal to ``default``.
        doc: One-line human explanation, harvested from the source that
            declared the parameter.
        detail: The long-form reasoning, where the source had room to give it --
            whole paragraphs, in the FALCON configs. Shown on demand rather than
            in the row, which is why it is kept apart from :attr:`doc`.
        section: Grouping header for the editor, e.g. ``"A* GLOBAL PLANNER"``.
        choices: Allowed values, when the source declared a closed set. Renders
            as a dropdown instead of a free-text box.
        syntax: One of :data:`SYNTAXES` -- how this parameter is written out.
        pinned: Render this parameter even when it equals its default. Set for
            parameters that were spelled out in the command we started from:
            they are part of that command's identity (``provider_type:=apriltag``
            selects the whole provider), not an override of it.
        source: Where the parameter was discovered, for the editor's tooltip.
    """

    name: str
    default: str
    value: str = ""
    doc: str = ""
    detail: str = ""
    section: str = ""
    choices: Sequence[str] = ()
    syntax: str = ROS2
    pinned: bool = False
    source: str = ""

    def __post_init__(self) -> None:
        if self.syntax not in SYNTAXES:
            raise ValueError(
                "unknown parameter syntax %r for %r (expected one of %s)"
                % (self.syntax, self.name, ", ".join(SYNTAXES)))
        self.default = str(self.default)
        # A freshly discovered parameter sits at its default; an explicit value
        # (e.g. one parsed out of a command) is preserved.
        self.value = str(self.value) if self.value != "" else self.default
        self.choices = tuple(str(c) for c in self.choices)

    @property
    def changed(self) -> bool:
        """True when the current value differs from the built-in default."""
        return self.value != self.default

    @property
    def rendered(self) -> bool:
        """True when this parameter belongs on the command line."""
        return self.pinned or self.changed

    def reset(self) -> None:
        """Put the parameter back to its built-in default."""
        self.value = self.default

    def tokens(self) -> list[str]:
        """The command-line tokens for this parameter, in its own syntax.

        Returns:
            A list of shell tokens (already split; the caller quotes them).
            Empty for a FLAG that is off, which is spelled by *absence*.
        """
        if self.syntax == SLOT:
            return []
        if self.syntax in (ROS2, ROS2_REMAP):
            flag = "-p" if self.syntax == ROS2 else "-r"
            return [flag, "%s:=%s" % (self.name, self.value)]
        if self.syntax in (ROSLAUNCH, ENV):
            joiner = ":=" if self.syntax == ROSLAUNCH else "="
            return ["%s%s%s" % (self.name, joiner, self.value)]
        if self.syntax == CLI:
            return ["--%s" % self.name, self.value]
        # FLAG: presence is the value.
        return ["--%s" % self.name] if self.value == "on" else []


class ParamSet:
    """An ordered, name-keyed collection of :class:`ParamSpec`.

    Order is the order parameters were added, which is the order they were
    declared in their source file -- so the editor shows them grouped the way
    their author grouped them, not alphabetically.
    """

    def __init__(self, params: Iterable[ParamSpec] = ()) -> None:
        self._by_name: dict[str, ParamSpec] = {}
        for param in params:
            self.add(param)

    def __iter__(self) -> Iterator[ParamSpec]:
        return iter(self._by_name.values())

    def __len__(self) -> int:
        return len(self._by_name)

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def __getitem__(self, name: str) -> ParamSpec:
        return self._by_name[name]

    def get(self, name: str, fallback: ParamSpec | None = None) -> ParamSpec | None:
        return self._by_name.get(name, fallback)

    def add(self, param: ParamSpec, *, override_default: bool = True) -> ParamSpec:
        """Add a parameter, or enrich the one already registered under its name.

        Discovery runs several sources over the same command, and they overlap
        by design: the launch file knows every ``<arg>`` that exists, the YAML
        config knows which of them the operator already re-defaulted, and the
        command we started from knows which are deliberately spelled out. Later
        sources therefore *refine* rather than replace -- a second source fills
        in a missing doc, and never blanks one an earlier source supplied.

        Args:
            param: The parameter to merge in.
            override_default: Whether this source may restate an existing
                parameter's default. False for a source that is authoritative
                about which parameters *exist* but not about what they are set
                to -- a launch file read after the config that re-defaults it
                would otherwise hand back the built-in the config replaced, and
                the editor would offer to "reset" to a value no run ever uses.

        Returns:
            The :class:`ParamSpec` now registered under ``param.name``.
        """
        existing = self._by_name.get(param.name)
        if existing is None:
            self._by_name[param.name] = param
            return param

        # The winning default is the one closest to what the command really runs
        # with, and the value follows it unless the operator has moved it.
        if override_default and param.default != existing.default:
            was_changed = existing.changed
            existing.default = param.default
            if not was_changed:
                existing.value = param.default
        if param.value != param.default:
            existing.value = param.value
        existing.doc = existing.doc or param.doc
        existing.detail = existing.detail or param.detail
        existing.section = existing.section or param.section
        existing.choices = existing.choices or param.choices
        existing.source = existing.source or param.source
        existing.pinned = existing.pinned or param.pinned
        # ROS2 and roslaunch describe the same knob differently; the syntax the
        # command itself uses is the one that can actually launch it.
        if param.pinned:
            existing.syntax = param.syntax
        return existing

    def extend(self, params: Iterable[ParamSpec], *,
               override_default: bool = True) -> None:
        for param in params:
            self.add(param, override_default=override_default)

    def rendered(self, syntax: str | None = None) -> list[ParamSpec]:
        """Parameters that belong on the command line, optionally one syntax."""
        return [p for p in self
                if p.rendered and (syntax is None or p.syntax == syntax)]

    def changed(self) -> list[ParamSpec]:
        """Parameters the operator has moved off their default."""
        return [p for p in self if p.changed]

    def sections(self) -> list[str]:
        """Section names in first-seen order, so the editor can group by them."""
        seen: list[str] = []
        for param in self:
            if param.section not in seen:
                seen.append(param.section)
        return seen

    def reset(self) -> None:
        """Put every parameter back to its built-in default."""
        for param in self:
            param.reset()

    def apply(self, values: dict[str, str]) -> list[str]:
        """Set saved values onto matching parameters.

        Args:
            values: ``name -> value``, e.g. as restored from disk.

        Returns:
            The names in ``values`` that no parameter claimed. A saved value for
            a parameter that no longer exists is reported rather than dropped:
            it usually means the underlying node renamed it, and silently
            forgetting the operator's setting is how a drone flies with a
            parameter nobody realised had stopped being applied.
        """
        unknown = []
        for name, value in values.items():
            param = self._by_name.get(name)
            if param is None:
                unknown.append(name)
            else:
                param.value = str(value)
        return unknown

    def as_dict(self) -> dict[str, str]:
        """The changed parameters as ``name -> value``, for persistence."""
        return {p.name: p.value for p in self.changed()}
