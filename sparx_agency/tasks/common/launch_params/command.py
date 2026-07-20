"""Take a command apart into parameters, and put it back together.

Two directions, and both are needed:

* :func:`parse` reads a command that was written by hand -- the ones the
  launcher has always shipped -- and recovers the parameters spelled out in it.
  Those become the *pinned* parameters: they are the command's identity, so they
  are re-emitted even at their default.
* :func:`render` writes a command back out from a head plus a
  :class:`~.spec.ParamSet`, emitting each parameter in its own syntax.

:func:`render_template` covers the commands :func:`parse` must not touch --
anything whose parameters live inside a quoted inner shell, such as
``docker exec falcon bash -lc '... roslaunch ...'``. Splitting those on
whitespace would tear the quoting apart, so they declare their parameters
explicitly and mark their slots with ``{name}`` / ``{params}`` instead.
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field

from .spec import CLI, ENV, FLAG, ROS2, ROSLAUNCH, ParamSet, ParamSpec

#: Marker that switches a ROS2 command line into parameter mode.
ROS_ARGS = "--ros-args"

#: ``name:=value`` -- a roslaunch arg, or the payload of a ``-p``.
_ASSIGN_RE = re.compile(r"^(?P<name>[A-Za-z_][\w./-]*):=(?P<value>.*)$", re.S)
#: ``NAME=value`` -- a shell environment override, only before the command word.
_ENV_RE = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>.*)$", re.S)
#: A token that introduces an option, as opposed to being a value like ``-1.57``.
_OPTION_RE = re.compile(r"^--?[A-Za-z]")

#: How far a rendered command may run before it is wrapped onto the next line.
_WRAP_COLUMNS = 96


@dataclass
class ParsedCommand:
    """A command split into the fixed part and the parameters it carries.

    Attributes:
        preamble: Whole statements that precede the command (a ``VAR=...`` setup
            line, a ``cd``), kept verbatim -- they are not parameters.
        head: Tokens up to the first parameter: the program and its
            subcommands. Re-emitted unchanged.
        tail: Tokens after the parameters that were not recognised as
            parameters. Re-emitted unchanged, after them.
        params: The parameters recovered from the command, in the order they
            appeared.
        raw: The statement verbatim, when it could not be split into tokens at
            all. Re-emitted exactly as given -- a command we failed to
            understand must come back out unharmed, not shell-quoted into
            something that no longer runs.
    """

    preamble: list[str] = field(default_factory=list)
    head: list[str] = field(default_factory=list)
    tail: list[str] = field(default_factory=list)
    params: list[ParamSpec] = field(default_factory=list)
    raw: str = ""


def logical_lines(command: str) -> list[str]:
    """Split a shell snippet into statements, joining ``\\``-continuations.

    Args:
        command: Raw command text, possibly spanning several lines.

    Returns:
        One string per statement, with continuations already joined.
    """
    joined = re.sub(r"\\\s*\n\s*", " ", command.strip())
    return [line.strip() for line in joined.splitlines() if line.strip()]


def parse(command: str, *, source: str = "command") -> ParsedCommand:
    """Recover the parameters spelled out in a hand-written command.

    Only the LAST statement is examined; anything before it is preamble. The
    parameter forms understood are ``-p k:=v`` (ROS2), ``k:=v`` (roslaunch),
    ``--opt value`` / ``--opt`` (argparse), and a leading ``K=v`` (environment).

    Args:
        command: The command text.
        source: Label recorded on each recovered parameter, for the editor.

    Returns:
        The :class:`ParsedCommand`. Every recovered parameter is ``pinned``:
        it was written out deliberately, so it must survive a re-render even
        when it happens to equal the underlying default.
    """
    statements = logical_lines(command)
    if not statements:
        return ParsedCommand()

    parsed = ParsedCommand(preamble=statements[:-1])
    try:
        tokens = shlex.split(statements[-1])
    except ValueError:
        # Unbalanced quoting: not something we can safely take apart, so keep
        # the statement whole rather than mangling it.
        parsed.raw = statements[-1]
        return parsed

    index, seen_command_word = 0, False
    while index < len(tokens):
        token = tokens[index]
        step = 1
        param = None

        env_match = _ENV_RE.match(token)
        assign_match = _ASSIGN_RE.match(token)

        if not seen_command_word and env_match and not assign_match:
            param = ParamSpec(name=env_match.group("name"), default=env_match.group("value"),
                              syntax=ENV, pinned=True, source=source)
        elif token == "-p" and index + 1 < len(tokens) and _ASSIGN_RE.match(tokens[index + 1]):
            inner = _ASSIGN_RE.match(tokens[index + 1])
            param = ParamSpec(name=inner.group("name"), default=inner.group("value"),
                              syntax=ROS2, pinned=True, source=source)
            step = 2
        elif assign_match:
            seen_command_word = True
            param = ParamSpec(name=assign_match.group("name"), default=assign_match.group("value"),
                              syntax=ROSLAUNCH, pinned=True, source=source)
        elif token.startswith("--") and token != ROS_ARGS:
            seen_command_word = True
            takes_value = index + 1 < len(tokens) and not _OPTION_RE.match(tokens[index + 1])
            param = ParamSpec(
                name=token[2:],
                default=tokens[index + 1] if takes_value else "on",
                syntax=CLI if takes_value else FLAG,
                pinned=True, source=source)
            step = 2 if takes_value else 1
        else:
            seen_command_word = True

        if param is None:
            (parsed.head if not parsed.params else parsed.tail).append(token)
        else:
            parsed.params.append(param)
        index += step

    return parsed


def _wrap(tokens: list[list[str]], indent: str = "  ") -> str:
    """Join pre-grouped token runs onto lines no wider than _WRAP_COLUMNS."""
    lines: list[str] = []
    current: list[str] = []
    width = 0
    for group in tokens:
        text = " ".join(shlex.quote(t) for t in group)
        if current and width + len(text) + 1 > _WRAP_COLUMNS:
            lines.append(" ".join(current))
            current, width = [], 0
        current.append(text)
        width += len(text) + 1
    if current:
        lines.append(" ".join(current))
    return (" \\\n" + indent).join(lines)


def render(parsed: ParsedCommand, params: ParamSet) -> str:
    """Write a command out from its fixed parts and its current parameters.

    Parameters are emitted grouped by syntax, in the order a shell expects to
    read them: environment overrides first (they precede the command word),
    then the head, then ROS2 ``-p`` pairs, then argparse options, then
    roslaunch ``k:=v`` args, then whatever tail the parse could not classify.

    Only parameters that are ``pinned`` or moved off their default are emitted
    -- the point of a 200-knob config is that a run states the handful it is
    changing, not all 200.

    Args:
        parsed: The fixed skeleton from :func:`parse`.
        params: Current parameter values.

    Returns:
        The command as multi-line shell text.
    """
    if parsed.raw:
        # A statement we could not tokenize: give it back exactly, with any
        # changed parameters appended where they would normally go. The operator
        # sees the result in the command box and can move them if it is wrong.
        extra = " ".join(shlex.quote(t) for p in params.rendered() for t in p.tokens())
        return "\n".join(parsed.preamble
                         + [parsed.raw + (" " + extra if extra else "")])

    head = list(parsed.head)
    ros2 = params.rendered(ROS2)
    if ros2 and ROS_ARGS not in head:
        # Discovery can surface node parameters a hand-written command never
        # spelled out; without the marker, ROS2 reads -p as the program's own.
        head.append(ROS_ARGS)

    groups: list[list[str]] = [head]
    for syntax in (ROS2, CLI, FLAG, ROSLAUNCH):
        groups.extend(p.tokens() for p in params.rendered(syntax) if p.tokens())
    if parsed.tail:
        groups.append(parsed.tail)

    env = " ".join(" ".join(p.tokens()) for p in params.rendered(ENV))
    body = _wrap([g for g in groups if g])
    if env:
        body = env + " " + body
    return "\n".join(parsed.preamble + [body])


class _Slots(dict):
    """Template mapping that names the offending slot instead of KeyError-ing."""

    def __missing__(self, key: str):
        raise KeyError(
            "command template refers to {%s}, which is not one of its declared "
            "parameters (%s)" % (key, ", ".join(sorted(self)) or "none"))


def render_template(template: str, params: ParamSet) -> str:
    """Fill a command template whose parameters cannot be parsed back out.

    Used for commands that wrap an inner shell in quotes -- ``docker exec ...
    bash -lc '...'`` -- where splitting on whitespace would destroy the quoting,
    and for scripts whose parameters do not sit in one contiguous run. Such a
    command names its slots instead:

    * ``{some_param}`` is replaced by that parameter's current value, always,
      because a slot is part of the command's shape rather than an override;
    * ``{env}`` is replaced by the environment overrides, ready to prefix the
      command word;
    * ``{params}`` is replaced by every other parameter that renders.

    A parameter used as a named slot is excluded from ``{env}``/``{params}``, so
    naming one is how a template says "this belongs here, not in the pile".

    Args:
        template: The command text with ``{...}`` slots.
        params: Current parameter values.

    Returns:
        The filled-in command.

    Raises:
        KeyError: If the template names a slot no parameter provides.
    """
    named = {p.name for p in params if "{%s}" % p.name in template}
    overflow = [p for p in params if p.name not in named and p.rendered]
    quote = lambda group: " ".join(shlex.quote(t) for t in group)  # noqa: E731

    slots = _Slots({p.name: p.value for p in params})
    environment = quote([t for p in overflow if p.syntax == ENV for t in p.tokens()])
    slots["env"] = environment + " " if environment else ""
    slots["params"] = " ".join(filter(None, (
        quote([t for p in overflow if p.syntax == syntax for t in p.tokens()])
        for syntax in (ROS2, CLI, FLAG, ROSLAUNCH))))
    return template.format_map(slots).strip()
