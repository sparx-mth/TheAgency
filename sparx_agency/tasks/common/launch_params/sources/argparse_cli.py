"""Every option a plain Python script accepts, read from its ``argparse`` setup.

The non-ROS members of the stack -- the XTEND bridge publisher, the demo mode
manager, the NavDP TensorRT server -- are argparse scripts, and ``add_argument``
already carries everything the editor wants: the flag, its default, its ``help``
text, and often a closed set of ``choices``.
"""
from __future__ import annotations

from pathlib import Path

from ..spec import CLI, FLAG, ParamSpec
from .pysource import PythonSource, as_command_value, keyword, literal

#: argparse actions that take no value on the command line.
_FLAG_ACTIONS = ("store_true", "store_false", "count")


def _long_option(call) -> str | None:
    """The long flag of an ``add_argument`` call, without its leading dashes.

    Positional arguments (no leading dash) are not options anyone can toggle,
    and a short-only flag has no readable name, so both answer ``None``.
    """
    for arg in call.args:
        text = literal(arg)
        if isinstance(text, str) and text.startswith("--"):
            return text[2:]
    return None


def discover(path: str | Path) -> list[ParamSpec]:
    """Read the ``add_argument`` calls out of an argparse script.

    Args:
        path: The script's ``.py`` file.

    Returns:
        One :class:`~..spec.ParamSpec` per long option, in declaration order.
        A ``store_true``-style option becomes a FLAG parameter, whose value is
        its own presence, so the editor offers it as on/off rather than as text.

    Raises:
        OSError: If the file cannot be read.
        SyntaxError: If it is not valid Python.
    """
    source = PythonSource.load(path)
    label = Path(path).name
    params = []

    for call in source.calls("add_argument"):
        name = _long_option(call)
        if name is None:
            continue
        action = literal(keyword(call, "action"))
        is_flag = isinstance(action, str) and action in _FLAG_ACTIONS
        choices = literal(keyword(call, "choices"))
        help_text = literal(keyword(call, "help"))
        params.append(ParamSpec(
            name=name,
            default="off" if is_flag else as_command_value(
                literal(keyword(call, "default"))),
            doc=str(help_text) if isinstance(help_text, str)
                else source.doc_above(call.lineno),
            section=source.heading_at(call.lineno),
            choices=("off", "on") if is_flag else tuple(
                as_command_value(c) for c in choices) if isinstance(
                    choices, (list, tuple)) else (),
            syntax=FLAG if is_flag else CLI,
            source=label,
        ))
    return params
