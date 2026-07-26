"""Every parameter a ROS2 node declares, read from its source.

``ros2 run <node> --ros-args -p k:=v`` accepts only parameters the node
``declare_parameter``-ed, and that list is the honest answer to "what can I
change here?" -- far longer than the handful any given launch command spells
out. The localization node declares thirty-odd; the command that starts it
names five.
"""
from __future__ import annotations

from pathlib import Path

from ..spec import ROS2, ParamSpec
from .pysource import PythonSource, as_command_value, keyword, literal


def discover(path: str | Path) -> list[ParamSpec]:
    """Read the ``declare_parameter`` calls out of a ROS2 node.

    Args:
        path: The node's ``.py`` file.

    Returns:
        One :class:`~..spec.ParamSpec` per declared parameter, in declaration
        order, documented from the comments around each call. Declarations
        whose name or default is not a literal are skipped: the editor cannot
        round-trip a value it cannot read.

    Raises:
        OSError: If the file cannot be read.
        SyntaxError: If it is not valid Python.
    """
    source = PythonSource.load(path)
    label = Path(path).name
    params = []

    for call in source.calls("declare_parameter"):
        name = literal(call.args[0]) if call.args else None
        if not isinstance(name, str):
            continue
        default = literal(call.args[1]) if len(call.args) > 1 else literal(
            keyword(call, "value"))
        params.append(ParamSpec(
            name=name,
            default=as_command_value(default),
            doc=source.trailing_doc(call.lineno) or source.doc_above(call.lineno),
            section=source.heading_at(call.lineno),
            choices=("true", "false") if isinstance(default, bool) else (),
            syntax=ROS2,
            source=label,
        ))
    return params
