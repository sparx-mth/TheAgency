"""Define, edit and render the parameters of any command a launcher can start.

A launcher normally hard-codes each command as a string, which means the only
parameters an operator can reach are the ones whoever wrote that string thought
to spell out. The localization node declares thirty-odd parameters and its
launch command names five; the object mission has some three hundred and names
four. The rest are invisible, and editing them means editing the launcher.

This package closes that gap in three steps:

1. **Discover** (:mod:`discovery`, :mod:`sources`) -- read the full parameter
   set out of whatever declared it: a ROS2 node's ``declare_parameter`` calls, a
   script's ``argparse`` setup, a launch file's ``<arg>`` elements, a commented
   YAML config. The comments come too, so every knob arrives explained.
2. **Edit** (:mod:`spec`, :mod:`editor`) -- present them grouped and searchable,
   each against its own default, with one click to put any of them (or all of
   them) back.
3. **Render** (:mod:`command`) -- write the command back out with the operator's
   values in it, emitting only what was deliberately set: the parameters the
   command already named, plus the ones actually moved off their default.

:mod:`store` keeps those overrides across sessions.

Typical use::

    found = discovery.discover(item.param_sources)
    parsed = command.parse(item.command)
    found.params.extend(parsed.params)      # what the command already names
    found.params.apply(store.get(item.key))  # what was saved last time
    ...
    text = command.render(parsed, found.params)
"""
from __future__ import annotations

from .command import ParsedCommand, parse, render, render_template
from .discovery import Discovery, Source, discover
from .spec import CLI, ENV, FLAG, ROS2, ROSLAUNCH, ParamSet, ParamSpec
from .store import ParamStore

__all__ = [
    "CLI", "ENV", "FLAG", "ROS2", "ROSLAUNCH",
    "Discovery", "ParamSet", "ParamSpec", "ParamStore", "ParsedCommand", "Source",
    "discover", "parse", "render", "render_template",
]
