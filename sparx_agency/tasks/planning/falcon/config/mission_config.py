#!/usr/bin/env python3
"""Read the object mission's YAML config and emit shell assignments for it.

``run_object_mission.sh`` spans three worlds that each want their parameters in a
different shape: host/docker values (paths, mounts), ``roslaunch`` ``<arg>`` values,
and the ROS2 detector sidecar's ``-p key:=value`` params. A roslaunch ``<arg>``
CANNOT be loaded from a rosparam YAML (``$(arg ...)`` is resolved at parse time,
before the parameter server exists), so the single config file is read HERE, on the
host, and fanned out to each world in its native form.

This module is the reader. It loads the YAML, validates it, and prints shell code
for the caller to ``eval``::

    eval "$(mission_config.py config/mission.yaml --launch-file .../object_mission.launch)"

Every value becomes a ``MISSION_CFG_*`` variable, which the script then applies with
``${VAR:-$MISSION_CFG_X}`` semantics so the precedence stays:

    command line  >  environment variable  >  this YAML  >  built-in default

The free-form ``launch:`` section is emitted as a ``MISSION_CFG_LAUNCH_ARGS`` array of
``key:=value`` strings, validated against the ``<arg>`` names actually declared by the
launch file, so a typo fails here with a suggestion rather than being silently ignored
(CLAUDE.md: prefer raising errors over silent fallbacks).

Deliberately ROS-free: this runs on the host before any ROS environment is sourced.
"""
import argparse
import difflib
import shlex
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - environment problem, not logic
    sys.exit("mission_config: PyYAML is required (pip install pyyaml)")

#: Emitted variable prefix, so config values never collide with the caller's own vars.
PREFIX = "MISSION_CFG_"

#: section -> {yaml key: emitted variable suffix}. The shape of the whole config.
SCHEMA = {
    "mission": {
        "map": "MAP",
        "selection_mode": "SELECTION_MODE",
        "nav_mode": "NAV_MODE",
        "seed": "SEED",
    },
    "detector": {
        "model": "MODEL",
        "engines_dir": "ENGINES_DIR",
        "weights_dir": "WEIGHTS_DIR",
        "text_weights": "TEXT_WEIGHTS",
        "conf_thresh": "CONF_THRESH",
        "init_target": "INIT_TARGET",
    },
    "paths": {
        "objects_dir": "OBJECTS_DIR",
        "objects_file": "OBJECTS_FILE",
    },
}

#: Free-form section: any key declared as an <arg> by the mission launch file.
LAUNCH_SECTION = "launch"


class ConfigError(ValueError):
    """The config file is malformed, or names something that does not exist."""


def _hint(name, known):
    """' Did you mean ...?' for a mistyped key, or '' when nothing is close."""
    close = difflib.get_close_matches(str(name), sorted(known), n=1)
    return "  Did you mean %r?" % close[0] if close else ""


def _scalar(section, key, value):
    """Render one YAML scalar the way the shell / roslaunch expects to receive it.

    YAML ``true`` must reach roslaunch as ``true`` (Python's ``str(True)`` would give
    ``True``, which roslaunch does not accept as a boolean).
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None or isinstance(value, (dict, list)):
        raise ConfigError("%s.%s must be a single value, got %r"
                          % (section, key, value))
    return str(value)


def launch_arg_names(launch_file):
    """Every ``<arg name=...>`` the launch file declares.

    Args:
        launch_file: Path to the mission launch XML.

    Returns:
        A set of declared arg names; empty if ``launch_file`` is None.

    Raises:
        ConfigError: If the file cannot be parsed.
    """
    if launch_file is None:
        return set()
    try:
        root = ET.parse(str(launch_file)).getroot()
    except (OSError, ET.ParseError) as e:
        raise ConfigError("cannot read launch file %s: %s" % (launch_file, e))
    return {a.get("name") for a in root.iter("arg") if a.get("name")}


def parse(raw, valid_launch_args=frozenset()):
    """Validate a loaded config mapping and split it into vars + launch args.

    Args:
        raw: The mapping loaded from YAML (``None``/empty is allowed: nothing set).
        valid_launch_args: Arg names the launch file declares. When non-empty, every
            key of the ``launch:`` section must be one of them.

    Returns:
        ``(variables, launch_args)`` -- a dict of ``SUFFIX -> str`` and a list of
        ``"key:=value"`` strings, both in file order.

    Raises:
        ConfigError: On an unknown section/key, a non-mapping section, or a
            non-scalar value.
    """
    if raw is None:
        return {}, []
    if not isinstance(raw, dict):
        raise ConfigError("config must be a YAML mapping, got %s"
                          % type(raw).__name__)

    known_sections = set(SCHEMA) | {LAUNCH_SECTION}
    variables, launch_args = {}, []

    for section, body in raw.items():
        if section not in known_sections:
            raise ConfigError("unknown section %r.%s" % (section, _hint(section, known_sections)))
        if body is None:
            continue
        if not isinstance(body, dict):
            raise ConfigError("section %r must be a mapping, got %s"
                              % (section, type(body).__name__))

        if section == LAUNCH_SECTION:
            for key, value in body.items():
                if valid_launch_args and key not in valid_launch_args:
                    raise ConfigError(
                        "launch.%s is not an arg of the mission launch file.%s"
                        % (key, _hint(key, valid_launch_args)))
                launch_args.append("%s:=%s" % (key, _scalar(section, key, value)))
            continue

        for key, value in body.items():
            if key not in SCHEMA[section]:
                raise ConfigError("unknown key %s.%s%s"
                                  % (section, key, _hint(key, SCHEMA[section])))
            variables[SCHEMA[section][key]] = _scalar(section, key, value)

    return variables, launch_args


def load(path, launch_file=None):
    """Load + validate the config file.

    Args:
        path: Path to the YAML config.
        launch_file: Optional launch XML whose ``<arg>`` names bound the ``launch:``
            section.

    Returns:
        ``(variables, launch_args)`` as per :func:`parse`.

    Raises:
        ConfigError: If the file cannot be read or is invalid.
    """
    try:
        text = Path(path).read_text()
    except OSError as e:
        raise ConfigError("cannot read config %s: %s" % (path, e))
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ConfigError("config %s is not valid YAML: %s" % (path, e))
    return parse(raw, launch_arg_names(launch_file))


def emit(variables, launch_args):
    """Render the shell code the caller evals. Every value is shell-quoted."""
    lines = ["%s%s=%s" % (PREFIX, name, shlex.quote(value))
             for name, value in sorted(variables.items())]
    lines.append("%sLAUNCH_ARGS=(%s)"
                 % (PREFIX, " ".join(shlex.quote(a) for a in launch_args)))
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("config", help="path to the mission YAML")
    p.add_argument("--launch-file", default=None,
                   help="launch XML whose <arg> names bound the launch: section")
    args = p.parse_args(argv)
    try:
        print(emit(*load(args.config, args.launch_file)))
    except ConfigError as e:
        # stderr + nonzero: the caller's `eval` then sets nothing and the script stops.
        sys.stderr.write("[ERROR] %s\n" % e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
