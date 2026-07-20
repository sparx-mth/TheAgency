"""Point a command at the files that declare its parameters, and collect them.

A launchable command declares *where* its knobs are documented -- the node it
runs, the launch file it includes, the config it reads -- and this resolves that
into one :class:`~.spec.ParamSet`. Several sources over one command is the
normal case, and they layer: the launch file says which arguments exist, the
YAML config says which of them the operator has already re-defaulted, and the
command itself says which are deliberately spelled out.

Discovery never raises for a source it cannot read. A moved node file must not
stop the launcher from starting the other fifteen commands -- but it must not
pass unnoticed either, so the failure is returned to the caller to put on
screen next to the parameters that did load.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

from .spec import ParamSet
from .sources import argparse_cli, ros2_node, roslaunch_xml, yaml_config

#: Repo root: .../sparx_agency/tasks/common/launch_params/discovery.py
REPO_ROOT = Path(__file__).resolve().parents[4]

#: Reader per source kind. Each takes a path and returns a list of ParamSpec.
READERS = {
    "ros2_node": ros2_node.discover,
    "argparse": argparse_cli.discover,
    "roslaunch": roslaunch_xml.discover,
    "yaml": yaml_config.discover,
}


@dataclass(frozen=True)
class Source:
    """Where one slice of a command's parameters is declared.

    Attributes:
        kind: A key of :data:`READERS`.
        path: Repo-relative path to the declaring file. Relative so the launcher
            reads the checkout it ships in, whatever machine that is on, while
            the command it builds still names the absolute path on the target.
        env_schema_from: For ``yaml`` sources whose groups are not all launch
            arguments: the repo-relative reader module exporting a ``SCHEMA``
            of ``{group: {key: ENV_VAR}}``. Loaded rather than copied, so the
            launcher cannot drift from the script that consumes the config.
        only_groups: For ``yaml`` sources: restrict to these top-level groups,
            when a command uses only part of a shared config.
        defines_defaults: Whether this source is authoritative about the VALUES
            it lists, not merely about which parameters exist. Set False for a
            launch file read after the config that overrides it: the file
            declares the complete set of arguments, but the config decides what
            a plain run of them actually uses, and that is what "reset" must
            return to.
    """

    kind: str
    path: str
    env_schema_from: str = ""
    only_groups: tuple[str, ...] = ()
    defines_defaults: bool = True

    def __post_init__(self) -> None:
        if self.kind not in READERS:
            raise ValueError("unknown parameter source kind %r (expected one of %s)"
                             % (self.kind, ", ".join(sorted(READERS))))

    @property
    def absolute(self) -> Path:
        return REPO_ROOT / self.path


def load_schema(module_path: str | Path) -> dict[str, dict[str, str]]:
    """Import ``SCHEMA`` from a config-reader module given by path.

    The reader modules live beside the shell scripts that use them and are not
    importable packages, so they are loaded by file path.

    Args:
        module_path: Absolute path to the ``.py`` file exporting ``SCHEMA``.

    Returns:
        The module's ``SCHEMA`` mapping.

    Raises:
        ImportError: If the module cannot be loaded or exports no ``SCHEMA``.
    """
    spec = importlib.util.spec_from_file_location("_launch_params_schema", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load a module from %s" % module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "SCHEMA"):
        raise ImportError("%s exports no SCHEMA" % module_path)
    return module.SCHEMA


@dataclass
class Discovery:
    """The outcome of reading every source of one command.

    Attributes:
        params: Everything that was found, layered in source order.
        problems: One human-readable line per source that could not be read.
    """

    params: ParamSet = field(default_factory=ParamSet)
    problems: list[str] = field(default_factory=list)


def discover(sources: tuple[Source, ...]) -> Discovery:
    """Read every source of a command, in order.

    Args:
        sources: The command's declared sources. Later ones refine earlier ones
            (see :meth:`~.spec.ParamSet.add`), so declare them coarse-to-fine:
            the launch file that lists every argument first, the config that
            re-defaults a few of them last.

    Returns:
        A :class:`Discovery` holding the merged parameters and any read failures.
    """
    found = Discovery()
    for source in sources:
        try:
            if source.kind == "yaml":
                schema = (load_schema(REPO_ROOT / source.env_schema_from)
                          if source.env_schema_from else None)
                discovered = yaml_config.discover(source.absolute, schema,
                                                  source.only_groups)
            else:
                discovered = READERS[source.kind](source.absolute)
            found.params.extend(discovered,
                                override_default=source.defines_defaults)
        except Exception as error:  # noqa: BLE001 - reported, never swallowed
            found.problems.append("%s (%s): %s: %s" % (
                source.path, source.kind, type(error).__name__, error))
    return found
