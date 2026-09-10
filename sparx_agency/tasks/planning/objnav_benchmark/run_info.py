"""What ``run.json`` records about how a run was produced, and how it is read back.

A number nobody can reproduce is a number nobody can check, so every results
directory carries its own provenance: the full configuration, the git
revision (``-dirty`` when the tree had changes), the command line, Python and
the platform -- written before the first episode, so even a killed run can be
explained -- and an entry for every resume, since a run resumed under
changed code mixes two revisions. Reading it back refuses a record of another
schema, or one written by hand, rather than resuming against a config it
cannot trust. :class:`MetricsLogger` decides when ``run.json`` is written.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import collections.abc
import json
import pathlib
import platform
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

from sparx_agency.tasks.planning.objnav_benchmark.errors import (
    HarnessError,
    ResultsError,
)
from sparx_agency.tasks.planning.objnav_benchmark.results_io import (
    git_revision,
    strict_json,
    strict_loads,
    utc_now,
)

#: The ``run.json`` format. Bump it when a field changes meaning.
RUN_SCHEMA = "objnav_run/1"

#: ``run.json`` timestamps: ISO 8601, UTC.
UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def argv_list(argv: Optional[Sequence[str]]) -> List[str]:
    """The command line to record: ``argv``, or the process's own when None.

    Raises:
        HarnessError: If ``argv`` is not a sequence of strings.
    """
    items = sys.argv if argv is None else argv
    if (isinstance(items, (str, bytes))
            or not isinstance(items, collections.abc.Sequence)
            or not all(isinstance(item, str) for item in items)):
        raise HarnessError("argv must be a sequence of strings, got %r"
                           % (argv,))
    return list(items)


def new_run_info(config_text: str, argv: List[str]) -> Dict[str, Any]:
    """A fresh run's ``run.json``: its provenance, not finished, no episodes, no resumes.

    Args:
        config_text: The run config as strict JSON text.
        argv: The command line.
    """
    return {
        "schema": RUN_SCHEMA,
        "started_utc": utc_now(UTC_FORMAT),
        "argv": argv,
        "git": git_revision(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config": json.loads(config_text),
        "finished_utc": None,
        "n_episodes": 0,
        "resumes": [],
    }


def resume_entry(argv: List[str], episodes_done: int) -> Dict[str, Any]:
    """One resume's entry in ``run.json``: when, by which command, at which revision, after how many episodes."""
    return {"utc": utc_now(UTC_FORMAT), "argv": argv, "git": git_revision(),
            "episodes_done": episodes_done}


def read_run_info(path: pathlib.Path) -> Dict[str, Any]:
    """``run.json`` as written, refused unless it is a run record of :data:`RUN_SCHEMA`.

    Raises:
        ResultsError: If the file is not strict JSON, or not a run record of
            this schema with a config and a list of resumes.
    """
    try:
        info = strict_loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ResultsError("%s is not strict JSON (%s)" % (path, exc)) from exc
    if (not isinstance(info, dict) or info.get("schema") != RUN_SCHEMA
            or "config" not in info
            or not isinstance(info.get("resumes"), list)):
        raise ResultsError(
            "%s is not a run record of schema %r; it was written by another "
            "version of the harness, or by hand" % (path, RUN_SCHEMA))
    return info


def differing_keys(recorded: Any, config: Mapping[str, Any]) -> List[str]:
    """The top-level config keys whose values differ, for a refused resume's message."""
    if not isinstance(recorded, dict):
        return ["the whole config"]
    return [key for key in sorted(set(recorded).union(config))
            if key not in recorded or key not in config
            or strict_json(recorded[key], key) != strict_json(config[key], key)]
