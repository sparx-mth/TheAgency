"""Where a run's results go and how they are written and read back: the layout, the revision, strict JSON.

Every run lives under :data:`DEFAULT_RESULTS_ROOT` -- outside the repository,
so writing results never dirties the revision they record -- in
``root/benchmark/split/stamp``. Each file is strict JSON: ``NaN`` and
``Infinity`` are not JSON, and plain ``json.dumps`` writes them anyway, so
a file it writes can fail to read back on the one resume that needs it. A file
replaced as a whole is replaced in one step, and an episodes file read back
reports a malformed or truncated line with its number instead of skipping it:
a skipped line is an episode that silently leaves the mean.

The rules of a run directory -- fresh or resumed, one experiment per
directory, no episode twice -- are :class:`MetricsLogger`'s, in ``logger.py``.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import datetime
import json
import os
import pathlib
import subprocess
from typing import Any, Iterator, Optional, Sequence, Tuple

from sparx_agency.tasks.planning.objnav_benchmark.checks import is_name
from sparx_agency.tasks.planning.objnav_benchmark.errors import (
    HarnessError,
    ResultsError,
)
from sparx_agency.tasks.planning.objnav_benchmark.records import EpisodeRecord

#: Where runs go when no directory is given: outside the repository, so
#: writing results never dirties the revision they record.
DEFAULT_RESULTS_ROOT = pathlib.Path.home() / "objnav_benchmark"

#: What :func:`git_revision` says when git cannot say.
UNKNOWN_REVISION = "unknown"

#: Seconds allowed per git command: a hung git must not hang a benchmark.
GIT_TIMEOUT_S = 10.0

#: Run-directory stamps: UTC to the second, so they sort as text.
STAMP_FORMAT = "%Y%m%dT%H%M%SZ"


def utc_now(fmt: str) -> str:
    """The current UTC time, formatted with ``fmt`` (a ``strftime`` format)."""
    return datetime.datetime.now(datetime.timezone.utc).strftime(fmt)


def _directory_level(name: str, value) -> str:
    """``value``, once it is known to name exactly one directory level."""
    if (not is_name(value) or value in (".", "..") or "/" in value
            or "\\" in value):
        raise HarnessError(
            "%s must name one directory level (no separators, not '.' or "
            "'..'), got %r" % (name, value))
    return value


def default_run_dir(benchmark: str, split: str,
                    root: Optional[pathlib.Path] = None,
                    stamp: Optional[str] = None) -> pathlib.Path:
    """Where a run's results go: ``root/benchmark/split/stamp``.

    Nothing is created here; :class:`MetricsLogger` creates the directory.

    Args:
        benchmark: Benchmark key (``"hm3d_v2"``).
        split: Dataset split.
        root: Results root; defaults to :data:`DEFAULT_RESULTS_ROOT`.
        stamp: The run's name; defaults to the UTC time now in
            :data:`STAMP_FORMAT`, e.g. ``20260910T141502Z``.

    Returns:
        The run directory.

    Raises:
        HarnessError: If ``benchmark``, ``split`` or ``stamp`` is blank,
            holds a path separator, or is ``.`` or ``..`` -- each would put
            the run somewhere other than this layout says.
    """
    base = DEFAULT_RESULTS_ROOT if root is None else pathlib.Path(root).expanduser()
    name = utc_now(STAMP_FORMAT) if stamp is None else stamp
    return (base / _directory_level("benchmark", benchmark)
            / _directory_level("split", split) / _directory_level("stamp", name))


def _git(args: Sequence[str], where: pathlib.Path) -> Optional[str]:
    """The standard output of one git command run in ``where``, or None if it failed."""
    # GIT_OPTIONAL_LOCKS=0: `git status` then never takes the index lock, so a
    # benchmark starting cannot collide with the user's own git command.
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    try:
        done = subprocess.run(["git"] + list(args), cwd=str(where), env=env,
                              capture_output=True, text=True,
                              timeout=GIT_TIMEOUT_S, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def git_revision(repo_root: Optional[pathlib.Path] = None) -> str:
    """The short commit of the code being run, with ``-dirty`` when the tree has changes.

    A deliberate copy of ``git_commit`` in
    ``tasks/planning/vlas/navdp/finetune/world_goal/logger.py`` rather than an
    import of it. That one asks git about the process's working directory, so
    a run launched from outside the checkout records ``unknown`` -- or, from
    inside another repository, that repository's commit; it reports a tree
    whose status git could not read as clean; and importing it would tie the
    benchmark harness to a fine-tuning package it has nothing to do with.
    This one describes the checkout the code lives in, and never claims a
    clean tree that git has not confirmed: a clean revision promises that the
    commit alone reproduces the run.

    Args:
        repo_root: A directory inside the checkout to describe. Defaults to
            this module's own directory -- the code actually running,
            whatever the working directory is.

    Returns:
        ``"<short sha>"`` for a clean tree; ``"<short sha>-dirty"`` when git
        reports changes (untracked files included) or cannot report the
        status; :data:`UNKNOWN_REVISION` when git is not installed, times
        out, or the directory is not in a repository.

    Raises:
        HarnessError: If ``repo_root`` is given and is not a directory.
    """
    if repo_root is None:
        where = pathlib.Path(__file__).resolve().parent
    else:
        where = pathlib.Path(repo_root).expanduser()
        if not where.is_dir():
            raise HarnessError(
                "git_revision: repo_root %s is not a directory; pass a "
                "directory inside the checkout, or None for this one" % where)
    head = _git(["rev-parse", "--short", "HEAD"], where)
    if head is None or not head.strip():
        return UNKNOWN_REVISION
    status = _git(["status", "--porcelain"], where)
    clean = status is not None and status.strip() == ""
    return head.strip() + ("" if clean else "-dirty")


def strict_json(value: Any, what: str, indent: Optional[int] = None) -> str:
    """``value`` as strict JSON text with sorted keys, or a ResultsError naming ``what``.

    Strict means it reads back as written: no ``NaN`` or infinity (plain
    ``json.dumps`` writes them as tokens JSON does not have), and only JSON's
    own types -- a numpy scalar, a set or an arbitrary object is refused, not
    stringified. Tuples become lists and number keys strings, as JSON itself
    demands. The logger and the runner both use it, so there is one
    definition of what a results file may hold.

    Args:
        value: The data.
        what: What it is, for the message (``"the run config"``).
        indent: Passed to :func:`json.dumps`; None writes one line.

    Returns:
        The JSON text, without a trailing newline.

    Raises:
        ResultsError: If ``value`` is not strict JSON.
    """
    try:
        return json.dumps(value, allow_nan=False, sort_keys=True, indent=indent)
    except (TypeError, ValueError) as exc:
        raise ResultsError(
            "%s is not strict JSON (%s); results hold only JSON's own types "
            "with finite numbers -- convert numpy scalars with float() or "
            "int(), and write a missing or infinite value as None"
            % (what, exc)) from exc


def _refuse_constant(token: str):
    raise ValueError("%s is not JSON: it was written without allow_nan=False"
                     % token)


def strict_loads(text: str) -> Any:
    """Parse JSON, refusing the ``NaN`` and ``Infinity`` tokens ``json.loads`` accepts.

    Raises:
        ValueError: If ``text`` is not strict JSON (a ``json.JSONDecodeError``
            for malformed text).
    """
    return json.loads(text, parse_constant=_refuse_constant)


def write_atomically(path: pathlib.Path, text: str) -> None:
    """Replace ``path`` with ``text`` and a newline in one step: a crash leaves the old file or the new, never half of one."""
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text + "\n", encoding="utf-8")
    os.replace(str(partial), str(path))


def _line_error(path: pathlib.Path, number: int, total: int,
                problem: str) -> ResultsError:
    if number == total:
        hint = ("a run killed while writing leaves its last line like "
                "this: delete that line and resume, and its episode runs "
                "again")
    else:
        hint = ("a line before the last is not a crash: the file was "
                "edited or two files were joined; repair it by hand or "
                "start a new run")
    return ResultsError("%s line %d %s; %s" % (path, number, problem, hint))


def read_episode_rows(path: pathlib.Path) -> Iterator[Tuple[int, EpisodeRecord]]:
    """Every row of an episodes file as a record, with its 1-based line number, in file order.

    Lazy: a line is parsed only once the previous record has been taken, so a
    caller that refuses a record (an episode already seen) stops there.

    Args:
        path: An ``episodes.jsonl``, one :class:`EpisodeRecord` row per line.

    Yields:
        ``(line number, record)``.

    Raises:
        ResultsError: On a last line without its end of line (a run killed
            while writing), or a line that is not a valid episode row; the
            message names the line and says how to recover.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    if text and not text.endswith("\n"):
        raise _line_error(path, len(lines), len(lines),
                          "is truncated (it has no end of line)")
    for number, line in enumerate(lines[:-1], start=1):
        try:
            record = EpisodeRecord.from_row(strict_loads(line))
        except ValueError as exc:  # a JSON error or a ResultsError
            raise _line_error(path, number, len(lines) - 1,
                              "is not a valid episode row (%s)" % exc) from exc
        yield number, record
