"""Keep one benchmark run on disk so it can be resumed, explained, and never mixed with another.

A results directory holds one experiment -- one agent on one benchmark split
under one configuration -- as three plain-text files and a lock:

``run.json``        how the run was produced (``run_info.py``). Written before
                    the first episode, so even a killed run can be explained;
                    updated at the finish, and by a resume once it logs or
                    finishes.
``episodes.jsonl``  one :class:`EpisodeRecord` per line, appended and flushed as
                    each episode ends: a crash loses one episode, not the sweep.
``summary.json``    the :class:`BenchmarkSummary`, written at the finish.
``run.lock``        held by the one logger writing the directory
                    (``run_lock.py``).

Every refusal here stops a quiet corruption of the numbers. One logger at a
time writes a directory: a resume of a run that is still going would append
its episodes a second time. A fresh run creates ``episodes.jsonl``
exclusively, so it never overwrites results. A resume must present exactly
the recorded config, or it would append a second experiment to the first, and
it touches ``run.json`` only once it logs or finishes, so a resume refused
before its first episode leaves a finished run looking finished. What the
rows may hold -- each episode once, one experiment, a summary of exactly them
-- is ``run_records.py``'s; how a file is written and read back (strict JSON,
atomic replacement, a bad line reported with its number), ``results_io.py``'s.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import collections.abc
import json
import pathlib
from typing import Any, Dict, FrozenSet, Mapping, Optional, Sequence, Tuple

from sparx_agency.tasks.planning.objnav_benchmark.errors import (
    HarnessError,
    ResultsError,
)
from sparx_agency.tasks.planning.objnav_benchmark.records import EpisodeRecord
from sparx_agency.tasks.planning.objnav_benchmark.results_io import (
    read_episode_rows,
    strict_json,
    strict_loads,
    utc_now,
    write_atomically,
)
from sparx_agency.tasks.planning.objnav_benchmark.run_info import (
    UTC_FORMAT,
    argv_list,
    differing_keys,
    new_run_info,
    read_run_info,
    resume_entry,
)
from sparx_agency.tasks.planning.objnav_benchmark.run_lock import RunLock
from sparx_agency.tasks.planning.objnav_benchmark.run_records import RunRecords
from sparx_agency.tasks.planning.objnav_benchmark.summaries import (
    BenchmarkSummary,
)


class MetricsLogger:
    """One run's results directory: ``run.json``, ``episodes.jsonl`` and ``summary.json``.

    Open it before the first episode, :meth:`log` each episode as it ends,
    and :meth:`finish` with the run's summary. It holds the directory's lock
    from construction until :meth:`finish` or :meth:`close` -- or until its
    process ends -- and refuses to write after that; used as a context manager
    it closes on the way out, so a run that raises releases the directory to
    its resume. With ``resume=True`` it reopens a directory a killed run left
    behind: :meth:`completed_episode_ids` says which episodes to skip, and
    :attr:`records` holds them for the summary.

    Attributes:
        EPISODES_FILE: One :class:`EpisodeRecord` row per line.
        RUN_FILE: How the run was produced.
        SUMMARY_FILE: The finished run's summary.
        LOCK_FILE: Locked by the logger writing the directory.
    """

    EPISODES_FILE = "episodes.jsonl"
    RUN_FILE = "run.json"
    SUMMARY_FILE = "summary.json"
    LOCK_FILE = "run.lock"

    def __init__(self, run_dir: pathlib.Path, config: Mapping[str, Any], *,
                 resume: bool = False,
                 argv: Optional[Sequence[str]] = None) -> None:
        """Open a results directory, fresh or resumed, and take its lock.

        Args:
            run_dir: The run's directory (see ``results_io.default_run_dir``),
                created when fresh.
            config: The full run configuration, strict JSON. A resume must
                pass exactly the recorded one.
            resume: Reopen a directory an earlier run left behind.
            argv: The command line to record; defaults to ``sys.argv``.

        Raises:
            TypeError: If ``config`` is not a mapping.
            HarnessError: If ``resume`` is not a bool or ``argv`` is not a
                sequence of strings.
            ResultsError: If ``config`` is not strict JSON; if a fresh run
                finds ``episodes.jsonl`` already there; if another logger is
                writing the directory; if a resume finds no run, a different
                config, or a malformed, truncated or repeated line.
        """
        if not isinstance(resume, bool):
            raise HarnessError(
                "resume must be a bool, got %r (the string 'false' would read "
                "as True and append to a run meant to start fresh)" % (resume,))
        if not isinstance(config, collections.abc.Mapping):
            raise TypeError("config must be a mapping, got %s"
                            % type(config).__name__)
        self._config_text = strict_json(dict(config), "the run config")
        self._argv = argv_list(argv)
        self._run_dir = pathlib.Path(run_dir).expanduser()
        self._rows = RunRecords(self.episodes_path)
        self._info: Dict[str, Any] = {}
        self._pending_resume: Optional[Dict[str, Any]] = None
        self._lock: Optional[RunLock] = None
        try:
            if resume:
                self._resume()
            else:
                self._start()
        except BaseException:
            self.close()
            raise

    def __repr__(self) -> str:
        return "MetricsLogger(%r, %d episodes)" % (str(self._run_dir),
                                                   len(self._rows.records))

    def __enter__(self) -> MetricsLogger:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    @property
    def run_dir(self) -> pathlib.Path:
        """The run's directory."""
        return self._run_dir

    @property
    def episodes_path(self) -> pathlib.Path:
        """The episodes file, ``run_dir / EPISODES_FILE``."""
        return self._run_dir / self.EPISODES_FILE

    @property
    def records(self) -> Tuple[EpisodeRecord, ...]:
        """Every episode in the file, in file order, exactly as a resume reads it."""
        return self._rows.records

    def completed_episode_ids(self) -> FrozenSet[str]:
        """The episodes already in the file: a resumed run skips these."""
        return self._rows.ids

    def log(self, record: EpisodeRecord) -> None:
        """Append one episode's row and flush it.

        The row is read back through :meth:`EpisodeRecord.from_row` before it
        is written, so :attr:`records` always equals what a resume will read.

        Raises:
            TypeError: If ``record`` is not an :class:`EpisodeRecord`.
            ResultsError: If the logger is closed; if the episode is already
                logged, belongs to another agent, benchmark or split than this
                run, or holds something that is not strict JSON. Nothing is
                written then.
        """
        self._check_open("log")
        if not isinstance(record, EpisodeRecord):
            raise TypeError("log needs an EpisodeRecord, got %s"
                            % type(record).__name__)
        self._rows.check(record, "log")
        line = strict_json(record.to_row(), "episode %r" % record.episode_id)
        stored = EpisodeRecord.from_row(strict_loads(line))
        self._enter_resume()
        # Open, append, close per episode: the row is on disk before the next
        # episode starts.
        with self.episodes_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
        self._rows.add(stored, "log")

    def finish(self, summary: BenchmarkSummary) -> pathlib.Path:
        """Write ``summary.json``, record the outcome in ``run.json``, and release the directory.

        Args:
            summary: The summary of exactly the episodes in the file.

        Returns:
            The path of ``summary.json``.

        Raises:
            TypeError: If ``summary`` is not a :class:`BenchmarkSummary`.
            ResultsError: If the logger is closed; if the summary describes
                other episodes than this logger holds; or if the file, read
                back, holds other rows than this logger wrote and read. The
                logger stays open then.
        """
        self._check_open("finish")
        if not isinstance(summary, BenchmarkSummary):
            raise TypeError("finish needs a BenchmarkSummary, got %s"
                            % type(summary).__name__)
        self._rows.check_summary(summary)
        self._rows.check_file()
        self._enter_resume()
        path = self._run_dir / self.SUMMARY_FILE
        write_atomically(path, strict_json(summary.to_dict(), "the summary",
                                           indent=1))
        self._info["finished_utc"] = utc_now(UTC_FORMAT)
        self._info["n_episodes"] = len(self._rows.records)
        self._write_run_info()
        self.close()
        return path

    def close(self) -> None:
        """Release the directory's lock, so another logger may resume it. Idempotent; :meth:`finish` calls it.

        A closed logger refuses :meth:`log` and :meth:`finish`: without the
        lock, its writes could interleave with another writer's.
        """
        if self._lock is not None:
            lock, self._lock = self._lock, None
            lock.release()

    # -- internals -------------------------------------------------------------

    def _check_open(self, where: str) -> None:
        if self._lock is None:
            raise ResultsError(
                "%s: the logger of %s is closed, so it no longer holds the "
                "directory's lock; open a resume to add to the run"
                % (where, self._run_dir))

    def _enter_resume(self) -> None:
        """Record a pending resume in ``run.json``, on its first log or finish.

        Deferred, so a resume the runner refuses before its first episode (a
        different episode list, another agent, a typo in an option) leaves
        ``run.json`` as it was -- a finished run still says it finished.
        """
        if self._pending_resume is None:
            return
        self._info["finished_utc"] = None
        self._info["resumes"].append(self._pending_resume)
        self._pending_resume = None
        self._write_run_info()

    def _write_run_info(self) -> None:
        write_atomically(self._run_dir / self.RUN_FILE,
                         strict_json(self._info, "run.json", indent=1))

    def _start(self) -> None:
        self._run_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Exclusive creation is atomic: of two runs pointed at one
            # directory, the second is refused here, before it writes a byte.
            with self.episodes_path.open("x", encoding="utf-8"):
                pass
        except FileExistsError as exc:
            raise ResultsError(
                "%s already holds a run's results, which are never "
                "overwritten; choose another directory, or pass resume=True "
                "to continue that run" % self.episodes_path) from exc
        self._lock = RunLock(self._run_dir / self.LOCK_FILE)
        self._info = new_run_info(self._config_text, self._argv)
        self._write_run_info()

    def _resume(self) -> None:
        run_path = self._run_dir / self.RUN_FILE
        if not run_path.is_file() or not self.episodes_path.is_file():
            raise ResultsError(
                "cannot resume %s: it lacks %s or %s, so there is no run to "
                "continue and no recorded config to check against; start a "
                "fresh run in a new directory"
                % (self._run_dir, self.RUN_FILE, self.EPISODES_FILE))
        # Locked before anything is read: a run still being written is
        # refused, not read half-way.
        self._lock = RunLock(self._run_dir / self.LOCK_FILE)
        self._info = read_run_info(run_path)
        recorded = self._info["config"]
        if strict_json(recorded, "the recorded config") != self._config_text:
            raise ResultsError(
                "cannot resume %s with a different config (%s differ): "
                "resuming with a different config would mix two experiments "
                "in one file; start a fresh run in a new directory"
                % (self._run_dir, ", ".join(differing_keys(
                    recorded, json.loads(self._config_text)))))
        for number, record in read_episode_rows(self.episodes_path):
            self._rows.add(record, "%s line %d" % (self.episodes_path, number))
        self._pending_resume = resume_entry(self._argv, len(self._rows.records))
