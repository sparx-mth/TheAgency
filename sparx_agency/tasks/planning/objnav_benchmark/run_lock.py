"""Let one writer at a time into a results directory, and let it go the moment that writer's process dies.

Two writers on one directory append the same episodes twice. The usual way
it happens: a detached sweep (tmux, nohup) is still running, and a resume is
started because the sweep looked dead. Each writer's own checks pass -- each
compares its summary with its own records -- and ``summary.json`` ends up
beside a file holding more rows than it describes. An exclusive lock, taken
before the first byte is written and held for the writer's lifetime, turns
the second writer away instead.

``fcntl.flock`` rather than a marker file: the kernel drops the lock when its
file is closed, which includes the process dying however it dies, so a
crashed run never leaves a stale lock that blocks its own resume. The lock
belongs to the open file, not the process, so two loggers in one process
exclude each other as two processes do. It is advisory: it binds every
:class:`MetricsLogger`, not an editor or a shell redirect.

POSIX only (``fcntl``), like every environment the benchmarks run in.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import fcntl
import pathlib
from typing import IO, Optional

from sparx_agency.tasks.planning.objnav_benchmark.errors import ResultsError


class RunLock:
    """An exclusive advisory lock on one file, held from construction until :meth:`release`.

    Args:
        path: The lock file; created if missing, never truncated. Its
            directory must exist.

    Raises:
        ResultsError: If another writer holds the lock.
        OSError: If the lock file cannot be opened or locked for another
            reason.
    """

    def __init__(self, path: pathlib.Path) -> None:
        self._path = pathlib.Path(path)
        handle = self._path.open("a", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise ResultsError(
                "%s: another process is writing this run (it holds %s); two "
                "writers would append the same episodes twice -- let it "
                "finish, or stop it, before resuming"
                % (self._path.parent, self._path)) from exc
        except BaseException:
            handle.close()
            raise
        self._handle: Optional[IO[str]] = handle

    def __repr__(self) -> str:
        return "RunLock(%r, %s)" % (str(self._path),
                                    "held" if self.held else "released")

    @property
    def held(self) -> bool:
        """Whether this object still holds the lock."""
        return self._handle is not None

    def release(self) -> None:
        """Drop the lock. Idempotent.

        Unlocked explicitly before the file is closed: a child process forked
        while it was held shares the open file, and closing only this copy
        would leave the lock held for as long as the child lives.
        """
        if self._handle is None:
            return
        handle, self._handle = self._handle, None
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
