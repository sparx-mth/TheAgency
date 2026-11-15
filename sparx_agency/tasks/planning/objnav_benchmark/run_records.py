"""What one results directory may hold: one experiment, each episode once, and a summary of exactly those episodes.

Every refusal here stops a quiet corruption of the numbers. An episode logged
twice counts twice in every mean; a row of another agent, benchmark or split
is a second experiment averaged into the first; and a ``summary.json`` beside
rows it does not describe -- another number of episodes, or rows a second
writer appended to the file -- is a number nobody can check.
:class:`RunRecords` holds the rows one logger has written or read, in file
order, and refuses each of these before it can happen; the files themselves
are :class:`MetricsLogger`'s.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import pathlib
from typing import FrozenSet, List, Set, Tuple

from sparx_agency.tasks.planning.objnav_benchmark.errors import ResultsError
from sparx_agency.tasks.planning.objnav_benchmark.records import EpisodeRecord
from sparx_agency.tasks.planning.objnav_benchmark.results_io import (
    read_episode_rows,
)
from sparx_agency.tasks.planning.objnav_benchmark.summaries import (
    BenchmarkSummary,
)


class RunRecords:
    """The episodes of one results directory, in file order, as one logger wrote or read them.

    Args:
        episodes_path: The directory's episodes file: named in every refusal,
            and read back by :meth:`check_file`.
    """

    def __init__(self, episodes_path: pathlib.Path) -> None:
        self._path = episodes_path
        self._records: List[EpisodeRecord] = []
        self._ids: Set[str] = set()

    @property
    def records(self) -> Tuple[EpisodeRecord, ...]:
        """Every episode held, in file order."""
        return tuple(self._records)

    @property
    def ids(self) -> FrozenSet[str]:
        """The episode ids held."""
        return frozenset(self._ids)

    def check(self, record: EpisodeRecord, where: str) -> None:
        """Refuse an episode already held, or one of another agent, benchmark or split.

        Raises:
            ResultsError: Naming ``where`` the record came from.
        """
        if record.episode_id in self._ids:
            raise ResultsError(
                "%s: episode %r is already in %s; an episode logged twice "
                "would count twice in every mean"
                % (where, record.episode_id, self._path))
        if self._records:
            first = self._records[0]
            run = (first.benchmark, first.split, first.agent)
            if (record.benchmark, record.split, record.agent) != run:
                raise ResultsError(
                    "%s: episode %r is %s/%s/%s but this run is %s/%s/%s; one "
                    "results directory holds one agent on one benchmark split"
                    % ((where, record.episode_id, record.benchmark,
                        record.split, record.agent) + run))

    def add(self, record: EpisodeRecord, where: str) -> None:
        """Hold ``record``, once :meth:`check` accepts it."""
        self.check(record, where)
        self._records.append(record)
        self._ids.add(record.episode_id)

    def check_summary(self, summary: BenchmarkSummary) -> None:
        """Refuse a summary of another agent, benchmark, split or number of episodes than those held.

        Raises:
            ResultsError: When ``summary`` describes other episodes.
        """
        described = (summary.benchmark, summary.split, summary.agent,
                     summary.overall.n_episodes)
        held = ((self._records[0].benchmark, self._records[0].split,
                 self._records[0].agent, len(self._records))
                if self._records else ("-", "-", "-", 0))
        if described != held:
            raise ResultsError(
                "finish: the summary describes %s/%s/%s over %d episodes but "
                "%s holds %s/%s/%s over %d; summary.json must describe exactly "
                "the episodes beside it"
                % (described + (self._path,) + held))

    def check_file(self) -> None:
        """Refuse when the episodes file, read back, holds other rows than these: another writer's, or an edit.

        Raises:
            ResultsError: Naming the first row that differs; or as
                ``read_episode_rows`` does, on a line it cannot read.
        """
        in_file = [record.episode_id for _, record in read_episode_rows(self._path)]
        ours = [record.episode_id for record in self._records]
        if in_file == ours:
            return
        row = next((i for i, (a, b) in enumerate(zip(in_file, ours)) if a != b),
                   min(len(in_file), len(ours)))
        raise ResultsError(
            "finish: %s holds %d rows but this logger wrote or read %d (row "
            "%d is %r in the file, %r here); another writer appended to it, "
            "or it was edited -- summary.json must describe exactly the "
            "episodes beside it"
            % (self._path, len(in_file), len(ours), row + 1,
               in_file[row] if row < len(in_file) else None,
               ours[row] if row < len(ours) else None))
