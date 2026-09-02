"""Reading, indexing and time-aligning the jsonl lanes of a recorded run.

A Sphera run is written by two recorders that cannot see each other's ROS graph
(see :mod:`.schema`): the ROS1 recorder in the FALCON container writes the plan,
the reference and the command we ask for; the ROS2 recorder in the vendor
container writes what the drone was actually told and what it actually did.
Neither can subscribe to the other, so the two recordings are joined here,
offline, on the host ``wall`` clock both of them stamp every row with.

:class:`Stream` is one lane: rows sorted and indexed so that "the newest row at
or before this instant" is a bisect on *either* clock. :class:`ClockOffset`
estimates ``wall - t`` for a recorder from every ``(t, wall)`` pair it wrote --
a median over the whole run rather than a single sample, with the spread kept
so that a bad join is reportable instead of silently shifting every panel.

Pure stdlib and Python 3.8 compatible, like the rest of the nav-debug package.
"""
from __future__ import annotations

import bisect
import json
import os
import statistics
from typing import Iterable, List, Optional, Sequence

# Two processes on one kernel clock should agree to well under a frame period;
# more than this and the cross-recorder join is worth flagging to the operator.
SPREAD_WARN_S = 0.05


def to_float(value) -> Optional[float]:
    """Parse a JSON/CSV scalar to float, or None for ``''`` / None / bad values."""
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_bool(value) -> bool:
    """Truthiness of a JSON/CSV scalar, accepting ``'true'``/``'1'`` strings."""
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def read_jsonl(path: str) -> List[dict]:
    """Every parseable object in a jsonl file; ``[]`` when it is absent.

    A recording is a diagnostic, not a database: an unreadable file or a
    half-written trailing line (the recorder was killed mid-flush) must cost
    that one row, never the replay.
    """
    rows = []  # type: List[dict]
    if not os.path.isfile(path):
        return rows
    try:
        handle = open(path, "r")
    except (OSError, IOError):
        return rows
    with handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def as_of_index(stamps: Sequence[float], t: float) -> Optional[int]:
    """Index of the latest stamp ``<= t``, or None if all of them are later."""
    if not stamps:
        return None
    j = bisect.bisect_right(stamps, t) - 1
    return j if j >= 0 else None


def _as_of(stamps, rows, t, max_age=None):
    """The row of the latest stamp ``<= t``, dropped if older than ``max_age``."""
    j = as_of_index(stamps, t)
    if j is None:
        return None
    if max_age is not None and t - stamps[j] > max_age:
        return None
    return rows[j]


class ClockOffset:
    """One recorder's ``wall - t`` offset, estimated from all of its rows.

    ``offset`` is None when the recorder wrote no wall clock at all (a run from
    before ``schema.row``); :meth:`to_wall` then degrades to the identity, which
    is right for a ROS clock that already runs on wall time and wrong -- but
    *reported*, via :attr:`known` -- for one that does not.
    """

    def __init__(self, offset: Optional[float], spread: float, samples: int) -> None:
        self.offset = offset
        self.spread = spread        # median absolute deviation, seconds
        self.samples = samples

    @classmethod
    def estimate(cls, rows: Iterable[dict]) -> "ClockOffset":
        """Median ``wall - t`` over every row that carries both clocks."""
        deltas = [to_float(r.get("wall")) - to_float(r.get("t"))
                  for r in rows
                  if to_float(r.get("t")) is not None
                  and to_float(r.get("wall")) is not None]
        if not deltas:
            return cls(None, 0.0, 0)
        offset = float(statistics.median(deltas))
        spread = float(statistics.median([abs(d - offset) for d in deltas]))
        return cls(offset, spread, len(deltas))

    @property
    def known(self) -> bool:
        """True when at least one row carried both clocks."""
        return self.offset is not None

    @property
    def suspect(self) -> bool:
        """True when the offset wandered enough during the run to skew a join."""
        return self.known and self.spread > SPREAD_WARN_S

    def to_wall(self, t: float) -> float:
        """ROS ``t`` -> host wall clock (identity when no offset was recorded)."""
        return t if self.offset is None else t + self.offset

    def to_ros(self, wall: float) -> float:
        """Host wall clock -> ROS ``t`` (identity when no offset was recorded)."""
        return wall if self.offset is None else wall - self.offset

    def describe(self) -> str:
        """One line for the operator: the offset, its spread and the sample count."""
        if not self.known:
            return "wall offset unknown (no wall clock in this recording)"
        return ("wall offset %+.3fs (spread %.3fs, n=%d)"
                % (self.offset, self.spread, self.samples))

    def __repr__(self) -> str:      # pragma: no cover - debugging aid
        return "ClockOffset(%s)" % self.describe()


class Stream:
    """One jsonl lane, indexed for as-of lookup on both clocks.

    Rows missing one of the two clocks are still usable: the missing stamp is
    filled from the lane's own :class:`ClockOffset`, so a lane joined by wall
    stays joinable even if a few rows lost their ROS stamp.
    """

    def __init__(self, rows: Iterable[dict], name: str = "",
                 clock: Optional[ClockOffset] = None) -> None:
        self.name = name
        usable = [r for r in rows
                  if to_float(r.get("t")) is not None
                  or to_float(r.get("wall")) is not None]
        self.clock = clock if clock is not None else ClockOffset.estimate(usable)
        pairs = []
        for row in usable:
            t, wall = to_float(row.get("t")), to_float(row.get("wall"))
            t = self.clock.to_ros(wall) if t is None else t
            wall = self.clock.to_wall(t) if wall is None else wall
            pairs.append((t, wall, row))
        pairs.sort(key=lambda p: p[0])
        self.rows = [p[2] for p in pairs]
        self._t = [p[0] for p in pairs]
        order = sorted(range(len(pairs)), key=lambda i: pairs[i][1])
        self._wall = [pairs[i][1] for i in order]
        self._wall_rows = [pairs[i][2] for i in order]

    @classmethod
    def from_file(cls, path: str, name: str = "") -> "Stream":
        """Load a lane from ``path`` (a missing file yields an empty stream)."""
        return cls(read_jsonl(path), name or os.path.basename(path))

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def has_wall(self) -> bool:
        """True when this lane carries the cross-recorder join key."""
        return self.clock.known

    def at(self, t: float, max_age: Optional[float] = None) -> Optional[dict]:
        """Newest row at or before ROS time ``t`` (None if absent or too old)."""
        return _as_of(self._t, self.rows, t, max_age)

    def at_wall(self, wall: float, max_age: Optional[float] = None) -> Optional[dict]:
        """Newest row at or before host time ``wall`` (None if absent/too old)."""
        return _as_of(self._wall, self._wall_rows, wall, max_age)
