"""Turn one recorded jsonl row into the frame dataclass that lane describes.

The contract in :mod:`.schema` is that a lane's rows use the field names of its
dataclass in :mod:`.frame` verbatim, so these builders are reflective: they walk
``dataclasses.fields`` and coerce each value to the type of that field's
default. A field added to :mod:`.frame` and written by a recorder is therefore
read here with no change, and a field a recorder does not write keeps its
default instead of raising -- which is what a partial recording must do.

Two conventions the recorders may use are absorbed here rather than pushed onto
the writers: a lane may nest its dataclasses under their frame attribute name
(``{"tracking": {...}, "terms": {...}}``) or write them flat in one row, and the
axis lane may write one row per tick or one row per axis.
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional

from sparx_agency.tasks.planning.nav_debug.frame import (
    Actuator, Altitude, AxisTrace, ControlTerms, MapStats, Reference, Tracking,
    Truth,
)
from sparx_agency.tasks.planning.nav_debug.sources import to_bool, to_float

_MISSING = dataclasses.MISSING
_SKIP = object()            # "this row has nothing usable for that field"
_AXIS_GROUP_S = 0.03        # per-axis rows this close together are one tick


def section(row: Optional[dict], key: str) -> Optional[dict]:
    """The nested ``key`` sub-object of ``row``, or ``row`` itself if it is flat.

    An **explicit null** is not the same as an absent key. The control trace
    writes ``"tracking": null`` on every tick where the tracker did not run --
    a muted demo mode, no reference yet, a tilt cut -- and falling back to the
    whole row there would build a Tracking of all-zero defaults, i.e. a panel
    reading "perfect tracking, no error" at exactly the moments the aircraft was
    not being flown. So a present-but-null section means "no data".
    """
    if not isinstance(row, dict):
        return None
    if key in row:
        nested = row[key]
        return nested if isinstance(nested, dict) else None
    return row      # a flat lane writes its fields at the top level


def build(cls, row: Optional[dict]):
    """Instantiate a :mod:`.frame` dataclass from a recorded row.

    Args:
        cls: The dataclass to build.
        row: One decoded jsonl row (or a nested sub-object of one).

    Returns:
        An instance of ``cls``, or None when ``row`` is not a dict or lacks a
        field the dataclass requires -- a lane that never recorded is simply
        absent from the frame.
    """
    if not isinstance(row, dict):
        return None
    kwargs = {}
    for f in dataclasses.fields(cls):
        value = _coerce(f, row[f.name]) if f.name in row else _SKIP
        if value is _SKIP:
            if _required(f):
                return None
            continue
        kwargs[f.name] = value
    try:
        return cls(**kwargs)
    except (TypeError, ValueError):
        return None


def _required(f) -> bool:
    """True when a field has neither a default nor a default factory."""
    return f.default is _MISSING and f.default_factory is _MISSING


def _coerce(f, value):
    """Coerce a recorded value to the type of its field's default."""
    if value is None:
        return _SKIP
    default = f.default
    if isinstance(default, bool):
        return to_bool(value)
    if isinstance(default, (int, float)):
        number = to_float(value)
        if number is None:
            return _SKIP
        return int(number) if isinstance(default, int) else number
    if isinstance(default, str):
        return str(value)
    if isinstance(default, tuple) or isinstance(value, (list, tuple)):
        return _tuple(value)
    if _required(f):
        number = to_float(value)      # every required field in frame.py is numeric
        return _SKIP if number is None else number
    return _scalar(value)


def _tuple(value):
    """A JSON array -> tuple, keeping strings (``limits``) as strings."""
    if not isinstance(value, (list, tuple)):
        return _SKIP
    return tuple(float(v) if isinstance(v, (int, float))
                 and not isinstance(v, bool) else v for v in value)


def _scalar(value):
    """An ``Optional[...]`` field's value, with ``'true'``/``'false'`` honoured."""
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "false"):
            return low == "true"
        number = to_float(value)
        return value if number is None else number
    return _SKIP


# ── one builder per lane ─────────────────────────────────────────────────────
def reference(row: Optional[dict]) -> Optional[Reference]:
    """The setpoint being chased, from a ``reference.jsonl`` row."""
    return build(Reference, section(row, "reference"))


def tracking(row: Optional[dict]) -> Optional[Tracking]:
    """The tracker's verdict, from a ``control.jsonl`` row."""
    return build(Tracking, section(row, "tracking"))


def control_terms(row: Optional[dict]) -> Optional[ControlTerms]:
    """The command broken into its terms, from a ``control.jsonl`` row."""
    return build(ControlTerms, section(row, "terms"))


def actuator(row: Optional[dict]) -> Optional[Actuator]:
    """What the drone was told, from a ``ros2/actuator.jsonl`` row."""
    return build(Actuator, section(row, "actuator"))


def altitude(row: Optional[dict]) -> Optional[Altitude]:
    """The vertical lane, from a ``ros2/altitude.jsonl`` row."""
    section_row = section(row, "altitude")
    if section_row is None:
        return None
    # The writer puts the outcome alongside the section, not inside it.
    if isinstance(row, dict) and "reason" in row and "reason" not in section_row:
        section_row = dict(section_row, reason=row.get("reason"))
    return build(Altitude, section_row)


def truth(row: Optional[dict]) -> Optional[Truth]:
    """Sphera ground truth, from a ``ros2/truth.jsonl`` row."""
    return build(Truth, section(row, "truth"))


def map_stats(row: Optional[dict]) -> Optional[MapStats]:
    """Map-quality counters, from a ``mapping.jsonl`` row."""
    return build(MapStats, section(row, "map_stats"))


def axes(row: Optional[dict]) -> List[AxisTrace]:
    """Every axis in one normalised ``ros2/axis_trace.jsonl`` row."""
    entries = row.get("axes") if isinstance(row, dict) else None
    if not isinstance(entries, list):
        return []
    built = [build(AxisTrace, e) for e in entries]
    return [a for a in built if a is not None]


def group_axis_rows(rows: List[dict]) -> List[dict]:
    """Normalise axis-trace rows to one row per tick: ``{t, wall, axes: [...]}``.

    The twist adapter may write one row carrying every axis, or one row per
    axis. Consecutive per-axis rows within :data:`_AXIS_GROUP_S`, with no axis
    name repeated, are folded into a single tick so that a frame resolves to the
    whole aircraft rather than to whichever axis was written last.
    """
    out = []            # type: List[dict]
    group = None
    for row in rows:
        if isinstance(row.get("axes"), list):
            out.append(row)
            group = None
            continue
        t = to_float(row.get("t"))
        name = str(row.get("name", ""))
        if group is None or t is None or _breaks(group, t, name):
            group = {"t": t, "wall": to_float(row.get("wall")),
                     "axes": [], "_names": set()}
            out.append(group)
        group["axes"].append(row)
        group["_names"].add(name)
    for row in out:
        row.pop("_names", None)
    return out


def _breaks(group: dict, t: float, name: str) -> bool:
    """True when this per-axis row starts a new tick rather than joining one."""
    start = group.get("t")
    return (start is None or t - start > _AXIS_GROUP_S
            or name in group.get("_names", ()))
