"""The route lane: what the planner had laid out at a given instant.

Each snapshot is one ``routes/<ms>.json`` holding whichever layers that chain
produces. On XTEND those are the A* chain's three -- raw, BEV-corrected, final
-- plus the goal and the pure-pursuit aim point. On Sphera there is no A* chain:
FALCON's own trajectory arrives as ``final`` and the path it has actually flown
as ``executed``, so plan and outcome can be compared on one map.

``executed`` is passed through only when :class:`~.frame.Routes` declares it, so
a recorder that writes the newer key stays loadable against an older
``frame.py`` instead of failing to construct the whole layer set.
"""
from __future__ import annotations

import dataclasses
import glob
import json
import os
from typing import List, Optional

from sparx_agency.tasks.planning.nav_debug.frame import Routes
from sparx_agency.tasks.planning.nav_debug.schema import ROUTES_DIR
from sparx_agency.tasks.planning.nav_debug.sources import as_of_index, read_jsonl

_FIELDS = {f.name for f in dataclasses.fields(Routes)}
_LAYERS = ("astar", "safe", "final", "executed")
_POINTS = ("goal", "lookahead")


class RouteSource:
    """Every recorded route snapshot of one run, resolvable by timestamp."""

    def __init__(self, run_dir: str) -> None:
        self.rows = sorted(read_jsonl(os.path.join(run_dir, "routes.jsonl"))
                           + _snapshots(run_dir),
                           key=lambda d: d.get("t", 0.0))
        self.stamps = [d.get("t", 0.0) for d in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def at(self, t: float) -> Routes:
        """The route layers latest at or before ``t`` (empty if none yet)."""
        j = as_of_index(self.stamps, t)
        if j is None:
            return Routes()
        row = self.rows[j]
        kwargs = dict((name, _polyline(row.get(name))) for name in _LAYERS
                      if name in _FIELDS)
        kwargs.update((name, _point(row.get(name))) for name in _POINTS)
        return Routes(**kwargs)


def _snapshots(run_dir: str) -> List[dict]:
    """Every ``routes/<ms>.json``, stamped from its payload or its filename."""
    out = []
    for path in glob.glob(os.path.join(run_dir, ROUTES_DIR, "*.json")):
        try:
            with open(path) as fh:
                row = json.load(fh)
        except (OSError, IOError, ValueError):
            continue
        if not isinstance(row, dict):
            continue
        if "t" not in row:
            try:
                row["t"] = float(os.path.splitext(os.path.basename(path))[0]) / 1000.0
            except ValueError:
                continue
        out.append(row)
    return out


def _polyline(value) -> Optional[List]:
    """A recorded layer -> ``[(x, y), ...]``; None when absent or malformed."""
    if not value:
        return None
    try:
        return [(float(p[0]), float(p[1])) for p in value]
    except (TypeError, ValueError, IndexError, KeyError):
        return None


def _point(value):
    """A recorded point -> ``(x, y)``; None when absent or malformed."""
    if not value:
        return None
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError, IndexError, KeyError):
        return None
