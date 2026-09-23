"""Carry a survey across flights, so one bad contact does not cost the building.

A hospital is bigger than a flight. Measured on this deployment, one System-2
decision costs about 22 s and buys about a metre of committed route, so covering
twenty rooms is hours of flying -- and a single capsize ends a flight outright,
because the SJTU plugin cannot right itself and ``/simple_drone/reset`` does not
do it either. A thirty-minute run that flips at minute nine has surveyed nine
minutes of building and thrown it away.

So the survey is written to disk and read back: the mask of what the camera has
seen, and the supervisor's own bookkeeping of which rooms are finished and which
have been given up on. The campaign harness already restarts the world and
re-ferries the aircraft between runs; with this, run *n+1* continues run *n*
rather than starting again, and a capsize costs one segment instead of
everything.

**It refuses to load state built against a different building.** A seen-mask is
just an array of cells; laid over a map with a different shape, resolution or
origin it is confidently wrong, and every number downstream would be wrong with
it. The provenance is checked and a mismatch is an error, not a silent reset.

ROS-free, numpy only, Python 3.8.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np

#: Bumped when the stored fields change shape. An older file is refused rather
#: than half-read.
FORMAT = 2


def save_survey(path, coverage, supervisor, extra=None):
    # type: (str, Any, Any, Optional[Dict]) -> str
    """Write the seen-mask and the supervisor's bookkeeping to one ``.npz``.

    Args:
        path: Destination file. Parent directories are created.
        coverage: A :class:`VisibilityCoverage`; its ``seen_mask`` is stored.
        supervisor: An :class:`ExplorationSupervisor`; its progress is stored.
        extra: Optional JSON-serialisable provenance, returned by
            :func:`load_survey` untouched.

    Returns:
        The path written.

    Written to a temporary file and renamed, because this is saved periodically
    during a flight and a run killed mid-write would otherwise leave the next
    one a truncated array to start from.
    """
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    book = {
        "accepted": sorted(int(r) for r in supervisor._accepted),
        "exhausted": sorted([str(k), int(v)] for k, v in supervisor._exhausted),
        "attempts": [[str(k), int(v), int(n)]
                     for (k, v), n in supervisor._attempts.items()],
        "issues": [[str(k), int(v), int(n)]
                   for (k, v), n in supervisor._issues.items()],
        # Added after the first surveys, and deliberately OPTIONAL on read: a
        # file written before it existed still loads, it just starts with no
        # remembered vantages, which costs a little repeated scanning and no
        # correctness. Bumping the format would have thrown away a survey that
        # was 43% of the way through a building.
        "scans": [[int(rid), [[float(vx), float(vy)] for vx, vy in points]]
                  for rid, points in supervisor._scans.items()],
    }
    temporary = path + ".part"
    np.savez_compressed(
        temporary,
        format=FORMAT,
        seen=coverage.seen_mask,
        shape=np.array(coverage.seen_mask.shape),
        resolution=float(coverage._resolution),
        origin=np.array([coverage._origin_x, coverage._origin_y]),
        cells_total=int(coverage.cells_total),
        book=json.dumps(book),
        extra=json.dumps(extra or {}))
    os.replace(temporary + ".npz", path)
    return path


def load_survey(path, coverage, supervisor, logger=None):
    # type: (Optional[str], Any, Any, object) -> bool
    """Restore a survey into a fresh coverage tracker and supervisor.

    Args:
        path: The ``.npz`` written by :func:`save_survey`, or None/missing to
            start a new survey.
        coverage: The tracker to fill. Must already be built against the same
            map the state was saved from.
        supervisor: The supervisor whose bookkeeping to restore.
        logger: Optional object with ``.info``/``.warn``.

    Returns:
        True if state was loaded, False if there was none to load.

    Raises:
        ValueError: The stored survey does not belong to this map, or was
            written by a different format. Continuing from a mask laid over the
            wrong building is worse than starting again, and silently starting
            again would hide it.
    """
    if not path or not os.path.isfile(path):
        if logger is not None and path:
            logger.info("no survey to resume at %s; starting a new one" % path)
        return False
    data = np.load(path, allow_pickle=False)
    stored = int(data["format"]) if "format" in data else 0
    if stored != FORMAT:
        raise ValueError("survey at %s is format %d, this is %d -- delete it or "
                         "keep it, but it cannot be resumed"
                         % (path, stored, FORMAT))
    seen = data["seen"].astype(bool)
    if seen.shape != coverage.seen_mask.shape:
        raise ValueError("survey at %s is %r, this building is %r"
                         % (path, seen.shape, coverage.seen_mask.shape))
    if abs(float(data["resolution"]) - coverage._resolution) > 1e-9:
        raise ValueError("survey at %s was surveyed at %.4f m per cell, this map "
                         "is %.4f" % (path, float(data["resolution"]),
                                      coverage._resolution))
    origin = data["origin"]
    if (abs(float(origin[0]) - coverage._origin_x) > 1e-6
            or abs(float(origin[1]) - coverage._origin_y) > 1e-6):
        raise ValueError("survey at %s has origin %r, this map has (%.2f, %.2f)"
                         % (path, tuple(origin), coverage._origin_x,
                            coverage._origin_y))

    coverage.restore_seen(seen)
    book = json.loads(str(data["book"]))
    supervisor._accepted = set(int(r) for r in book.get("accepted", ()))
    supervisor._exhausted = set((str(k), int(v)) for k, v in book.get("exhausted", ()))
    supervisor._attempts = {(str(k), int(v)): int(n)
                            for k, v, n in book.get("attempts", ())}
    supervisor._issues = {(str(k), int(v)): int(n)
                          for k, v, n in book.get("issues", ())}
    supervisor._scans = {int(rid): [(float(vx), float(vy)) for vx, vy in points]
                         for rid, points in book.get("scans", ())}
    if logger is not None:
        logger.info("resumed survey %s: %.1f%% seen, %d areas finished, "
                    "%d orders retired"
                    % (path, 100.0 * coverage.fraction_seen,
                       len(supervisor._accepted), len(supervisor._exhausted)))
    return True
