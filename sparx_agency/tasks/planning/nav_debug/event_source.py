"""The event lane: the planner's own account of why it changed its mind.

Two chains narrate into it and they share no vocabulary. The XTEND A* chain
says ``periodic replan``, ``rotated``, ``obstacle on route``, ``blockage``,
``boxed in``; FALCON's exploration FSM says none of those and instead reports
plan failures, FSM transitions, frontier/viewpoint choices, recovery and
``exploration finished``. :func:`classify_event` buckets both, so one banner
colours a run from either stack.

The recorder writes its own ``kind`` on every row; this classifier is the
fallback for rows that carry none, and the two are deliberately kept in step
(see ``nav_debug_sources.classify`` in the FALCON adapter).
"""
from __future__ import annotations

import os
from typing import Optional

from sparx_agency.tasks.planning.nav_debug.frame import ReplanEvent
from sparx_agency.tasks.planning.nav_debug.schema import EVENTS_FILE
from sparx_agency.tasks.planning.nav_debug.sources import (
    as_of_index, read_jsonl, to_float,
)

WINDOW_S = 6.0      # a banner lingers this long after its event fired

# First match wins. The A* vocabulary is tested first so an XTEND run
# classifies exactly as it always did.
_BUCKETS = (
    ("boxed_in", ("boxed in",)),
    ("blockage", ("blockage", "unseen obstacle")),
    ("obstacle", ("obstacle on route", "collision")),
    ("rotation", ("rotat",)),
    ("time", ("periodic",)),
    ("plan_fail", ("plan fail", "failed to plan", "no path", "search fail",
                   "no traj", "traj fail", "unreachable")),
    ("finish", ("finish", "exploration complete", "mission complete",
                "all explored", "no frontier")),
    ("frontier", ("frontier", "viewpoint", "coverage", "next goal", "new goal")),
    ("recovery", ("recovery", "recover", "relocaliz", "lost localization",
                  "stuck", "escape", "emergency", "backtrack")),
    ("fsm", ("fsm", "plan_traj", "pub_traj", "exec_traj", "gen_new_traj",
             "replan_traj", "wait_target", "state change")),
)


def classify_event(text: str) -> str:
    """Bucket a raw planner event string into a coarse replan ``kind``.

    Args:
        text: The publisher's own string, from either nav chain.

    Returns:
        One of ``boxed_in``, ``blockage``, ``obstacle``, ``rotation``, ``time``,
        ``plan_fail``, ``finish``, ``frontier``, ``recovery``, ``fsm``, or
        ``info`` when nothing matches.
    """
    lowered = (text or "").lower()
    for kind, keywords in _BUCKETS:
        for keyword in keywords:
            if keyword in lowered:
                return kind
    return "info"


class EventSource:
    """Every recorded event of one run, latched onto the frames that follow it."""

    def __init__(self, run_dir: str, window_s: float = WINDOW_S) -> None:
        self.window_s = window_s
        self.rows = sorted(read_jsonl(os.path.join(run_dir, EVENTS_FILE)),
                           key=lambda d: d.get("t", 0.0))
        self.stamps = [d.get("t", 0.0) for d in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def at(self, t: float) -> Optional[ReplanEvent]:
        """The newest non-``info`` event still inside the banner window at ``t``."""
        j = as_of_index(self.stamps, t)
        while j is not None and j >= 0:
            row = self.rows[j]
            stamp = float(row.get("t", 0.0))
            if t - stamp > self.window_s:
                return None
            text = str(row.get("text", ""))
            kind = str(row.get("kind") or classify_event(text))
            if kind == "info":      # skip pure info; surface the last real replan
                j -= 1
                continue
            return ReplanEvent(stamp=stamp, kind=kind, text=text, age_s=t - stamp,
                               xy=_xy(row))
        return None


def _xy(row: dict):
    """The event's world position, when it reported one."""
    x, y = to_float(row.get("x")), to_float(row.get("y"))
    return None if x is None or y is None else (x, y)
