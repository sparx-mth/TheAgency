"""Detect a metric family that has silently stopped being produced.

The campaign has lost real time to instrumentation that broke without saying
so: a hardcoded epoch prefix in a regex that matched nothing once wall time
rolled over (and therefore read as "no events"), `rosout` duplicating FSM lines
by a rotation-dependent amount, and coverage gaps that deleted the stalls they
should have recorded. Each was found by accident, long after it started.

A dead probe is worse than a loud failure because the number it feeds keeps
looking plausible. So this compares a recent window against an older one and
reports any family that used to be produced and is not any more.

ONE DISTINCTION MATTERS, and getting it wrong makes this tool useless: a key
that is ABSENT or NULL is broken, while a key present and EMPTY is usually
healthy. ``collapse_signature: []`` means the run matched no failure shape and
``data_gaps: []`` means the recording had no holes -- both are the good case,
and an earlier version of this check flagged them as failures precisely because
it conflated the two.

Usage:
    python -m sparx_agency.tools.falcon_campaign.check_instrumentation
"""

import json
from collections import Counter

from sparx_agency.tools.falcon_campaign import config as C

#: Dotted paths into ``metrics.json`` that every healthy run should populate.
FAMILIES = (
    "motion.distance_m", "motion.per_minute", "actuation",
    "tracking.pos_err_m.mean", "tracking.heartbeats", "altitude.ranger_m",
    "coverage.final_m3", "health", "exploration", "clearance",
    "config.max_vel", "collapse_signature", "data_gaps",
)
#: Runs in the recent window, and in the older window compared against.
RECENT = 10
OLDER = 50


def state(metrics, path):
    """Classify one dotted path as ``ABSENT``, ``NULL``, ``empty`` or ``ok``."""
    current = metrics
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return "ABSENT"
        current = current[key]
    if current is None:
        return "NULL"
    if isinstance(current, (list, dict)) and not current:
        return "empty"
    return "ok"


def _load(runs):
    """Read the metrics of each run, skipping unreadable ones."""
    out = []
    for run in runs:
        path = run / "metrics.json"
        if not path.is_file():
            continue
        try:
            out.append(json.loads(path.read_text()))
        except (ValueError, OSError):
            continue
    return out


def check(runs_dir):
    """Report families that stopped being produced. Returns the broken ones."""
    runs = sorted(runs_dir.glob("2026*Z"))
    recent, older = _load(runs[-RECENT:]), _load(runs[-RECENT - OLDER:-RECENT])
    if not recent or not older:
        print("not enough runs to compare (%d recent, %d older)"
              % (len(recent), len(older)))
        return []
    broken = []
    print("%-28s %-24s %s" % ("family", "last %d" % len(recent),
                              "prior %d" % len(older)))
    for family in FAMILIES:
        now = Counter(state(m, family) for m in recent)
        was = Counter(state(m, family) for m in older)
        # Only ABSENT/NULL count as broken, and only when the older window was
        # producing the family cleanly -- otherwise this reports a probe that
        # never worked as one that just died.
        now_bad = now["ABSENT"] + now["NULL"]
        was_bad = was["ABSENT"] + was["NULL"]
        died = now_bad == len(recent) and was_bad < len(older) / 2.0
        if died:
            broken.append(family)
        print("%-28s %-24s %s%s" % (family, dict(now), dict(was),
                                    "   <-- STOPPED" if died else ""))
    print("\n%s" % ("BROKEN: " + ", ".join(broken) if broken
                    else "every family still being produced."))
    return broken


if __name__ == "__main__":
    check(C.RUNS_DIR)
