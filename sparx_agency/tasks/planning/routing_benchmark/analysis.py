"""Turning a results file into the handful of numbers worth looking at.

Pure aggregation: reads the JSON the runner wrote, groups it, and returns
tables. Nothing here plots anything or prints anything, so the same summaries
back both the terminal report and the published page and cannot disagree with
each other.

**Medians, not means.** Distances are strongly right-skewed -- one scenario
where a planner chases a decoy to the far end of a cross-shaped building
produces an outlier several times the typical value, and a mean would let a
handful of those decide the answer. The median says what usually happens, which
is the question.

Standard library only.
"""
from __future__ import annotations

import json
import pathlib
import statistics
from typing import Dict, Iterable, List, Sequence, Tuple

#: Presentation order, best-motivated first.
PLANNER_ORDER = ("rpt_star", "f_rpt_star", "nearest_2opt", "nearest",
                 "greedy", "random")

#: Oracle regimes from most to least informative -- the axis every headline
#: chart is grouped by.
ORACLE_ORDER = ("perfect", "accurate", "noisy", "decoy", "uninformative",
                "adversarial")


def load(path):
    # type: (pathlib.Path) -> Dict
    """Read a results file."""
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def rows_where(rows, **conditions):
    # type: (Sequence[Dict], object) -> List[Dict]
    """The rows matching every named field value."""
    return [row for row in rows
            if all(row.get(field) == value
                   for field, value in conditions.items())]


def overall(rows):
    # type: (Sequence[Dict]) -> List[Dict]
    """One line per planner, across every scenario.

    Returns:
        Per planner: the median expected distance to find the object, how much
        worse that is than the best planner on the same scenario, the median
        planning time, and how often it produced its own answer.
    """
    out = []
    for planner in PLANNER_ORDER:
        mine = rows_where(rows, planner=planner)
        if not mine:
            continue
        out.append({
            "planner": planner,
            "scenarios": len(mine),
            "distance_m": statistics.median(r["distance_m"] for r in mine),
            "mission_time_s": statistics.median(r["mission_time_s"]
                                                for r in mine),
            "ratio_to_best": statistics.median(r["ratio_to_best"]
                                               for r in mine),
            "regret_m": statistics.median(r["regret_m"] for r in mine),
            "plan_ms": 1000.0 * statistics.median(r["plan_seconds"]
                                                  for r in mine),
            "plan_ms_p95": 1000.0 * _quantile(
                [r["plan_seconds"] for r in mine], 0.95),
            "solved_pct": 100.0 * sum(1 for r in mine if r["solved"]) / len(mine),
        })
    return out


def by_field(rows, field, order=None):
    # type: (Sequence[Dict], str, Sequence) -> List[Dict]
    """Median distance per planner, grouped by one scenario factor.

    Args:
        rows: The result rows.
        field: Which factor to group by -- ``oracle``, ``n_rooms``,
            ``topology``.
        order: The order to present the groups in; discovered and sorted if
            omitted.

    Returns:
        One entry per group, with a ``planners`` mapping inside it and the
        saving RPT* makes against the best baseline.
    """
    groups = order or sorted({row[field] for row in rows})
    out = []
    for group in groups:
        subset = rows_where(rows, **{field: group})
        if not subset:
            continue
        per_planner = {}
        for planner in PLANNER_ORDER:
            mine = rows_where(subset, planner=planner)
            if mine:
                per_planner[planner] = {
                    "distance_m": statistics.median(r["distance_m"]
                                                    for r in mine),
                    "mission_time_s": statistics.median(r["mission_time_s"]
                                                        for r in mine),
                    "plan_ms": 1000.0 * statistics.median(r["plan_seconds"]
                                                          for r in mine),
                    "solved_pct": (100.0 * sum(1 for r in mine if r["solved"])
                                   / len(mine)),
                }
        out.append({
            field: group,
            "scenarios": len(subset) // max(1, len(per_planner)),
            "agreement": statistics.median(r["agreement"] for r in subset),
            "clairvoyant_m": statistics.median(r["clairvoyant_m"]
                                               for r in subset),
            "planners": per_planner,
            "saving_pct": _saving(per_planner),
        })
    return out


def head_to_head(rows, champion="rpt_star"):
    # type: (Sequence[Dict], str) -> List[Dict]
    """How often the champion beats each rival, scenario by scenario.

    A per-scenario comparison rather than a comparison of aggregates, because
    two planners can have identical medians while one wins nearly every
    individual case. Ties inside a tenth of a percent are counted as ties.

    Returns:
        One entry per rival, with win/tie/loss counts and the median relative
        saving where it does win.
    """
    paired = {}                                 # type: Dict[str, Dict[str, float]]
    for row in rows:
        paired.setdefault(row["scenario"], {})[row["planner"]] = row["distance_m"]

    out = []
    for rival in PLANNER_ORDER:
        if rival == champion:
            continue
        wins = ties = losses = 0
        savings = []                            # type: List[float]
        for scores in paired.values():
            if champion not in scores or rival not in scores:
                continue
            mine, theirs = scores[champion], scores[rival]
            if theirs <= 0.0:
                continue
            relative = mine / theirs
            if relative < 0.999:
                wins += 1
                savings.append(100.0 * (1.0 - relative))
            elif relative > 1.001:
                losses += 1
            else:
                ties += 1
        out.append({
            "rival": rival,
            "wins": wins, "ties": ties, "losses": losses,
            "win_pct": 100.0 * wins / max(1, wins + ties + losses),
            "median_saving_pct": statistics.median(savings) if savings else 0.0,
        })
    return out


def _saving(per_planner, champion="rpt_star"):
    """How much less the champion flies than the best planner that is not it."""
    if champion not in per_planner:
        return 0.0
    rivals = [values["distance_m"] for name, values in per_planner.items()
              if name not in (champion, "f_rpt_star")]
    if not rivals:
        return 0.0
    best_rival = min(rivals)
    if best_rival <= 0.0:
        return 0.0
    return 100.0 * (1.0 - per_planner[champion]["distance_m"] / best_rival)


def _quantile(values, fraction):
    """A simple order-statistic quantile; no interpolation, no numpy."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]
