"""Campaigns by storey: SR / SPL / DTG on the same-storey and cross-storey episodes, and the storey transitions taken.

An MP3D ObjectNav episode may put its goal on another storey than its start (17.7 % of val, entry 043), and the
method's search either stays on the start storey (``allow_stair_traversal`` False, the default) or climbs. This
report reads one or several campaign roots and tabulates, for every arm and paired against the first:

* overall SR (Wilson 95 %), SPL, SoftSPL, DTG, steps, STOP / step limit, and against the first arm the paired
  dSR and dSPL with bootstrap intervals, the exact McNemar p and the flips;
* the same against the **same-storey** and **cross-storey** episodes -- same storey when a goal view point lies
  within ``--storey-m`` (0.5) of the start height, the rule of entry 043 -- and, where the level survey
  (``sets/mp3d_val_levels.csv``) is given, by its ``start_level == goal_level`` as a second reading;
* per category: n, SR / SPL / DTG per arm, and SR per arm on the same- and cross-storey halves;
* the **storey transitions per episode**: the method's ``floor_arrival`` events (the floor atlas settling on another
  storey, planned or not) and its completed stair traversals (``completed_transitions`` with a destination), both
  from the full per-episode report (``--full-records``); and, from each episode's recording (``trajectory.csv``),
  the storey changes the agent's own height shows against the survey's level heights -- the mean per episode, the
  distribution, and for the cross-storey episodes how many ended on the goal's storey and the SR with and without
  a transition. A lean record or a run without recordings leaves the method's or the trajectory's column empty.

Every number is over the episodes every arm ran (``paired_ab.load_rows``). Usage::

    python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.storey_report \\
        stairs-off=runs/gt_oracle_mp3d_2195ep_14b_X stairs-on=runs/gt_oracle_mp3d_2195ep_14b_stairs_Y \\
        [--episodes-dir ~/datasets/objectnav/mp3d/v1/val] [--levels-csv mp3d/sets/mp3d_val_levels.csv] \\
        [--storey-m 0.5] [--markdown out.md] [--json out.json]

Standard library beyond the repository's ``stats`` and ``paired_ab``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from sparx_agency.tasks.planning.objnav_benchmark.stats import bootstrap_mean_interval, mcnemar_exact_p, wilson_interval
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.single_storey_failures import (
    load_goals, published_episode, recordings_by_episode)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.paired_ab import load_rows

DEFAULT_EPISODES_DIR = Path.home() / "datasets" / "objectnav" / "mp3d" / "v1" / "val"
DEFAULT_LEVELS_CSV = Path(__file__).resolve().parent / "sets" / "mp3d_val_levels.csv"
#: A goal view point within this height of the start is on the start's storey (entry 043's rule).
DEFAULT_STOREY_M = 0.5
#: A trajectory height is "on" a survey level when within this much of it; between levels it is on the stairs.
LEVEL_SNAP_M = 0.35


# -- the episodes' storeys ---------------------------------------------------------------------

def storey_facts(ids: Sequence[str], episodes_dir: Path, levels_csv: Optional[Path], storey_m: float) -> Dict[str, dict]:
    """Per episode: the start height, the nearest goal view point's height offset, ``same_storey``, the survey's levels."""
    survey = {}
    if levels_csv is not None and Path(levels_csv).is_file():
        with open(levels_csv, newline="") as stream:
            for record in csv.DictReader(stream):
                survey[record["episode"]] = record
    goals_cache, out = {}, {}
    for episode_id in ids:
        scene, index = episode_id.split("/")
        if scene not in goals_cache:
            goals_cache[scene] = load_goals(Path(episodes_dir), scene)
        published = published_episode(goals_cache[scene], int(index))
        start_z = published["start"][2]
        dz = min((abs(v[2] - start_z) for v in published["view_points"]), default=float("nan"))
        record = survey.get(episode_id) or {}
        levels = [float(v) for v in record.get("level_heights_m", "").split()] if record else []
        goal_levels = sorted({_nearest_level(v[2], levels) for v in published["view_points"]} - {None}) if levels else []
        out[episode_id] = {"start_z_m": round(start_z, 3), "nearest_view_point_dz_m": round(dz, 3),
                           "same_storey": dz <= storey_m,
                           "survey_same_level": (record.get("start_level") == record.get("goal_level")) if record else None,
                           "levels_m": levels, "start_level": _nearest_level(start_z, levels) if levels else None,
                           "goal_levels": goal_levels}
    return out


def _nearest_level(z: float, levels: Sequence[float], snap_m: float = LEVEL_SNAP_M) -> Optional[int]:
    """The index of the survey level within ``snap_m`` of ``z``, or None (on the stairs, or no survey)."""
    if not levels:
        return None
    index = min(range(len(levels)), key=lambda i: abs(levels[i] - z))
    return index if abs(levels[index] - z) <= snap_m else None


# -- the transitions -----------------------------------------------------------------------------

def method_transitions(row: dict) -> Optional[dict]:
    """The multi-storey policy's own count: ``floor_arrival`` events and completed stair traversals; None for a lean row."""
    policy = (row.get("agent_info") or {}).get("policy") or {}
    building = policy.get("building")
    if not isinstance(building, dict):
        return None
    events = building.get("events") or []
    completed = building.get("completed_transitions") or []
    if not isinstance(events, list) or not isinstance(completed, list):
        return None
    arrivals = [e for e in events if isinstance(e, dict) and e.get("event") == "floor_arrival"]
    return {"floor_arrivals": len(arrivals),
            "completed_stair_transitions": sum(1 for c in completed if isinstance(c, dict) and "destination" in c),
            "stair_transition_attempts": len(completed),
            "floor_revisions": int(policy.get("floor_revisions") or 0)}


def trajectory_transitions(recording: Optional[Path], facts: dict) -> Optional[dict]:
    """Storey changes the recorded height shows: consecutive settled levels that differ; the final level."""
    if recording is None or not facts.get("levels_m"):
        return None
    path = Path(recording) / "trajectory.csv"
    if not path.is_file():
        return None
    with open(path, newline="") as stream:
        heights = [float(r["z"]) for r in csv.DictReader(stream) if r.get("z")]
    if not heights:
        return None
    settled = [lvl for lvl in (_nearest_level(z, facts["levels_m"]) for z in heights) if lvl is not None]
    changes = sum(1 for a, b in zip(settled, settled[1:]) if a != b)
    final = settled[-1] if settled else None
    return {"level_changes": changes, "final_level": final,
            "ended_on_goal_level": None if final is None or not facts["goal_levels"] else final in facts["goal_levels"],
            "max_rise_m": round(max(heights) - heights[0], 2), "max_drop_m": round(heights[0] - min(heights), 2)}


# -- the tables ---------------------------------------------------------------------------------

def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def group_summary(rows: Dict[str, dict], ids: Sequence[str]) -> dict:
    picked = [rows[k] for k in ids]
    n = len(picked)
    successes = sum(1 for r in picked if r["success"])
    dtg = [float(r["distance_to_goal_m"]) for r in picked if r.get("distance_to_goal_m") is not None]
    return {"n": n, "successes": successes, "sr": successes / n if n else float("nan"),
            "sr_ci": list(wilson_interval(successes, n)) if n else [float("nan")] * 2,
            "spl": _mean([float(r["spl"]) for r in picked]),
            "soft_spl": _mean([float(r.get("soft_spl", 0.0)) for r in picked]),
            "dtg_m": _mean(dtg), "steps": _mean([float(r["steps"]) for r in picked]),
            "stops": sum(1 for r in picked if r.get("stop_called")),
            "limits": sum(1 for r in picked if not r.get("stop_called")),
            "wrong_stops_3m": sum(1 for r in picked if r.get("stop_called") and not r["success"]
                                  and float(r.get("distance_to_goal_m") or 0.0) >= 3.0)}


def paired_summary(base: Dict[str, dict], other: Dict[str, dict], ids: Sequence[str], resamples: int) -> Optional[dict]:
    if not ids:
        return None
    first = [bool(base[k]["success"]) for k in ids]
    second = [bool(other[k]["success"]) for k in ids]
    dsr = [float(b) - float(a) for a, b in zip(first, second)]
    dspl = [float(other[k]["spl"]) - float(base[k]["spl"]) for k in ids]
    ddtg = [float(other[k]["distance_to_goal_m"]) - float(base[k]["distance_to_goal_m"]) for k in ids
            if other[k].get("distance_to_goal_m") is not None and base[k].get("distance_to_goal_m") is not None]
    out = {"n": len(ids), "sr_diff": _mean(dsr), "spl_diff": _mean(dspl), "dtg_diff_m": _mean(ddtg),
           "to_success": [k for k, a, b in zip(ids, first, second) if b and not a],
           "to_failure": [k for k, a, b in zip(ids, first, second) if a and not b]}
    if len(ids) >= 2:
        out["sr_ci"] = list(bootstrap_mean_interval(dsr, resamples=resamples, seed_key="storey/sr"))
        out["spl_ci"] = list(bootstrap_mean_interval(dspl, resamples=resamples, seed_key="storey/spl"))
        out["mcnemar_p"] = mcnemar_exact_p(first, second)
    return out


def transition_summary(rows: Dict[str, dict], ids: Sequence[str], facts: Dict[str, dict],
                       recordings: Dict[str, Path]) -> dict:
    """The transitions of one arm over ``ids``: the method's counts and the trajectories', overall and by storey group."""
    method = {k: method_transitions(rows[k]) for k in ids}
    trajectory = {k: trajectory_transitions(recordings.get(k), facts[k]) for k in ids}
    out = {"method_rows": sum(1 for v in method.values() if v is not None),
           "trajectory_rows": sum(1 for v in trajectory.values() if v is not None), "groups": {}}
    for name, member in (("all", lambda k: True), ("same_storey", lambda k: facts[k]["same_storey"]),
                         ("cross_storey", lambda k: not facts[k]["same_storey"])):
        group = [k for k in ids if member(k)]
        m = [method[k] for k in group if method[k] is not None]
        t = [trajectory[k] for k in group if trajectory[k] is not None]
        arrivals = [v["floor_arrivals"] for v in m]
        changes = [v["level_changes"] for v in t]
        entry = {"n": len(group),
                 "method": None if not m else {
                     "mean_floor_arrivals": _mean(arrivals),
                     "episodes_with_arrival": sum(1 for a in arrivals if a > 0),
                     "arrivals_histogram": dict(sorted(Counter(min(a, 3) for a in arrivals).items())),
                     "mean_completed_stair_transitions": _mean([v["completed_stair_transitions"] for v in m]),
                     "stair_transition_attempts": sum(v["stair_transition_attempts"] for v in m),
                     "completed_stair_transitions": sum(v["completed_stair_transitions"] for v in m)},
                 "trajectory": None if not t else {
                     "mean_level_changes": _mean(changes),
                     "episodes_with_change": sum(1 for c in changes if c > 0),
                     "changes_histogram": dict(sorted(Counter(min(c, 3) for c in changes).items())),
                     "ended_on_goal_level": sum(1 for v in t if v["ended_on_goal_level"] is True),
                     "ended_on_goal_level_known": sum(1 for v in t if v["ended_on_goal_level"] is not None)}}
        if t:
            with_change = [k for k in group if trajectory[k] is not None and trajectory[k]["level_changes"] > 0]
            without = [k for k in group if trajectory[k] is not None and trajectory[k]["level_changes"] == 0]
            entry["sr_with_change"] = (sum(1 for k in with_change if rows[k]["success"]), len(with_change))
            entry["sr_without_change"] = (sum(1 for k in without if rows[k]["success"]), len(without))
        out["groups"][name] = entry
    out["per_episode"] = {k: {"method": method[k], "trajectory": trajectory[k]} for k in ids}
    return out


def parse_arms(specs: Sequence[str]) -> List[tuple]:
    arms = []
    for spec in specs:
        name, sep, roots = spec.partition("=")
        if not sep:
            name, roots = spec.rstrip("/").split("/")[-1], spec
        arms.append((name, roots))
    if not arms:
        raise ValueError("name at least one campaign root")
    return arms


def _recordings(roots: str) -> Dict[str, Path]:
    out = {}
    for root in roots.split(","):
        for scene_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
            out.update(recordings_by_episode(scene_dir))
    return out


def report(arms: Sequence[tuple], *, episodes_dir: Path = DEFAULT_EPISODES_DIR, levels_csv: Optional[Path] = DEFAULT_LEVELS_CSV,
           storey_m: float = DEFAULT_STOREY_M, resamples: int = 10000, transitions: bool = True) -> dict:
    """Every table as data; :func:`render` writes the Markdown."""
    loaded = [(name, load_rows(roots)) for name, roots in arms]
    common = set(loaded[0][1])
    for _, rows in loaded[1:]:
        common &= set(rows)
    ids = sorted(common)
    if not ids:
        raise ValueError("no episode was run by every campaign named")
    facts = storey_facts(ids, episodes_dir, levels_csv, storey_m)
    groups = OrderedDict((("all", ids),
                          ("same_storey", [k for k in ids if facts[k]["same_storey"]]),
                          ("cross_storey", [k for k in ids if not facts[k]["same_storey"]])))
    survey_groups = OrderedDict((("survey_same_level", [k for k in ids if facts[k]["survey_same_level"] is True]),
                                 ("survey_other_level", [k for k in ids if facts[k]["survey_same_level"] is False])))
    base_name, base = loaded[0]
    categories = sorted({base[k]["target_category"] for k in ids})
    out = {"arms": [name for name, _ in loaded], "episodes": len(ids), "storey_m": storey_m,
           "sizes": {name: len(rows) for name, rows in loaded},
           "group_sizes": {name: len(members) for name, members in groups.items()},
           "survey_group_sizes": {name: len(members) for name, members in survey_groups.items()},
           "per_arm": {}, "paired": {}, "by_category": {}, "transitions": {}}
    for name, rows in loaded:
        out["per_arm"][name] = {g: group_summary(rows, members) for g, members in list(groups.items()) + list(survey_groups.items())}
        out["by_category"][name] = {
            c: {"all": group_summary(rows, [k for k in ids if rows[k]["target_category"] == c]),
                "same_storey": group_summary(rows, [k for k in groups["same_storey"] if rows[k]["target_category"] == c]),
                "cross_storey": group_summary(rows, [k for k in groups["cross_storey"] if rows[k]["target_category"] == c])}
            for c in categories}
        if name != base_name:
            out["paired"][name] = {g: paired_summary(base, rows, members, resamples)
                                   for g, members in list(groups.items()) + list(survey_groups.items())}
            out["paired"][name]["by_category"] = {
                c: paired_summary(base, rows, [k for k in ids if rows[k]["target_category"] == c], resamples) for c in categories}
        if transitions:
            recordings = _recordings(dict(arms)[name])
            out["transitions"][name] = transition_summary(rows, ids, facts, recordings)
    out["storey_facts"] = facts
    return out


def _pct(value: float) -> str:
    return "n/a" if value != value else "%.1f %%" % (100.0 * value)


def _ci(values: Sequence[float]) -> str:
    return "[%.1f, %.1f]" % (100.0 * values[0], 100.0 * values[1])


def _group_line(name: str, summary: dict, pair: Optional[dict]) -> str:
    cells = [name, "%d" % summary["n"], "%s %s" % (_pct(summary["sr"]), _ci(summary["sr_ci"])) if summary["n"] else "n/a",
             _pct(summary["spl"]), _pct(summary["soft_spl"]), "%.2f" % summary["dtg_m"] if summary["n"] else "n/a",
             "%.0f" % summary["steps"] if summary["n"] else "n/a", "%d / %d" % (summary["stops"], summary["limits"]),
             "%d" % summary["wrong_stops_3m"]]
    if pair is None:
        cells += ["--", "--", "--", "--"]
    else:
        cells += ["%+.1f pts %s" % (100 * pair["sr_diff"], _ci(pair["sr_ci"]) if "sr_ci" in pair else ""),
                  "%.3f" % pair["mcnemar_p"] if "mcnemar_p" in pair else "--",
                  "%+.1f pts %s" % (100 * pair["spl_diff"], _ci(pair["spl_ci"]) if "spl_ci" in pair else ""),
                  "%+.2f m; flips +%d / -%d" % (pair["dtg_diff_m"], len(pair["to_success"]), len(pair["to_failure"]))]
    return "| " + " | ".join(cells) + " |"


def render(data: dict) -> str:
    names = data["arms"]
    base = names[0]
    lines = ["Episodes in every campaign: %d (same storey %d, cross storey %d at %.1f m; the level survey: same %d, other %d)%s" % (
        data["episodes"], data["group_sizes"]["same_storey"], data["group_sizes"]["cross_storey"], data["storey_m"],
        data["survey_group_sizes"]["survey_same_level"], data["survey_group_sizes"]["survey_other_level"],
        "" if all(n == data["episodes"] for n in data["sizes"].values())
        else " (rows per campaign: %s)" % ", ".join("%s %d" % item for item in data["sizes"].items())), ""]
    header = ("| arm / group | n | SR [95 %%] | SPL | SoftSPL | DTG (m) | steps | STOP / limit | wrong STOPs >= 3 m "
              "| dSR vs %s [95 %%] | McNemar p | dSPL [95 %%] | dDTG; flips |" % base)
    rule = "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    for group, title in (("all", "Overall"), ("same_storey", "Same storey (a goal view point within %.1f m of the start height)" % data["storey_m"]),
                         ("cross_storey", "Cross storey (every goal view point farther than that)"),
                         ("survey_same_level", "Level survey: start_level == goal_level"),
                         ("survey_other_level", "Level survey: goal on another level")):
        lines += ["## " + title, "", header, rule]
        for name in names:
            pair = (data["paired"].get(name) or {}).get(group) if name != base else None
            lines.append(_group_line(name, data["per_arm"][name][group], pair))
        lines.append("")
    lines += ["## Per category", "", "| category | n | " + " | ".join("SR %s" % n for n in names) + " | "
              + " | ".join("SPL %s" % n for n in names) + " | " + " | ".join("DTG %s" % n for n in names)
              + (" | dSR %s vs %s" % (names[1], base) if len(names) > 1 else "") + " |",
              "|---|---|" + "---|" * (3 * len(names) + (1 if len(names) > 1 else 0))]
    categories = sorted(data["by_category"][base])
    for c in categories:
        cells = ["%d" % data["by_category"][base][c]["all"]["n"]]
        cells += [_pct(data["by_category"][n][c]["all"]["sr"]) for n in names]
        cells += [_pct(data["by_category"][n][c]["all"]["spl"]) for n in names]
        cells += ["%.2f" % data["by_category"][n][c]["all"]["dtg_m"] for n in names]
        if len(names) > 1:
            pair = data["paired"][names[1]]["by_category"][c]
            cells.append("%+.1f pts (+%d / -%d)" % (100 * pair["sr_diff"], len(pair["to_success"]), len(pair["to_failure"])) if pair else "--")
        lines.append("| %s | %s |" % (c, " | ".join(cells)))
    lines += ["", "## Per category by storey (SR; n same / n cross)", "",
              "| category | n same | " + " | ".join("same %s" % n for n in names) + " | n cross | "
              + " | ".join("cross %s" % n for n in names) + " |", "|---|---|" + "---|" * (2 * len(names) + 1)]
    for c in categories:
        same = data["by_category"][base][c]["same_storey"]["n"]
        cross = data["by_category"][base][c]["cross_storey"]["n"]
        cells = ["%d" % same] + [_pct(data["by_category"][n][c]["same_storey"]["sr"]) if same else "--" for n in names]
        cells += ["%d" % cross] + [_pct(data["by_category"][n][c]["cross_storey"]["sr"]) if cross else "--" for n in names]
        lines.append("| %s | %s |" % (c, " | ".join(cells)))
    if data["transitions"]:
        lines += ["", "## Storey transitions per episode", "",
                  "Method: the floor atlas's `floor_arrival` events (settling on another storey, planned or not) and the "
                  "completed stair traversals, from the full per-episode report. Trajectory: changes of the settled survey "
                  "level along the recorded height (within %.2f m of a level; between levels is the stairs)." % LEVEL_SNAP_M, "",
                  "| arm | group | n | floor arrivals / ep | episodes with an arrival | arrivals 0 / 1 / 2 / 3+ | completed stair traversals (attempts) "
                  "| level changes / ep (trajectory) | episodes with a change | changes 0 / 1 / 2 / 3+ | ended on the goal's level | SR with a change | SR without |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name in names:
            summary = data["transitions"][name]
            for group in ("all", "same_storey", "cross_storey"):
                entry = summary["groups"][group]
                m, t = entry["method"], entry["trajectory"]
                hist = lambda h: " / ".join(str(h.get(i, 0)) for i in range(4))
                cells = [name, group, "%d" % entry["n"],
                         "%.2f" % m["mean_floor_arrivals"] if m else "--", "%d" % m["episodes_with_arrival"] if m else "--",
                         hist(m["arrivals_histogram"]) if m else "--",
                         "%d (%d)" % (m["completed_stair_transitions"], m["stair_transition_attempts"]) if m else "--",
                         "%.2f" % t["mean_level_changes"] if t else "--", "%d" % t["episodes_with_change"] if t else "--",
                         hist(t["changes_histogram"]) if t else "--",
                         "%d of %d" % (t["ended_on_goal_level"], t["ended_on_goal_level_known"]) if t else "--",
                         "%d / %d" % entry["sr_with_change"] if "sr_with_change" in entry else "--",
                         "%d / %d" % entry["sr_without_change"] if "sr_without_change" in entry else "--"]
                lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    for name in names[1:]:
        pair = data["paired"][name]["all"]
        lines += ["%s against %s: %d to success, %d to failure" % (name, base, len(pair["to_success"]), len(pair["to_failure"])),
                  "  to success: %s" % (", ".join(pair["to_success"]) or "none"),
                  "  to failure: %s" % (", ".join(pair["to_failure"]) or "none"), ""]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("arms", nargs="+", help="name=root[,root...]; the first is the baseline")
    parser.add_argument("--episodes-dir", type=Path, default=DEFAULT_EPISODES_DIR)
    parser.add_argument("--levels-csv", type=Path, default=DEFAULT_LEVELS_CSV,
                        help="the level survey (sets/mp3d_val_levels.csv); 'none' for no survey reading")
    parser.add_argument("--storey-m", type=float, default=DEFAULT_STOREY_M)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--no-transitions", action="store_true", help="skip the recordings (faster)")
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--json", type=Path, help="the tables as JSON (without the per-episode transitions)")
    args = parser.parse_args(argv)
    levels = None if str(args.levels_csv) == "none" else args.levels_csv
    data = report(parse_arms(args.arms), episodes_dir=args.episodes_dir, levels_csv=levels, storey_m=args.storey_m,
                  resamples=args.resamples, transitions=not args.no_transitions)
    text = render(data)
    print(text, end="")
    if args.markdown:
        args.markdown.write_text(text)
    if args.json:
        slim = dict(data, transitions={name: {k: v for k, v in t.items() if k != "per_episode"} for name, t in data["transitions"].items()})
        args.json.write_text(json.dumps(slim, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
