"""Per-episode summaries of a closing trace and the report for tests T1-T5, T8 and the inputs of T7b of the verification diagnosis."""
import collections
import math
import statistics

REAL_GOAL_RADIUS_M = 1.5
NEAR_MISS_RADIUS_M = 1.5
NEAR_GOAL_APPROACH_M = 2.0
AT_GOAL_APPROACH_M = 1.0
LONG_ESCAPE_STEPS = 100
MIN_SHORTEST_PATH_M = 0.5


def distance_to_nearest(point, centres):
    return min(math.dist(point, c) for c in centres)


def is_hit(row):
    return bool(row["target_boxes"])


def all_clipped(row):
    return all(box["clipped"] for box in row["target_boxes"])


def longest_consecutive_hits(rows):
    best = run = 0
    last_hit_step = None
    for row in rows:
        if not is_hit(row):
            run, last_hit_step = 0, None
            continue
        run = run + 1 if last_hit_step is not None and row["step"] == last_hit_step + 1 else 1
        last_hit_step = row["step"]
        best = max(best, run)
    return best


def takeover_windows(rows):
    """``[(rows of one takeover, the row right after it)]``: a takeover is a run of rows with the closing active."""
    windows, start = [], None
    for index, row in enumerate(rows):
        if row["active"] and start is None:
            start = index
        elif not row["active"] and start is not None:
            windows.append((rows[start:index], row))
            start = None
    if start is not None:
        windows.append((rows[start:], None))
    return windows


def describe_window(window, following, goals):
    anchor = next((r["candidate_xy"] for r in reversed(window) if r["candidate_xy"]), None)
    return {"first_step": window[0]["step"], "steps": window[-1]["step"] - window[0]["step"] + 1,
            "hit_frames": sum(is_hit(r) for r in window), "longest_run": longest_consecutive_hits(window),
            "clipped_hit_frames": sum(is_hit(r) and all_clipped(r) for r in window), "locked": any(r["locked"] for r in window),
            "max_conf": max(r["best_conf"] for r in window), "ended_by": following["last_release"] if following else None,
            "on_real_goal": None if anchor is None else distance_to_nearest(anchor, goals) <= REAL_GOAL_RADIUS_M}


def outcome_of(record, final_xy, goals):
    if record["success"]:
        return "success"
    if not record["stop_called"]:
        return "timeout"
    near = final_xy is not None and distance_to_nearest(final_xy, goals) <= NEAR_MISS_RADIUS_M
    return "near_miss_stop" if near else "wrong_stop"


def stop_room_of(rows):
    room = next((r["candidate_room"] for r in reversed(rows) if r.get("candidate_room")), None)
    return None if room is None else {"label": room["label"], "strength": room["strength"]}


def summarise_episode(episode_id, record, rows, poses, goals):
    phases = record["agent_info"]["policy"]["exploration_metrics"]["actions_by_phase"]
    final_xy = tuple(poses[-1]) if poses else None
    outside_takeover_hits = [r for r in rows if not r["active"] and is_hit(r)]
    shortest = record["shortest_path_m"]
    return {"episode": episode_id, "category": record["target_category"], "outcome": outcome_of(record, final_xy, goals),
            "steps": record["steps"], "distance_to_goal_m": record["distance_to_goal_m"], "spl": record["spl"],
            "path_ratio": record["path_length_m"] / shortest if shortest > MIN_SHORTEST_PATH_M else None,
            "phase_steps": dict(phases), "escape_steps": phases.get("escape", 0),
            "windows": [describe_window(w, f, goals) for w, f in takeover_windows(rows)],
            "frames_dropped_by_border": sum(all_clipped(r) for r in outside_takeover_hits),
            "frames_seen_not_taken": sum(not all_clipped(r) for r in outside_takeover_hits),
            "closest_approach_m": min((distance_to_nearest(tuple(p), goals) for p in poses), default=None),
            "stop_to_goal_m": None if final_xy is None else distance_to_nearest(final_xy, goals),
            "memory_sight_refusals": rows[-1]["counters"]["memory_sight_refusals"] if rows else 0,
            "stop_room": stop_room_of(rows)}


def window_line(summary, window):
    return "  - %s: step %d, %d steps, %d hit frames (longest run %d, %d clipped), locked=%s, ended by %s" % (
        summary["episode"], window["first_step"], window["steps"], window["hit_frames"], window["longest_run"],
        window["clipped_hit_frames"], window["locked"], window["ended_by"])


def section_lock_gate(summaries):
    real = [(s, w) for s in summaries for w in s["windows"] if w["on_real_goal"]]
    unlocked = [(s, w) for s, w in real if not w["locked"]]
    scattered = [(s, w) for s, w in unlocked if w["hit_frames"] >= 2 and w["longest_run"] < 2]
    false_windows = sum(1 for s in summaries for w in s["windows"] if w["on_real_goal"] is False)
    lines = ["## T1 lock gate (2 consecutive frames within 12 steps)", "",
             "takeovers on a real goal: %d, ended without a lock: %d, of those with 2+ hit frames but never 2 in a row: %d "
             "(the gate, not the detector). Takeovers on non-goal objects: %d." % (len(real), len(unlocked), len(scattered), false_windows)]
    return lines + [window_line(s, w) for s, w in unlocked]


def section_border(summaries):
    dropped = [s for s in summaries if s["frames_dropped_by_border"]]
    near = [s for s in dropped if s["closest_approach_m"] is not None and s["closest_approach_m"] <= NEAR_GOAL_APPROACH_M]
    lines = ["## T2 border clipping", "",
             "episodes with target frames dropped by the border: %d, of those that came within %.0f m of a goal: %d." % (len(dropped), NEAR_GOAL_APPROACH_M, len(near))]
    return lines + ["  - %s: %d dropped frames, closest approach %.1f m, %s" % (s["episode"], s["frames_dropped_by_border"], s["closest_approach_m"], s["outcome"]) for s in near]


def section_memory_refusals(summaries):
    hit = [s for s in summaries if s["memory_sight_refusals"] and s["closest_approach_m"] is not None and s["closest_approach_m"] <= AT_GOAL_APPROACH_M]
    lines = ["## T3 memory-STOP refusals", "", "episodes with refusals that came within %.0f m of a goal: %d." % (AT_GOAL_APPROACH_M, len(hit))]
    return lines + ["  - %s: %d refusals, closest approach %.1f m, %s" % (s["episode"], s["memory_sight_refusals"], s["closest_approach_m"], s["outcome"]) for s in hit]


def section_escape(summaries):
    stuck = [s for s in summaries if s["escape_steps"] >= LONG_ESCAPE_STEPS]
    lines = ["## T4 escape trap", "", "episodes with %d+ steps in escape: %d of %d." % (LONG_ESCAPE_STEPS, len(stuck), len(summaries))]
    return lines + ["  - %s: %d escape steps, %s" % (s["episode"], s["escape_steps"], s["outcome"]) for s in stuck]


def section_near_miss(summaries):
    misses = [s for s in summaries if s["outcome"] == "near_miss_stop"]
    lines = ["## T5 near-miss stops", "", "stops within %.1f m of a goal centre that still failed: %d." % (NEAR_MISS_RADIUS_M, len(misses))]
    return lines + ["  - %s: stopped %.2f m from the goal centre, distance to goal %.2f m" % (s["episode"], s["stop_to_goal_m"], s["distance_to_goal_m"]) for s in misses]


def section_stop_rooms(summaries):
    lines = ["## T7b inputs: the room at each stop (the model replay reads these)", ""]
    for name, outcome in (("false stops", "wrong_stop"), ("successes", "success")):
        group = [s for s in summaries if s["outcome"] == outcome]
        labelled = [s for s in group if s["stop_room"] is not None]
        labels = collections.Counter(s["stop_room"]["label"] for s in labelled)
        lines += ["%s: %d (%d with a room label, %d without)" % (name, len(group), len(labelled), len(group) - len(labelled)),
                  "  room labels: %s" % dict(labels.most_common())]
    return lines


def section_spl(summaries):
    wins = [s for s in summaries if s["outcome"] == "success"]
    ratios = [s["path_ratio"] for s in wins if s["path_ratio"] is not None]
    steps = collections.Counter()
    for s in wins:
        for phase, n in s["phase_steps"].items():
            steps[phase.split("/")[0]] += n
    total = sum(steps.values()) or 1
    lines = ["## T8 where SPL goes", ""]
    if ratios:
        lines.append("successes: %d; path over shortest path: median %.2f, mean %.2f, share above 3x: %.0f%%." % (
            len(wins), statistics.median(ratios), statistics.mean(ratios), 100 * sum(r > 3 for r in ratios) / len(ratios)))
    return lines + ["  - %s: %.0f%% of actions" % (phase, 100 * n / total) for phase, n in steps.most_common(8)]


def render_report(summaries):
    outcomes = collections.Counter(s["outcome"] for s in summaries)
    head = ["# Closing trace report", "", "episodes: %d, outcomes: %s" % (len(summaries), dict(outcomes)), ""]
    sections = (section_lock_gate, section_border, section_memory_refusals, section_escape, section_near_miss, section_stop_rooms, section_spl)
    return "\n".join(head + [line for section in sections for line in section(summaries) + [""]])
