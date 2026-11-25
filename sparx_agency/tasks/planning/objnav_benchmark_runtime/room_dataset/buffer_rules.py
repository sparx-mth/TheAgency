"""Choosing a few views from a buffer of frames kept along the planner's own path: the earliest, evenly spaced in time, most floor, most different headings, or random."""
import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.view_selection import MAX_NEAR_SHARE, MIN_IN_ROOM_SHARE


def usable(frames):
    return {n: f for n, f in frames.items() if f["near"] <= MAX_NEAR_SHARE and f["inside"] >= MIN_IN_ROOM_SHARE}


def angle_gap(a, b):
    gap = abs(a - b) % (2 * np.pi)
    return min(gap, 2 * np.pi - gap)


def buffer_picks(frames, rule, count, rng):
    """The names of up to ``count`` usable frames under ``rule`` (``first``, ``even_time``, ``floor``, ``heading_spread``, ``random``), in the order the rule takes them."""
    good = usable(frames)
    by_time = sorted(good, key=lambda n: (good[n]["step"], n))
    if rule == "first":
        return by_time[:count]
    if rule == "even_time":
        spots = np.linspace(0, len(by_time) - 1, min(count, len(by_time))).round().astype(int) if by_time else []
        return [by_time[i] for i in spots]
    if rule == "random":
        return [by_time[i] for i in rng.permutation(len(by_time))[:count]]
    by_floor = sorted(good, key=lambda n: (-good[n]["floor_share"], n))
    if rule == "floor":
        return by_floor[:count]
    picks = by_floor[:1]
    while by_floor and len(picks) < min(count, len(by_floor)):
        rest = [n for n in by_floor if n not in picks]
        picks.append(max(rest, key=lambda n: min(angle_gap(good[n]["yaw"], good[p]["yaw"]) for p in picks)))
    return picks
