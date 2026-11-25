"""The planner's own room visits, read from a recording: when it arrived in a room and when it released it, which floor region a pose is in, and which steps give the peek frames."""
import re

import numpy as np

EVENT = re.compile(r"^\s*(\d+) (\w+)\((.*)\)$")
ROOM = re.compile(r"\broom=(\d+)")
PEEK_STEPS_IN = 4
PEEK_WINDOW = 30
MAX_REGION_GAP_M = 1.0
SAME_FLOOR_M = 1.0


def parse_events(text):
    """``(step, name, room or None)`` for each complete event line of the recording reviewer's ``events`` output."""
    found = []
    for line in text.splitlines():
        match = EVENT.match(line)
        if match:
            room = ROOM.search(match.group(3))
            found.append((int(match.group(1)), match.group(2), int(room.group(1)) if room else None))
    return found


def visit_intervals(events, last_step):
    """``(room, arrival step, end step)``: a visit ends at its room's release, at the next arrival, or at ``last_step``."""
    visits, open_visit = [], None
    for step, name, room in sorted(events, key=lambda e: e[0]):
        if name == "arrive" and room is not None:
            if open_visit:
                visits.append((open_visit[0], open_visit[1], step))
            open_visit = (room, step)
        elif name == "release" and open_visit and room == open_visit[0]:
            visits.append((open_visit[0], open_visit[1], step))
            open_visit = None
    if open_visit:
        visits.append((open_visit[0], open_visit[1], last_step))
    return visits


def peek_steps(arrival, end):
    """``(arrival step, step four actions in, steps of the first thirty actions of the visit)``, all within the visit."""
    inside = min(end, arrival + PEEK_STEPS_IN)
    return arrival, inside, list(range(arrival, min(end, arrival + PEEK_WINDOW) + 1))


def region_at(position, cells, regions):
    """The region of the floor cell nearest the habitat position ``(x, y, z)`` on its floor; None if none lies within ``MAX_REGION_GAP_M``."""
    cells = np.asarray(cells, dtype=float)
    near = np.abs(cells[:, 1] - position[1]) < SAME_FLOOR_M
    if not near.any():
        return None
    gap = np.hypot(cells[near, 0] - position[0], cells[near, 2] - position[2])
    best = int(np.argmin(gap))
    return int(np.asarray(regions)[near][best]) if gap[best] <= MAX_REGION_GAP_M else None


def pose_visits(region_by_step, min_steps):
    """``{region: (first step, last step)}`` of the first run of at least ``min_steps`` consecutive steps spent inside each region; shorter runs are passing through."""
    visits, start = {}, 0
    for step in range(1, len(region_by_step) + 1):
        if step == len(region_by_step) or region_by_step[step] != region_by_step[start]:
            region = region_by_step[start]
            if region is not None and region not in visits and step - start >= min_steps:
                visits[region] = (start, step - 1)
            start = step
    return visits
