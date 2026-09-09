#!/usr/bin/env python3
"""Message -> record conversion for ``nav_debug_recorder_node``.

Pure functions: no ROS, no file IO, no state. Each one turns one subscribed
message into the plain dict the recorder appends, so the node stays wiring and
the record shapes stay readable and testable off-board.

Field names match :mod:`sparx_agency.tasks.planning.nav_debug.frame` exactly --
the offline session loader reads them by name, so a rename here silently empties
a panel there.

Two nav chains feed this recorder. The XTEND click-to-fly chain narrates in the
A* vocabulary (``boxed in``, ``blockage``, ``periodic replan``); the Sphera
chain is FALCON's own exploration FSM, which says none of those words and
instead reports ``plan fail``, FSM transitions, frontier/viewpoint choices and
``exploration finished``. :func:`classify` covers both, because a run recorded
on either chain must bucket into the same vocabulary the offline player colours.

Python 3.8 compatible (runs in the FALCON Noetic container).
"""
import json
import math

import numpy as np

#: Reference speed (m/s) below which the setpoint counts as parked. ``traj_server``
#: republishes a trajectory's frozen endpoint with FRESH stamps, so "the reference
#: is not stale" does not imply "the reference is moving"; this separates them.
DEFAULT_MOVING_EPS = 0.05

#: FALCON's own ``/planning/replan`` verdicts (``std_msgs/Int32``), read off
#: ``exploration_fsm.cpp``: 0 while it plans a fresh trajectory, 1 when the live
#: trajectory was found in collision, 2 once exploration is finished (republished
#: every tick for as long as the FSM sits in FINISH).
REPLAN_VERDICTS = {
    0: ("fsm", "replan: FALCON is planning a new trajectory"),
    1: ("obstacle", "replan: collision detected on the live trajectory"),
    2: ("finish", "replan: exploration finished"),
}


#: Event buckets in priority order, the first matching keyword winning. The A*
#: buckets come first so an XTEND run classifies exactly as it always did. This
#: table is a deliberate copy of ``nav_debug.session._EVENT_BUCKETS``: the
#: recorder must classify identically to the offline loader, and it cannot
#: depend on that package being importable inside the Noetic container.
EVENT_BUCKETS = (
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


def classify(text):
    """Bucket a raw event string into a coarse ``kind``.

    Args:
        text: The publisher's own string, from any narrating topic.

    Returns:
        str: The first matching :data:`EVENT_BUCKETS` kind, else ``info``.
    """
    t = (text or "").lower()
    for kind, keywords in EVENT_BUCKETS:
        for keyword in keywords:
            if keyword in t:
                return kind
    return "info"


def replan_event(value):
    """FALCON's ``/planning/replan`` verdict -> ``(kind, text)``."""
    verdict = int(value)
    return REPLAN_VERDICTS.get(verdict, ("fsm", "replan: verdict %d" % verdict))


def reference_row(msg, now_s, moving_eps=DEFAULT_MOVING_EPS):
    """``quadrotor_msgs/PositionCommand`` -> the fields of ``frame.Reference``.

    Args:
        msg: One setpoint off ``/planning/pos_cmd``.
        now_s: Receipt time in the recorder's clock, seconds.
        moving_eps: Reference speed above which ``moving`` is True.

    Returns:
        dict: ``frame.Reference``'s fields, plus the acceleration and the
        trajectory flag, which have no field there but say why a reference
        moved (or why ``traj_server`` refused to emit a live one).
    """
    stamp = msg.header.stamp.to_sec()
    vel, acc = msg.velocity, msg.acceleration
    speed = math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z)
    return {
        "x": msg.position.x, "y": msg.position.y, "z": msg.position.z,
        "yaw": msg.yaw, "yaw_dot": msg.yaw_dot,
        "vx": vel.x, "vy": vel.y, "vz": vel.z,
        "ax": acc.x, "ay": acc.y, "az": acc.z,
        "age_s": max(0.0, now_s - stamp) if stamp > 0.0 else 0.0,
        "traj_id": int(msg.trajectory_id),
        "traj_flag": int(msg.trajectory_flag),
        "moving": bool(speed > moving_eps),
    }


def control_row(data, now_s, wall_s):
    """Decode the follower's control-trace payload into a row.

    The payload is already shaped as ``frame.Tracking`` + ``frame.ControlTerms``
    by its publisher, so it passes through verbatim; only the two clocks are
    lifted out, because the publisher's own stamps beat our receipt time.

    Args:
        data: The ``std_msgs/String`` payload (JSON object).
        now_s: Fallback ROS time if the payload carries none.
        wall_s: Fallback wall clock if the payload carries none.

    Returns:
        tuple: ``(t, wall, fields)``, or None if the payload is unusable.
    """
    try:
        payload = json.loads(data or "")
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    t = _num(payload.pop("t", None), now_s)
    wall = _num(payload.pop("wall", None), wall_s)
    return t, wall, payload


def thinking_text(data):
    """The narration line inside a ``/nav/thinking`` JSON payload.

    Falls back to the raw string: a malformed narration is still evidence, and
    a diagnostic must not drop it over a decode.
    """
    try:
        payload = json.loads(data or "")
    except (ValueError, TypeError):
        return str(data or "")
    if isinstance(payload, dict):
        return str(payload.get("text") or "")
    return str(data or "")


def path_xy(msg, max_points=0):
    """``nav_msgs/Path`` -> ``[[x, y], ...]`` in world metres.

    FALCON's executed path is a monotonically growing marker -- ``traj_server``
    appends a point per 100 Hz tick for the whole flight and republishes the
    whole vector -- so an unbounded copy of it, re-serialized into every route
    snapshot, costs O(snapshots x length). ``max_points`` decimates by a uniform
    stride (always keeping the last point, so the head of the path is exact) to
    bound both the record and the work done in the callback.

    Args:
        msg: The path message.
        max_points: Cap on the returned point count; 0 means no cap.

    Returns:
        The path as ``[[x, y], ...]``.
    """
    poses = msg.poses
    n = len(poses)
    if max_points and n > max_points:
        stride = (n + max_points - 1) // max_points
        kept = [poses[i] for i in range(0, n, stride)]
        if kept[-1] is not poses[-1]:
            kept.append(poses[-1])
        poses = kept
    return [[p.pose.position.x, p.pose.position.y] for p in poses]


def grid_from(msg):
    """``nav_msgs/OccupancyGrid`` -> (H, W) int8 array on the ROS convention."""
    return np.asarray(msg.data, np.int8).reshape(msg.info.height, msg.info.width)


def grid_counts(grid):
    """Occupancy census of a BEV grid, as ``frame.MapStats`` names it.

    A plan is only as good as the map under it: a grid that is 99% unknown
    explains a refusal to plan far better than any planner log line.
    """
    return {
        "occupied_cells": int(np.count_nonzero(grid > 0)),
        "free_cells": int(np.count_nonzero(grid == 0)),
        "unknown_cells": int(np.count_nonzero(grid < 0)),
    }


def _num(value, default):
    """``value`` as a float, or ``default`` when it is missing or not a number."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
