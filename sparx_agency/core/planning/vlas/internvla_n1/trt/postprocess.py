"""Turn System-1's trajectory candidates into the actions the robot executes.

This is ``internnav/model/utils/vln_utils.py::traj_to_actions``, reproduced in
numpy with no torch. It is the **decision function**: everything upstream of it
is tensors, and everything downstream is a command.

Two details are easy to get wrong and both change behaviour:

* the 32 candidates are **averaged, not selected** -- and averaged *after*
  integration, so the mean of the paths, not the path of the mean deltas;
* ``dx``/``dy`` are divided by 4 before integrating. That factor is the
  normalisation the policy was trained with, and dropping it scales every
  distance by 4.

The action alphabet is upstream's: 1 forward, 2 turn left, 3 turn right. 0 is
STOP and is filtered out by the caller before the queue is built.

Numpy-only, and Python-3.8 clean.
"""
from __future__ import annotations

#: Metres advanced by one forward action.
STEP_SIZE_M = 0.25
#: Degrees turned by one turn action.
TURN_ANGLE_DEG = 15.0
#: How far along the path to aim, in path indices.
LOOKAHEAD = 4
#: Stop walking the path once within this distance of its end.
GOAL_TOLERANCE_M = 0.2
#: ``traj_to_actions`` divides dx/dy by this before integrating.
DELTA_SCALE = 4.0
#: Actions the agent keeps from one System-1 call (``S1Output(idx=...[:4])``).
ACTIONS_KEPT = 4

#: Action indices, matching ``InternVLAN1Net.actions2idx``.
STOP, FORWARD, TURN_LEFT, TURN_RIGHT = 0, 1, 2, 3


def mean_path(deltas):
    """Integrate candidate deltas into the single path the agent acts on.

    Args:
        deltas: ``(B, T, 3)`` predicted per-step deltas; only the first two
            channels are positional, the third is a heading difference upstream
            does not use for position.

    Returns:
        numpy.ndarray of shape ``(T + 1, 2)``: the mean XY path, starting at the
        origin.

    Raises:
        ValueError: if ``deltas`` is not 3-D with at least 2 channels.
    """
    import numpy as np

    array = np.asarray(deltas, dtype=np.float64)
    if array.ndim != 3 or array.shape[-1] < 2:
        raise ValueError(
            "mean_path expects (B, T, >=2) deltas, got %s" % (array.shape,))
    xy = array[:, :, :2] / DELTA_SCALE
    paths = np.zeros((xy.shape[0], xy.shape[1] + 1, 2), dtype=np.float64)
    paths[:, 1:] = np.cumsum(xy, axis=1)
    return paths.mean(axis=0)


def discrete_actions(path, max_actions=None):
    """Walk a path into discrete actions, as upstream does.

    Args:
        path: ``(T + 1, 2)`` XY path from :func:`mean_path`.
        max_actions: stop after this many actions. Upstream's loop has **no
            iteration bound**: a path whose points do not approach its own end
            keeps the ``while`` going. The caller only ever consumes the first
            four, so bounding it costs nothing and removes a hang.

    Returns:
        List[int] of action indices.
    """
    import numpy as np

    path = np.asarray(path, dtype=np.float64)
    limit = 4 * len(path) if max_actions is None else int(max_actions)
    turn = np.deg2rad(TURN_ANGLE_DEG)
    actions = []
    yaw = 0.0
    position = path[0]
    goal = path[-1]

    def normalize(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    while len(actions) < limit:
        if np.linalg.norm(position - goal) <= GOAL_TOLERANCE_M:
            break
        nearest = int(np.argmin(np.linalg.norm(path - position, axis=1)))
        target = path[min(nearest + LOOKAHEAD, len(path) - 1)]
        direction = target - position
        if np.linalg.norm(direction) < 1e-6:
            break
        delta_yaw = normalize(np.arctan2(direction[1], direction[0]) - yaw)
        turns = int(round(delta_yaw / turn))
        if turns > 0:
            actions.extend([TURN_LEFT] * turns)
        elif turns < 0:
            actions.extend([TURN_RIGHT] * (-turns))
        yaw = normalize(yaw + turns * turn)
        advanced = position + STEP_SIZE_M * np.array([np.cos(yaw), np.sin(yaw)])
        if np.linalg.norm(advanced - goal) > np.linalg.norm(position - goal):
            break
        actions.append(FORWARD)
        position = advanced
    return actions


def action_queue(deltas, keep=ACTIONS_KEPT):
    """The full decision: candidate deltas in, executable action queue out.

    Args:
        deltas: ``(B, T, 3)`` candidate trajectory deltas from System 1.
        keep: how many actions the agent queues.

    Returns:
        List[int]: at most ``keep`` actions, STOP filtered out, the first of
        which is the command executed immediately.
    """
    actions = discrete_actions(mean_path(deltas))
    return [a for a in actions if a != STOP][:int(keep)]
