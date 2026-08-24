"""Turn InternVLA-N1's answer into a body-frame trajectory a follower can fly.

InternVLA-N1's System 1 predicts, per call, a fan of candidate trajectories that
:func:`~sparx_agency.core.planning.vlas.internvla_n1.trt.postprocess.mean_path`
integrates into a single ``(T + 1, 2)`` XY path in the robot **body frame** --
``+x`` forward, ``+y`` left, metres, the robot at the origin. That path is the
exact shape NavDP emits (``(T, >=2)`` ``(forward, left)``), which is why it can be
flown the same way: anchored at the pose it was asked from and pursued as a route
(see :mod:`~sparx_agency.core.planning.vlas.common.plan_commit`).

This module is the single seam that produces that body trajectory, from whichever
form the server hands back:

* **the continuous path** -- the faithful output. When the server returns the
  System-1 trajectory (the candidate deltas, or the integrated mean path), that
  *is* the trajectory to fly, and :func:`trajectory_from_deltas` /
  :func:`trajectory_from_path` shape it into ``(N, 3)`` ``(forward, left, yaw)``.
* **a single discrete action** -- what the deployed VLN-CE agent server returns
  today (one of STOP / FORWARD / TURN_LEFT / TURN_RIGHT). It carries no XY curve,
  so :func:`trajectory_from_action` renders it as one short, followable body step
  in the action's own direction: a forward action advances, a turn action places
  a waypoint one step ahead rotated by the turn angle, so an XY pursuit follower
  curves toward it instead of stalling on a zero-length path.

The continuous path is preferred wherever the server exposes it; the action step
is the fallback that keeps the aircraft moving against a server that only speaks
the discrete alphabet.

Numpy-only and Python-3.8 clean: nothing here is heavier than numpy, so importing
it is free and it survives the Noetic container the rest of ``core`` must.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from sparx_agency.core.planning.vlas.internvla_n1.trt import postprocess

#: Action indices, from ``postprocess`` (matching ``InternVLAN1Net.actions2idx``).
STOP = postprocess.STOP
FORWARD = postprocess.FORWARD
TURN_LEFT = postprocess.TURN_LEFT
TURN_RIGHT = postprocess.TURN_RIGHT

#: Metres a single MOVE_FORWARD action advances (VLN-CE / InternVLA-N1 default;
#: matches ``postprocess.STEP_SIZE_M``).
STEP_SIZE_M = postprocess.STEP_SIZE_M
#: Degrees a single TURN_LEFT / TURN_RIGHT action rotates (matches
#: ``postprocess.TURN_ANGLE_DEG``).
TURN_ANGLE_DEG = postprocess.TURN_ANGLE_DEG

#: Keys a server might carry a continuous System-1 trajectory under, most
#: specific first. Searched top-level then inside ``action[0]``.
_TRAJECTORY_KEYS = ("trajectory", "trajectories", "traj", "mean_path", "path", "waypoints")


def heading_column(xy):
    # type: (np.ndarray) -> np.ndarray
    """Append a yaw channel to an XY path: each vertex looks along its next step.

    The last vertex keeps the heading of the segment that reached it -- a route
    has no tangent past its own end, and holding the final heading is what keeps
    the aircraft's nose still as it decelerates onto the goal rather than
    snapping east.

    Args:
        xy: ``(N, 2)`` body-frame ``(forward, left)`` path, ``N >= 1``.

    Returns:
        ``(N, 3)`` float64 ``(forward, left, yaw)``; yaw is radians CCW from
        ``+x``.

    Raises:
        ValueError: ``xy`` is not ``(N, 2)`` with at least one row.
    """
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] < 1:
        raise ValueError("heading_column expects (N, 2) with N >= 1, got %r"
                         % (xy.shape,))
    yaw = np.zeros((xy.shape[0],), dtype=np.float64)
    if xy.shape[0] >= 2:
        deltas = np.diff(xy, axis=0)
        segment_yaw = np.arctan2(deltas[:, 1], deltas[:, 0])
        # A degenerate (zero-length) segment has no heading; carry the previous
        # one forward rather than the atan2(0, 0) == 0 that points east.
        lengths = np.hypot(deltas[:, 0], deltas[:, 1])
        last = 0.0
        for i in range(segment_yaw.shape[0]):
            if lengths[i] > 1e-9:
                last = float(segment_yaw[i])
            yaw[i] = last
        yaw[-1] = yaw[-2]
    return np.concatenate([xy, yaw[:, None]], axis=1)


def trajectory_from_deltas(deltas):
    # type: (np.ndarray) -> np.ndarray
    """Integrate System-1 candidate deltas into the body trajectory to fly.

    This is the faithful output: the same mean-of-paths
    :func:`~sparx_agency.core.planning.vlas.internvla_n1.trt.postprocess.mean_path`
    the deployed agent turns into actions, kept as a curve instead.

    Args:
        deltas: ``(B, T, >=2)`` candidate per-step deltas from System 1.

    Returns:
        ``(T + 1, 3)`` body-frame ``(forward, left, yaw)`` starting at the
        origin.
    """
    return heading_column(postprocess.mean_path(deltas))


def trajectory_from_path(path):
    # type: (Any) -> np.ndarray
    """Shape an already-integrated body path into ``(N, 3)``.

    Accepts a ``(N, >=2)`` array, or a batched ``(B, N, >=2)`` from which the
    first item is taken (the server's single-robot batch), exactly as NavDP's
    ``best_trajectory`` takes ``result["trajectory"][0]``.

    Args:
        path: body-frame ``(forward, left[, yaw])`` waypoints, optionally
            batched.

    Returns:
        ``(N, 3)`` ``(forward, left, yaw)`` -- an existing yaw column is kept,
        otherwise one is derived from the route tangents.

    Raises:
        ValueError: the payload is not a usable ``(N, >=2)`` path.
    """
    array = np.asarray(path, dtype=np.float64)
    if array.ndim == 3:
        array = array[0]
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 2:
        raise ValueError("trajectory_from_path expects (N, >=2) waypoints, got %r"
                         % (np.shape(path),))
    if array.shape[1] >= 3:
        return np.concatenate([array[:, :2], array[:, 2:3]], axis=1)
    return heading_column(array[:, :2])


def trajectory_from_action(action_index, step_m=STEP_SIZE_M, turn_deg=TURN_ANGLE_DEG):
    # type: (int, float, float) -> Optional[np.ndarray]
    """Render one discrete VLN action as a short, followable body step.

    The deployed agent server answers with a single action, not a curve. A
    forward action advances ``step_m``; a turn action places one waypoint
    ``step_m`` ahead rotated by ``turn_deg`` in the turn's direction, so an XY
    pursuit follower curves toward it rather than stalling on a zero-length
    path. STOP has no motion and returns ``None``.

    **A turn is an approximation, and knowing which way it errs matters.**
    Upstream a turn is a pure rotation: ``trajectory_to_discrete_actions_
    close_to_goal`` (``vln_utils.py``) advances ``pos`` only on action ``1``
    and turns change ``yaw`` alone, by a hard-coded ``turn_angle_deg=15``. A
    rotation is not a path, though, and everything downstream of here -- the
    plan-commit executor, the pure-pursuit follower -- consumes paths, so a
    zero-length one is rejected as TOO_SHORT and re-inferred on the next tick
    for ever. The compromise is a short arc: the aircraft rotates toward the
    bent waypoint and then flies the reach. Set the follower's
    ``stop_turn_rad`` BELOW ``turn_deg`` so it rotates first and translates
    second; above it, the turn is flown as a sideways crab and the aircraft
    never actually looks anywhere new.

    Args:
        action_index: ``0`` STOP, ``1`` FORWARD, ``2`` TURN_LEFT, ``3`` TURN_RIGHT.
        step_m: forward reach of the rendered step, metres.
        turn_deg: heading offset a turn action bends the step by, degrees.

    Returns:
        ``(2, 3)`` body-frame ``(forward, left, yaw)`` from the origin, or
        ``None`` for STOP / an unknown action.
    """
    index = int(action_index)
    if index == FORWARD:
        heading = 0.0
    elif index == TURN_LEFT:
        heading = np.deg2rad(float(turn_deg))
    elif index == TURN_RIGHT:
        heading = -np.deg2rad(float(turn_deg))
    else:  # STOP or unknown
        return None
    reach = float(step_m)
    target = np.array([reach * np.cos(heading), reach * np.sin(heading)],
                      dtype=np.float64)
    xy = np.stack([np.zeros(2, dtype=np.float64), target], axis=0)
    return heading_column(xy)


def trajectory_from_response(raw, step_m=STEP_SIZE_M, turn_deg=TURN_ANGLE_DEG):
    # type: (Dict[str, Any], float, float) -> Optional[np.ndarray]
    """Find the body trajectory in a raw server response, preferring the curve.

    Searches the known continuous-trajectory keys (top-level, then inside
    ``action[0]``) and shapes the first one found. A caller that wants the
    discrete fallback should read ``action`` itself and call
    :func:`trajectory_from_action`; this function returns ``None`` when the
    response carries no continuous path, so the caller can fall back cleanly.

    Args:
        raw: the parsed server JSON (``StepResponse.raw_response``).
        step_m: unused here; accepted so the two producers share a signature.
        turn_deg: unused here; accepted so the two producers share a signature.

    Returns:
        ``(N, 3)`` body-frame trajectory, or ``None`` if the response holds no
        continuous path.
    """
    if not isinstance(raw, dict):
        return None
    candidates = [raw]  # type: List[Dict[str, Any]]
    inner = raw.get("action")
    if isinstance(inner, list) and inner and isinstance(inner[0], dict):
        candidates.append(inner[0])
    for holder in candidates:
        for key in _TRAJECTORY_KEYS:
            value = holder.get(key)
            if value is None:
                continue
            try:
                return trajectory_from_path(value)
            except (ValueError, TypeError):
                continue
    return None


