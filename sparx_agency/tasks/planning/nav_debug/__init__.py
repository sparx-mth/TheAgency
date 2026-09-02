"""Offline (and, later, live) visual debugging for the FALCON navigation stack.

Loads a recorded run -- the per-tick certainty CSV plus the BEV maps, the three
route layers (raw A* -> corrected -> final flown) and the replan events written
by :mod:`nav_debug_recorder_node` -- and renders, for any moment, one debug
screen that answers "what did A* plan, what did the drone want to do, and why":

  * the BEV map with the raw / corrected / final routes, the target waypoint, the
    drone pose + a localization trail and the drift vector, plus a banner naming
    the active replan reason (time / rotation / blocking) and localization state;
  * a telemetry panel with TWO ROLL/PITCH/YAW gauge stacks -- the command WE send
    (``cmd_vel``) and the command the converter sends the DRONE (``cmd_nav`` axis
    counts) -- confidence, the "why", and short command/confidence history strips.

ROS-free and drone-agnostic: the offline player runs on the dev PC and the same
renderer can be driven live by a viewer node inside the Noetic container. Kept
Python 3.8 compatible for that reason.

The offline loader lives in :mod:`.session` (assembly) over :mod:`.timeline`
(the frame spine), :mod:`.sources` (jsonl lanes + the cross-recorder clock) and
:mod:`.records` (row -> dataclass). A Sphera run adds FALCON's exploration
lanes -- the reference being chased, the tracker's verdict and the terms behind
it, the map-quality counters, and, joined on the host wall clock from the ROS2
recorder, what the drone was told and what it actually did.

Importing this package must stay cheap: :mod:`.schema` is the on-disk contract
shared with the two ROS recorders, and the Foxy container that runs one of them
has no numpy or cv2. So the names below are re-exported **lazily** (PEP 562) --
``from ... import schema`` pulls in stdlib only, while ``from ... import
NavSession`` still works and loads numpy on demand.
"""
import importlib

# name -> the submodule that defines it, resolved on first attribute access.
_EXPORTS = {
    "Actuator": "frame", "Altitude": "frame", "AxisTrace": "frame",
    "BevMap": "frame", "ControlTerms": "frame", "Drift": "frame",
    "GaugeScales": "frame", "MapStats": "frame", "NavFrame": "frame",
    "Quality": "frame", "Reference": "frame", "ReplanEvent": "frame",
    "Routes": "frame", "Tracking": "frame", "Truth": "frame",
    "NavSession": "session", "classify_event": "session",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    """Resolve a re-exported name on first use (PEP 562)."""
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    value = getattr(importlib.import_module("%s.%s" % (__name__, module)), name)
    globals()[name] = value      # cache, so later lookups skip __getattr__
    return value


def __dir__():
    return sorted(list(globals()) + __all__)
