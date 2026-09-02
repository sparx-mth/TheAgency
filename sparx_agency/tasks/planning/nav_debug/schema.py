"""The on-disk contract for a nav-debug run, shared by every writer and reader.

A Sphera run is recorded by **two** processes that cannot see each other's ROS
graph: a ROS1 recorder in the ``falcon`` Noetic container (the plan, the
reference and the command we ask for) and a ROS2 recorder in the ``it`` Foxy
container (what the drone was actually told, and what it actually did). Neither
side can subscribe to the other -- ``bridge.yaml`` carries neither the actuator
topics nor the Sphera ground truth -- so the join happens offline, here.

Every row written by either side carries ``wall`` (``time.time()``, host clock)
as well as its own ROS ``t``. Both containers share the host kernel clock, so
``wall`` is the join key and needs no offset estimation; ``t`` is kept so a run
can still be read against ROS-time artefacts like the certainty CSV.

This module is pure stdlib and Python 3.8 compatible on purpose: it is imported
by the Noetic recorder, by the Foxy recorder and by the 3.12 offline player.
"""
from __future__ import annotations

SCHEMA_VERSION = 2

# ── run-folder layout ────────────────────────────────────────────────────────
# Written by the ROS1 recorder (falcon container).
TELEMETRY_FILE = "telemetry.jsonl"     # pose + the command we ask for
REFERENCE_FILE = "reference.jsonl"     # the point being chased, at rate
CONTROL_FILE = "control.jsonl"         # the tracker's verdict + why
EVENTS_FILE = "events.jsonl"           # replan / FSM / blockage events
MAPPING_FILE = "mapping.jsonl"         # map-quality stats per update
ROUTES_DIR = "routes"                  # <ms>.json route layers
BEV_DIR = "bev"                        # <ms>.npy occupancy + <ms>.json geometry
BEV_CONF_DIR = "bev_conf"              # <ms>.npy per-cell confidence
MANIFEST_FILE = "manifest.json"

# Written by the ROS2 recorder (it container), under this subdirectory.
ROS2_DIR = "ros2"
ACTUATOR_FILE = "actuator.jsonl"       # cmd_nav request + ManualControl sent
TRUTH_FILE = "truth.jsonl"             # Sphera ground truth + vehicle state
AXIS_TRACE_FILE = "axis_trace.jsonl"   # twist-adapter per-axis internals
ALTITUDE_FILE = "altitude.jsonl"       # the vertical lane, end to end

# ── diagnostic topics (all additive, all std_msgs/String JSON) ───────────────
# JSON on a String topic rather than a custom msg: the three publishers live in
# three containers across two ROS versions, and only the vendor images can build
# vendor messages. A String costs no build step anywhere.
CONTROL_TRACE_TOPIC = "/nav_debug/control_trace"        # ROS1, the follower
AXIS_TRACE_TOPIC = "/R1/nav_debug/axis_trace"           # ROS2, the twist adapter
ALTITUDE_TRACE_TOPIC = "/R1/nav_debug/altitude_trace"   # ROS2, the command unit


def row(t, wall, **fields):
    """One jsonl row: the two clocks first, then the payload.

    Args:
        t: The writer's own ROS time, seconds.
        wall: Host wall clock (``time.time()``), seconds -- the cross-recorder
            join key.
        **fields: The record's payload.

    Returns:
        A dict ready for ``json.dumps``, with floats rounded to keep a
        20 Hz-for-an-hour recording readable and small.
    """
    out = {"t": _r(t, 3), "wall": _r(wall, 3)}
    for key, value in fields.items():
        out[key] = _round(value)
    return out


def _round(value):
    """Round floats (recursively through lists/dicts) to 4 decimals."""
    if isinstance(value, float):
        return _r(value, 4)
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(v) for v in value]
    return value


def _r(value, digits):
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None
