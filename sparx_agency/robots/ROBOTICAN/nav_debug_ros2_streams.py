"""Vendor ROS 2 messages -> the rows :mod:`nav_debug.frame` expects.

Pure functions, no ROS imports: the recorder node owns the subscriptions and
this module owns the translation, so the mapping from a vendor message to an
``Actuator``/``Truth`` field -- and the jsonl row built from it -- can be read
and tested without a ROS graph.

Nothing here corrects anything. Positions are not handedness-flipped, roll/pitch
signs are not normalised, and no value is filtered -- a recording that applied a
correction cannot un-apply it once the correction turns out to be wrong. The one
conversion made is battery fraction to percent, which the field name demands.

Python 3.8 compatible: this runs under ROS 2 Foxy in the vendor container.
"""
from __future__ import annotations

import json
import math

from sparx_agency.robots.ROBOTICAN.nav_debug_ros2_imports import schema

#: Streams whose arrival makes an ``actuator.jsonl`` row worth writing.
ACTUATOR_AGE_STREAMS = ("cmd_nav", "manual")

#: Streams whose age is reported alongside the ``truth.jsonl`` values.
TRUTH_AGE_STREAMS = ("velocity", "attitude", "sphera", "state", "status")

#: ``RoosterState.flight_mode`` values, mirroring ``helpers.rooster_unit``'s
#: constants. Duplicated rather than imported so a recorder never drags a
#: flight-control module (and its publishers) into its own process.
FLIGHT_MODES = {
    0: "none", 1: "ground_roll", 2: "manual", 3: "position",
    4: "altitude", 5: "acro", 6: "stabilized",
}


def finite(value):
    """``value`` as a float, or ``None`` when it is missing, NaN or infinite.

    ``RoosterState.ranger`` is legitimately ``inf`` before the first rangefinder
    sample, and ``json.dumps`` would emit a bare ``Infinity`` that no strict JSON
    reader accepts -- one such value makes the whole run unreadable.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def flight_mode_name(value):
    """Flight-mode integer -> the short name ``frame.Truth.flight_mode`` holds."""
    try:
        code = int(value)
    except (TypeError, ValueError):
        return ""
    return FLIGHT_MODES.get(code, str(code))


def cmd_nav_fields(text):
    """``/R1/cmd_nav`` JSON -> ``frame.Actuator``'s request half.

    Args:
        text: The ``std_msgs/String`` payload.

    Returns:
        ``cmd_nav`` (the ``[x, y, r]`` a ``move`` asked for, else ``None``),
        ``action`` and ``value``. A payload that will not parse is kept under
        ``raw`` instead of being dropped -- a malformed publisher is itself the
        finding.
    """
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return {"cmd_nav": None, "action": None, "value": None, "raw": text}
    if not isinstance(payload, dict):
        return {"cmd_nav": None, "action": None, "value": None, "raw": text}
    axes = payload.get("axes")
    triple = None
    if isinstance(axes, dict):
        triple = [finite(axes.get(axis)) for axis in ("x", "y", "r")]
    return {"cmd_nav": triple,
            "action": payload.get("action"),
            "value": finite(payload.get("value"))}


def manual_fields(msg):
    """``fcu_driver_interfaces/ManualControl`` -> ``frame.Actuator``'s sent half.

    This is the last message before Sphera's physics, and the only one that
    carries the ``z`` axis the altitude loop owns.
    """
    return {"manual": [finite(msg.x), finite(msg.y), finite(msg.z), finite(msg.r)],
            "buttons": int(msg.buttons)}


def velocity_fields(msg):
    """``/R1/velocity_truth`` (``geometry_msgs/TwistStamped``) -> achieved world
    velocity, which is what ``frame.Truth.vx/vy/vz`` mean."""
    linear, angular = msg.twist.linear, msg.twist.angular
    return {"vx": finite(linear.x), "vy": finite(linear.y), "vz": finite(linear.z),
            "yaw_rate": finite(angular.z)}


def attitude_fields(msg):
    """``/R1/attitude_rpy`` (``geometry_msgs/Vector3``) -> roll/pitch/yaw, rad.

    The pose the whole stack consumes is yaw-only by contract, so this topic is
    the only place real roll and pitch exist. Signs are unverified upstream;
    they are recorded raw and consumers should compare magnitudes.
    """
    return {"roll": finite(msg.x), "pitch": finite(msg.y), "yaw": finite(msg.z)}


def sphera_fields(msg):
    """``sphera_common_interfaces/SpheraPawnState`` -> the simulator's own pose.

    ``msg.velocity`` is deliberately not read: the field is declared m/s but is
    all-zero in this build, so achieved velocity must come from
    ``/R1/velocity_truth`` (which ``rooster_ground_truth_localization`` derives)
    and never from here.
    """
    location, rotation = msg.location, msg.rotation
    return {"x": finite(location.x), "y": finite(location.y), "z": finite(location.z),
            "roll": finite(rotation.roll), "pitch": finite(rotation.pitch),
            "yaw": finite(rotation.yaw)}


def state_fields(msg):
    """``rooster_manager_interfaces/RoosterState`` -> ``frame.Truth``'s vehicle
    fields. ``percentage`` is a fraction in [0, 1]; ``battery_pct`` is percent."""
    fraction = finite(msg.percentage)
    return {"battery_pct": None if fraction is None else fraction * 100.0,
            "armed": bool(msg.armed), "airborne": bool(msg.airborne),
            "flight_mode": flight_mode_name(msg.flight_mode),
            "ranger_m": finite(msg.ranger)}


def trace_fields(text):
    """A diagnostic String trace, made safe to merge into ``schema.row()``.

    The axis and altitude traces are appended verbatim -- their publishers own
    their own field names. The one thing that cannot pass through is a ``t`` or
    ``wall`` key: ``schema.row`` owns those as the cross-recorder join keys, and
    a publisher sending its own would silently overwrite them, so they are kept
    under ``src_t``/``src_wall``.

    Args:
        text: The ``std_msgs/String`` payload.

    Returns:
        Keyword fields for ``schema.row``; unparseable input lands in ``raw``.
    """
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return {"raw": text}
    if not isinstance(payload, dict):
        return {"raw": payload}
    fields = {}
    for key, value in payload.items():
        fields["src_" + key if key in ("t", "wall") else key] = value
    return fields


def actuator_row(t, wall, held, ages):
    """One ``actuator.jsonl`` row: what was asked for, and what was sent.

    ``cmd_nav`` is the twist adapter's request; ``manual`` is the
    ``ManualControl`` the command unit really published, which is the only
    message Sphera's physics acts on. They differ whenever the altitude loop
    writes the throttle axis or a second publisher injects a command.

    Args:
        t: Recorder ROS time, seconds.
        wall: Host wall clock, the cross-recorder join key.
        held: The sample-and-hold map, keyed by stream name.
        ages: Seconds since each stream last arrived; ``None`` means never.

    Returns:
        A ``schema.row`` dict on ``frame.Actuator``'s field names.
    """
    cmd = held.get("cmd_nav") or {}
    manual = held.get("manual") or {}
    fields = {"cmd_nav": cmd.get("cmd_nav"), "action": cmd.get("action"),
              "value": cmd.get("value"), "manual": manual.get("manual"),
              "buttons": manual.get("buttons"),
              "cmd_nav_age_s": ages.get("cmd_nav"),
              "manual_age_s": ages.get("manual")}
    if cmd.get("raw") is not None:      # only a payload that would not parse
        fields["raw"] = cmd["raw"]
    return schema.row(t, wall, **fields)


def truth_row(t, wall, held, ages):
    """One ``truth.jsonl`` row: achieved velocity, attitude and vehicle state.

    The flat fields match ``frame.Truth``; ``sphera`` carries the simulator's
    raw pawn pose and ``ages`` says how old each contribution is, because
    "velocity is 0.0" and "the velocity publisher died" are indistinguishable
    in a value column.

    Args:
        t: Recorder ROS time, seconds.
        wall: Host wall clock, the cross-recorder join key.
        held: The sample-and-hold map, keyed by stream name.
        ages: Seconds since each stream last arrived; ``None`` means never.

    Returns:
        A ``schema.row`` dict on ``frame.Truth``'s field names.
    """
    velocity = held.get("velocity") or {}
    attitude = held.get("attitude") or {}
    state = held.get("state") or {}
    status = held.get("status") or {}
    return schema.row(
        t, wall,
        vx=velocity.get("vx"), vy=velocity.get("vy"), vz=velocity.get("vz"),
        yaw_rate=velocity.get("yaw_rate"),
        roll=attitude.get("roll"), pitch=attitude.get("pitch"),
        yaw=attitude.get("yaw"),
        battery_pct=state.get("battery_pct"), armed=state.get("armed"),
        airborne=state.get("airborne"), flight_mode=state.get("flight_mode", ""),
        ranger_m=state.get("ranger_m"), status=status.get("status", ""),
        sphera=held.get("sphera"),
        ages=dict((name, ages.get(name)) for name in TRUTH_AGE_STREAMS))
