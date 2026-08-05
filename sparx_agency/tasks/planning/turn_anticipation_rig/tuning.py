"""The controller tuning the rig flies: whatever the drone is flying today.

Read out of ``tasks/planning/falcon/config/mission.yaml`` rather than copied
into this file, because a rig that quietly flies last month's gains proves
nothing about this month's drone. Only the dials that change the *shape* of a
turn are mapped — speeds, yaw caps, capture radii, the force envelope and the
cross-track loop. The confidence, blockage and escape machinery is left at its
defaults: none of it is exercised here (the rig gives a perfect pose and puts no
walls in the way), and mapping it would only be more to keep in step.

A missing or unreadable mission.yaml is not an error: the core defaults are a
valid controller, and the run says which set it used.
"""
from __future__ import annotations

import math
import pathlib
from typing import Any, Dict, Optional

from sparx_agency.core.planning.trackers.drift_pid import (
    DriftPidParams, EnvelopeParams, PidGains, YawLookaheadParams,
)

#: The deployed FALCON mission configuration (see the module docstring).
MISSION_YAML = (pathlib.Path(__file__).resolve().parents[2]
                / "planning" / "falcon" / "config" / "mission.yaml")


def deployed_dials():
    # type: () -> Dict[str, Any]
    """The ``dp_*`` block of mission.yaml, or ``{}`` if it cannot be read."""
    try:
        import yaml
    except ImportError:
        return {}
    if not MISSION_YAML.exists():
        return {}
    try:
        loaded = yaml.safe_load(MISSION_YAML.read_text()) or {}
    except Exception:               # a malformed file is not worth a traceback
        return {}
    launch = loaded.get("launch") or {}
    return dict((k, v) for k, v in launch.items() if k.startswith("dp_"))


def controller_params(yaw_lookahead=None, dials=None):
    # type: (Optional[YawLookaheadParams], Optional[Dict[str, Any]]) -> DriftPidParams
    """Build the controller tuning, with the anticipation configured as asked.

    Args:
        yaw_lookahead: The anticipation's own params. None leaves it off.
        dials: Override the mission.yaml read (used by the tests).

    Returns:
        Validated controller params.
    """
    d = deployed_dials() if dials is None else dials
    default = DriftPidParams()

    def num(key, fallback):
        try:
            return float(d[key])
        except (KeyError, TypeError, ValueError):
            return float(fallback)

    def deg(key, fallback_rad):
        try:
            return math.radians(float(d[key]))
        except (KeyError, TypeError, ValueError):
            return float(fallback_rad)

    envelope = EnvelopeParams(
        max_vx=num("dp_max_vx", default.envelope.max_vx),
        max_vx_back=num("dp_max_vx_back", default.envelope.max_vx_back),
        max_vy=num("dp_max_vy", default.envelope.max_vy),
        max_wz=num("dp_max_wz", default.envelope.max_wz),
        max_translation=num("dp_max_translation", default.envelope.max_translation),
        combined_effort=num("dp_combined_effort", default.envelope.combined_effort),
        min_vx=num("dp_min_vx", default.envelope.min_vx),
        min_vy=num("dp_min_vy", default.envelope.min_vy),
        min_wz=deg("dp_min_wz_deg", default.envelope.min_wz),
        accel_xy=num("dp_accel_xy", default.envelope.accel_xy),
        decel_xy=num("dp_decel_xy", default.envelope.decel_xy),
        accel_wz=num("dp_accel_wz", default.envelope.accel_wz),
        decel_wz=num("dp_decel_wz", default.envelope.decel_wz),
    )
    lateral = PidGains(
        kp=num("dp_lat_kp", default.lateral_pid.kp),
        ki=num("dp_lat_ki", default.lateral_pid.ki),
        kd=num("dp_lat_kd", default.lateral_pid.kd),
        i_limit=num("dp_lat_i_limit", default.lateral_pid.i_limit),
        d_tau_s=num("dp_lat_d_tau_s", default.lateral_pid.d_tau_s),
        deadband=num("dp_lat_deadband_m", default.lateral_pid.deadband),
        out_limit=num("dp_lat_max", default.lateral_pid.out_limit),
    )
    yaw = PidGains(
        kp=num("dp_yaw_kp", default.yaw_pid.kp),
        ki=num("dp_yaw_ki", default.yaw_pid.ki),
        kd=num("dp_yaw_kd", default.yaw_pid.kd),
        i_limit=num("dp_yaw_i_limit", default.yaw_pid.i_limit),
        d_tau_s=num("dp_yaw_d_tau_s", default.yaw_pid.d_tau_s),
        deadband=deg("dp_yaw_deadband_deg", default.yaw_pid.deadband),
        out_limit=num("dp_yaw_max", default.yaw_pid.out_limit),
    )
    return DriftPidParams(
        cruise_speed=num("dp_cruise_speed", default.cruise_speed),
        cruise_speed_straight=num("dp_cruise_straight",
                                  default.cruise_speed_straight),
        approach_yaw_rate=num("dp_approach_yaw_rate", default.approach_yaw_rate),
        track_yaw_rate=num("dp_track_yaw_rate", default.track_yaw_rate),
        pos_radius=num("dp_pos_radius", default.pos_radius),
        slow_radius=num("dp_slow_radius", default.slow_radius),
        arrive_speed_min=num("dp_arrive_speed_min", default.arrive_speed_min),
        lookahead_m=num("dp_lookahead_m", default.lookahead_m),
        yaw_engage_rad=deg("dp_yaw_engage_deg", default.yaw_engage_rad),
        yaw_release_rad=deg("dp_yaw_release_deg", default.yaw_release_rad),
        travel_cone_rad=deg("dp_travel_cone_deg", default.travel_cone_rad),
        translate_suppress_rad=deg("dp_suppress_deg",
                                   default.translate_suppress_rad),
        translate_suppress_floor=num("dp_suppress_floor",
                                     default.translate_suppress_floor),
        passed_bearing_rad=deg("dp_passed_bearing_deg",
                               default.passed_bearing_rad),
        hold_deadband_m=num("dp_hold_deadband_m", default.hold_deadband_m),
        forward_track_frac=num("dp_forward_track_frac",
                               default.forward_track_frac),
        lateral_turn_frac=num("dp_lateral_turn_frac", default.lateral_turn_frac),
        turn_pitch_bias=num("dp_turn_pitch_bias", default.turn_pitch_bias),
        turn_side_cone_rad=deg("dp_turn_side_cone_deg",
                               default.turn_side_cone_rad),
        envelope=envelope, lateral_pid=lateral, yaw_pid=yaw,
        yaw_lookahead=yaw_lookahead or YawLookaheadParams(),
    )


def anticipation(dials=None, **overrides):
    # type: (Optional[Dict[str, Any]], **Any) -> YawLookaheadParams
    """The anticipation's params from mission.yaml, forced on, plus overrides.

    ``dp_yaw_lookahead_rate`` follows ``dp_track_yaw_rate`` when unset, exactly
    as the ROS adapter defaults it — the schedule runs in TRACK and may not ask
    for more rotation than that.
    """
    d = deployed_dials() if dials is None else dials
    base = YawLookaheadParams()

    def num(key, fallback):
        try:
            return float(d[key])
        except (KeyError, TypeError, ValueError):
            return float(fallback)

    track_rate = num("dp_track_yaw_rate", DriftPidParams().track_yaw_rate)
    values = dict(
        enabled=True,
        start_m=num("dp_yaw_lookahead_start_m", base.start_m),
        align_m=num("dp_yaw_lookahead_align_m", base.align_m),
        corner_rad=math.radians(num("dp_yaw_lookahead_corner_deg",
                                    math.degrees(base.corner_rad))),
        confirm_m=num("dp_yaw_lookahead_confirm_m", base.confirm_m),
        max_offset_rad=math.radians(num("dp_yaw_lookahead_max_deg",
                                        math.degrees(base.max_offset_rad))),
        catchup_rad=math.radians(num("dp_yaw_lookahead_catchup_deg",
                                     math.degrees(base.catchup_rad))),
        rate=num("dp_yaw_lookahead_rate", track_rate),
        side_cone_rad=math.radians(num("dp_yaw_lookahead_cone_deg",
                                       math.degrees(base.side_cone_rad))),
        feedforward=num("dp_yaw_lookahead_ff", base.feedforward),
    )
    values.update(overrides)
    return YawLookaheadParams(**values)
