#!/usr/bin/env python3
"""rosparams -> :class:`DriftPidParams`, and the telemetry the controller emits.

All tuning is namespaced ``~dp_*`` so it can never collide with the one-axis
follower's params or with ``~mx_* / ~pp_* / ~ra_*``. Angles are exposed in
DEGREES (``*_deg``) and converted here, matching the convention the rest of the
node uses; everything else is SI.

The controller itself is ROS-free and lives in
:mod:`sparx_agency.core.planning.trackers.drift_pid`; this module is only the
translation layer, plus the two publishers that make the controller's internal
state visible to an operator:

    /falcon/drift            what the drone has learned it must fight
    /falcon/blockage         "there is something here the camera cannot see"

The second is the seam between control and planning. The controller runs the
reflexes; when they are spent it says so exactly once, and the *planner* decides
what to do about it.
"""
import json
import math

import rospy
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String

from sparx_agency.core.planning.trackers.drift_pid import (
    BlockageParams,
    ConfidenceParams,
    DriftPidFollower,
    DriftPidParams,
    EnvelopeParams,
    EscapeParams,
    PidGains,
)


def param_bool(name, default):
    """rosparam bool that refuses to guess.

    ``bool(rospy.get_param(...))`` turns ANY non-empty string truthy, so a typo
    like "fales" silently reads as True -- roslaunch only coerces the exact
    strings 'true'/'false'. This accepts the obvious spellings and raises on
    anything else rather than flying on a misread flag.
    """
    value = rospy.get_param(name, default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    raise ValueError(
        "rosparam %s=%r is not a boolean. Use true/false -- roslaunch passes "
        "params as strings and a typo would otherwise read as True." % (name, value))


def build_drift_pid_params(G):
    """Assemble :class:`DriftPidParams` from ``~dp_*`` rosparams.

    Args:
        G: ``rospy.get_param``.

    Returns:
        The fully-populated, validated params. Any inconsistent combination
        raises out of the dataclass validators at construction, so the node fails
        loudly at startup rather than flying a nonsense envelope.
    """
    envelope = EnvelopeParams(
        max_vx=float(G("~dp_max_vx", 0.25)),
        max_vx_back=float(G("~dp_max_vx_back", 0.12)),
        max_vy=float(G("~dp_max_vy", 0.12)),
        max_wz=float(G("~dp_max_wz", 0.40)),
        max_translation=float(G("~dp_max_translation", 0.25)),
        combined_effort=float(G("~dp_combined_effort", 1.4)),
        min_vx=float(G("~dp_min_vx", 0.06)),
        min_vy=float(G("~dp_min_vy", 0.06)),
        min_wz=math.radians(float(G("~dp_min_wz_deg", 10.0))),
        release_frac=float(G("~dp_release_frac", 0.5)),
        cmd_zero_eps=float(G("~dp_cmd_zero_eps", 1e-3)),
        accel_xy=float(G("~dp_accel_xy", 0.35)),
        decel_xy=float(G("~dp_decel_xy", 0.60)),
        accel_wz=float(G("~dp_accel_wz", 1.2)),
        decel_wz=float(G("~dp_decel_wz", 2.0)),
    )
    lateral_pid = PidGains(
        kp=float(G("~dp_lat_kp", 0.55)),
        ki=float(G("~dp_lat_ki", 0.06)),
        kd=float(G("~dp_lat_kd", 0.12)),
        i_limit=float(G("~dp_lat_i_limit", 0.05)),
        d_tau_s=float(G("~dp_lat_d_tau_s", 0.4)),
        deadband=float(G("~dp_lat_deadband_m", 0.03)),
        out_limit=float(G("~dp_lat_max", 0.10)),
    )
    forward_pid = PidGains(
        kp=float(G("~dp_fwd_kp", 0.50)),
        ki=float(G("~dp_fwd_ki", 0.05)),
        kd=float(G("~dp_fwd_kd", 0.10)),
        i_limit=float(G("~dp_fwd_i_limit", 0.05)),
        d_tau_s=float(G("~dp_fwd_d_tau_s", 0.4)),
        deadband=float(G("~dp_fwd_deadband_m", 0.05)),
        out_limit=float(G("~dp_fwd_max", 0.10)),
    )
    yaw_pid = PidGains(
        kp=float(G("~dp_yaw_kp", 0.90)),
        ki=float(G("~dp_yaw_ki", 0.08)),
        kd=float(G("~dp_yaw_kd", 0.15)),
        i_limit=float(G("~dp_yaw_i_limit", 0.08)),
        d_tau_s=float(G("~dp_yaw_d_tau_s", 0.35)),
        deadband=math.radians(float(G("~dp_yaw_deadband_deg", 2.0))),
        out_limit=float(G("~dp_yaw_max", 0.35)),
    )
    confidence = ConfidenceParams(
        conf_full=float(G("~dp_conf_full", 0.35)),
        conf_min=float(G("~dp_conf_min", 0.10)),
        speed_floor=float(G("~dp_speed_floor", 0.35)),
        gain_floor=float(G("~dp_gain_floor", 0.25)),
        conf_integrate=float(G("~dp_conf_integrate", 0.18)),
        conf_hold=float(G("~dp_conf_hold", 0.05)),
        max_age_s=float(G("~dp_max_age_s", 0.6)),
        coast_speed_scale=float(G("~dp_coast_speed_scale", 0.5)),
        eff_floor=float(G("~dp_eff_floor", 0.15)),
        eff_full=float(G("~dp_eff_full", 0.60)),
        eff_speed_floor=float(G("~dp_eff_speed_floor", 0.5)),
        latency_s=float(G("~dp_latency_s", 0.12)),
        std_ref_m=float(G("~dp_std_ref_m", 0.05)),
        std_deadband_gain=float(G("~dp_std_deadband_gain", 0.6)),
        deadband_extra_max_m=float(G("~dp_deadband_extra_max_m", 0.15)),
        yaw_scale_floor=float(G("~dp_yaw_scale_floor", 0.0)),
    )
    blockage = BlockageParams(
        enabled=param_bool("~dp_block_enabled", True),
        window_s=float(G("~dp_block_window_s", 1.2)),
        min_cmd_vx=float(G("~dp_block_min_vx", 0.07)),
        min_cmd_wz=math.radians(float(G("~dp_block_min_wz_deg", 12.0))),
        min_cmd_distance_m=float(G("~dp_block_min_dist_m", 0.06)),
        min_cmd_yaw_rad=math.radians(float(G("~dp_block_min_yaw_deg", 7.0))),
        progress_frac=float(G("~dp_block_progress_frac", 0.30)),
        confirm_ticks=int(G("~dp_block_confirm_ticks", 5)),
        clear_ticks=int(G("~dp_block_clear_ticks", 3)),
        stale_clear_s=float(G("~dp_block_stale_clear_s", 4.0)),
        use_effectiveness=param_bool("~dp_block_use_effectiveness", True),
        eff_floor=float(G("~dp_eff_floor", 0.15)),
    )
    escape = EscapeParams(
        brake_s=float(G("~dp_escape_brake_s", 0.4)),
        back_s=float(G("~dp_escape_back_s", 0.7)),
        back_speed=float(G("~dp_escape_back_speed", 0.10)),
        probe_s=float(G("~dp_escape_probe_s", 0.8)),
        probe_speed=float(G("~dp_escape_probe_speed", 0.10)),
        settle_s=float(G("~dp_escape_settle_s", 0.5)),
        yaw_probe_s=float(G("~dp_escape_yaw_probe_s", 1.0)),
        yaw_escape_invert=param_bool("~dp_escape_yaw_invert", False),
        max_attempts=int(G("~dp_escape_max_attempts", 2)),
    )
    return DriftPidParams(
        cruise_speed=float(G("~dp_cruise_speed", 0.18)),
        cruise_speed_straight=float(G("~dp_cruise_straight", 0.0)),
        approach_yaw_rate=float(G("~dp_approach_yaw_rate", 0.35)),
        track_yaw_rate=float(G("~dp_track_yaw_rate",
                               float(G("~dp_approach_yaw_rate", 0.35)))),
        pos_radius=float(G("~dp_pos_radius",
                           float(G("~pos_acquisition_radius", 0.30)))),
        slow_radius=float(G("~dp_slow_radius", 0.80)),
        arrive_speed_min=float(G("~dp_arrive_speed_min", 0.08)),
        lookahead_m=float(G("~dp_lookahead_m", 0.60)),
        yaw_engage_rad=math.radians(float(G("~dp_yaw_engage_deg", 22.0))),
        yaw_release_rad=math.radians(float(G("~dp_yaw_release_deg", 9.0))),
        travel_cone_rad=math.radians(float(G("~dp_travel_cone_deg", 70.0))),
        translate_suppress_rad=math.radians(float(G("~dp_suppress_deg", 110.0))),
        translate_suppress_floor=float(G("~dp_suppress_floor", 0.15)),
        passed_bearing_rad=math.radians(float(G("~dp_passed_bearing_deg", 110.0))),
        hold_deadband_m=float(G("~dp_hold_deadband_m", 0.06)),
        forward_track_frac=float(G("~dp_forward_track_frac", 0.0)),
        lateral_turn_frac=float(G("~dp_lateral_turn_frac", 0.40)),
        turn_pitch_bias=float(G("~dp_turn_pitch_bias", 0.0)),
        settle_map_updates=int(G("~dp_settle_map_updates", 0)),
        lateral_pid=lateral_pid,
        forward_pid=forward_pid,
        yaw_pid=yaw_pid,
        drift_leak_s=float(G("~dp_drift_leak_s", 180.0)),
        envelope=envelope,
        confidence=confidence,
        blockage=blockage,
        escape=escape,
    )


def build_drift_pid(G):
    """Construct the controller from rosparams."""
    return DriftPidFollower(build_drift_pid_params(G))


class DriftTelemetryPublisher(object):
    """Publishes what the controller has learned, and where it got stuck."""

    def __init__(self, drift_topic="/falcon/drift",
                 blockage_topic="/falcon/blockage", rate_hz=2.0):
        """Create the publishers.

        Args:
            drift_topic: JSON String topic carrying the learned per-axis drift
                and the current tracking errors.
            blockage_topic: PointStamped topic naming, in the world frame, a spot
                the drone could not get through. Latched, because the planner may
                come up or restart after the event and must not miss it.
            rate_hz: How often the drift telemetry is published. The blockage
                report is event-driven and ignores this.
        """
        self.period = 1.0 / float(rate_hz) if rate_hz > 0 else 0.0
        self._last = rospy.Time(0)
        self.drift_pub = rospy.Publisher(drift_topic, String, queue_size=1,
                                         latch=True)
        self.blockage_pub = rospy.Publisher(blockage_topic, PointStamped,
                                            queue_size=5, latch=True)

    def publish_drift(self, telemetry, force=False):
        """Publish the drift telemetry, throttled to the configured rate."""
        if self.period <= 0.0:
            return
        now = rospy.Time.now()
        if not force and (now - self._last).to_sec() < self.period:
            return
        self._last = now
        self.drift_pub.publish(String(data=json.dumps({
            "drift_vy": round(telemetry.drift_vy, 4),
            "drift_vx": round(telemetry.drift_vx, 4),
            "drift_wz": round(telemetry.drift_wz, 4),
            "cross_track_m": round(telemetry.cross_track_m, 4),
            "along_track_m": round(telemetry.along_track_m, 4),
            "heading_err_deg": round(math.degrees(telemetry.heading_err_rad), 2),
            "effort": round(telemetry.effort, 3),
            "speed_scale": round(telemetry.speed_scale, 3),
            "lead_s": round(telemetry.lead_s, 3),
            "deadband_extra_m": round(telemetry.deadband_extra_m, 3),
            "authority": telemetry.authority,
            "blocked_axis": telemetry.blocked_axis,
            "escape_state": telemetry.escape_state,
        })))

    def publish_blockage(self, pose, frame_id, axis):
        """Report an impassable spot to the planner, in the world frame.

        Args:
            pose: The drone's pose when it gave up (``Pose2D``).
            frame_id: Frame the path and pose are expressed in.
            axis: Which axis was blocked, for the log line.
        """
        msg = PointStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = frame_id
        msg.point.x = float(pose.x)
        msg.point.y = float(pose.y)
        msg.point.z = 0.0
        self.blockage_pub.publish(msg)
        rospy.logwarn("drift_pid: blocked on the %s axis at (%.2f, %.2f) and the "
                      "escapes did not clear it -- reporting to the planner",
                      axis, pose.x, pose.y)
