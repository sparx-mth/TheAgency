"""Every knob of the target approach, and the core objects it builds (ROS-free).

Split out of ``target_approach_node`` so the *tuning* can be reasoned about and
tested without an rclpy context — the node then does nothing but declare
:data:`PARAM_DEFAULTS`, read the values back, and hand the plain mapping to the
builders here.

The one relationship worth stating out loud, because nothing else enforces it
and getting it wrong produces an aircraft that flies the whole approach
correctly and then simply never lands:

    ``land_range_m`` must be **greater** than ``target_range_m``.

``target_range_m`` is where the servo stops closing and holds station;
``land_range_m`` is where the state machine commits to its terminal LAND. Set
the land range *inside* the hover standoff and the drone settles into
HOVER_LOCK at the standoff, never gets closer, and hovers in front of the
object until the timeout. :func:`build_fsm` therefore checks it and raises.

The defaults are chosen so the *only* thing this feature changes about today's
working mission is what happens after ``/target_seen`` — before that the node
is inert, and ``enabled`` is the single switch that keeps it that way for good.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping

from sparx_agency.core.mapping.tracking import (DETECTOR_TRACKER, LOCK_MODES,
                                                DetectionOnlyConfig,
                                                ObjectLockTracker,
                                                TargetTrackerConfig,
                                                make_lock_tracker)
from sparx_agency.core.planning.visual_servo import (ApproachFSMConfig,
                                                     ConfirmationGateConfig,
                                                     ReSearchConfig,
                                                     ReSearchPolicy,
                                                     VisualApproachStateMachine,
                                                     VisualServoController,
                                                     VisualServoParams)
from sparx_agency.robots.SJTU.adapters.topics import (
    CMD_VEL, FRONT_CAMERA_INFO, FRONT_DEPTH_CAMERA_INFO, FRONT_DEPTH_IMAGE,
    FRONT_IMAGE, LAND, STATE)

EXTERNAL_CTRL_TOPIC = "/scene_graph/external_ctrl"
"""Latched ``std_msgs/Bool`` that mutes the FALCON b-spline follower.

True means "this node owns ``cmd_vel``". It is republished every
``external_ctrl_period_s`` because the follower treats a stale assertion as a
release — a single latched True would un-mute mid-approach and put two
publishers on the aircraft's only control input.
"""

STATUS_TOPIC = "/target_approach/info"
"""Latched ``std_msgs/String`` JSON status, shape from ``approach_info_payload``."""

PARAM_DEFAULTS: Dict[str, Any] = {
    # ── switches and wiring ──────────────────────────────────────────
    "enabled": True,
    "target_seen_topic": "/target_seen",
    "info_topic": "/target_seen/info",
    # Sim topics come from robots/SJTU/adapters/topics.py, never spelled by
    # hand: that module is pinned against the plugin by the SJTU tests, so a
    # namespace rename moves every consumer at once instead of leaving this
    # node subscribed to a name nothing publishes (which fails as "no data",
    # not as an error).
    "rgb_topic": FRONT_IMAGE,
    "depth_topic": FRONT_DEPTH_IMAGE,
    "rgb_info_topic": FRONT_CAMERA_INFO,
    "depth_info_topic": FRONT_DEPTH_CAMERA_INFO,
    "state_topic": STATE,
    "cmd_topic": CMD_VEL,
    "land_topic": LAND,
    "external_ctrl_topic": EXTERNAL_CTRL_TOPIC,
    "status_topic": STATUS_TOPIC,
    "server_url": "http://127.0.0.1:8092",

    # ── rates ────────────────────────────────────────────────────────
    "approach_rate_hz": 10.0,
    "detect_period_s": 0.5,
    "external_ctrl_period_s": 1.0,
    "server_timeout_s": 5.0,
    "approach_timeout_s": 120.0,

    # ── acquisition ──────────────────────────────────────────────────
    # min_score matches the object mapper's conf_min for the same reason:
    # YOLO-World scores the open-vocabulary hospital prompts ("wheelchair",
    # "hospital bed", "medical trolley") in the 0.3-0.4 band, so a higher gate
    # here would confirm nothing for a whole flight and read as "never seen".
    "n_confirm": 3,
    "min_score": 0.25,
    "confirm_iou": 0.4,
    "soft_confirm_min_score": 0.05,
    "lock_mode": DETECTOR_TRACKER,
    "max_predict_s": 0.4,
    "max_unconfirmed_s": 3.0,

    # ── servo ────────────────────────────────────────────────────────
    # Deliberately slow: a metre-scale closure inside a room the aircraft has
    # already flown into, not a transit.
    "vx_max": 0.35,
    "kp_yaw": 1.2,
    "max_yaw_rate": 0.6,
    "center_tol": 0.15,
    "target_range_m": 0.5,
    "slowdown_range_m": 2.0,

    # ── depth ────────────────────────────────────────────────────────
    "min_depth_m": 0.30,
    "max_depth_m": 8.0,
    "max_stamp_gap_s": 0.25,

    # ── terminal land ────────────────────────────────────────────────
    "land_range_m": 1.0,
    "land_confirm_ticks": 4,
    "acquire_stop_s": 0.5,
    "recover_timeout_s": 6.0,
    "land_repeat_period_s": 1.0,
    "land_settle_s": 12.0,
}
"""Every declared parameter with its default, in one mapping.

The node declares from this and reads straight back into a plain dict, so the
parameter list and the tuning it feeds cannot drift apart the way a
declare-here / read-there pair does.
"""


def build_servo(params: Mapping[str, Any]) -> VisualServoController:
    """The visual-servo control law, in ``holonomic`` mode.

    ``holonomic`` — yaw, forward and lateral crab commanded together — is the
    right mode for this aircraft precisely because the SJTU plugin takes a full
    body Twist. The ``yaw_forward_xor`` alternative exists for platforms that
    reject a Twist mixing forward and yaw, which this one does not.

    Args:
        params: Resolved parameter values (see :data:`PARAM_DEFAULTS`).

    Returns:
        The configured controller.
    """
    return VisualServoController(VisualServoParams(
        mode="holonomic",
        kp_yaw=float(params["kp_yaw"]),
        max_yaw_rate=float(params["max_yaw_rate"]),
        vx_max=float(params["vx_max"]),
        use_depth=True,
        target_range_m=float(params["target_range_m"]),
        slowdown_range_m=float(params["slowdown_range_m"]),
        center_tol=float(params["center_tol"])))


def build_fsm(params: Mapping[str, Any]) -> VisualApproachStateMachine:
    """The SEARCH/ACQUIRE_STOP/APPROACH/HOVER_LOCK/RECOVER/LAND machine.

    ``land_at_goal`` and ``scan_land_revolutions`` are left off: this approach
    has no coordinate goal and no room sweep of its own — it is handed an
    already-found target and only ever closes on what the camera sees.

    Args:
        params: Resolved parameter values.

    Returns:
        The configured state machine, with the terminal LAND armed.

    Raises:
        ValueError: If ``land_range_m`` is not strictly greater than
            ``target_range_m``. Inside the servo's hover standoff the LAND
            trigger can never fire: the drone would settle into HOVER_LOCK at
            the standoff and hover there until the approach timed out, having
            flown the whole approach correctly.
        ValueError: If ``acquire_stop_s`` is not strictly positive. The core
            default is 0.0, which is right for the single-writer XTEND stack
            this machine was written for and wrong here: ACQUIRE_STOP is the
            one tick of deliberate zeros that lets the follower mute cross the
            ROS2 -> bridge -> ROS1 hop before we command anything for real.
            With it at 0 the first command out is a live servo velocity racing
            an un-muted 50 Hz follower for the same aircraft.
    """
    land_range_m = float(params["land_range_m"])
    target_range_m = float(params["target_range_m"])
    acquire_stop_s = float(params["acquire_stop_s"])
    if acquire_stop_s <= 0.0:
        raise ValueError(
            "acquire_stop_s (%.2f) must be > 0 in this deployment: it is the "
            "settle window that lets the follower mute reach the ROS1 side "
            "before this node commands the aircraft. Zero means the first "
            "command races an un-muted follower." % (acquire_stop_s,))
    if land_range_m <= target_range_m:
        raise ValueError(
            "land_range_m (%.2f) must be > target_range_m (%.2f): the servo "
            "stops closing at target_range_m, so a land range inside it can "
            "never be reached and the drone would hover until the timeout."
            % (land_range_m, target_range_m))
    return VisualApproachStateMachine(ApproachFSMConfig(
        acquire_stop_s=acquire_stop_s,
        recover_timeout_s=float(params["recover_timeout_s"]),
        land_range_m=land_range_m,
        land_confirm_ticks=int(params["land_confirm_ticks"])))


def build_gate_config(params: Mapping[str, Any]) -> ConfirmationGateConfig:
    """Acquisition gate tuning (the target class is supplied at arm time)."""
    return ConfirmationGateConfig(n_confirm=int(params["n_confirm"]),
                                  min_score=float(params["min_score"]))


def build_tracker(params: Mapping[str, Any]) -> ObjectLockTracker:
    """The detect-once / track-many object lock.

    ``detector_tracker`` (the default) is what makes a 2 Hz detector usable by
    a 10 Hz servo: optical flow carries the box between POSTs and each fresh
    detection re-seeds it, bounding drift to the detector's inter-arrival time.
    The ``detector``-only path holds the last box for twice ``detect_period_s``
    and is there for when the server keeps up with the camera.

    Args:
        params: Resolved parameter values.

    Returns:
        The configured tracker.

    Raises:
        ValueError: If ``lock_mode`` is not a known mode.
    """
    mode = str(params["lock_mode"]).strip().lower()
    if mode not in LOCK_MODES:
        raise ValueError("lock_mode must be one of %s, got %r"
                         % (list(LOCK_MODES), params["lock_mode"]))
    return make_lock_tracker(
        mode,
        tracker_config=TargetTrackerConfig(
            input_is_bgr=True,
            max_predict_s=float(params["max_predict_s"]),
            max_unconfirmed_s=float(params["max_unconfirmed_s"])),
        detection_config=DetectionOnlyConfig(
            max_det_age_s=max(0.1, 2.0 * float(params["detect_period_s"]))))


def build_recovery(params: Mapping[str, Any]) -> ReSearchPolicy:
    """Where to look when the box is lost, bounded by the FSM's give-up timer.

    The policy's own ``max_search_s`` is set to the state machine's
    ``recover_timeout_s`` so the two agree on how long re-searching lasts; the
    node acts on the machine's decision, never on the policy's ``give_up``.
    """
    return ReSearchPolicy(ReSearchConfig(
        max_search_s=float(params["recover_timeout_s"])))
