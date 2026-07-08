"""Visual servoing onto a detected object (ROS-free).

The control half of the "lock onto a named object and approach it" capability:

  * :class:`TargetConfirmationGate` — flip from search to approach on N consecutive
    detections (pose-free);
  * :class:`VisualServoController` — turn a tracked box (+ optional depth range)
    into a body-frame velocity that centres and closes on the target;
  * :class:`ReSearchPolicy` — where to look when the track is lost;
  * :class:`VisualApproachStateMachine` — the SEARCH / APPROACH / HOVER_LOCK /
    RECOVER switch that decides when this node drives ``/cmd_vel``.

The tracker feeding this lives in
:mod:`sparx_agency.core.planning.visual_tracking`; the detector in
:mod:`sparx_agency.core.mapping.detection`.
"""
from __future__ import annotations

from sparx_agency.core.planning.visual_servo.params import VisualServoParams
from sparx_agency.core.planning.visual_servo.interface import (
    VisualServoRequest,
    VisualServoResult,
)
from sparx_agency.core.planning.visual_servo.controller import VisualServoController
from sparx_agency.core.planning.visual_servo.confirmation_gate import (
    ConfirmationGateConfig,
    ConfirmationState,
    TargetConfirmationGate,
    label_matches,
    select_target_detection,
)
from sparx_agency.core.planning.visual_servo.recovery import (
    ReSearchConfig,
    ReSearchDecision,
    ReSearchPolicy,
    infer_exit_side,
)
from sparx_agency.core.planning.visual_servo.state_machine import (
    ApproachFSMConfig,
    ApproachDecision,
    VisualApproachStateMachine,
    SEARCH,
    APPROACH,
    HOVER_LOCK,
    RECOVER,
)

__all__ = [
    "VisualServoParams",
    "VisualServoRequest",
    "VisualServoResult",
    "VisualServoController",
    "ConfirmationGateConfig",
    "ConfirmationState",
    "TargetConfirmationGate",
    "label_matches",
    "select_target_detection",
    "ReSearchConfig",
    "ReSearchDecision",
    "ReSearchPolicy",
    "infer_exit_side",
    "ApproachFSMConfig",
    "ApproachDecision",
    "VisualApproachStateMachine",
    "SEARCH",
    "APPROACH",
    "HOVER_LOCK",
    "RECOVER",
]
