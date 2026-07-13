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
:mod:`sparx_agency.core.mapping.tracking`; the detector in
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
    select_overlapping_target_detection,
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
    SCAN,
    ACQUIRE_STOP,
    APPROACH,
    HOVER_LOCK,
    RECOVER,
)
from sparx_agency.core.planning.visual_servo.force_shaping import (
    FORCE_MODES,
    AxisForceProfile,
    CommandForceShaper,
    shape_axis_force,
)
from sparx_agency.core.planning.visual_servo.pulse_shaper import PulseShaper
from sparx_agency.core.planning.visual_servo.gait import (
    ClosureGait,
    ClosureGaitConfig,
)
from sparx_agency.core.planning.visual_servo.scan_search import (
    ScanSearchConfig,
    ScanSearchPolicy,
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
    "select_overlapping_target_detection",
    "ReSearchConfig",
    "ReSearchDecision",
    "ReSearchPolicy",
    "infer_exit_side",
    "ApproachFSMConfig",
    "ApproachDecision",
    "VisualApproachStateMachine",
    "SEARCH",
    "SCAN",
    "ACQUIRE_STOP",
    "APPROACH",
    "HOVER_LOCK",
    "RECOVER",
    "FORCE_MODES",
    "AxisForceProfile",
    "CommandForceShaper",
    "shape_axis_force",
    "PulseShaper",
    "ClosureGait",
    "ClosureGaitConfig",
    "ScanSearchConfig",
    "ScanSearchPolicy",
]
