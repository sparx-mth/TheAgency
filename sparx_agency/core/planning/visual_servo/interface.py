"""Request/result contract for the visual-servo controller.

Deliberately *not* the trajectory-tracker contract
(:mod:`sparx_agency.core.planning.interfaces.tracker`): a visual servo is reactive
and trajectory-free — its input is a tracked box + camera model (+ optional metric
range), not a reference ``Trajectory``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from sparx_agency.core.common.types import ControlCommand, Intrinsics, KinematicLimits
from sparx_agency.core.common.types.perception import Track2D


@dataclass(frozen=True)
class VisualServoRequest:
    """Inputs to one visual-servo step.

    Attributes:
        track: The tracked target this frame (its ``bbox_xyxy`` drives centring;
            ``area_frac`` is the fallback proximity proxy).
        intrinsics: Camera model of the frame the track lives in.
        range_m: Metric range to the target (m) from depth, or None. When present
            and ``params.use_depth`` is set, it drives the approach/terminal logic
            in place of the area proxy. Compute with
            :func:`...mapping.depth.depth_bbox_fusion.bbox_to_xyz_cam_from_depth`.
        limits: Optional kinematic limits; when given they cap the raw params.
        dt: Seconds since the previous step (for output smoothing).
        options: Algorithm-specific extras.
    """

    track: Track2D
    intrinsics: Intrinsics
    range_m: Optional[float] = None
    limits: Optional[KinematicLimits] = None
    dt: float = 0.05
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VisualServoResult:
    """Output of one visual-servo step.

    Attributes:
        command: Body-frame velocity command (REP-103) for the platform.
        at_target: True when the target is centred *and* close enough — the
            success condition ("in front of and very close to the object"). The
            mission FSM transitions to hover-lock on this.
        centered: True when ``|x_offset| <= center_tol``.
        x_offset: Normalised horizontal offset of the box centre in ``[-1, 1]``.
        y_offset: Normalised vertical offset of the box centre in ``[-1, 1]``.
        area_frac: Box area / image area (proximity proxy).
        range_m: The metric range used, or None if unavailable.
        mode: Sub-mode label ("holonomic", or "YAW"/"ADVANCE" in xor mode).
        metadata: Diagnostics.
    """

    command: ControlCommand
    at_target: bool
    centered: bool
    x_offset: float
    y_offset: float
    area_frac: float
    range_m: Optional[float] = None
    mode: str = "holonomic"
    metadata: Dict[str, Any] = field(default_factory=dict)
