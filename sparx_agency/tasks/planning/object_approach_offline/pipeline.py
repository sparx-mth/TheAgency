"""Offline, ROS-free replay of the object-approach mission stack for one target.

Composes the same pieces ``tasks/planning/falcon/adapter/scripts/object_approach_node.py``
wires to ROS -- :class:`TargetConfirmationGate` (acquire on N consecutive
detections), :class:`TargetTracker` (LK box tracking + motion prediction),
:class:`VisualServoController` (tracked box -> body velocity),
:class:`VisualApproachStateMachine` (SEARCH/APPROACH/HOVER_LOCK/RECOVER), and
:class:`ReSearchPolicy` (where to look when lost) -- so a folder of frames can be
driven through the exact mission logic with no ROS, no Falcon, and no live drone
connection. See :mod:`.run_folder_target_lock` for the CLI that drives this over a
folder of RGB frames.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from sparx_agency.core.common.types import ControlCommand, Intrinsics, KinematicLimits
from sparx_agency.core.common.types.perception import Detection2D, Track2D
from sparx_agency.core.mapping.depth.depth_bbox_fusion import bbox_to_xyz_cam_from_depth
from sparx_agency.core.planning.visual_servo import (
    ApproachFSMConfig,
    ConfirmationGateConfig,
    ReSearchConfig,
    ReSearchPolicy,
    TargetConfirmationGate,
    VisualApproachStateMachine,
    VisualServoController,
    VisualServoParams,
    VisualServoRequest,
)
from sparx_agency.core.mapping.tracking import (
    DetectionOnlyConfig,
    TargetTrackerConfig,
    make_lock_tracker,
)


def _cap(value: float, limit: Optional[float]) -> float:
    """Same saturation :meth:`VisualServoController._cap` applies internally."""
    return float(value) if limit is None else float(min(value, limit))


@dataclass(frozen=True)
class FrameResult:
    """Everything one call to :meth:`TargetLockPipeline.step` decided, for logging/HUD.

    Attributes:
        stamp_s: Timestamp passed in for this frame.
        dt: Seconds since the previous step.
        target: The locked-onto label.
        detections: All detector output this frame (target + any distractors).
        target_detection: The matching target detection this frame (above
            ``min_score``), or None. Its presence is the "YOLO sees it now" signal
            that colours the HUD box green (vs orange for tracking-only); its box
            is what the green rectangle is drawn from.
        confirmed: Acquisition gate has seen ``n_confirm`` consecutive hits.
        streak: Current consecutive-hit count.
        track: Current :class:`Track2D`, or None before the tracker is ever seeded.
        fsm_mode: One of SEARCH/APPROACH/HOVER_LOCK/RECOVER.
        at_target: The servo reports centred and close enough (hover-lock condition).
        x_offset: Normalised horizontal box-centre offset in [-1, 1], or None.
        y_offset: Normalised vertical box-centre offset in [-1, 1], or None.
        area_frac: Box area / image area, or None.
        range_m: Metric range used by the servo, or None (area proxy / no track).
        command: The body-frame velocity that would be published to ``/cmd_vel``
            this frame, or None while the mission planner (not this pipeline) is
            driving (SEARCH).
        cmd_source: Where ``command`` came from: ``"servo:<mode>"``,
            ``"recovery:<phase>"``, or ``"planner (visual approach passive)"``.
        gauge_max_vx: Full-scale forward/back speed (m/s) the servo can command --
            for normalising a PITCH display, not itself a command.
        gauge_max_vy: Full-scale lateral speed (m/s) the servo can command -- for
            normalising a ROLL display.
        gauge_max_yaw_rate: Full-scale yaw rate (rad/s) the servo can command --
            for normalising a YAW display.
    """

    stamp_s: float
    dt: float
    target: str
    detections: List[Detection2D]
    target_detection: Optional[Detection2D]
    confirmed: bool
    streak: int
    track: Optional[Track2D]
    fsm_mode: str
    at_target: bool
    x_offset: Optional[float]
    y_offset: Optional[float]
    area_frac: Optional[float]
    range_m: Optional[float]
    command: Optional[ControlCommand]
    cmd_source: str
    gauge_max_vx: float
    gauge_max_vy: float
    gauge_max_yaw_rate: float


class TargetLockPipeline:
    """Drive one named target through acquire -> track -> servo -> FSM, frame by frame."""

    def __init__(self, target: str, intrinsics: Intrinsics,
                 limits: Optional[KinematicLimits] = None,
                 servo_params: Optional[VisualServoParams] = None,
                 gate_config: Optional[ConfirmationGateConfig] = None,
                 tracker_config: Optional[TargetTrackerConfig] = None,
                 fsm_config: Optional[ApproachFSMConfig] = None,
                 recovery_config: Optional[ReSearchConfig] = None,
                 lock_mode: str = "detector_tracker",
                 detection_config: Optional[DetectionOnlyConfig] = None,
                 reseed_on_detection: bool = True) -> None:
        """Args:
            target: Object label to lock onto (matched fuzzily; see
                :func:`~core.planning.visual_servo.label_matches`).
            intrinsics: Camera model of the frames that will be passed to :meth:`step`.
            limits: Optional kinematic caps applied to the servo output.
            lock_mode: How to close on the object — ``"detector_tracker"`` (default:
                detector seeds an optical-flow tracker propagated every frame) or
                ``"detector"`` (the detector's box alone, no tracking). See
                :func:`~core.mapping.tracking.make_lock_tracker`.
            detection_config: Config for the ``"detector"`` lock mode (freshness
                window); ignored for ``"detector_tracker"``.
            reseed_on_detection: Re-seed the tracker from every fresh matching
                detection (bounds drift; mirrors the live node's default).
        """
        self.target = str(target).strip().lower()
        self.intr = intrinsics
        self.limits = limits
        self.reseed_on_detection = reseed_on_detection

        params = servo_params or VisualServoParams()
        self.tracker = make_lock_tracker(lock_mode, tracker_config, detection_config)
        self.servo = VisualServoController(params, default_limits=limits)
        # Full-scale references for a gauge display -- the same caps
        # VisualServoController.step applies internally (params, floored by any
        # kinematic limit), computed once since neither changes per frame.
        self.gauge_max_vx = _cap(params.vx_max, limits and limits.max_speed_xy)
        self.gauge_max_vy = _cap(params.max_lateral_speed, limits and limits.max_speed_xy)
        self.gauge_max_yaw_rate = _cap(params.max_yaw_rate, limits and limits.max_yaw_rate)
        self.gate = TargetConfirmationGate(self.target,
                                           gate_config or ConfirmationGateConfig())
        self.recovery = ReSearchPolicy(recovery_config or ReSearchConfig())
        self.fsm = VisualApproachStateMachine(fsm_config or ApproachFSMConfig())
        self._prev_stamp: Optional[float] = None

    def step(self, bgr: np.ndarray, stamp_s: float, detections: List[Detection2D],
             depth_m: Optional[np.ndarray] = None) -> FrameResult:
        """Advance the mission by one frame and return what it decided.

        Args:
            bgr: The frame the detections were made on (feeds the LK tracker).
            stamp_s: Monotonic timestamp (s) of this frame.
            detections: This frame's detector output (target + any distractors).
            depth_m: Optional aligned metric depth (HxW, meters) for range-gated
                approach/terminal logic; without it the servo falls back to the
                box area fraction.
        """
        stamp = float(stamp_s)
        dt = 0.0 if self._prev_stamp is None else max(0.0, stamp - self._prev_stamp)
        self._prev_stamp = stamp

        state = self.gate.update(detections)
        # Re-feed the tracker on every matching detection when asked to (bounds an
        # optical-flow tracker's drift), and ALWAYS for a non-propagating tracker
        # (detector-only) -- it has no state to coast on, so "seed once" would
        # freeze the lock and abandon a still-visible target.
        reseed = self.reseed_on_detection or not self.tracker.propagates
        need_seed = (not self.tracker.has_target and state.confirmed) or \
                    (self.tracker.has_target and reseed)
        if need_seed and state.best is not None:
            self.tracker.on_detection(bgr, state.best, stamp)

        track = self.tracker.on_frame(bgr, stamp) if self.tracker.has_target else None
        track_valid = bool(track is not None and track.valid)

        res = None
        range_m = None
        if track_valid:
            range_m = self._range_to(track, depth_m)
            res = self.servo.step(VisualServoRequest(
                track=track, intrinsics=self.intr, range_m=range_m, dt=dt))
        at_target = bool(res is not None and res.at_target)

        dec = self.fsm.update(confirmed=state.confirmed, track_valid=track_valid,
                              at_target=at_target, dt=dt)
        if dec.reset_acquisition:
            self.gate.reset()
            self.tracker.reset()

        command, cmd_source = self._select_command(dec, res, track)

        return FrameResult(
            stamp_s=stamp, dt=dt, target=self.target, detections=list(detections),
            target_detection=state.best,
            confirmed=state.confirmed, streak=state.streak, track=track,
            fsm_mode=dec.mode, at_target=at_target,
            x_offset=None if res is None else res.x_offset,
            y_offset=None if res is None else res.y_offset,
            area_frac=None if res is None else res.area_frac,
            range_m=range_m, command=command, cmd_source=cmd_source,
            gauge_max_vx=self.gauge_max_vx, gauge_max_vy=self.gauge_max_vy,
            gauge_max_yaw_rate=self.gauge_max_yaw_rate)

    def _select_command(self, dec, res, track):
        if not dec.drive_cmd_vel:
            return None, "planner (visual approach passive)"
        if dec.mode == "RECOVER" or res is None:
            rec = self.recovery.command(track, dec.lost_for_s,
                                        self.intr.width, self.intr.height)
            return rec.command, "recovery:%s" % rec.phase
        return res.command, "servo:%s" % res.mode

    def _range_to(self, track: Track2D, depth_m: Optional[np.ndarray]) -> Optional[float]:
        """Metric range (m) to the tracked box from depth, or None."""
        if depth_m is None:
            return None
        if depth_m.shape[:2] != (self.intr.height, self.intr.width):
            raise ValueError(
                "depth shape %r does not match intrinsics %dx%d"
                % (depth_m.shape[:2], self.intr.height, self.intr.width))
        x1, y1, x2, y2 = (int(v) for v in track.bbox_xyxy)
        xyz = bbox_to_xyz_cam_from_depth(depth_m, (x1, y1, x2, y2),
                                         self.intr.fx, self.intr.fy,
                                         self.intr.cx, self.intr.cy)
        return None if xyz is None else float(xyz[2])
