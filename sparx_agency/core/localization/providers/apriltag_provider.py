"""AprilTag-based localization provider (pure Python, no ROS2)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from sparx_agency.core.common.types.geometry import Pose3D
from sparx_agency.core.common.types.perception import Observation
from sparx_agency.core.localization.base import BaseLocalizationProvider, LocalizationEstimate
from sparx_agency.core.localization.command_motion_model import (
    CommandMotionModel,
    CommandMotionParams,
)
from sparx_agency.core.localization.providers.tag_persistence import (
    TagPersistence,
    TagPersistenceParams,
)
from sparx_agency.core.localization.tag_triangulation import TagWorldPose
from sparx_agency.tasks.localization.common.apriltag_cv_common import (
    load_camera_calib_yaml,
    make_detector,
)
from sparx_agency.tasks.localization.common.apriltag_pnp import (
    TagDetection,
    estimate_camera_pose,
    pose_confidence,
)

# OpenCV camera frame → ROS/world frame convention
_CV_TO_ROS = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)


def _load_tag_world_map(path: str, default_size: float) -> Tuple[Dict[int, TagWorldPose], Dict[int, float]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"tag_map_path does not exist: {path}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    tags = data.get("tags", data)
    out_poses: Dict[int, TagWorldPose] = {}
    out_sizes: Dict[int, float] = {}
    for k, v in tags.items():
        tid = int(k)
        xyz = tuple(float(x) for x in v["xyz"])
        rpy = tuple(float(x) for x in v["rpy"])
        out_poses[tid] = TagWorldPose(xyz=xyz, rpy=rpy)
        out_sizes[tid] = float(v.get("size", default_size))
    if not out_poses:
        raise ValueError(f"No tags loaded from: {path}")
    return out_poses, out_sizes


class AprilTagLocalizationProvider(BaseLocalizationProvider):
    """Localization from AprilTags.

    Wraps :func:`apriltag_pnp.estimate_camera_pose`: a single joint PnP over the
    pooled corners of all visible tags (>= 2), or a disambiguated IPPE-square
    solve for a single tag. Every detected, mapped tag is used — a far or small
    tag still adds a real constraint, and on the deployed map dropping one
    measurably *worsens* the fix.

    Filtering is confidence-adaptive rather than fixed-gain. The old constant
    ``alpha=0.1`` bought jump-rejection with ~0.9 s of lag on every movement,
    which is most of a second of stale position for a controller to fly on. Here
    the gain follows :func:`pose_confidence`, so a crisp multi-tag fix is taken
    almost immediately while a lone distant tag is averaged down — and an
    outright impossible jump is rejected by ``max_jump_m`` before it reaches the
    filter at all.

    Args:
        alpha: Filter gain at full confidence (1.0 = no smoothing at all).
        alpha_min: Filter gain at zero confidence — the floor that keeps a poor
            fix contributing something rather than freezing the estimate.
        max_jump_m: Reject a fix that moves the drone further than this in one
            frame. 0 disables the gate.
        max_jump_frames: Consecutive rejections allowed before a jump is accepted
            anyway, so a genuine re-localisation can never be locked out forever.
        margin_keep: Hysteresis lower rail — a tag already in the used set stays
            down to this ``decision_margin`` (new tags still need ``min_margin``).
            Set equal to ``min_margin`` to disable.
        roi_rescue: Re-detect a just-lost tag inside a 2x-upscaled crop around
            its last position before accepting that it is gone (sub-ms, and only
            runs for tags that actually vanished).
        cmd_trust_max: Ceiling on the command prior fed via :meth:`set_command`
            (see :class:`CommandMotionModel`). 0 ignores commands entirely.
    """

    source_name = "apriltag"

    def __init__(
        self,
        tag_map_path: str,
        camera_calib_path: str,
        tag_size_m: float = 0.13,
        tag_family: str = "tag36h11",
        min_margin: float = 10.0,
        alpha: float = 0.8,
        nthreads: int = 2,
        fuse_method: str = "avg_translation_keep_first_rotation",
        alpha_min: float = 0.02,
        max_jump_m: float = 1.0,
        max_jump_frames: int = 3,
        process_std_m: float = 0.05,
        margin_keep: float = 5.0,
        roi_rescue: bool = True,
        cmd_trust_max: float = 0.7,
    ) -> None:
        self._tag_map, self._tag_sizes = _load_tag_world_map(tag_map_path, tag_size_m)
        self._default_tag_size = float(tag_size_m)
        self._calib = load_camera_calib_yaml(camera_calib_path)
        self._detector = make_detector(tag_family, nthreads=nthreads)
        self._min_margin = float(min_margin)
        self._alpha = float(alpha)
        self._alpha_min = float(alpha_min)
        self._max_jump_m = float(max_jump_m)
        self._max_jump_frames = int(max_jump_frames)
        self._process_var = float(process_std_m) ** 2
        self._fuse_method = fuse_method

        # Set-flicker defence: margin hysteresis + ROI rescue of dropouts. The
        # rescue reuses the SAME detector instance -- a second used Detector
        # makes pupil_apriltags segfault at interpreter teardown.
        self._persist = TagPersistence(self._detector, TagPersistenceParams(
            enter_margin=float(min_margin), keep_margin=float(margin_keep),
            rescue=bool(roi_rescue)))
        # Command prior with earned trust; its effectiveness doubles as the
        # "commanded but not moving" stuck signal published downstream.
        self._cmd = CommandMotionModel(CommandMotionParams(
            trust_max=float(cmd_trust_max)))
        self._raw_cmd = [0.0, 0.0, 0.0]   # unscaled commanded motion since last accept
        self._last_meas: Optional[np.ndarray] = None
        self._last_yaw_raw: Optional[float] = None
        self._last_stamp: Optional[float] = None

        self._P = float(process_std_m) ** 2
        self._pos: Optional[np.ndarray] = None
        self._filtered_q: Optional[np.ndarray] = None
        # Previous camera position in world for temporal disambiguation of the
        # single-tag planar (IPPE) pose ambiguity.
        self._prev_cam_pos: Optional[np.ndarray] = None
        self._rejected: int = 0
        self._healthy = True

        # Core stays importable without ROS2: rclpy only supplies a nicer logger.
        try:
            import rclpy.logging
            self._log = rclpy.logging.get_logger("apriltag_provider")
        except ImportError:
            import logging
            self._log = logging.getLogger("apriltag_provider")
            self._log.warn = self._log.warning  # rclpy-logger API shim

    def is_healthy(self) -> bool:
        return self._healthy

    def reset(self) -> None:
        self._pos = None
        self._filtered_q = None
        self._prev_cam_pos = None
        self._rejected = 0
        self._P = self._process_var
        self._persist.reset()
        self._cmd.reset()
        self._raw_cmd = [0.0, 0.0, 0.0]
        self._last_meas = None
        self._last_yaw_raw = None
        self._last_stamp = None

    def set_command(self, vx: float, vy: float, wz: float, stamp_sec: float) -> None:
        """Feed the twist currently commanded to the platform (body frame).

        Called by the node layer from the command topic. The commands are used
        only through :class:`CommandMotionModel`, i.e. scaled by how much of the
        commanded motion the drone has recently PROVEN to achieve — a drone
        pressed against a wall stops being believed within about a second.
        """
        self._cmd.set_command(vx, vy, wz, stamp_sec)

    def _gain(self, pos_std_m: float) -> float:
        """Filter gain for this frame, from the fix's own expected error.

        A fixed gain has to choose between tracking quickly and rejecting jumps,
        and the old 0.1 chose rejection — at the cost of ~0.9 s of lag on every
        move. Weighting by variance removes the choice: this is the scalar Kalman
        gain ``P / (P + R)``, where ``R`` is how wrong this measurement expects to
        be and ``P`` is how wrong we already were.

        The behaviour falls out of the arithmetic rather than being tuned in. A
        crisp multi-tag fix has tiny ``R``, so the gain approaches 1 and the pose
        is taken almost as-is. A lone distant tag has ``R`` two orders of magnitude
        larger, so it barely moves the estimate — and when the good fix returns,
        ``R`` collapses and the estimate snaps to it in a frame or two instead of
        crawling back over a second.

        ``_process_var`` is how far the drone may genuinely move between frames.
        It is what stops the filter becoming overconfident and ignoring reality:
        without it ``P`` would shrink toward zero and the pose would freeze.
        """
        self._P += self._process_var
        R = max(1e-6, float(pos_std_m) ** 2)
        K = self._P / (self._P + R)
        self._P *= (1.0 - K)
        return float(max(self._alpha_min, min(self._alpha, K)))

    def update(self, obs: Observation) -> Optional[LocalizationEstimate]:
        if obs.rgb is None:
            return None

        frame = obs.rgb.image
        stamp_sec = obs.rgb.stamp_sec

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        dets = self._detector.detect(gray)

        # Hysteresis + ROI rescue: the per-frame defence against the visible tag
        # SET changing, which is what actually makes the published pose jump.
        kept = self._persist.filter(gray, dets, self._tag_map.keys())
        rescued = [d.tag_id for d in kept if d.rescued]
        if rescued:
            self._log.info(f"[apriltag] rescued {rescued} via ROI re-detect")

        tag_dets: List[TagDetection] = [
            TagDetection(tag_id=d.tag_id, corners=d.corners,
                         size_m=self._tag_sizes.get(d.tag_id, self._default_tag_size))
            for d in kept]

        if not tag_dets:
            return None

        # One joint PnP over all tags (>= 2), or a disambiguated IPPE-square
        # solve for a single tag, seeded with the previous position.
        est = estimate_camera_pose(
            tag_dets, self._tag_map, self._calib.K, self._calib.D,
            prev_cam_pos_world=self._prev_cam_pos,
        )
        if est is None:
            return None

        confidence = pose_confidence(est)

        world_T_ros = est.world_T_cam @ _CV_TO_ROS
        meas = np.array([world_T_ros[0, 3], world_T_ros[1, 3], world_T_ros[2, 3]],
                        dtype=float)
        q_raw = np.array(_quat_from_matrix(world_T_ros), dtype=float)
        yaw_raw = math.atan2(2.0 * (q_raw[3] * q_raw[2] + q_raw[0] * q_raw[1]),
                             1.0 - 2.0 * (q_raw[1] * q_raw[1] + q_raw[2] * q_raw[2]))

        # Command prior: move the filtered state by the commanded motion, scaled
        # by how much of it the drone has recently PROVEN to achieve. This is what
        # keeps the pose current through a low-confidence stretch (single far tag,
        # short dropout) instead of freezing — while a stuck drone, whose commands
        # demonstrably do nothing, moves the prior almost not at all.
        if self._pos is not None and self._cmd.enabled:
            yaw_f = math.atan2(
                2.0 * (self._filtered_q[3] * self._filtered_q[2]
                       + self._filtered_q[0] * self._filtered_q[1]),
                1.0 - 2.0 * (self._filtered_q[1] * self._filtered_q[1]
                             + self._filtered_q[2] * self._filtered_q[2]))
            dx, dy, dyaw, rdx, rdy, rdyaw = self._cmd.consume(stamp_sec, yaw_f)
            self._pos = self._pos + np.array([dx, dy, 0.0])
            if abs(dyaw) > 1e-9:
                half = 0.5 * dyaw
                qd = np.array([0.0, 0.0, math.sin(half), math.cos(half)])
                self._filtered_q = _quat_mul(qd, self._filtered_q)
                self._filtered_q /= np.linalg.norm(self._filtered_q)
            self._raw_cmd[0] += rdx
            self._raw_cmd[1] += rdy
            self._raw_cmd[2] += rdyaw

        # Jump gate. A fix that puts the drone a metre from where it was one frame
        # ago is not a fast drone, it is a bad solve -- most often the visible tag
        # set changing under us. Drop it, but only for a few frames in a row: a
        # genuine re-localisation (or a gate mis-tuned for the real flight speed)
        # must never be able to lock the estimate out permanently. The gate
        # compares against the command-predicted pose, so genuinely commanded
        # motion does not eat into the jump budget.
        if (self._pos is not None and self._max_jump_m > 0.0
                and float(np.linalg.norm(meas - self._pos)) > self._max_jump_m
                and self._rejected < self._max_jump_frames):
            self._rejected += 1
            self._log.warn(
                "[apriltag] REJECT jump %.2fm (>%.2f) conf=%.2f n_tags=%d used=%s "
                "-- holding previous pose (%d/%d)"
                % (float(np.linalg.norm(meas - self._pos)), self._max_jump_m,
                   confidence, est.n_tags, sorted(est.used_tag_ids),
                   self._rejected, self._max_jump_frames))
            return None
        self._rejected = 0
        self._prev_cam_pos = est.world_T_cam[:3, 3].copy()

        # Teach the command model: what the camera says actually happened over
        # the interval, against the RAW commanded motion over that same interval.
        # Only confident fixes teach (the model enforces its own floor) — the
        # measured deltas are between accepted measurements, not filtered poses,
        # so the filter's own smoothing cannot flatter the commands.
        if self._cmd.enabled and self._last_meas is not None \
                and self._last_yaw_raw is not None:
            dyaw_meas = math.atan2(math.sin(yaw_raw - self._last_yaw_raw),
                                   math.cos(yaw_raw - self._last_yaw_raw))
            self._cmd.observe(float(meas[0] - self._last_meas[0]),
                              float(meas[1] - self._last_meas[1]), dyaw_meas,
                              self._raw_cmd[0], self._raw_cmd[1],
                              self._raw_cmd[2], confidence)
        elif self._cmd.enabled and self._last_meas is None:
            # First accepted fix: flush any command backlog so it cannot dump
            # a stale burst of "motion" into the second fix's interval.
            self._cmd.consume(stamp_sec, 0.0)
        self._raw_cmd = [0.0, 0.0, 0.0]
        self._last_meas = meas.copy()
        self._last_yaw_raw = yaw_raw
        self._last_stamp = stamp_sec

        # Position spread measured on the deployed map: ~1 cm for a crisp
        # multi-tag fix rising to ~30 cm for a lone distant tag. This drives the
        # filter gain, so it has to be an honest error estimate rather than a
        # cosmetic score.
        pos_std = 0.01 + 0.30 * (1.0 - confidence) ** 2
        gain = self._gain(pos_std)
        if self._pos is None:
            self._pos = meas
        else:
            self._pos = gain * meas + (1.0 - gain) * self._pos
        x, y, z = (float(v) for v in self._pos)

        if self._filtered_q is None:
            self._filtered_q = q_raw
        else:
            if np.dot(self._filtered_q, q_raw) < 0.0:
                q_raw = -q_raw
            self._filtered_q = gain * q_raw + (1.0 - gain) * self._filtered_q
            self._filtered_q /= np.linalg.norm(self._filtered_q)

        qx, qy, qz, qw = self._filtered_q
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

        eff = self._cmd.effectiveness_lin if self._cmd.enabled else None
        used_ids = sorted(est.used_tag_ids)
        self._log.info(
            f"[apriltag] pose: conf={confidence:.2f} gain={gain:.2f} n_tags={est.n_tags} "
            f"used={used_ids} rms={est.reproj_rms_px:.2f}px worst={est.worst_tag_rms_px:.2f}px "
            f"amb={est.ambiguity:.2f} geom={est.geometry:.2f} "
            f"min_px={est.min_tag_px:.0f} far={est.max_tag_dist_m:.1f}m"
            + (f" eff={eff:.2f}/{self._cmd.effectiveness_yaw:.2f}" if eff is not None else ""))

        return LocalizationEstimate(
            pose=Pose3D(x=float(x), y=float(y), z=float(z), yaw=yaw),
            source=self.source_name,
            confidence=confidence,
            stamp_sec=stamp_sec,
            pos_std_m=pos_std,
            yaw_std_rad=0.02 + 0.20 * (1.0 - confidence) ** 2,
            cmd_effectiveness=eff,
        )


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a ⊗ b for (x, y, z, w) quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def _quat_from_matrix(M: np.ndarray) -> Tuple[float, float, float, float]:
    """Extract quaternion (x, y, z, w) from a 4×4 rotation matrix."""
    R = M[:3, :3]
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)