"""Flight-recording schema + reader (numpy, no torch).

A fine-tune recording is an on-disk directory produced from a synchronized drone
flight. This module defines the expected layout, loads it, and provides the
body-frame transforms the label generator needs (future path, goal). It is
deliberately format-simple so a rosbag exporter can target it.

Layout::

    recording_dir/
      intrinsics.json     {"width","height","fx","fy","cx","cy"}
      meta.json           {"rate_hz","camera_height_m","pitch_deg", "frames": N}
      depth/000000.npy    (H, W) float32 meters, one per frame
                          -- or .png, (H, W) uint16 millimetres
      rgb/000000.jpg      (H, W, 3) uint8, co-registered (optional for label-gen)
                          -- or .png
      poses.npy           (N, >=4) float32, columns 0-3 = [t, x, y, yaw] world frame

Two extractors write this layout and they do not agree on file *extensions*:
``bag_extract`` (real rosbags) defaults to PNG for colour, ``sim_extract``
(Isaac Sim) to JPEG for colour and PNG for depth. Rather than force one, the
reader accepts either for both — a recording is defined by its arrays, not by
which lossless container they arrived in.

``poses.npy`` may carry **more** than four columns. Columns 0-3 are fixed and are
all this module reads; a simulated flight appends full 6-DoF ground truth after
them (altitude, attitude quaternion, velocities — see
:data:`~...datasets.sim_extract.POSE_COLUMNS`), reachable through
:meth:`FlightRecording.pose_full`. Older four-column recordings still load.

The goal is supplied per-fine-tune (a body point for NavDP, an image for FlowNav);
for auto-labels the goal defaults to the pose ``lookahead`` frames ahead.

:func:`synthesize_recording` builds a tiny synthetic one for tests and to
document the schema. Real recordings come from ``bag_extract`` (rosbags) or
``sim_extract`` (``tasks/planning/sim_flight_recording/``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from sparx_agency.core.common.types import Intrinsics


DEPTH_EXTENSIONS = (".npy", ".png")
RGB_EXTENSIONS = (".jpg", ".png", ".jpeg")


def _frame_file(directory: Path, index: int, extensions) -> Optional[Path]:
    """The first of ``extensions`` that exists for frame ``index``, or None."""
    for extension in extensions:
        candidate = directory / f"{index:06d}{extension}"
        if candidate.exists():
            return candidate
    return None


@dataclass(frozen=True)
class FlightRecording:
    """A loaded flight recording (lazy per-frame depth/rgb access)."""

    root: Path
    intrinsics: Intrinsics
    poses: np.ndarray          # (N, >=4), columns 0-3 = [t, x, y, yaw]
    rate_hz: float
    camera_height_m: float
    pitch_deg: float
    depth_scale_m: float = 1.0  # metres per stored unit, for integer depth images

    @property
    def num_frames(self) -> int:
        return int(self.poses.shape[0])

    def depth(self, i: int) -> np.ndarray:
        """Frame ``i``'s depth in **metres**, whatever it was stored as.

        Raises:
            FileNotFoundError: If no depth frame exists at that index.
        """
        path = _frame_file(self.root / "depth", i, DEPTH_EXTENSIONS)
        if path is None:
            raise FileNotFoundError(
                f"no depth frame {i:06d} in {self.root / 'depth'} "
                f"(tried {', '.join(DEPTH_EXTENSIONS)})"
            )
        if path.suffix == ".npy":
            return np.load(path).astype(np.float32)
        import cv2
        stored = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if stored is None:
            raise FileNotFoundError(f"could not decode depth image {path}")
        return stored.astype(np.float32) * self.depth_scale_m

    def rgb(self, i: int) -> Optional[np.ndarray]:
        path = _frame_file(self.root / "rgb", i, RGB_EXTENSIONS)
        if path is None:
            return None
        import cv2
        bgr = cv2.imread(str(path))
        return None if bgr is None else bgr[:, :, ::-1].copy()

    def pose(self, i: int) -> np.ndarray:
        return self.poses[i, 1:4]  # (x, y, yaw)

    def pose_full(self, i: int) -> np.ndarray:
        """Every column stored for frame ``i``.

        Four values for a legacy recording, fifteen for a simulated one (see
        ``sim_extract.POSE_COLUMNS``). Callers that need altitude or attitude
        must check the width rather than assume it.
        """
        return self.poses[i]

    @property
    def has_full_pose(self) -> bool:
        """True if the recording carries more than ``[t, x, y, yaw]``."""
        return self.poses.ndim == 2 and self.poses.shape[1] > 4

    def future_path_body(self, i: int, horizon: int, stride: int = 1) -> np.ndarray:
        """The flown-future world path expressed in frame ``i``'s body FLU frame.

        Returns ``(K, 2)`` ``[fwd, left]`` starting at the origin (the robot). Used
        as the seed the PF/ESDF corrector refines into the label.
        """
        x0, y0, yaw0 = self.pose(i)
        idx = range(i, min(i + horizon * stride + 1, self.num_frames), stride)
        pts = self.poses[list(idx), 1:3] - np.array([x0, y0])
        # rotate world offsets into the body FLU frame (fwd = +x, left = +y)
        fwd = pts[:, 0] * np.cos(yaw0) + pts[:, 1] * np.sin(yaw0)
        left = -pts[:, 0] * np.sin(yaw0) + pts[:, 1] * np.cos(yaw0)
        out = np.stack([fwd, left], axis=1).astype(np.float32)
        return out if out.shape[0] >= 2 else np.array([[0.0, 0.0], [0.1, 0.0]], np.float32)

    def goal_body(self, i: int, lookahead: int) -> Tuple[float, float]:
        """Auto-goal: the pose ``lookahead`` frames ahead, in body FLU (fwd, left)."""
        j = min(i + lookahead, self.num_frames - 1)
        x0, y0, yaw0 = self.pose(i)
        dx, dy = self.poses[j, 1] - x0, self.poses[j, 2] - y0
        fwd = dx * np.cos(yaw0) + dy * np.sin(yaw0)
        left = -dx * np.sin(yaw0) + dy * np.cos(yaw0)
        return float(fwd), float(left)


def load_recording(root) -> FlightRecording:
    """Load a recording directory into a :class:`FlightRecording`."""
    root = Path(root)
    intr = json.loads((root / "intrinsics.json").read_text())
    meta = json.loads((root / "meta.json").read_text())
    poses = np.load(root / "poses.npy").astype(np.float32)
    return FlightRecording(
        root=root,
        intrinsics=Intrinsics(width=intr["width"], height=intr["height"],
                              fx=intr["fx"], fy=intr["fy"], cx=intr["cx"], cy=intr["cy"]),
        poses=poses,
        rate_hz=float(meta.get("rate_hz", 10.0)),
        camera_height_m=float(meta.get("camera_height_m", 1.0)),
        pitch_deg=float(meta.get("pitch_deg", 0.0)),
        depth_scale_m=float(meta.get("depth_scale_m", 1.0)),
    )


def synthesize_recording(root, num_frames: int = 12, obstacle: bool = True) -> FlightRecording:
    """Write a tiny synthetic recording (for tests / schema documentation)."""
    root = Path(root)
    (root / "depth").mkdir(parents=True, exist_ok=True)
    h, w = 120, 160
    intr = {"width": w, "height": h, "fx": 120.0, "fy": 120.0, "cx": 80.0, "cy": 40.0}
    (root / "intrinsics.json").write_text(json.dumps(intr))
    (root / "meta.json").write_text(json.dumps(
        {"rate_hz": 10.0, "camera_height_m": 1.0, "pitch_deg": 0.0, "frames": num_frames}))
    # straight forward flight along +x
    t = np.arange(num_frames, dtype=np.float32)
    poses = np.stack([t * 0.1, t * 0.25, np.zeros_like(t), np.zeros_like(t)], axis=1)
    np.save(root / "poses.npy", poses.astype(np.float32))
    for i in range(num_frames):
        depth = np.full((h, w), 8.0, np.float32)
        if obstacle:
            depth[:, : w // 2] = 2.5   # near wall on the left half
        np.save(root / "depth" / f"{i:06d}.npy", depth)
    return load_recording(root)
